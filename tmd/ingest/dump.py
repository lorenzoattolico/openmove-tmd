"""
tmd.ingest.dump — esporta GPS/IMU/labels da MongoDB a data/raw/ (Parquet).

Differenze rispetto al vecchio scripts/dump_raw_data.py (tutto il resto è fedele:
schemi, dedup, partizione mese, stato crash-safe):
  * GPS: cursore PER-UTENTE (era globale = bug latenza), query dal MIN dei cursori.
  * OVERLAP 1h → 2 giorni (copre la latenza ≤ 1 giorno).
  * --since <data>  → floor INCLUSIVO (da 00:00 del giorno).
  * --until <data>  → tetto INCLUSIVO: una data nuda copre TUTTO il giorno (fino a
                      23:59:59.999) → snapshot deterministico per il freeze. Con orario
                      esplicito ('YYYY-MM-DD HH:MM') usa l'istante esatto.
  * --dry-run       → stampa i range che interrogherebbe, SENZA toccare Mongo.
  * --raw-dir       → directory output (default data/raw; sandbox per i test).

  Date interpretate in ORA LOCALE ITALIANA (Europe/Rome): il deployment è a Trento,
  quindi "19 mag–8 giu" sono giorni di calendario italiani. I confini vengono
  convertiti in epoch-ms (UTC) per le query. Il log mostra l'ora italiana.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tmd.config import CityConfig
from tmd.ingest.mongo_source import (
    load_gps, load_imu_for_user, load_labels, get_imu_user_ids, LOCAL_TZ,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

OVERLAP_MS = 2 * 86_400_000   # 2 giorni — copre la latenza ≤ 1 giorno
DAY_MS     = 86_400_000        # LOCAL_TZ (ora italiana) importato da mongo_source


# ── Schemi PyArrow (fedeli) ───────────────────────────────────────────────────

_GPS_SCHEMA = pa.schema([
    pa.field("userId",    pa.dictionary(pa.int16(), pa.string())),
    pa.field("timestamp", pa.int64()),
    pa.field("latitude",  pa.float32()),
    pa.field("longitude", pa.float32()),
    pa.field("accuracy",  pa.float32()),
    pa.field("speed",     pa.float32()),
    pa.field("bearing",   pa.float32()),
])
_IMU_SCHEMA = pa.schema([
    pa.field("timestamp", pa.int64()),
    pa.field("acc_x", pa.float32()), pa.field("acc_y", pa.float32()),
    pa.field("acc_z", pa.float32()), pa.field("gyr_x", pa.float32()),
    pa.field("gyr_y", pa.float32()), pa.field("gyr_z", pa.float32()),
])
_LABEL_SCHEMA = pa.schema([
    pa.field("userId",      pa.string()),
    pa.field("started_at",  pa.int64()),
    pa.field("finished_at", pa.int64()),
    pa.field("mode_tmd",    pa.string()),
])


# ── Logica cursore (PURA — testabile senza Mongo) ─────────────────────────────

def _gps_since_ms(cursors: dict, full: bool, overlap_ms: int,
                  since_floor_ms: int | None) -> int | None:
    """GPS: riparte dal MINIMO dei cursori per-utente (mai dal max globale → no bug)."""
    if full or not cursors:
        return since_floor_ms
    since = min(cursors.values()) - overlap_ms
    if since_floor_ms is not None:
        since = max(since, since_floor_ms)
    return since


def _imu_since_ms(last_ms: int | None, full: bool, overlap_ms: int,
                  existing_max: int | None, since_floor_ms: int | None) -> int | None:
    """IMU: cursore per-utente (− overlap), con fallback al max del file esistente."""
    if not full and last_ms:
        since = last_ms - overlap_ms
    else:
        since = (existing_max - overlap_ms) if (existing_max and not full) else None
    if since_floor_ms is not None:
        since = max(since, since_floor_ms) if since else since_floor_ms
    return since


# ── Stato incrementale ────────────────────────────────────────────────────────

def _state_file(raw_dir: Path) -> Path:
    return raw_dir / ".dump_state.json"


def load_state(raw_dir: Path) -> dict:
    p = _state_file(raw_dir)
    if p.exists():
        st = json.loads(p.read_text())
        # backward-compat: gps_last_ms scalare/None → dict per-utente
        if not isinstance(st.get("gps_last_ms"), dict):
            st["gps_last_ms"] = {}
        return st
    return {"gps_last_ms": {}, "imu_last_ms": {}, "labels_lookback_days": 60}


def save_state(raw_dir: Path, state: dict) -> None:
    f = _state_file(raw_dir)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state, indent=2))


# ── Helpers Parquet (fedeli) ──────────────────────────────────────────────────

def _df_to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    cols = {}
    for field in schema:
        if field.name not in df.columns:
            cols[field.name] = pa.array([None] * len(df), type=field.type)
        else:
            series = df[field.name]
            try:
                cols[field.name] = pa.array(series.values, type=field.type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                cols[field.name] = pa.array(series.tolist(), type=field.type)
    return pa.table(cols, schema=schema)


def _read_existing_max_ts(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        meta = pq.read_metadata(path)
        max_ts = None
        for rg_idx in range(meta.num_row_groups):
            rg = meta.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                col = rg.column(col_idx)
                if col.path_in_schema == "timestamp" and col.statistics:
                    if col.statistics.max is not None:
                        v = col.statistics.max
                        max_ts = v if max_ts is None else max(max_ts, v)
        return int(max_ts) if max_ts is not None else None
    except Exception:
        return None


def _append_and_dedup(existing_path: Path, new_df: pd.DataFrame, schema: pa.Schema,
                      dedup_cols: list[str], sort_col: str = "timestamp") -> int:
    if new_df.empty:
        return pq.read_metadata(existing_path).num_rows if existing_path.exists() else 0
    new_table = _df_to_table(new_df, schema)
    if existing_path.exists():
        existing_table = pq.read_table(existing_path)
        schema_cols = {f.name for f in schema}
        extra = [c for c in existing_table.schema.names if c not in schema_cols]
        if extra:
            existing_table = existing_table.drop(extra)
        combined = pa.concat_tables([existing_table, new_table], promote_options="default")
    else:
        combined = new_table
    df_combined = combined.to_pandas()
    df_combined = (df_combined.drop_duplicates(subset=dedup_cols, keep="last")
                   .sort_values(sort_col, kind="stable").reset_index(drop=True))
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_df_to_table(df_combined, schema), existing_path,
                   row_group_size=50_000, compression="snappy")
    return len(df_combined)


def _fmt_local(ms: int | None, default: str) -> str:
    """Formatta un epoch-ms in ora locale italiana (per il log)."""
    if not ms:
        return default
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")


def _log_range(tag: str, since_ms: int | None, until_ms: int | None) -> None:
    s = _fmt_local(since_ms, "(tutto)")
    u = _fmt_local(until_ms, "(now)")
    log.info(f"{tag}: da {s} a {u} (ora IT)")


# ── Dump per canale ───────────────────────────────────────────────────────────

def dump_gps(cfg, state, raw_dir, full, since_floor_ms=None, until_ms=None,
             dry_run=False) -> dict:
    cursors = dict(state.get("gps_last_ms") or {})        # {uid: ms} per-utente
    since_ms = _gps_since_ms(cursors, full, OVERLAP_MS, since_floor_ms)
    _log_range("GPS", since_ms, until_ms)
    if dry_run:
        return state

    df_gps = load_gps(cfg, since_ms, until_ms)
    log.info(f"GPS: {len(df_gps):,} fix")
    if df_gps.empty:
        return state

    gps_dir = raw_dir / "gps"
    df_gps["year_month"] = pd.to_datetime(df_gps["timestamp"], unit="ms").dt.strftime("%Y-%m")
    for ym, grp in df_gps.groupby("year_month"):
        part = gps_dir / f"year_month={ym}" / "part.parquet"
        n = _append_and_dedup(part, grp.drop(columns=["year_month"]), _GPS_SCHEMA,
                              dedup_cols=["userId", "timestamp"])
        log.info(f"  GPS {ym}: {n:,} righe totali")
    for uid, g in df_gps.groupby("userId"):
        cursors[str(uid)] = max(cursors.get(str(uid), 0), int(g["timestamp"].max()))
    state["gps_last_ms"] = cursors
    return state


def dump_imu(cfg, state, raw_dir, full, since_floor_ms=None, until_ms=None,
             dry_run=False) -> dict:
    imu_dir = raw_dir / "imu"
    now_ms = until_ms if until_ms is not None else int(pd.Timestamp.now().timestamp() * 1000)
    imu_last_ms = dict(state.get("imu_last_ms") or {})

    if dry_run:
        # offline: utenti da stato + file locali, niente Mongo
        user_ids = set(imu_last_ms)
        if imu_dir.exists():
            user_ids |= {p.stem for p in imu_dir.glob("*.parquet")}
    else:
        user_ids = get_imu_user_ids(cfg)
    log.info(f"IMU: {len(user_ids)} utenti")

    for uid in sorted(user_ids):
        uid_path = imu_dir / f"{uid}.parquet"
        existing_max = _read_existing_max_ts(uid_path) if not full else None
        since_uid = _imu_since_ms(imu_last_ms.get(uid), full, OVERLAP_MS,
                                  existing_max, since_floor_ms)
        _log_range(f"IMU {uid[:12]}", since_uid, now_ms)
        if dry_run:
            continue
        try:
            df_imu = load_imu_for_user(cfg, uid, since_uid, now_ms)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.warning(f"  IMU {uid[:12]}: errore ({type(e).__name__}: {e}) — skip")
            continue
        if df_imu.empty:
            continue
        try:
            n = _append_and_dedup(uid_path, df_imu, _IMU_SCHEMA, dedup_cols=["timestamp"])
        except Exception as e:
            log.warning(f"  IMU {uid[:12]}: errore scrittura ({e}) — skip")
            continue
        imu_last_ms[uid] = int(df_imu["timestamp"].max())
        log.info(f"  IMU {uid[:12]}: +{len(df_imu):,} nuovi  totale={n:,}")

    state["imu_last_ms"] = imu_last_ms
    return state


def dump_labels(cfg, state, raw_dir, lookback_days, since_floor_ms=None,
                until_ms=None, dry_run=False) -> dict:
    now_ms = until_ms if until_ms is not None else int(pd.Timestamp.now().timestamp() * 1000)
    since_ms = now_ms - int(lookback_days) * DAY_MS
    if since_floor_ms is not None:
        since_ms = max(since_ms, since_floor_ms)
    _log_range("Labels", since_ms, until_ms)
    if dry_run:
        return state

    df_new = load_labels(cfg, since_ms, until_ms)
    log.info(f"Labels: {len(df_new):,} segmenti")
    if df_new.empty:
        return state
    df_new = df_new.rename(columns={
        "started_at_ms": "started_at", "finished_at_ms": "finished_at",
        "mode": "mode_tmd", "label": "mode_tmd",
    })
    for col in ["userId", "started_at", "finished_at", "mode_tmd"]:
        if col not in df_new.columns:
            log.warning(f"  Labels: colonna '{col}' mancante — skip")
            return state
    labels_path = raw_dir / "labels.parquet"
    n = _append_and_dedup(labels_path, df_new, _LABEL_SCHEMA,
                          dedup_cols=["userId", "started_at"], sort_col="started_at")
    log.info(f"Labels: {n:,} segmenti totali")
    state["labels_lookback_days"] = lookback_days
    return state


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mongo → data/raw/ (Parquet)")
    p.add_argument("--city", default="trento")
    p.add_argument("--since", default=None,
                   help="Floor INCLUSIVO (YYYY-MM-DD), ora italiana: non scaricare "
                        "prima di 00:00 (Europe/Rome) del giorno.")
    p.add_argument("--until", default=None,
                   help="Tetto INCLUSIVO (YYYY-MM-DD), ora italiana: copre tutto il "
                        "giorno (fino a 23:59:59.999 Europe/Rome). Snapshot "
                        "deterministico per il freeze.")
    p.add_argument("--full", action="store_true", help="Ri-scarica tutto (ignora cursori).")
    p.add_argument("--only", default=None, choices=["gps", "imu", "labels"])
    p.add_argument("--labels-lookback", type=int, default=60)
    p.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"),
                   help="Directory output (default data/raw; usa una sandbox per i test).")
    p.add_argument("--dry-run", action="store_true",
                   help="Stampa i range che interrogherebbe, senza toccare Mongo.")
    return p.parse_args()


def _to_ms(date_str: str | None) -> int | None:
    """Floor INCLUSIVO in ORA LOCALE ITALIANA (Europe/Rome). Una data nuda
    'YYYY-MM-DD' = 00:00 italiane di quel giorno; con orario esplicito usa
    quell'istante (ora italiana)."""
    if not date_str:
        return None
    return int(pd.Timestamp(date_str, tz=LOCAL_TZ).timestamp() * 1000)


def _until_to_ms(date_str: str | None) -> int | None:
    """Tetto temporale INCLUSIVO in ORA LOCALE ITALIANA (Europe/Rome). Una data
    nuda 'YYYY-MM-DD' copre tutto il giorno italiano (fino a 23:59:59.999, perché
    la query usa $lte); con un orario esplicito usa l'istante esatto.
    Il calcolo "fine giornata" localizza la mezzanotte del giorno dopo e sottrae
    1 ms → robusto anche su cambi DST."""
    if not date_str:
        return None
    if ":" not in date_str:                      # data nuda → fine giornata italiana inclusa
        next_midnight = (pd.Timestamp(date_str) + pd.Timedelta(days=1)).normalize()
        t = pd.Timestamp(next_midnight, tz=LOCAL_TZ) - pd.Timedelta(milliseconds=1)
    else:
        t = pd.Timestamp(date_str, tz=LOCAL_TZ)
    return int(t.timestamp() * 1000)


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / "tmd" / "configs" / "cities" / f"{args.city}.yaml"
    if not cfg_path.exists():
        log.error(f"Config non trovato: {cfg_path}")
        sys.exit(1)
    cfg = CityConfig.from_yaml(cfg_path)

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(raw_dir)

    since_floor_ms = _to_ms(args.since)
    until_ms = _until_to_ms(args.until)
    if args.full and args.only != "labels":
        state["gps_last_ms"] = {}
        state["imu_last_ms"] = {}

    run = {c: args.only in (None, c) for c in ("gps", "imu", "labels")}
    t0 = time.time()

    if run["gps"]:
        log.info("\n── GPS ──")
        state = dump_gps(cfg, state, raw_dir, args.full, since_floor_ms, until_ms, args.dry_run)
        if not args.dry_run:
            save_state(raw_dir, state)
    if run["imu"]:
        log.info("\n── IMU ──")
        state = dump_imu(cfg, state, raw_dir, args.full, since_floor_ms, until_ms, args.dry_run)
        if not args.dry_run:
            save_state(raw_dir, state)
    if run["labels"]:
        log.info("\n── Labels ──")
        state = dump_labels(cfg, state, raw_dir, args.labels_lookback,
                            since_floor_ms, until_ms, args.dry_run)
        if not args.dry_run:
            save_state(raw_dir, state)

    log.info(f"\nFatto in {time.time()-t0:.0f}s → {raw_dir}"
             + (" (DRY-RUN: niente scritto)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
