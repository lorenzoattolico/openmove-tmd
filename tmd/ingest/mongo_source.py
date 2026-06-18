"""
tmd.ingest.mongo_source — sorgente live MongoDB (GPS / IMU / labels) per il dump.

pymongo è confinato QUI (e in dump.py). Portato fedele da tmd/data/mongo_reader.py;
rimossi i pezzi morti nel path del dump: i loader diagnostici (load_imu all-users,
load_imu_timestamps_for_user) e la CACHE IMU (serviva solo al vecchio hack di
run_pipeline, eliminato allo Step 2 — il dump non l'ha mai usata).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING

from tmd.config import CityConfig

# PyMongoArrow (opzionale, ~16x su bulk reads). Fallback automatico se assente.
try:
    from pymongoarrow.api import aggregate_pandas_all
    _HAS_PMA = True
except ImportError:
    _HAS_PMA = False

# Le date del CLI/freeze sono in ora locale italiana. Trento → Europe/Rome.
# Conta soprattutto per le label: i motiontag started_at sono STRINGHE ISO in ora
# locale ('2026-05-21T10:44:33+02:00') e Mongo le confronta LESSICOGRAFICAMENTE →
# i bound della query vanno emessi nello STESSO fuso, altrimenti il confronto sballa
# di ~2h ai bordi (in CEST). (GPS/IMU usano epoch-ms interi → già corretti.)
LOCAL_TZ = "Europe/Rome"


# ── Client singleton + .env ───────────────────────────────────────────────────

_CLIENT: Optional[MongoClient] = None


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Carica KEY=VALUE da .env senza sovrascrivere variabili già impostate."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _make_client() -> MongoClient:
    _load_dotenv()
    uri = os.environ.get("MONGO_URI")
    if not uri:
        raise RuntimeError(
            "MONGO_URI non impostato. Crea .env nella project root "
            "(copia .env.example) o esporta MONGO_URI nell'ambiente."
        )
    return MongoClient(
        uri,
        maxPoolSize=10,
        maxIdleTimeMS=120_000,
        socketTimeoutMS=None,          # query bulk via port-forward durano minuti
        connectTimeoutMS=30_000,
        serverSelectionTimeoutMS=30_000,
    )


def _get_client() -> MongoClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _make_client()
    return _CLIENT


def _reset_client() -> None:
    global _CLIENT
    try:
        if _CLIENT is not None:
            _CLIENT.close()
    except Exception:
        pass
    _CLIENT = None


def _get_db(db_name: str):
    return _get_client()[db_name]


# ── Pipeline IMU + fallback cursor ────────────────────────────────────────────

_IMU_COLS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
_BATCH_SIZE = 10_000   # round-trip corti → meno timeout port-forward


def _imu_pipeline(uid: str, t0_ms: Optional[int], t1_ms: Optional[int]) -> list:
    match: dict = {"userId": uid}
    ts: dict = {}
    if t0_ms is not None: ts["$gte"] = t0_ms
    if t1_ms is not None: ts["$lte"] = t1_ms
    if ts: match["timestamp"] = ts
    return [
        {"$match": match},
        {"$project": {
            "_id": 0, "timestamp": 1,
            "acc_x": "$acceleration.x", "acc_y": "$acceleration.y",
            "acc_z": "$acceleration.z",
            "gyr_x": "$gyroscope.x", "gyr_y": "$gyroscope.y",
            "gyr_z": "$gyroscope.z",
        }},
    ]


def _cursor_to_imu_df(cursor) -> pd.DataFrame:
    ts_list, ax, ay, az, gx, gy, gz = [], [], [], [], [], [], []
    for doc in cursor:
        ts_list.append(doc["timestamp"])
        ax.append(doc["acc_x"]); ay.append(doc["acc_y"]); az.append(doc["acc_z"])
        gx.append(doc["gyr_x"]); gy.append(doc["gyr_y"]); gz.append(doc["gyr_z"])
    if not ts_list:
        return pd.DataFrame(columns=_IMU_COLS)
    return pd.DataFrame({
        "timestamp": np.array(ts_list, dtype=np.int64),
        "acc_x": np.array(ax, dtype=np.float32), "acc_y": np.array(ay, dtype=np.float32),
        "acc_z": np.array(az, dtype=np.float32), "gyr_x": np.array(gx, dtype=np.float32),
        "gyr_y": np.array(gy, dtype=np.float32), "gyr_z": np.array(gz, dtype=np.float32),
    })


def _cursor_fallback(cfg: CityConfig, uid: str, t0_ms, t1_ms,
                     max_retries: int = 3) -> pd.DataFrame:
    """Cursor PyMongo con retry interattivo su drop di connessione (port-forward)."""
    pipeline = _imu_pipeline(uid, t0_ms, t1_ms)
    hint = {"userId": ASCENDING, "timestamp": ASCENDING}
    for attempt in range(max_retries):
        try:
            collection = _get_db(cfg.db_name)[cfg.collections["imu"]]
            cursor = collection.aggregate(pipeline, allowDiskUse=False,
                                          batchSize=_BATCH_SIZE, hint=hint)
            return _cursor_to_imu_df(cursor)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [WARN] Connessione persa ({attempt+1}/{max_retries}): {e}")
                print("  [WARN] Riavvia il port-forward e premi Invio per riprovare...")
                input()
                _reset_client()
            else:
                raise
    return pd.DataFrame(columns=_IMU_COLS)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_gps(cfg: CityConfig, since_ms: Optional[int] = None,
             until_ms: Optional[int] = None) -> pd.DataFrame:
    """GPS expo: un documento = un fix."""
    query: dict = {}
    if since_ms: query.setdefault("timestamp", {})["$gte"] = since_ms
    if until_ms: query.setdefault("timestamp", {})["$lte"] = until_ms
    docs = list(
        _get_db(cfg.db_name)[cfg.collections["gps"]].find(
            query,
            {"_id": 0, "userId": 1, "timestamp": 1, "latitude": 1, "longitude": 1,
             "accuracy": 1, "speed": 1, "bearing": 1},
        ).batch_size(50_000)
    )
    if not docs:
        return pd.DataFrame(columns=["userId", "timestamp", "latitude", "longitude",
                                     "accuracy", "speed", "bearing"])
    df = pd.DataFrame(docs)
    for col in ("accuracy", "speed", "bearing"):
        if col not in df.columns:
            df[col] = float("nan")
    return df.astype({"timestamp": "int64"})


def get_imu_user_ids(cfg: CityConfig, since_ms: Optional[int] = None,
                     until_ms: Optional[int] = None) -> set:
    """userId dalla collezione IMU via .distinct() — non carica record."""
    query: dict = {}
    if since_ms: query.setdefault("timestamp", {})["$gte"] = since_ms
    if until_ms: query.setdefault("timestamp", {})["$lte"] = until_ms
    return set(_get_db(cfg.db_name)[cfg.collections["imu"]].distinct("userId", query))


def load_imu_for_user(cfg: CityConfig, uid: str, t0_ms: Optional[int] = None,
                      t1_ms: Optional[int] = None) -> pd.DataFrame:
    """IMU completo (acc+gyr) per un utente in [t0_ms, t1_ms]. PyMongoArrow + fallback."""
    pipeline = _imu_pipeline(uid, t0_ms, t1_ms)
    collection = _get_db(cfg.db_name)[cfg.collections["imu"]]
    hint = {"userId": ASCENDING, "timestamp": ASCENDING}

    if _HAS_PMA:
        try:
            df = aggregate_pandas_all(collection, pipeline, hint=hint, allowDiskUse=False)
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(np.int64)
            for c in ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"):
                if c in df.columns:
                    df[c] = df[c].astype(np.float32)
        except Exception as e:
            print(f"  [WARN] PyMongoArrow fallback per uid={uid[:8]}: {e}")
            _reset_client()
            df = _cursor_fallback(cfg, uid, t0_ms, t1_ms)
    else:
        df = _cursor_fallback(cfg, uid, t0_ms, t1_ms)
    return df


def load_labels(cfg: CityConfig, since_ms: Optional[int] = None,
                until_ms: Optional[int] = None) -> pd.DataFrame:
    query: dict = {}
    ts_filter: dict = {}
    # started_at è stringa ISO in ora locale ('...+02:00') confrontata come stringa:
    # i bound vanno nello STESSO fuso (Europe/Rome), non in UTC (vedi LOCAL_TZ).
    if since_ms: ts_filter["$gte"] = pd.Timestamp(since_ms, unit="ms", tz="UTC").tz_convert(LOCAL_TZ).isoformat()
    if until_ms: ts_filter["$lte"] = pd.Timestamp(until_ms, unit="ms", tz="UTC").tz_convert(LOCAL_TZ).isoformat()
    if ts_filter:
        query["started_at"] = ts_filter

    docs = list(_get_db(cfg.db_name)[cfg.collections["labels"]].find(
        query, {"_id": 0, "userId": 1, "started_at": 1, "attributes": 1}
    ))
    if not docs:
        return pd.DataFrame()

    rows = []
    for d in docs:
        attr = d.get("attributes", {})
        mode = attr.get("mode_key")
        if mode:
            mode_tmd = cfg.label_map.get(mode)
            if not mode_tmd:
                continue
        elif attr.get("purpose_key"):
            mode_tmd = "Still"
            mode = f"stay_{attr['purpose_key']}"
        else:
            continue
        try:
            t0 = int(pd.Timestamp(attr["started_at"]).timestamp() * 1000)
            t1 = int(pd.Timestamp(attr["finished_at"]).timestamp() * 1000)
        except Exception:
            continue
        rows.append({
            "userId": d["userId"], "started_at_ms": t0, "finished_at_ms": t1,
            "mode_tmd": mode_tmd, "mode_key": mode,
            "detected_mode_key": attr.get("detected_mode_key"),
        })
    return pd.DataFrame(rows)
