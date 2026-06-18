"""
scripts/load_geolife.py — GeoLife (Microsoft Research Asia) → features_geolife.parquet.

GeoLife è GPS-only (niente IMU, niente accuracy). Costruisce finestre 120s e calcola
le feature **gruppo B** (cinematica GPS) RIUSANDO `tmd.features.gps_kinematic.compute`
→ parità con Trento/SHL garantita. Niente A (IMU assente), niente C (no GTFS/OSM Pechino),
niente D (no accuracy). Serve allo Stadio 1 del transfer N=3 (zero-shot Trento/SHL → GeoLife).

Label: walk→Walk, car/taxi→Car, bus→Bus, train/subway→Train (Bike e airplane/boat/run/
motorcycle scartati). **Niente Still** (GeoLife non etichetta il fermo) → eval su 4 classi.

PLT: 6 righe header da saltare; col0=lat, col1=lon, col4=giorni-dal-1899 (GMT).
labels.txt: StartTime\tEndTime\tMode (GMT).

Da project root: /opt/miniconda3/envs/tmd/bin/python scripts/load_geolife.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from tmd.features import gps_kinematic as gk  # noqa: E402

GEOLIFE = PROJECT_ROOT / "data" / "external" / "raw" / "geolife" / "Data"
OUT = PROJECT_ROOT / "data" / "processed" / "features_geolife.parquet"
WIN_MS = 120_000          # finestra 120s (tumbling, dentro un singolo segmento)
MIN_PTS = 3
DAYS_1899_TO_1970 = 25569  # giorni da 1899-12-30 a 1970-01-01

MODE_MAP = {"walk": "Walk", "car": "Car", "taxi": "Car",
            "bus": "Bus", "train": "Train", "subway": "Train"}


def read_user_points(user_dir: Path) -> np.ndarray | None:
    """Tutti i punti GPS dell'utente: array (N,3) = [ts_ms, lat, lon], ordinato per ts."""
    plts = sorted((user_dir / "Trajectory").glob("*.plt"))
    if not plts:
        return None
    frames = []
    for p in plts:
        try:
            d = pd.read_csv(p, skiprows=6, header=None, usecols=[0, 1, 4],
                            names=["lat", "lon", "days"], engine="c")
            frames.append(d)
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    ts = ((df["days"].to_numpy() - DAYS_1899_TO_1970) * 86_400_000.0).round().astype(np.int64)
    arr = np.column_stack([ts, df["lat"].to_numpy(), df["lon"].to_numpy()])
    return arr[np.argsort(arr[:, 0])]


def read_labels(user_dir: Path) -> list[tuple[int, int, str]]:
    f = user_dir / "labels.txt"
    if not f.exists():
        return []
    lab = pd.read_csv(f, sep="\t", skiprows=1, header=None,
                      names=["start", "end", "mode"])
    out = []
    for _, r in lab.iterrows():
        cls = MODE_MAP.get(str(r["mode"]).strip().lower())
        if cls is None:
            continue
        t0 = pd.to_datetime(r["start"], format="%Y/%m/%d %H:%M:%S").value // 1_000_000
        t1 = pd.to_datetime(r["end"],   format="%Y/%m/%d %H:%M:%S").value // 1_000_000
        out.append((int(t0), int(t1), cls))
    return out


def main():
    users = sorted([d for d in GEOLIFE.iterdir() if d.is_dir() and (d / "labels.txt").exists()])
    print(f"Utenti con labels: {len(users)}")
    rows = []
    for ui, ud in enumerate(users):
        labels = read_labels(ud)
        if not labels:
            continue
        pts = read_user_points(ud)
        if pts is None or len(pts) < MIN_PTS:
            continue
        ts_all = pts[:, 0]
        seg = 0
        for (t0, t1, cls) in labels:
            lo = int(np.searchsorted(ts_all, t0, "left"))
            hi = int(np.searchsorted(ts_all, t1, "right"))
            if hi - lo < MIN_PTS:
                continue
            sub = pts[lo:hi]            # punti del segmento (ts ordinato)
            seg += 1
            # finestre tumbling 120s dentro il segmento
            w0 = sub[0, 0]
            while w0 + WIN_MS <= sub[-1, 0] + WIN_MS:   # include ultima parziale
                wlo = int(np.searchsorted(sub[:, 0], w0, "left"))
                whi = int(np.searchsorted(sub[:, 0], w0 + WIN_MS, "left"))
                w0 += WIN_MS
                if whi - wlo < MIN_PTS:
                    continue
                win = sub[wlo:whi]
                feats = gk.compute(win[:, 1], win[:, 2], win[:, 0])  # lat, lon, ts_ms
                feats["label"] = cls
                feats["userId"] = f"geolife_{ud.name}"
                feats["session_id"] = f"geolife_{ud.name}_s{seg}"
                feats["ts_start"] = int(win[0, 0])
                feats["ts_end"] = int(win[-1, 0])
                feats["gps_frac"] = (whi - wlo) / (WIN_MS / 1000.0)
                feats["city"] = "geolife"
                feats["split"] = "validate"
                rows.append(feats)
        if (ui + 1) % 15 == 0:
            print(f"  ...{ui+1}/{len(users)} utenti, {len(rows)} finestre")

    df = pd.DataFrame(rows)
    # colonne C/D = NaN (assenti) per compatibilità con labeler/eval che le cercano
    for c in ["C_osm_rail_prop", "C_bus_stops_prop", "D_gap_fraction", "D_has_reliable_gps"]:
        if c not in df.columns:
            df[c] = np.nan
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n── features_geolife.parquet ──")
    print(f"  {len(df):,} finestre · {df['userId'].nunique()} utenti · {df['session_id'].nunique()} segmenti")
    print("  label:"); print(df["label"].value_counts().to_string())
    print("\n  sanity — B_speed_mean (m/s) per classe (atteso Walk<Bus≈Car<Train):")
    print(df.groupby("label")["B_speed_mean"].median().round(2).to_string())
    print(f"\nSalvato: {OUT}")


if __name__ == "__main__":
    main()
