"""
tmd.cli.run_pipeline — orchestratore one-shot (Step 5).

raw (data/raw_freeze) → sessioni R1 → finestre + feature A/B/C/D → GT motiontag → parquet.
Niente incrementale/stale: processa tutto una volta (adatto al freeze). La logica
sta nella libreria (tmd.datasets/sessions/features); qui solo il cablaggio.

Uso:
    python -m tmd.cli.run_pipeline --city trento [--groups A,B,C,D] [--out PATH]
                                    [--max-users N]   # N = solo per test rapidi
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.config import CityConfig
from tmd.datasets import OpenMoveRaw
from tmd.sessions import build_sessions_for_user
from tmd.features import extract_session
from tmd.features.quality import filter_all
from tmd.features.gt_labels import assign_labels_for_session

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(city: str = "trento", groups=("A", "B", "C", "D"),
        raw_dir=None, out=None, win_s: float = 120.0, step_s: float = 60.0,
        max_users: int | None = None, all_features: bool = False):
    cfg = CityConfig.from_yaml(PROJECT_ROOT / "tmd" / "configs" / "cities" / f"{city}.yaml")
    raw = OpenMoveRaw(raw_dir or (PROJECT_ROOT / "data" / "raw_freeze"))
    out = Path(out) if out else PROJECT_ROOT / "data" / "v2" / f"features_{city}.parquet"

    gps_all = raw.gps()
    labels  = raw.labels()
    idx_path = PROJECT_ROOT / "data" / "processed" / f"spatial_index_{city}.pkl"
    idx_spatial = pickle.load(open(idx_path, "rb")) if idx_path.exists() else {}

    gap_s = cfg.session["gap_s"]
    min_s = cfg.session["min_duration_s"]
    users = sorted(set(gps_all["userId"].unique()) | set(raw.imu_user_ids()))
    if max_users:
        users = users[:max_users]

    records, gps_records = [], []
    for uid in users:
        df_gps_u = gps_all[gps_all["userId"] == uid].reset_index(drop=True)
        df_gps_u, _ = filter_all(df_gps_u, pd.DataFrame(), cfg)     # filtro accuracy GPS
        df_imu_u = raw.imu(uid)
        ts_gps = df_gps_u["timestamp"].to_numpy(np.float64)
        ts_imu = (df_imu_u["timestamp"].to_numpy(np.float64)
                  if not df_imu_u.empty else np.array([]))

        for s in build_sessions_for_user(uid, ts_gps, ts_imu, gap_s, min_s):
            if s["type"] == "gps_only":     # filter_sessions: senza IMU niente feature
                continue
            df_feat = extract_session(
                df_imu_u, df_gps_u, s["t0_ms"], s["t1_ms"], idx_spatial,
                source="openmove", groups=list(groups),
                fs=s.get("imu_hz", 50.0), win_s=win_s, step_s=step_s,
                apply_drop_list=not all_features,
                meta={"session_id": s["session_id"], "userId": uid,
                      "sess_type": s["type"], "imu_hz": round(s.get("imu_hz", 50.0), 1),
                      "city": cfg.city},
            )
            if df_feat.empty:
                continue
            df_feat["label"] = assign_labels_for_session(df_feat, labels, uid)
            df_feat["label_source"] = df_feat["label"].apply(
                lambda x: "motiontag" if pd.notna(x) else None)
            records.append(df_feat)
            if s["n_gps"] > 0:   # GPS della sessione (taggato session_id) per il labeling infra
                gw = df_gps_u[(df_gps_u["timestamp"] >= s["t0_ms"]) &
                              (df_gps_u["timestamp"] <= s["t1_ms"])]
                if not gw.empty:
                    gps_records.append(
                        gw[["userId", "timestamp", "latitude", "longitude"]]
                        .assign(session_id=s["session_id"]))

    df_out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out, index=False)
    if gps_records:
        pd.concat(gps_records, ignore_index=True).to_parquet(
            out.parent / f"gps_sessions_{city}.parquet", index=False)
    return df_out, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="trento")
    ap.add_argument("--groups", default="A,B,C,D")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--win-s", type=float, default=120.0)
    ap.add_argument("--step-s", type=float, default=60.0)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--all-features", action="store_true",
                    help="non applicare DROP_LIST: estrae TUTTE le feature (variable-selection su OpenMove)")
    a = ap.parse_args()
    df, out = run(a.city, tuple(a.groups.split(",")), a.raw_dir, a.out,
                  a.win_s, a.step_s, a.max_users, a.all_features)
    print(f"{len(df):,} finestre × {df.shape[1]} col → {out}")
    if "sess_type" in df.columns:
        print("per sess_type:", df["sess_type"].value_counts().to_dict())
    if "label" in df.columns:
        print(f"con GT label: {df['label'].notna().sum():,} ({100*df['label'].notna().mean():.0f}%)")


if __name__ == "__main__":
    main()
