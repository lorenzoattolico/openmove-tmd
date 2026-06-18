"""
e2_imu_raw.py — EDA raw IMU sul freeze (Fase 1a · task E2).

Scopo:    caratterizzare l'IMU GREZZO del freeze: sampling rate (Hz) per utente,
          ETEROGENEITA' DEVICE via ampiezza acc a riposo (iOS in g ~1.0 vs Android in m/s^2 ~9.81)
          -> sanity della normalizzazione (tmd quality.normalize_and_filter_imu), copertura/utente.
Input:    data/raw_freeze/imu/<userId>.parquet   (1 file = 1 utente; 111.7M righe tot; offline)
Output:   research/figures/e2_imu_device.{png,pdf}   (split iOS/Android)
          research/figures/e2_imu_hz.{png,pdf}       (sampling rate per utente)
          research/figures/e2_imu_peruser.{png,pdf}  (n campioni per utente)
Alimenta: thesis/eda.md §1 (E2)
Sez.tesi: 3.1 Deployment (schema sensori, eterogeneita' device) / 4.x normalizzazione

Memory-safe: processa 1 file (utente) alla volta, tiene solo scalari per-utente.
Run: /opt/miniconda3/envs/tmd/bin/python research/e2_imu_raw.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMU_DIR = ROOT / "data" / "raw_freeze" / "imu"
FIG = ROOT / "thesis" / "figures"
G_MS2 = 9.80665          # 1 g in m/s^2
SPLIT = 5.0              # soglia |acc| mediana: <SPLIT => iOS (g), >=SPLIT => Android (m/s^2)


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def per_user_stats() -> pd.DataFrame:
    files = sorted(IMU_DIR.glob("*.parquet"))
    if not files:
        sys.exit(f"Nessun IMU in {IMU_DIR} — eseguito il freeze?")
    rows = []
    for f in files:
        df = pd.read_parquet(f, columns=["timestamp", "acc_x", "acc_y", "acc_z",
                                         "gyr_x", "gyr_y", "gyr_z"])
        ts = np.sort(df["timestamp"].to_numpy(np.int64))
        dt = np.diff(ts) / 1000.0
        dt = dt[(dt > 0) & (dt < 1.0)]                     # intervalli di campionamento (no gap sessione)
        hz = (1.0 / np.median(dt)) if dt.size else np.nan
        accmag = np.sqrt(df.acc_x.to_numpy(np.float64) ** 2
                         + df.acc_y.to_numpy(np.float64) ** 2
                         + df.acc_z.to_numpy(np.float64) ** 2)
        gyrmag = np.sqrt(df.gyr_x.to_numpy(np.float64) ** 2
                         + df.gyr_y.to_numpy(np.float64) ** 2
                         + df.gyr_z.to_numpy(np.float64) ** 2)
        rows.append({
            "user": f.stem[:8],
            "n": len(df),
            "dur_h": (ts[-1] - ts[0]) / 3.6e6 if len(ts) > 1 else 0.0,
            "hz": hz,
            "accmag_med": float(np.median(accmag)),
            "gyrmag_med": float(np.median(gyrmag)),
        })
        del df, accmag, gyrmag
    return pd.DataFrame(rows)


def main() -> None:
    s = per_user_stats()
    s["device"] = np.where(s.accmag_med < SPLIT, "iOS (g)", "Android (m/s²)")
    n_ios = int((s.device == "iOS (g)").sum())
    n_and = int((s.device == "Android (m/s²)").sum())

    # ── riepilogo numerico (→ thesis/eda.md) ──────────────────────────────────
    print("=" * 64)
    print("E2 — EDA RAW IMU (freeze)")
    print("=" * 64)
    print(f"utenti IMU: {len(s)} | campioni totali: {s.n.sum():,}")
    print(f"sampling rate Hz  mediana {s.hz.median():.0f}  range [{s.hz.min():.0f}, {s.hz.max():.0f}]")
    print(f"durata IMU/utente (h)  mediana {s.dur_h.median():.1f}  p90 {s.dur_h.quantile(.9):.1f}  max {s.dur_h.max():.1f}")
    print(f"campioni/utente  mediana {s.n.median():,.0f}  p10 {s.n.quantile(.1):,.0f}  max {s.n.max():,}")
    print(f"DEVICE split (|acc| mediana):  iOS≈1g: {n_ios}  |  Android≈9.81 m/s²: {n_and}")
    print(f"  |acc| mediana iOS:     {s.loc[s.device=='iOS (g)','accmag_med'].median():.2f} (atteso ~1.0)")
    print(f"  |acc| mediana Android: {s.loc[s.device=='Android (m/s²)','accmag_med'].median():.2f} (atteso ~9.81)")
    print(f"  Hz mediana iOS {s.loc[s.device=='iOS (g)','hz'].median():.0f} | Android {s.loc[s.device=='Android (m/s²)','hz'].median():.0f}")

    # ── figures (LABELS IN ENGLISH) ───────────────────────────────────────────
    # 1) device split via resting |acc|
    plt.figure(figsize=(6.5, 4))
    plt.hist(s.accmag_med, bins=40)
    plt.axvline(1.0, ls="--", c="tab:green", label="1 g (iOS)")
    plt.axvline(G_MS2, ls="--", c="tab:orange", label="9.81 m/s² (Android)")
    plt.xlabel("per-user median |acceleration| (raw units)"); plt.ylabel("users")
    plt.title("IMU device heterogeneity (raw acc units)"); plt.legend(); plt.grid(alpha=.3)
    savefig("e2_imu_device")

    # 2) sampling rate per user
    plt.figure(figsize=(6.5, 4))
    plt.hist(s.hz.dropna(), bins=30)
    plt.xlabel("sampling rate (Hz)"); plt.ylabel("users")
    plt.title("IMU sampling rate per user"); plt.grid(alpha=.3)
    savefig("e2_imu_hz")

    # 3) samples per user (heterogeneity, log)
    pu = s.sort_values("n", ascending=False)
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(pu)), pu.n.values)
    plt.yscale("log")
    plt.xlabel("user (sorted by #samples)"); plt.ylabel("IMU samples (freeze window)")
    plt.title("IMU samples per user (heterogeneity)"); plt.grid(alpha=.3, axis="y")
    savefig("e2_imu_peruser")

    print(f"\nfigure → {FIG.relative_to(ROOT)}/e2_imu_*.png|pdf")


if __name__ == "__main__":
    main()
