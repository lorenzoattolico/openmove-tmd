"""
e1_gps_raw.py — EDA raw GPS sul freeze (Fase 1a · task E1).

Scopo:    caratterizzare il segnale GPS GREZZO del freeze: volume/copertura,
          sampling rate, accuracy, struttura dei gap (duty-cycling), eterogeneita' per-utente.
Input:    data/raw_freeze/gps/year_month=*/part.parquet   (offline, no Mongo)
Output:   research/figures/e1_gps_accuracy.{png,pdf}
          research/figures/e1_gps_interval.{png,pdf}
          research/figures/e1_gps_peruser.{png,pdf}
Alimenta: thesis/eda.md §1 (E1)
Sez.tesi: 3.1 Deployment OpenMove / 3.2 guasto GPS (caratterizzazione raw)

Run: /opt/miniconda3/envs/tmd/bin/python research/e1_gps_raw.py
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
GPS_DIR = ROOT / "data" / "raw_freeze" / "gps"
FIG = ROOT / "thesis" / "figures"
TZ = "Europe/Rome"


def load_gps() -> pd.DataFrame:
    parts = sorted(GPS_DIR.glob("year_month=*/part.parquet"))
    if not parts:
        sys.exit(f"Nessun GPS in {GPS_DIR} — eseguito il freeze?")
    df = pd.concat(
        [pd.read_parquet(p, columns=["userId", "timestamp", "accuracy", "speed", "bearing"])
         for p in parts],
        ignore_index=True,
    )
    df["userId"] = df["userId"].astype(str)
    return df.sort_values(["userId", "timestamp"]).reset_index(drop=True)


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main() -> None:
    df = load_gps()
    n, nu = len(df), df.userId.nunique()
    ts = pd.to_datetime(df.timestamp, unit="ms", utc=True).dt.tz_convert(TZ)

    per_user = df.groupby("userId").size().sort_values(ascending=False)
    acc = pd.to_numeric(df.accuracy, errors="coerce").dropna()
    dt = (df.groupby("userId").timestamp.diff() / 1000.0).dropna()
    dt = dt[dt > 0]
    days_per_user = (pd.DataFrame({"u": df.userId.values, "d": ts.dt.date.values})
                     .groupby("u").d.nunique())
    sp_ok = pd.to_numeric(df.speed, errors="coerce").notna().mean()
    br_ok = pd.to_numeric(df.bearing, errors="coerce").notna().mean()

    # ── riepilogo numerico (→ thesis/eda.md) ──────────────────────────────────
    print("=" * 64)
    print("E1 — EDA RAW GPS (freeze)")
    print("=" * 64)
    print(f"fix totali: {n:,} | utenti con GPS: {nu}")
    print(f"fix/utente   mediana {per_user.median():.0f}  p10 {per_user.quantile(.1):.0f}  "
          f"p90 {per_user.quantile(.9):.0f}  max {per_user.max():,}")
    print(f"giorni attivi/utente  mediana {days_per_user.median():.0f}  "
          f"range [{days_per_user.min()}, {days_per_user.max()}]")
    print(f"accuracy (m)  mediana {acc.median():.1f}  <5m {100*(acc<5).mean():.1f}%  "
          f"<10m {100*(acc<10).mean():.1f}%  >50m {100*(acc>50).mean():.1f}%")
    print(f"intervallo inter-fix (s)  mediana {dt.median():.1f}  p90 {dt.quantile(.9):.1f}  "
          f"<=2s {100*(dt<=2).mean():.1f}%  >60s {100*(dt>60).mean():.1f}%")
    print(f"speed presente {100*sp_ok:.1f}%  |  bearing presente {100*br_ok:.1f}%")

    # ── figures (LABELS IN ENGLISH — go into the EN thesis, STYLE_GUIDE) ──────
    # 1) accuracy CDF (over all fixes; view clipped at 100 m)
    plt.figure(figsize=(6, 4))
    a = np.sort(acc.values)
    plt.plot(a, np.linspace(0, 100, len(a)))
    plt.xlim(0, 100); plt.axvline(5, ls="--", c="grey", label="5 m")
    plt.xlabel("GPS accuracy (m)"); plt.ylabel("CDF (%)")
    plt.title("GPS fix accuracy (CDF)"); plt.legend(); plt.grid(alpha=.3)
    savefig("e1_gps_accuracy")

    # 2) inter-fix interval (duty-cycling): dense ~1 Hz sampling vs off-gaps
    plt.figure(figsize=(6, 4))
    d = dt.values; d = d[d <= 3600]
    plt.hist(np.log10(d), bins=60)
    plt.yscale("log")
    plt.xlabel("log10(inter-fix interval, s)"); plt.ylabel("count (log)")
    plt.title("GPS inter-fix interval (duty-cycling)"); plt.grid(alpha=.3)
    savefig("e1_gps_interval")

    # 3) per-user heterogeneity (#fix, log)
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(per_user)), per_user.values)
    plt.yscale("log")
    plt.xlabel("user (sorted by #fix)"); plt.ylabel("GPS fixes (freeze window)")
    plt.title("GPS fixes per user (heterogeneity)"); plt.grid(alpha=.3, axis="y")
    savefig("e1_gps_peruser")

    print(f"\nfigure → {FIG.relative_to(ROOT)}/e1_gps_*.png|pdf")


if __name__ == "__main__":
    main()
