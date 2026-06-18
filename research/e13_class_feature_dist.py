"""
e13_class_feature_dist.py — distribuzioni per-classe delle feature chiave (Fase 1c · task E13).

Scopo:    interpretabilità FISICA: ogni feature chiave ha un pattern per-classe sensato?
          - C_osm_rail_prop (vicinanza rotaia)   → Train alto
          - C_bus_stops_prop (vicinanza fermate)  → Bus alto
          - B_stop_frac (frazione fermo)          → Still/Bus alti, Car/Train bassi
          - A_lin_mag_iqr (variabilità moto IMU, NORMALIZZATO) → Walk/Bike vs motorizzati
          Rende concreto il framing fisico (figure interpretabili per Cap.4.5/5).
Input:    data/v2/features_trento.parquet (163 + label GT + gps_frac)
Output:   research/figures/e13_class_feature_dist.{png,pdf}
Alimenta: thesis/eda.md (E13)
Sez.tesi: 4.5 / 5

Run: /opt/miniconda3/envs/tmd/bin/python research/e13_class_feature_dist.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
ORDER = ["Still", "Walk", "Bike", "Bus", "Car", "Train"]
FEATS = [("C_osm_rail_prop", "rail proximity (Train↑)"),
         ("C_bus_stops_prop", "bus-stop proximity (Bus↑)"),
         ("B_stop_frac", "stop fraction (Still/Bus↑)"),
         ("A_lin_mag_iqr", "IMU motion variability (norm.)")]


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    g = df[(df.gps_frac > 0.5) & df.label.notna()]
    classes = [c for c in ORDER if c in g.label.unique()]
    print("=" * 64); print("E13 — distribuzioni per-classe feature chiave (GPS-present)"); print("=" * 64)
    print(f"finestre: {len(g):,}")
    print(f"\nmediana per classe:")
    hdr = "feature".ljust(20) + "".join(c[:6].rjust(8) for c in classes)
    print(hdr)
    for f, _ in FEATS:
        if f not in g.columns:
            print(f"  {f}: ASSENTE"); continue
        meds = [g[g.label == c][f].median() for c in classes]
        print(f"  {f:18s}" + "".join(f"{m:8.2f}" for m in meds))

    # ── figura 2x2 (EN) ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (f, desc) in zip(axes.ravel(), FEATS):
        if f not in g.columns:
            ax.set_visible(False); continue
        data = [g[g.label == c][f].dropna().values for c in classes]
        ax.boxplot(data, tick_labels=classes, showfliers=False)
        ax.set_title(f"{f}\n{desc}", fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Per-class distributions of key features (physical interpretability)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"e13_class_feature_dist.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nfigura → research/figures/e13_class_feature_dist.png|pdf")


if __name__ == "__main__":
    main()
