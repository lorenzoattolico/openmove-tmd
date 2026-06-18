"""
e8_trips_gt.py — caratterizzazione viaggi/finestre GT (Fase 1c · task E8).

Scopo:    verificare la SEPARABILITÀ FISICA dei modi GT (fonda il labeling fisico):
          - velocità per modo (B_speed_mean/max) su finestre GPS-present → ordinamento fisico + overlap;
          - stop_frac per modo (asse di separazione Bus/Car vs Train/Walk);
          - durata dei viaggi GT per modo (da labels.parquet).
Input:    data/v2/features_trento_full.parquet (B_speed_*, B_stop_frac, label, gps_frac)
          data/raw_freeze/labels.parquet (started_at/finished_at/mode_tmd)
Output:   research/figures/e8_speed_by_mode.{png,pdf}  · e8_duration_by_mode.{png,pdf}
Alimenta: thesis/eda.md (E8)
Sez.tesi: 3.4 riferimento / 4.4 labeling fisico

Run: /opt/miniconda3/envs/tmd/bin/python research/e8_trips_gt.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "v2" / "features_trento_full.parquet"
LAB = ROOT / "data" / "raw_freeze" / "labels.parquet"
FIG = ROOT / "thesis" / "figures"
ORDER = ["Walk", "Bike", "Bus", "Car", "Train"]   # atteso fisico (lento→veloce)


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    df = pd.read_parquet(FULL, columns=["B_speed_mean", "B_speed_max", "B_stop_frac", "label", "gps_frac"])
    g = df[(df.gps_frac > 0.5) & df.label.notna()].copy()   # velocità solo dove il GPS c'è

    print("=" * 64); print("E8 — separabilità fisica modi GT (finestre GPS-present)"); print("=" * 64)
    print(f"finestre GPS-present con GT: {len(g):,}")
    print(f"\n{'modo':7s} {'n':>5s} {'speed_mean(m/s)':>16s} {'speed_max':>10s} {'stop_frac':>10s}")
    classes = [c for c in ORDER if c in g.label.unique()]
    for c in classes:
        s = g[g.label == c]
        print(f"{c:7s} {len(s):5d} {s.B_speed_mean.median():16.2f} "
              f"{s.B_speed_max.median():10.2f} {s.B_stop_frac.median():10.2f}")

    # overlap Car vs Bus (la coppia critica)
    car, bus = g[g.label == "Car"].B_speed_mean.dropna(), g[g.label == "Bus"].B_speed_mean.dropna()
    if len(car) and len(bus):
        ov = max(0.0, min(car.quantile(.75), bus.quantile(.75)) - max(car.quantile(.25), bus.quantile(.25)))
        print(f"\nCar vs Bus speed: Car IQR[{car.quantile(.25):.1f},{car.quantile(.75):.1f}] "
              f"Bus IQR[{bus.quantile(.25):.1f},{bus.quantile(.75):.1f}] → overlap IQR ~{ov:.1f} m/s "
              f"({'FORTE' if ov > 1 else 'modesto'} → coppia difficile)")

    # durate viaggi GT (da labels)
    lab = pd.read_parquet(LAB)
    lab["dur_min"] = (lab.finished_at - lab.started_at) / 60000.0
    lab = lab[(lab.dur_min > 0) & (lab.dur_min < 600)]
    print("\ndurata viaggi GT (min) per modo: mediana")
    print({m: round(lab[lab.mode_tmd == m].dur_min.median(), 1) for m in classes if m in lab.mode_tmd.unique()})

    # ── figure (EN) ──
    plt.figure(figsize=(7, 4))
    data = [g[g.label == c].B_speed_mean.clip(upper=30).dropna().values for c in classes]
    plt.violinplot(data, showmedians=True)
    plt.xticks(range(1, len(classes) + 1),
               [f"{c}\n(n={len(d):,})".replace(",", ".") for c, d in zip(classes, data)])  # C8: n per violino
    # label allineate alla dottrina Cap.3.4 (reference, non GT); niente title in-immagine (C8)
    plt.ylabel("mean GPS speed (m/s)"); plt.xlabel("Reference mode")
    plt.grid(alpha=.3, axis="y"); savefig("e8_speed_by_mode")

    plt.figure(figsize=(7, 4))
    dd = [lab[lab.mode_tmd == c].dur_min.clip(upper=120).values for c in classes if c in lab.mode_tmd.unique()]
    lbl = [c for c in classes if c in lab.mode_tmd.unique()]
    plt.boxplot(dd, labels=lbl, showfliers=False)
    plt.ylabel("trip duration (min)"); plt.xlabel("GT mode")
    plt.title("Per-mode GT trip duration"); plt.grid(alpha=.3, axis="y")
    savefig("e8_duration_by_mode")

    print(f"\nfigure → research/figures/e8_{{speed_by_mode,duration_by_mode}}.png|pdf")


if __name__ == "__main__":
    main()
