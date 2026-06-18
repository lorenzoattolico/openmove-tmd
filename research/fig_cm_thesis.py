"""
research/fig_cm_thesis.py — confusion matrix in VESTE TESI (Cap.6.1).

Scopo:    CM del modello canonico in formato pubblicabile: etichette EN, niente titolo-debug,
          DUE pannelli che raccontano la storia giusta: GPS-present (dominio operativo) vs ALL
          (dove si vede il collasso Walk/moving→Still del no-GPS). Predizioni RAW (Convenzioni #2).
Input:    data/v2/processed/eval_trento_20260612_202507.parquet (eval rolling-OOF del canonico)
Output:   research/figures/cm_thesis.{png,pdf} (da copiare in manuscript/images/)
Alimenta: Cap.6.1 (STRUCTURE §6.1). Numeri → results.md RQ1.

Run: python research/fig_cm_thesis.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "data/v2/processed/eval_trento_20260612_202507.parquet"
FIVE = ["Still", "Walk", "Car", "Bus", "Train"]


def panel(ax, y, p, title):
    cm = confusion_matrix(y, p, labels=FIVE)
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(FIVE)):
        for j in range(len(FIVE)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]:,})".replace(",", "."),
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(FIVE)), FIVE, rotation=30, ha="right")
    ax.set_yticks(range(len(FIVE)), FIVE)
    ax.set_xlabel("Predicted"); ax.set_title(title, fontsize=11)


def main():
    ev = pd.read_parquet(EV)
    ev = ev[ev.label.isin(FIVE)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    g = ev[ev.gps_frac > 0.5]
    panel(axes[0], g.label, g.predicted_class, f"GPS-present (operational domain, n={len(g):,})".replace(",", "."))
    panel(axes[1], ev.label, ev.predicted_class, f"All windows (n={len(ev):,})".replace(",", "."))
    axes[0].set_ylabel("True (MotionTag reference)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"research/figures/cm_thesis.{ext}", dpi=150, bbox_inches="tight")
    print("figura → research/figures/cm_thesis.{png,pdf}")


if __name__ == "__main__":
    main()
