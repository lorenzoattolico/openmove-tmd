"""
research/rq3_1_gps_strata_fig.py — RQ3.1: figura headline del CONFINE OPERATIVO.

Scopo:    la figura-chiave della tesi (Cap.6 confine operativo): macro-F1 del modello canonico per
          strato GPS **3-way** (absent=0 / sparse / present), baseline vs **GPS-dropout** (3.2).
          Mostra il floor IMU-only reale (0.18) — che il binario mascherava — e il recupero chirurgico
          del GPS-dropout sullo zero-GPS, senza degradare present/sparse.
Metodo:   F1 macro (5 classi moving) per strato dai due eval rolling-OOF (baseline 233509, dropout 101222).
Input:    data/v2/processed/eval_trento_20260612_202507.parquet (baseline) ·
          data/v2/processed/eval_trento_20260612_203011.parquet (+gps-dropout)
Output:   research/figures/rq3_1_gps_strata.{png,pdf}
Alimenta: thesis/results.md §RQ3 (headline confine operativo). Sez.tesi: 6.x.

Run: python research/rq3_1_gps_strata_fig.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "research/figures"
FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
BASE = ROOT / "data/v2/processed/eval_trento_20260612_202507.parquet"
DROP = ROOT / "data/v2/processed/eval_trento_20260612_203011.parquet"
STRATA = [("GPS-absent\n(=0)", lambda f: f == 0),
          ("GPS-sparse\n(0–0.5)", lambda f: (f > 0) & (f <= 0.5)),
          ("GPS-present\n(>0.5)", lambda f: f > 0.5)]


def macro(y, p):
    seen = [c for c in FIVE if (y == c).sum() > 0]
    return f1_score(y, p, labels=seen, average="macro", zero_division=0)


def strata_f1(path):
    ev = pd.read_parquet(path)
    ev = ev[ev.label.isin(FIVE)]
    return [macro(ev[m(ev.gps_frac)].label.values, ev[m(ev.gps_frac)].predicted_class.values)
            for _, m in STRATA]


def main():
    base, drop = strata_f1(BASE), strata_f1(DROP)
    print("strato        baseline  +dropout")
    for (lab, _), b, d in zip(STRATA, base, drop):
        print(f"  {lab.replace(chr(10),' '):<20} {b:.3f}    {d:.3f}")

    from decimal import Decimal, ROUND_HALF_UP
    def lbl(v):  # round-half-up vero (dal 3-dec), coerente col testo (0.175→0.18)
        return str(Decimal(str(round(v, 3))).quantize(Decimal("0.01"), ROUND_HALF_UP))

    x = np.arange(len(STRATA))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    cols = ["tab:red", "tab:orange", "tab:blue"]   # absent/sparse/present
    bars = ax.bar(x, base, 0.55, color=cols)
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.012,
                lbl(r.get_height()), ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in STRATA])
    ax.set_ylabel("macro-F1 (5 moving classes)"); ax.set_ylim(0, 1.0)
    # niente title in-immagine (C8): la caption LaTeX racconta la figura
    ax.axhline(0.293, ls=":", color="dimgray", lw=1.2)
    ax.annotate("binary “absent” = 0.29\n(masks the 0.18 floor)", xy=(1.0, 0.293),
                xytext=(1.15, 0.45), fontsize=9, color="dimgray", ha="center", style="italic",
                arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.8))
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"rq3_1_gps_strata.{ext}", dpi=150, bbox_inches="tight")
    print("figura → rq3_1_gps_strata.{png,pdf}")


if __name__ == "__main__":
    main()
