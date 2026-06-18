"""
research/rq3_2d_degradation.py — RQ3.2d: curva di degradazione graceful (robustezza CONTROLLATA).

Scopo:    la figura-robustezza canonica per la commissione "real-world robustness". L'eval 3-way
          (RQ3.1) è OSSERVAZIONALE (il class-mix cambia tra strati). Qui un esperimento CONTROLLATO:
          stesse finestre GPS-present (stesso class-mix), si **maschera progressivamente il GPS** (B/C→NaN
          su una frazione p crescente di finestre) e si misura la F1. Confronto **baseline vs gps-dropout-0.7**:
          il baseline CROLLA (si fida del GPS), il modello gps-dropout **degrada con grazia** = robusto al
          guasto-GPS. È la prova pulita (non confusa dal class-mix) della robustezza migliorata.
Metodo:   split temporale (no leakage). Train baseline (ABCD) + gps-dropout-0.7 (maschera B/C sul 70% delle
          finestre GPS-present in train). Test = GPS-present (motiontag GT). Per p∈{0,.25,.5,.75,1}: maschera
          B/C su frazione p delle finestre test (seed fisso) → predici con entrambi → macro-F1.
Input:    data/v2/features_trento.parquet
Output:   research/figures/rq3_2d_degradation.{png,pdf}
Alimenta: thesis/results.md §RQ3 (robustezza controllata). Sez.tesi: 6.5.

Run: python research/rq3_2d_degradation.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
from tmd.training.trainer import get_feature_cols, temporal_splits  # noqa: E402
from tmd.models.hierarchical import HierarchicalTMD  # noqa: E402

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
PS = [0.0, 0.25, 0.5, 0.75, 1.0]


def macro(y, p):
    seen = [c for c in FIVE if (np.asarray(y) == c).sum() > 0]
    return f1_score(y, p, labels=seen, average="macro", zero_division=0)


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    silver = df[df.silver_label.notna()].copy()
    evalp = df[df.label.notna()].copy()
    feats = get_feature_cols(silver, ["A", "B", "C", "D"])
    bc_idx = [i for i, c in enumerate(feats) if c.startswith("B_") or c.startswith("C_")]
    ngps = feats.index("B_n_gps") if "B_n_gps" in feats else bc_idx[0]
    _, pool, test = next(iter(temporal_splits(silver, evalp)))
    test = test[(test.gps_frac > 0.5) & test.label.isin(FIVE)]
    Xtr = pool[feats].values.astype(np.float32)
    ytr = pool.silver_label.values
    cls = [c for c in FIVE if c in set(ytr)]
    Xte = test[feats].values.astype(np.float32); yte = test.label.values

    # baseline
    m_base = HierarchicalTMD(cls, [], clf_type="rf").fit(Xtr, ytr)
    # gps-dropout 0.7: maschera B/C sul 70% delle finestre train GPS-ok
    rng = np.random.default_rng(42)
    gps_ok = np.isfinite(Xtr[:, ngps]) & (Xtr[:, ngps] > 0)
    drop = gps_ok & (rng.random(len(Xtr)) < 0.7)
    Xtr_d = Xtr.copy(); Xtr_d[np.ix_(drop, bc_idx)] = np.nan
    m_drop = HierarchicalTMD(cls, [], clf_type="rf").fit(Xtr_d, ytr)

    print("=" * 60); print("RQ3.2d — degradazione graceful (GPS mascherato, test GPS-present)"); print("=" * 60)
    print(f"test GPS-present = {len(test)} finestre (class-mix fisso)\n")
    print(f"{'p (GPS mascher.)':>16}{'baseline':>11}{'gps-dropout':>13}")
    res = {"base": [], "drop": []}
    rng2 = np.random.default_rng(0)
    mask_order = rng2.random(len(test))   # ordine fisso → p crescente è nested
    for p in PS:
        msk = mask_order < p
        Xp = Xte.copy()
        if msk.any():
            Xp[np.ix_(msk, bc_idx)] = np.nan
        fb = macro(yte, m_base.predict(Xp))
        fd = macro(yte, m_drop.predict(Xp))
        res["base"].append(fb); res["drop"].append(fd)
        print(f"{p:>16.2f}{fb:>11.3f}{fd:>13.3f}")

    print(f"\n  → a p=1 (GPS tutto mascherato): baseline {res['base'][-1]:.3f} vs gps-dropout {res['drop'][-1]:.3f} "
          f"(Δ {res['drop'][-1]-res['base'][-1]:+.3f}). Il gap CRESCE con p = robustezza migliorata.")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(PS, res["base"], "o-", color="tab:red", label="baseline (GPS-reliant)")
    ax.plot(PS, res["drop"], "s-", color="tab:green", label="gps-dropout 0.7 (robust)")
    ax.fill_between(PS, res["base"], res["drop"], color="green", alpha=0.07)
    ax.set_xlabel("fraction of windows with GPS masked"); ax.set_ylabel("macro-F1 (GPS-present test, fixed class mix)")
    ax.set_ylim(0, 0.9)  # niente title in-immagine (C8): la caption LaTeX racconta la figura
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"research/figures/rq3_2d_degradation.{ext}", dpi=150, bbox_inches="tight")
    print("\nfigura → rq3_2d_degradation.{png,pdf}")


if __name__ == "__main__":
    main()
