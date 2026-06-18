"""
research/rq2_1_feature_groups.py — RQ2.1: ablation feature-group A/B/C/D × strato GPS.

Scopo:    risolvere il paradosso "se l'IMU (A) da solo non fa male, perché senza GPS facciamo 0.18?".
          Tesi: NON è che l'IMU peggiora — è che il modello PIENO (ABCD) non USA l'IMU quando il GPS
          manca (L1 Still/Moving si appoggia alle feature GPS imputate → collassa a Still). Un modello
          A-only (o GPS-dropout), costretto a leggere l'IMU, recupera Walk/Still sullo zero-GPS.
          Quantifica per ogni sottoinsieme di gruppi la macro-F1 per strato GPS **3-way**.
Metodo:   per ogni subset {A,B,C,AB,BC,ABC,ABCD,AD}: train rolling silver (RF gerarchico), eval
          stratificato 3-way (absent=0 / sparse / present). Per A e ABCD anche per-classe sullo zero-GPS.
Input:    data/v2/features_trento.parquet
Output:   research/figures/rq2_1_feature_groups.{png,pdf} + riepilogo stdout
Alimenta: thesis/results.md §RQ2/RQ3 (feature-group × confine). Sez.tesi: 4.5 / 6.5.

⚠ PROXY (conv. 8 di results.md): qui l'ABCD e' un RF RI-ALLENATO (HierarchicalTMD default), NON il
  modello canonico del registry → present/sparse differiscono ≤0.02 dall'headline (canonico = present
  0.796 / sparse 0.572 / ALL 0.672; qui ~0.79/~0.55/0.67). L'absent 0.175 e' identico (= il claim).
  Per CITARE in tesi usa i numeri del registry (results.md), non l'output di questo script.

Run: python research/rq2_1_feature_groups.py
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
from tmd.training.trainer import get_feature_cols, rolling_splits  # noqa: E402
from tmd.models.hierarchical import HierarchicalTMD  # noqa: E402

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
SUBSETS = [["A"], ["B"], ["C"], ["A", "B"], ["B", "C"], ["A", "B", "C"],
           ["A", "B", "C", "D"], ["A", "D"]]
STRATA = [("absent", lambda f: f == 0), ("sparse", lambda f: (f > 0) & (f <= 0.5)),
          ("present", lambda f: f > 0.5)]


def macro(y, p):
    seen = [c for c in FIVE if (np.asarray(y) == c).sum() > 0]
    return f1_score(y, p, labels=seen, average="macro", zero_division=0)


def matrix(df, feats):
    X = np.full((len(df), len(feats)), np.nan, np.float32)
    for i, c in enumerate(feats):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    return X


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    silver = df[df.silver_label.notna()].copy()
    evalp = df[df.label.notna()].copy()

    print("=" * 70)
    print("RQ2.1 — feature-group A/B/C/D × strato GPS (rolling silver, media-fold)")
    print("=" * 70)
    print(f"{'groups':<10}{'n_feat':>7}{'absent':>9}{'sparse':>9}{'present':>9}{'ALL':>8}")
    rows = {}
    perclass_absent = {}
    for groups in SUBSETS:
        feats = get_feature_cols(silver, groups)
        # accumula predizioni OOF per strato sui fold
        oof = []
        for _, dtr, dte in rolling_splits(silver, evalp):
            dte = dte[dte.label.isin(FIVE)]
            if len(dte) < 10:
                continue
            cls = [c for c in FIVE if c in set(dtr.silver_label)]
            m = HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(dtr, feats), dtr.silver_label.values)
            d = dte[["label", "gps_frac"]].copy(); d["pred"] = m.predict(matrix(dte, feats))
            oof.append(d)
        oof = pd.concat(oof, ignore_index=True)
        key = "".join(groups)
        st = {nm: macro(oof[fn(oof.gps_frac)].label.values, oof[fn(oof.gps_frac)].pred.values)
              for nm, fn in STRATA}
        st["ALL"] = macro(oof.label.values, oof.pred.values)
        rows[key] = st
        print(f"{key:<10}{len(feats):>7}{st['absent']:>9.3f}{st['sparse']:>9.3f}{st['present']:>9.3f}{st['ALL']:>8.3f}")
        if key in ("A", "ABCD"):
            ab = oof[oof.gps_frac == 0]
            perclass_absent[key] = {c: f1_score(ab.label.values, ab.pred.values, labels=[c],
                                                 average="macro", zero_division=0) for c in FIVE}

    print("\n── per-classe sullo ZERO-GPS (absent) — A-only vs ABCD ──")
    print(f"  {'class':<7}" + "".join(f"{k:>10}" for k in perclass_absent))
    for c in FIVE:
        print(f"  {c:<7}" + "".join(f"{perclass_absent[k][c]:>10.2f}" for k in perclass_absent))

    # ── figura: macro per strato, per subset ──
    keys = list(rows); x = np.arange(len(keys)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (nm, col) in enumerate([("absent", "tab:red"), ("sparse", "tab:orange"), ("present", "tab:blue")]):
        ax.bar(x + (i - 1) * w, [rows[k][nm] for k in keys], w, label=f"GPS-{nm}", color=col)
    ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1)
    # niente title in-immagine (C8): la caption LaTeX racconta la figura
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"research/figures/rq2_1_feature_groups.{ext}", dpi=150, bbox_inches="tight")
    print("\nfigura → rq2_1_feature_groups.{png,pdf}")


if __name__ == "__main__":
    main()
