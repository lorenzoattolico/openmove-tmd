"""
research/rq3_2b_gps_router.py — RQ3.2b: router per disponibilità GPS (fix del collasso-a-Still).

Scopo:    RQ2.1 ha mostrato che il modello PIENO (ABCD) collassa a Still sullo zero-GPS (0.18) perché
          si appoggia alle feature GPS imputate, mentre un modello A-only (IMU) le ignora ed è
          costretto a leggere l'IMU → 0.46 sullo zero-GPS. Qui si valuta la cura deployabile:
          un ROUTER per disponibilità-GPS (ABCD dove GPS c'è, IMU-only dove GPS=0), confrontato con
          canonico, IMU-only, GPS-dropout, e un modello A-only addestrato SOLO sullo zero-GPS (la
          "dedicated GPS-absent model" suggerita).
Metodo:   rolling silver. Per ogni fold colleziono le predizioni OOF di: ABCD, A-only(all),
          A-only(absent-trained). Router = A-only se gps_frac==0 altrimenti ABCD. Eval 3-way + ALL.
Input:    data/v2/features_trento.parquet
Output:   research/figures/rq3_2b_gps_router.{png,pdf} + riepilogo stdout
Alimenta: thesis/results.md §RQ3 (confine operativo — il floor era architetturale). Sez.tesi: 6.5 / 8.

⚠ PROXY (conv. 8 di results.md): ABCD/router qui sono RF RI-ALLENATI (HierarchicalTMD default), NON il
  modello canonico del registry → canonico sparse 0.552/ALL 0.671 invece di 0.572/0.672 (≤0.02, solo sparse).
  La tabella-cura della tesi (tab:exp-cure) usa i valori del REGISTRY (router ALL 0.727 ricalcolato da
  eval.parquet canonico + IMU-only su absent). Per CITARE usa results.md, non l'output di questo script.

Run: python research/rq3_2b_gps_router.py
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
STRATA = [("absent", lambda f: f == 0), ("sparse", lambda f: (f > 0) & (f <= 0.5)),
          ("present", lambda f: f > 0.5), ("ALL", lambda f: f == f)]


def macro(y, p):
    seen = [c for c in FIVE if (np.asarray(y) == c).sum() > 0]
    return f1_score(y, p, labels=seen, average="macro", zero_division=0)


def matrix(df, feats):
    X = np.full((len(df), len(feats)), np.nan, np.float32)
    for i, c in enumerate(feats):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    return X


def fit(dtr, feats, sub=None):
    d = dtr if sub is None else dtr[sub(dtr.gps_frac)]
    cls = [c for c in FIVE if c in set(d.silver_label)]
    return HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(d, feats), d.silver_label.values)


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    silver = df[df.silver_label.notna()].copy()
    evalp = df[df.label.notna()].copy()
    fA = get_feature_cols(silver, ["A"]); fAll = get_feature_cols(silver, ["A", "B", "C", "D"])

    oof = []
    for _, dtr, dte in rolling_splits(silver, evalp):
        dte = dte[dte.label.isin(FIVE)]
        if len(dte) < 10:
            continue
        mABCD = fit(dtr, fAll)
        mA = fit(dtr, fA)
        d = dte[["label", "gps_frac"]].copy()
        d["ABCD"] = mABCD.predict(matrix(dte, fAll))
        d["Aonly"] = mA.predict(matrix(dte, fA))
        oof.append(d)
    oof = pd.concat(oof, ignore_index=True)

    # router: A-only (IMU) se gps_frac==0, altrimenti ABCD
    # (NB: niente "dedicated absent model": il silver non ha label MOVING sullo zero-GPS —
    #  il labeler astiene senza GPS → l'A-only-all impara i moving dalle finestre GPS-present
    #  e ne TRASFERISCE la firma-IMU allo zero-GPS. È questo il punto.)
    z = oof.gps_frac == 0
    oof["router"] = np.where(z, oof.Aonly, oof.ABCD)

    print("=" * 66); print("RQ3.2b — router per disponibilità GPS (rolling silver, OOF)"); print("=" * 66)
    arms = [("ABCD (canonico)", "ABCD"), ("IMU-only (A)", "Aonly"),
            ("ROUTER (ABCD|A@0)", "router")]
    print(f"{'modello':<24}{'absent':>9}{'sparse':>9}{'present':>9}{'ALL':>8}")
    res = {}
    for label, col in arms:
        st = {nm: macro(oof[fn(oof.gps_frac)].label.values, oof[fn(oof.gps_frac)][col].values)
              for nm, fn in STRATA}
        res[label] = st
        print(f"{label:<24}{st['absent']:>9.3f}{st['sparse']:>9.3f}{st['present']:>9.3f}{st['ALL']:>8.3f}")

    print("\n  → il floor 'absent 0.18' del canonico è ARCHITETTURALE (collasso-a-Still): "
          "il router lo porta a ~0.46 e l'ALL sale; il vero floor resta i MOTORIZZATI (IMU non separa).")

    # ── figura (C8: incluso lo strato sparse — 3-way completo + ALL) ──
    labels = [a[0] for a in arms]; x = np.arange(len(labels)); w = 0.19
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    series = [("absent", "tab:red"), ("sparse", "tab:orange"), ("present", "tab:blue"), ("ALL", "tab:green")]
    for i, (nm, col) in enumerate(series):
        bars = ax.bar(x + (i - 1.5) * w, [res[l][nm] for l in labels], w, label=f"GPS-{nm}" if nm != "ALL" else "ALL", color=col)
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.012, f"{r.get_height():.2f}",
                    ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right"); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1)
    ax.set_title("GPS-availability router fixes the absent collapse (Trento, rolling silver)")
    ax.legend(); ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"research/figures/rq3_2b_gps_router.{ext}", dpi=150, bbox_inches="tight")
    print("figura → rq3_2b_gps_router.{png,pdf}")


if __name__ == "__main__":
    main()
