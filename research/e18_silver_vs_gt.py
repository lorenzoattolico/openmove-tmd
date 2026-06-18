"""
e18_silver_vs_gt.py — qualità del silver-labeling vs GT (Fase 1c · task E18).

Scopo:    quanto è accurata la weak-supervision (silver_label) DOVE abbiamo la GT?
          Confusione GT×silver + precision/recall per classe + accuratezza per strato GPS.
          Valida l'uso del silver per addestrare sui dati NON etichettati (Cap.4.4/6.4) e
          quantifica l'errore indotto dal silver (e dove si concentra).
Labeler:  silver = tmd `label_windows_universal` (soglie fisiche + infra prior, window-mode,
          label-free). **Bike escluso per design** → mai emesso (limite noto, vedi E8/E20).
Input:    data/v2/features_trento.parquet (label GT + silver_label + gps_frac)
Output:   research/figures/e18_confusion.{png,pdf} (row-normalized = recall)
          research/figures/e18_prec_recall.{png,pdf} · e18_silver_vs_gt.csv (metriche per classe)
Alimenta: thesis/eda.md (E18)
Sez.tesi: 4.4 weak-supervision / 6.4

Nota: analisi sulle finestre con GT E silver presenti (commit del labeler); l'ABSTAIN → E19.
Run: /opt/miniconda3/envs/tmd/bin/python research/e18_silver_vs_gt.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
ORDER = ["Still", "Walk", "Bike", "Bus", "Car", "Train"]


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    print("=" * 70); print("E18 — silver-labeling vs GT (qualità weak-supervision)"); print("=" * 70)

    gt_yes, sv_yes = df.label.notna(), df.silver_label.notna()
    n_gt = gt_yes.sum()
    print(f"finestre GT={n_gt} | silver={sv_yes.sum()} | entrambi={int((gt_yes&sv_yes).sum())} | "
          f"GT-ma-ABSTAIN={int((gt_yes&~sv_yes).sum())} ({100*(gt_yes&~sv_yes).sum()/n_gt:.0f}% → E19)")

    both = df[gt_yes & sv_yes].copy()
    classes = [c for c in ORDER if c in both.label.unique()]
    svcls = [c for c in ORDER if c in both.silver_label.unique()]
    print(f"\nclassi GT: {classes} | classi emesse dal silver: {svcls} (Bike mai emesso = limite by-design)")

    # ── confusione GT×silver ──
    ct = pd.crosstab(both.label, both.silver_label).reindex(index=classes, columns=svcls, fill_value=0)
    print(f"\nConfusione GT(righe)×silver(col), n={len(both)}:")
    print(ct.to_string())
    recall_rownorm = ct.div(ct.sum(axis=1), axis=0)

    # ── precision/recall/F1 (silver come predittore della GT) ──
    p, r, f1, sup = precision_recall_fscore_support(both.label, both.silver_label, labels=classes, zero_division=0)
    met = pd.DataFrame({"precision": p, "recall": r, "f1": f1, "support": sup}, index=classes)
    met.round(3).to_csv(FIG / "e18_silver_vs_gt.csv")
    print("\nMetriche per classe (silver vs GT):"); print(met.round(2).to_string())

    acc = (both.label == both.silver_label).mean()
    nb = both[both.label != "Bike"]
    acc_nb = (nb.label == nb.silver_label).mean()
    print(f"\nAccuratezza silver: TUTTE {acc:.3f} | esclusa Bike {acc_nb:.3f} (Bike strutturalmente persa)")
    # dove va la Bike
    if "Bike" in classes:
        bk = ct.loc["Bike"]
        print(f"GT-Bike (n={int(bk.sum())}) → silver: " + ", ".join(f"{c} {100*bk[c]/bk.sum():.0f}%" for c in bk[bk>0].index))

    # ── per strato GPS — ATTENZIONE artefatto class-mix (Simpson) ──
    print("\nPer strato GPS (silver usa GPS+infra). ⚠ l'aggregato inganna → class-mix:")
    strata = [("GPS-present(>0.5)", both.gps_frac > 0.5), ("GPS-absent(<=0.5)", both.gps_frac <= 0.5)]
    for nm, m in strata:
        s = both[m]
        still_pct = 100 * (s.label == "Still").mean()
        print(f"  {nm:18s} n={len(s):5d}  acc-aggreg={s.eval('label==silver_label').mean():.3f}  (Still={still_pct:.0f}% → gonfia l'aggregato)")
    print("  → onesto = accuratezza ENTRO-classe per strato (controlla il mix):")
    movcls = [c for c in classes if c != "Bike"]
    hdr = "  " + "class".ljust(7) + "".join(nm.split("(")[0].rjust(12) for nm, _ in strata)
    print(hdr)
    for c in movcls:
        cells = []
        for nm, m in strata:
            s = both[(both.label == c) & m]
            cells.append((f"{(s.label==s.silver_label).mean():.2f}(n{len(s)})" if len(s) else "NA").rjust(12))
        print("  " + c.ljust(7) + "".join(cells))
    print("  → silver MOLTO migliore con GPS sui modi in movimento (Bus 0.73→0.17, Car 0.87→0.43, Train 0.95→0.37);")
    print("    l'inversione aggregata è solo Still-dominance del GPS-absent (91%).")

    # ── coverage per classe (commit-rate: di GT-X, quante etichetta il silver) → ponte a E19 ──
    print("\nCommit-rate per classe GT (silver NON-abstain | GT=X):")
    cov = df[gt_yes].groupby("label").apply(lambda g: g.silver_label.notna().mean(), include_groups=False)
    for c in classes:
        print(f"  {c:6s} {100*cov[c]:.0f}%  (abstain {100*(1-cov[c]):.0f}%)")

    # ── FIG 1: confusione row-normalized (recall) ──
    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(recall_rownorm.values, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, label="fraction of GT row (recall)")
    plt.xticks(range(len(svcls)), svcls); plt.yticks(range(len(classes)), classes)
    plt.xlabel("silver label"); plt.ylabel("GT label")
    for i in range(len(classes)):
        for j in range(len(svcls)):
            v = recall_rownorm.values[i, j]
            if v > 0.01:
                plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > 0.5 else "black", fontsize=8)
    plt.title("Silver vs GT confusion (row-normalized)\nBike never emitted by physical labeler")
    savefig("e18_confusion")

    # ── FIG 2: precision/recall per classe ──
    x = np.arange(len(classes)); w = 0.38
    plt.figure(figsize=(7, 4.2))
    plt.bar(x - w/2, met.precision, w, label="precision", color="tab:blue")
    plt.bar(x + w/2, met.recall, w, label="recall", color="tab:orange")
    plt.xticks(x, classes); plt.ylim(0, 1.05); plt.ylabel("score")
    # niente title in-immagine (C8): la caption LaTeX racconta la figura (agreement 0.87 / 0.90 ex-Bike)
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y")
    for i, c in enumerate(classes):
        plt.text(i, 0.02, f"n={int(met.support[c])}", ha="center", fontsize=7, rotation=90, color="gray")
    savefig("e18_prec_recall")

    print("\nfigure → e18_confusion · e18_prec_recall | tabella → e18_silver_vs_gt.csv")


if __name__ == "__main__":
    main()
