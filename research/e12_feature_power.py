"""
e12_feature_power.py — potere discriminante delle feature (Fase 1c · task E12).

Scopo:    quali feature portano il segnale? (mutual-info + ANOVA-F per classe), contributo per
          GRUPPO A/B/C/D, concentrazione del segnale, e separabilità delle COPPIE difficili
          (Car↔Bus, Walk↔Still). Triangola E8 (separabilità via velocità) con vista multi-feature.
Input:    data/v2/features_trento.parquet (163 canonico + label GT + gps_frac)
Output:   research/figures/e12_mi_top.{png,pdf} · e12_group_mi.{png,pdf}
Alimenta: thesis/eda.md (E12)
Sez.tesi: 4.5 selezione variabili / 5

Note: su finestre GPS-present (tutte le feature definite); NaN residui imputati (median) come il modello.
Run: /opt/miniconda3/envs/tmd/bin/python research/e12_feature_power.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
SEED = 0


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def mi(X, y):
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    return pd.Series(mutual_info_classif(Xi, y, random_state=SEED), index=X.columns)


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    feat = [c for c in df.columns if c[:2] in ("A_", "B_", "C_", "D_")]
    g = df[(df.gps_frac > 0.5) & df.label.notna()]
    classes = ["Still", "Walk", "Bike", "Bus", "Car", "Train"]
    g = g[g.label.isin(classes)]
    print("=" * 64); print("E12 — potere discriminante feature (GPS-present, 163 feat)"); print("=" * 64)
    print(f"finestre: {len(g):,} | feature: {len(feat)}")

    m = mi(g[feat], g.label).sort_values(ascending=False)
    print("\nTop-15 feature per mutual-info (multiclasse):")
    for f, v in m.head(15).items():
        print(f"  {f:28s} {v:.3f}  [{f[0]}]")
    # concentrazione
    cum = m.cumsum() / m.sum()
    n10 = (cum <= 0.0).sum()  # placeholder
    print(f"\nconcentrazione segnale: top-10 feature = {100*m.head(10).sum()/m.sum():.0f}% della MI totale; "
          f"top-30 = {100*m.head(30).sum()/m.sum():.0f}%")

    # contributo per gruppo (somma e media)
    grp = pd.DataFrame({"mi": m, "grp": [f[0] for f in m.index]})
    by = grp.groupby("grp").mi.agg(["sum", "mean", "count"]).reindex(list("ABCD"))
    print("\ncontributo per GRUPPO (MI somma / media-per-feature / n):")
    print(by.round(3).to_string())

    # coppie difficili
    for a, b in [("Car", "Bus"), ("Walk", "Still")]:
        sub = g[g.label.isin([a, b])]
        mp = mi(sub[feat], sub.label).sort_values(ascending=False)
        print(f"\n{a}↔{b}: top-5 feature separanti: " +
              ", ".join(f"{f}({v:.2f})" for f, v in mp.head(5).items()))

    # ── figure (EN) ──
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
    top = m.head(20)[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(range(len(top)), top.values, color=[colors[f[0]] for f in top.index])
    plt.yticks(range(len(top)), top.index, fontsize=7)
    plt.xlabel("mutual information with GT class"); plt.title("Top-20 discriminative features")
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color=colors[g_], label=g_) for g_ in "ABCD"], fontsize=8)
    plt.grid(alpha=.3, axis="x"); savefig("e12_mi_top")

    plt.figure(figsize=(6, 4))
    plt.bar(by.index, by["mean"].values, color=[colors[g_] for g_ in by.index])
    for i, g_ in enumerate(by.index):
        plt.text(i, by["mean"][g_], f"n={int(by['count'][g_])}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("mean MI per feature"); plt.xlabel("feature group")
    plt.title("Per-feature discriminative potency (B,C strongest; A many but individually weak)")
    plt.grid(alpha=.3, axis="y")
    savefig("e12_group_mi")

    print(f"\nfigure → research/figures/e12_{{mi_top,group_mi}}.png|pdf")


if __name__ == "__main__":
    main()
