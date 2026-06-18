"""
e28_class_prior_shift.py — shift del PRIOR di classe cross-dataset (Fase 1c · task E28).

Scopo:    completa il quadro domain-shift. E14 = covariate shift (P(x) sulle feature); E28 = label/
          prior shift (P(y), la distribuzione delle classi). Conta per il transfer: il posterior
          P(y|x) ∝ P(x|y)·P(y) → se il prior P(y) differisce, il transfer va RICALIBRATO sul target.
          Spiega perché l'accuratezza transfer soffre anche dove la struttura per-classe è preservata
          (E27), e perché il modal-split (E29) — che ri-stima le quote — è robusto al prior-shift.
Input:    data/v2/features_trento.parquet · data/processed/features_shl_bootstrap.parquet
          data/processed/features_geolife.parquet (tutti con `label`)
Output:   research/figures/e28_class_prior.{png,pdf} · e28_class_prior.csv
Alimenta: thesis/eda.md (E28)
Sez.tesi: 6.3 transfer / 3.3

Lettura: Trento Still-heavy (50.9%) vs SHL bilanciato vs GeoLife Walk-heavy (no Still) → prior-shift
         enorme → il transfer crudo è mal-calibrato sul prior; serve prior-correction / il modal-split
         lo aggira ri-stimando le quote sul target.
Run: /opt/miniconda3/envs/tmd/bin/python research/e28_class_prior_shift.py
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
ALL = ["Still", "Walk", "Bike", "Bus", "Car", "Train", "Subway", "Run"]
SHARED = ["Walk", "Bus", "Car", "Train"]   # comuni a Trento/SHL/GeoLife
DSETS = {"Trento": "data/v2/features_trento.parquet",
         "SHL": "data/v2/features_shl_full.parquet",  # tmd-aligned (230)
         "GeoLife": "data/processed/features_geolife.parquet"}


def tvd(p, q):
    return 0.5 * np.abs(p - q).sum()


def main():
    print("=" * 70); print("E28 — shift del PRIOR di classe cross-dataset"); print("=" * 70)
    priors = {}
    for nm, p in DSETS.items():
        lab = pd.read_parquet(ROOT / p, columns=["label"]).label.dropna()
        priors[nm] = lab.value_counts(normalize=True) * 100
    full = pd.DataFrame(priors).reindex(ALL).dropna(how="all")
    full.to_csv(FIG / "e28_class_prior.csv")
    print("\nPrior di classe per dataset (% sulle finestre etichettate):")
    print(full.round(1).fillna(0).to_string())

    # shared classes (renormalizzate) + TVD pairwise
    sh = pd.DataFrame({k: priors[k].reindex(SHARED).fillna(0) for k in DSETS})
    sh = sh / sh.sum() * 100
    print(f"\nClassi condivise {SHARED} (ri-normalizzate %):")
    print(sh.round(1).to_string())
    print("\nTVD pairwise del prior (classi condivise):")
    pairs = [("Trento", "SHL"), ("Trento", "GeoLife"), ("SHL", "GeoLife")]
    for a, b in pairs:
        print(f"  {a}↔{b}: {tvd(sh[a].values/100, sh[b].values/100)*100:.0f}%")

    print("\nOsservazioni:")
    print(f"  - Trento Still {full.loc['Still','Trento']:.0f}% (E9: stay lunghi) vs SHL {full.loc['Still','SHL']:.0f}% vs GeoLife 0% (no Still).")
    print(f"  - SHL bilanciato by-design; GeoLife solo modi di viaggio; Trento dominato da Still+Car.")
    print("  - → transfer crudo mal-calibrato sul prior (sovra-predice Still); serve prior-correction.")
    print("  - → il modal-split (E29) ri-stima le quote sul target → robusto al prior-shift.")

    # ── FIG: prior per dataset ──
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    # (a) full distribution per dataset
    f = full.fillna(0)
    x = np.arange(len(f)); w = 0.26
    for i, nm in enumerate(DSETS):
        ax[0].bar(x + (i-1)*w, f[nm].values, w, label=nm)
    ax[0].set_xticks(x); ax[0].set_xticklabels(f.index, rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("class share %"); ax[0].set_title("Class prior per dataset (label shift)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    # (b) shared classes renormalized
    xs = np.arange(len(SHARED))
    for i, nm in enumerate(DSETS):
        ax[1].bar(xs + (i-1)*w, sh[nm].values, w, label=nm)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(SHARED); ax[1].set_ylabel("share % (renormalized)")
    ax[1].set_title("Shared moving classes: prior still shifts\n(travel-mode mix differs by city)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    fig.suptitle("Label/prior shift across datasets (complements E14 covariate shift)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"e28_class_prior.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print("\nfigura → e28_class_prior | tabella → e28_class_prior.csv")


if __name__ == "__main__":
    main()
