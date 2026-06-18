"""
e5_windows.py — EDA livello FINESTRA sul freeze (Fase 1c · task E5).

Scopo:    - distribuzione gps_frac a livello FINESTRA (bimodalità netta 0-vs-pieno);
          - GPS-ASSENTE prima/dopo PULITO (RQ3.5): freeze vs vecchio snapshot, stesso orizzonte (≤1 giu),
            3-way zero / sparso / presente -> il dump a cursore corretto recupera GPS o no?
          - distribuzione GT per strato GPS (measurement floor: moving-GT senza GPS).
          (silver vs GT -> dopo label_silver, 1d: il silver non esiste ancora.)
Input:    data/v2/features_trento_full.parquet   (freeze, nuovo)
          data/processed/features_trento.parquet (vecchio snapshot, per il confronto prima/dopo)
Output:   research/figures/e5_window_gpsfrac.{png,pdf}   (bimodalità window-level)
          research/figures/e5_gt_by_stratum.{png,pdf}    (GT per strato GPS)
Alimenta: thesis/eda.md §3 (E5)
Sez.tesi: 3.2 guasto GPS / 6.5 confine operativo

Run: /opt/miniconda3/envs/tmd/bin/python research/e5_windows.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data" / "v2" / "features_trento_full.parquet"
OLD = ROOT / "data" / "processed" / "features_trento.parquet"
FIG = ROOT / "thesis" / "figures"
JUN1 = pd.Timestamp("2026-06-01 23:59:59", tz="Europe/Rome").timestamp() * 1000  # orizzonte vecchio


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def strata(gf: pd.Series) -> tuple[float, float, float]:
    """% zero (==0), sparso (0<.<=0.5), presente (>0.5)."""
    gf = pd.to_numeric(gf, errors="coerce")
    return (100 * (gf == 0).mean(), 100 * ((gf > 0) & (gf <= 0.5)).mean(), 100 * (gf > 0.5).mean())


def main() -> None:
    if not NEW.exists():
        sys.exit(f"Manca {NEW}")
    new = pd.read_parquet(NEW, columns=["gps_frac", "label", "ts_start"])
    old = pd.read_parquet(OLD, columns=["gps_frac"]) if OLD.exists() else None
    new_pre = new[new.ts_start <= JUN1]   # stesso orizzonte del vecchio (≤1 giu)

    # ── GPS-assente prima/dopo (RQ3.5) ────────────────────────────────────────
    print("=" * 64); print("E5 — EDA FINESTRE (freeze)"); print("=" * 64)
    print(f"finestre: {len(new):,} | con GT: {new.label.notna().sum():,} ({100*new.label.notna().mean():.0f}%)")
    print("\n3-way GPS [% zero / sparso / presente] — ASSENTE = zero+sparso:")
    for nm, gf in [("VECCHIO (data/processed, ≤1 giu)", old.gps_frac if old is not None else None),
                   ("FREEZE ≤1 giu (apples-to-apples)", new_pre.gps_frac),
                   ("FREEZE tutto (19 mag–8 giu)",       new.gps_frac)]:
        if gf is None:
            print(f"  {nm:34s}: (vecchio non trovato)"); continue
        z, s, p = strata(gf)
        print(f"  {nm:34s}: zero {z:4.1f} | sparso {s:4.1f} | presente {p:4.1f}  → ASSENTE {z+s:4.1f}%  (n={len(gf):,})")
    print("  rif. doc vecchio: ASSENTE ~59.5%")

    # ── GT per strato GPS (measurement floor) ─────────────────────────────────
    gt = new[new.label.notna()].copy()
    gf = pd.to_numeric(gt.gps_frac, errors="coerce")
    gt["stratum"] = np.where(gf == 0, "zero", np.where(gf <= 0.5, "sparse", "present"))
    moving = ["Walk", "Car", "Bus", "Train"]
    ct = pd.crosstab(gt.stratum, gt.label)
    print("\nGT per strato (conteggi):")
    print(ct.to_string())
    mov_zero = gt[(gt.stratum == "zero") & gt.label.isin(moving)]
    print(f"\nmoving-GT in finestre ZERO-GPS: {len(mov_zero):,} "
          f"({100*len(mov_zero)/max(len(gt[gt.label.isin(moving)]),1):.0f}% dei moving-GT) "
          f"→ non classificabili senza GPS (measurement floor)")

    # ── figure (EN) ───────────────────────────────────────────────────────────
    # 1) window-level GPS coverage: TRE strati colorati (absent / sparse / present),
    #    non la sola linea 0.5 — mostra anche il terzo picco sparse (~0.1-0.2). C8: no title.
    gf_all = pd.to_numeric(new.gps_frac, errors="coerce").dropna()
    n_abs = int((gf_all == 0).sum())
    pos = gf_all[gf_all > 0].clip(upper=1.2)
    z, s, p = strata(new.gps_frac)
    col_abs, col_sp, col_pr = "tab:red", "tab:orange", "tab:blue"
    edges = np.linspace(0, 1.2, 49)
    counts, _ = np.histogram(pos, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    w = edges[1] - edges[0]
    colors = [col_sp if c <= 0.5 else col_pr for c in centers]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(centers, counts, width=w, color=colors, align="center")
    ax.bar([0.0], [n_abs], width=w, color=col_abs, align="center")  # spike ==0 = absent
    ax.set_yscale("log")
    ax.axvline(0.5, ls="--", c="grey", lw=1)
    handles = [mpatches.Patch(color=col_abs, label=f"absent (=0): {z:.1f}%"),
               mpatches.Patch(color=col_sp, label=f"sparse (0–0.5]: {s:.1f}%"),
               mpatches.Patch(color=col_pr, label=f"present (>0.5): {p:.1f}%")]
    ax.legend(handles=handles, loc="upper center", framealpha=.9)
    ax.set_xlabel("window GPS coverage (gps_frac = #GPS fixes / 120)")
    ax.set_ylabel("windows (log)")
    ax.grid(alpha=.3)
    savefig("e5_window_gpsfrac")

    # 2) GT per strato (stacked, ordine strati)
    order = [s for s in ["zero", "sparse", "present"] if s in ct.index]
    ct = ct.reindex(order)
    plt.figure(figsize=(7, 4))
    bottom = np.zeros(len(ct))
    for cls in ct.columns:
        plt.bar(ct.index, ct[cls].values, bottom=bottom, label=cls)
        bottom += ct[cls].values
    plt.xlabel("GPS stratum"); plt.ylabel("GT-labeled windows")
    plt.title("GT class distribution by GPS stratum (measurement floor)")
    plt.legend(fontsize=7, ncol=2); plt.grid(alpha=.3, axis="y")
    savefig("e5_gt_by_stratum")

    print(f"\nfigure → research/figures/e5_{{window_gpsfrac,gt_by_stratum}}.png|pdf")


if __name__ == "__main__":
    main()
