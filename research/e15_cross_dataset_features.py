"""
e15_cross_dataset_features.py — feature comuni cross-dataset (Fase 1c · task E15).

Scopo:    definire il BACKBONE trasferibile. Due livelli:
          (1) Trento163 ∩ SHL = set TRANSFER canonico (**163, set pieno** con SHL ri-allineato a 230).
              NB: il "142" di prima era un ARTEFATTO del SHL-parquet stale (185); ri-estratto SHL con
              tmd (`features_shl_full.parquet`, 230) → 0 breakers, il 163 transferisce intero.
          (2) Trento163 ∩ SHL ∩ GeoLife = backbone UNIVERSALE (3 geografie indipendenti).
          Incrocia E14 (KS): nel set transfer, quante sono ad alto shift (candidate al drop in ablation 1e)?
Input:    data/v2/features_trento.parquet (163 canonico) · data/v2/features_trento_full.parquet (230)
          data/processed/features_shl_bootstrap.parquet (SHL 185) · data/processed/features_geolife.parquet (GeoLife 27, GPS-only)
          research/figures/e14_domain_shift.csv (KS per feature, opzionale)
Output:   research/figures/e15_transfer_set.csv  (il set transfer — DELIVERABLE per il modello)
          research/figures/e15_backbone_3way.csv     (backbone universale GPS-only)
          research/figures/e15_transfer_breakers.csv (21 feature 163-only assenti su SHL)
          research/figures/e15_cross_dataset.{png,pdf} · e15_transfer_availability.{png,pdf}
Alimenta: thesis/eda.md (E15)
Sez.tesi: 6.3 transfer / 4.5 selezione

Nota: intersezione PER NOME — tutte e 3 le pipeline usano la stessa estrazione (nomi coerenti).
      GeoLife è GPS-only (A=0): il backbone 3-vie è per costruzione senza IMU.
Run: /opt/miniconda3/envs/tmd/bin/python research/e15_cross_dataset_features.py
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
GROUPS = list("ABCD")


def feats(p):
    c = pd.read_parquet(ROOT / p).columns.tolist()
    return set(x for x in c if x[:2] in ("A_", "B_", "C_", "D_"))


def by_group(s):
    cc = Counter(x[0] for x in s)
    return {g: cc.get(g, 0) for g in GROUPS}


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    T163 = feats("data/v2/features_trento.parquet")            # 163 canonico (OpenMove)
    Tf = feats("data/v2/features_trento_full.parquet")          # 230 (tutte)
    S = feats("data/v2/features_shl_full.parquet")  # tmd-aligned (230)
    G = feats("data/processed/features_geolife.parquet")        # 27 (GPS-only)

    print("=" * 70)
    print("E15 — feature comuni cross-dataset (backbone trasferibile)")
    print("=" * 70)
    print(f"per-dataset:  Trento163={len(T163)} {by_group(T163)} | Trento_full={len(Tf)} {by_group(Tf)}")
    print(f"              SHL={len(S)} {by_group(S)} | GeoLife={len(G)} {by_group(G)} (GPS-only, A=0)")

    transfer = T163 & S                # set transfer pieno (163 con SHL allineato)
    backbone = T163 & S & G            # backbone universale
    breakers = T163 - S                # 163-only assenti su SHL → NaN → rompono il transfer

    print(f"\n── set TRANSFER canonico (Trento163 ∩ SHL) = {len(transfer)}  {by_group(transfer)}")
    print(f"── backbone UNIVERSALE (∩ GeoLife)         = {len(backbone)}  {by_group(backbone)}  (GPS-only)")
    print(f"── transfer-breakers (163-only, NaN su SHL)= {len(breakers)}  {by_group(breakers)}")
    print("   breakers:", ", ".join(sorted(breakers)))

    # perché il backbone si restringe: chi manca dove (solo B/C/D, dove vive GeoLife)
    geoBCD = {x for x in G}
    miss_in_S = sorted(x for x in geoBCD if x in Tf and x not in S)
    miss_in_G = sorted(x for x in (Tf & S) if x[0] in "BCD" and x not in G)
    print(f"\nbackbone-shrink: B/C/D in Trento+GeoLife ma NON in SHL ({len(miss_in_S)}): {miss_in_S}")
    print(f"                 B/C/D in Trento+SHL ma NON in GeoLife ({len(miss_in_G)}): {miss_in_G}")

    # ── overlay E14 KS sul set transfer: quante ad alto shift (ablation 1e candidate al drop)? ──
    ks = None
    try:
        ks = pd.read_csv(FIG / "e14_domain_shift.csv").set_index("feature")["ks"]
    except Exception:
        pass
    tdf = pd.DataFrame({"feature": sorted(transfer)})
    tdf["grp"] = tdf.feature.str[0]
    tdf["in_backbone_3way"] = tdf.feature.isin(backbone)
    if ks is not None:
        tdf["ks_trento_shl"] = tdf.feature.map(ks)
        hi = tdf.ks_trento_shl > 0.5
        print(f"\nKS (E14) sul set transfer ({len(transfer)}): {hi.sum()} feature ad alto shift (KS>0.5) — candidate al DROP in ablation 1e")
        print("   high-KS nel set:", ", ".join(tdf[hi].sort_values('ks_trento_shl', ascending=False).feature))
        print(f"   KS mediano del set: {tdf.ks_trento_shl.median():.2f}")
    tdf.sort_values(["grp", "feature"]).to_csv(FIG / "e15_transfer_set.csv", index=False)
    pd.DataFrame({"feature": sorted(backbone), "grp": [f[0] for f in sorted(backbone)]}).to_csv(
        FIG / "e15_backbone_3way.csv", index=False)
    pd.DataFrame({"feature": sorted(breakers), "grp": [f[0] for f in sorted(breakers)]}).to_csv(
        FIG / "e15_transfer_breakers.csv", index=False)

    # ── FIG 1: conteggi per gruppo — per-dataset + intersezioni chiave ──
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
    nT, nS, nG, nX, nB = len(T163), len(S), len(G), len(transfer), len(backbone)
    cols = {f"Trento\n({nT})": T163, f"SHL\n({nS})": S, f"GeoLife\n({nG})": G,
            f"T∩SHL\n({nX})": transfer, f"T∩SHL∩Geo\n({nB})": backbone}
    x = np.arange(len(cols))
    plt.figure(figsize=(9, 5))
    bottom = np.zeros(len(cols))
    for g in GROUPS:
        vals = np.array([by_group(s)[g] for s in cols.values()])
        plt.bar(x, vals, bottom=bottom, color=colors[g], label=f"{g}")
        bottom += vals
    plt.xticks(x, list(cols.keys()), fontsize=9)
    plt.ylabel("feature count"); plt.legend(title="group", fontsize=8)
    plt.title("Cross-dataset feature availability (transferable backbone)")
    plt.grid(alpha=.3, axis="y"); savefig("e15_cross_dataset")

    # ── FIG 2: l'allineamento conta — 163 transferisce pieno su SHL allineato (0 persi);
    #    col parquet stale ne perdeva 21 (artefatto, non limite dati) ──
    try:
        S_stale = feats("data/processed/features_shl_bootstrap.parquet")
    except Exception:
        S_stale = set()
    kept_now = by_group(transfer)                          # 163 ∩ SHL-aligned (= tutto 163)
    lost_stale = by_group(T163 - S_stale) if S_stale else {g: 0 for g in GROUPS}
    gx = np.arange(len(GROUPS)); w = 0.38
    plt.figure(figsize=(7, 4.5))
    plt.bar(gx - w/2, [len([f for f in T163 if f[0] == g]) - lost_stale[g] for g in GROUPS], w,
            color="tab:gray", label=f"stale SHL-parquet (185): {sum(lost_stale.values())} lost→NaN")
    plt.bar(gx + w/2, [kept_now[g] for g in GROUPS], w,
            color="tab:green", label=f"aligned SHL (tmd, 230): 0 lost → set pieno {nX}")
    for i, g in enumerate(GROUPS):
        if lost_stale[g]:
            plt.text(i - w/2, len([f for f in T163 if f[0] == g]) - lost_stale[g], f"-{lost_stale[g]}",
                     ha="center", va="bottom", fontsize=8, color="tab:red")
    plt.xticks(gx, GROUPS); plt.ylabel("features of canonical 163-set available on SHL")
    plt.title("The '142' was an artifact: aligning SHL (tmd) recovers the full 163-set")
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y"); savefig("e15_transfer_availability")

    print(f"\ndeliverable → e15_transfer_set.csv (set transfer = {nX} feature, set pieno)")
    print("figure → e15_cross_dataset.{png,pdf} · e15_transfer_availability.{png,pdf}")


if __name__ == "__main__":
    main()
