"""
e27_physical_universality.py — universalità FISICA per-classe cross-dataset (Fase 1c · task E27).

Scopo:    scioglie il CAVEAT di E14 (KS marginale conflà class-mix + domain). Domanda:
          le firme FISICHE per-classe sono PRESERVATE cross-dataset, anche dove il livello
          assoluto shifta? Se l'ORDINAMENTO per-classe è preservato (Spearman alto) → lo shift
          è "calibrazione" (trasferibile con normalizzazione), NON "struttura rotta" (negative
          transfer). Confronta entro-classe (rimuove il confondente class-mix di E14).
Metodo:   per ogni feature comune → mediana per-classe in ciascun dataset → vettore su classi;
          Spearman tra i vettori dei dataset. Regola duale (dove il sensore è informativo):
          GPS (B/C/D) su gps_frac>0.5 (dominio operativo); IMU (A) su tutte le finestre etichettate.
Input:    data/v2/features_trento.parquet (GT label) · data/processed/features_shl_bootstrap.parquet (label)
          data/processed/features_geolife.parquet (label, GPS-only) · research/figures/e14_domain_shift.csv (KS)
Output:   research/figures/e27_universality_heatmap.{png,pdf}  (z-score per-classe backbone × 3 dataset)
          research/figures/e27_ordering_highshift.{png,pdf}    (Spearman delle high-shift di E14)
          research/figures/e27_universality.csv                (Spearman per feature)
Alimenta: thesis/eda.md (E27)
Sez.tesi: 6.3 transfer / 4.4-4.5

Classi comuni: 3-vie (Trento∩SHL∩GeoLife) = Walk/Bus/Car/Train · 2-vie (Trento∩SHL) += Still/Bike.
Run: /opt/miniconda3/envs/tmd/bin/python research/e27_physical_universality.py
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=ConstantInputWarning)  # vettori per-classe costanti → rho=nan (escluso)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
GPS_THR = 0.5
MIN_N = 30  # min finestre per (classe,feature) per accettare la mediana


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def class_medians(df, feat, classes):
    """mediana per-classe di `feat`, regola duale GPS/IMU. Ritorna Series indicizzata su classes (NaN se <MIN_N)."""
    sub = df if feat[0] == "A" else df[df.gps_frac > GPS_THR]
    out = {}
    for c in classes:
        v = sub.loc[sub.label == c, feat].dropna()
        out[c] = v.median() if len(v) >= MIN_N else np.nan
    return pd.Series(out, index=classes)


def main():
    T = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    S = pd.read_parquet(ROOT / "data/v2/features_shl_full.parquet")  # tmd-aligned (230)
    G = pd.read_parquet(ROOT / "data/processed/features_geolife.parquet")
    for d in (T, S, G):
        d.dropna(subset=["label"], inplace=True)

    cls_T, cls_S, cls_G = set(T.label.unique()), set(S.label.unique()), set(G.label.unique())
    ORDER = ["Still", "Walk", "Bike", "Bus", "Car", "Train"]
    cls2 = [c for c in ORDER if c in cls_T & cls_S]          # 2-vie Trento∩SHL
    cls3 = [c for c in ORDER if c in cls_T & cls_S & cls_G]  # 3-vie incl GeoLife
    print("=" * 70); print("E27 — universalità FISICA per-classe cross-dataset"); print("=" * 70)
    print(f"classi 2-vie (Trento∩SHL): {cls2}\nclassi 3-vie (∩GeoLife):   {cls3}")

    def feats(df):
        return set(c for c in df.columns if c[:2] in ("A_", "B_", "C_", "D_"))
    common2 = sorted(feats(T) & feats(S))            # = 142
    backbone = sorted(feats(T) & feats(S) & feats(G))  # = 13 GPS-only

    # ── Spearman per feature (2-vie, 6 classi) ──
    rows = []
    for f in common2:
        mt, ms = class_medians(T, f, cls2), class_medians(S, f, cls2)
        ok = mt.notna() & ms.notna()
        if ok.sum() >= 4:
            rho = spearmanr(mt[ok], ms[ok]).correlation
            rows.append({"feature": f, "grp": f[0], "rho_TS": rho, "n_cls": int(ok.sum())})
    d = pd.DataFrame(rows).set_index("feature")
    # KS di E14 per incrocio
    try:
        ks = pd.read_csv(FIG / "e14_domain_shift.csv").set_index("feature")["ks"]
        d["ks_E14"] = d.index.map(ks)
    except Exception:
        d["ks_E14"] = np.nan
    d.to_csv(FIG / "e27_universality.csv")

    print(f"\n── 2-vie Trento↔SHL ({len(d)} feature, {len(cls2)} classi) ──")
    print(f"Spearman mediano: {d.rho_TS.median():.2f} | frazione con ordinamento PRESERVATO (ρ≥0.6): {100*(d.rho_TS>=0.6).mean():.0f}%")
    print("Spearman medio per gruppo:", d.groupby("grp").rho_TS.median().round(2).to_dict())

    # 🔑 disentangle del caveat E14: le high-shift (KS>0.5) preservano l'ordinamento?
    hi = d[d.ks_E14 > 0.5].sort_values("ks_E14", ascending=False)
    if len(hi):
        print(f"\n🔑 high-shift di E14 (KS>0.5, n={len(hi)}): ordinamento per-classe preservato?")
        for f, r in hi.iterrows():
            print(f"  {f:24s} KS {r.ks_E14:.2f} → Spearman {r.rho_TS:+.2f}  {'(preservato)' if r.rho_TS>=0.6 else '(ROTTO)' if r.rho_TS<0.2 else '(parziale)'}")
        print(f"  → Spearman mediano delle high-shift: {hi.rho_TS.median():+.2f} "
              f"({'SHIFT = CALIBRAZIONE (struttura preservata, transfer ok con norm.)' if hi.rho_TS.median()>=0.6 else 'STRUTTURA ROTTA (negative-transfer reale)'})")

    # ── 3-vie sul backbone (13 GPS, 4 classi) ──
    print(f"\n── 3-vie sul backbone ({len(backbone)} GPS-feat, {len(cls3)} classi: {cls3}) ──")
    Z = {}  # per heatmap: feature -> {dataset: zscored per-class vector}
    rho3 = []
    for f in backbone:
        mt, ms, mg = class_medians(T, f, cls3), class_medians(S, f, cls3), class_medians(G, f, cls3)
        def z(s):
            sd = s.std(ddof=0)
            return (s - s.mean()) / sd if sd > 1e-12 else s * 0
        Z[f] = {"Trento": z(mt), "SHL": z(ms), "GeoLife": z(mg)}
        pairs = {}
        for a, b, nm in [(mt, ms, "T-S"), (mt, mg, "T-G"), (ms, mg, "S-G")]:
            ok = a.notna() & b.notna()
            pairs[nm] = spearmanr(a[ok], b[ok]).correlation if ok.sum() >= 3 else np.nan
        rho3.append({"feature": f, **pairs})
    r3 = pd.DataFrame(rho3).set_index("feature")
    print("Spearman 3-vie (mediana per coppia di dataset):",
          {k: round(np.nanmedian(r3[k]), 2) for k in ["T-S", "T-G", "S-G"]})

    # ── FIG 1: heatmap z-score per-classe — backbone × (3 dataset × 4 classi) ──
    feats_h = [f for f in backbone if all(Z[f][ds].notna().sum() >= 3 for ds in ("Trento", "SHL", "GeoLife"))]
    DS = ["Trento", "SHL", "GeoLife"]
    M = np.full((len(feats_h), len(DS) * len(cls3)), np.nan)
    for i, f in enumerate(feats_h):
        for j, ds in enumerate(DS):
            for k, c in enumerate(cls3):
                M[i, j * len(cls3) + k] = Z[f][ds].get(c, np.nan)
    plt.figure(figsize=(10, max(3, 0.5 * len(feats_h))))
    im = plt.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)
    plt.colorbar(im, label="per-class median (z-scored within dataset)")
    plt.yticks(range(len(feats_h)), feats_h, fontsize=8)
    plt.xticks(range(len(DS) * len(cls3)), cls3 * len(DS), rotation=90, fontsize=7)
    for j in range(1, len(DS)):
        plt.axvline(j * len(cls3) - 0.5, color="k", lw=1.5)
    for j, ds in enumerate(DS):
        plt.text(j * len(cls3) + len(cls3) / 2 - 0.5, -0.75, ds, ha="center", fontsize=10, weight="bold")
    plt.title("Physical per-class signatures across 3 datasets\n(same color pattern = universal structure)",
              fontsize=10, pad=24)
    savefig("e27_universality_heatmap")

    # ── FIG 2: Spearman ordinamento delle high-shift di E14 (caveat disentangle) ──
    if len(hi):
        hh = hi.sort_values("rho_TS")
        plt.figure(figsize=(7, max(3, 0.4 * len(hh))))
        colors = ["tab:green" if v >= 0.6 else "tab:orange" if v >= 0.2 else "tab:red" for v in hh.rho_TS]
        plt.barh(range(len(hh)), hh.rho_TS.values, color=colors)
        plt.yticks(range(len(hh)), [f"{f}\n(KS={hh.ks_E14[f]:.2f})" for f in hh.index], fontsize=7)
        plt.axvline(0.6, color="k", ls="--", lw=1, label="ordering preserved (ρ≥0.6)")
        plt.xlabel("Spearman of per-class ranking (Trento vs SHL)")
        plt.title("Do the high-shift features (E14) keep their per-class ordering?")
        plt.legend(fontsize=8); plt.grid(alpha=.3, axis="x"); plt.xlim(-1.05, 1.05)
        savefig("e27_ordering_highshift")

    print("\nfigure → e27_universality_heatmap · e27_ordering_highshift | tabella → e27_universality.csv")


if __name__ == "__main__":
    main()
