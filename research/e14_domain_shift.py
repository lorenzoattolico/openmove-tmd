"""
e14_domain_shift.py — domain-shift Trento↔SHL per feature (Fase 1c · task E14).

Scopo:    per ogni feature COMUNE ai due dataset, quanto shifta la distribuzione (KS 2-sample)?
          → feature trasferibili (KS basso) vs a rischio negative-transfer (KS alto). Sblocca il
          transfer (con E15 definisce il set trasferibile) e valida i FLAG transfer-risk di E6.
Input:    data/v2/features_trento_full.parquet (Trento, 230) · data/processed/features_shl_bootstrap.parquet (SHL)
          research/figures/e6_feature_audit.csv (FLAG transfer-risk)
Output:   research/figures/e14_ks_top.{png,pdf} · e14_ks_by_group.{png,pdf} · e14_domain_shift.csv
Alimenta: thesis/eda.md (E14)
Sez.tesi: 6.3 transfer / 4.5

Caveat: KS sulla MARGINALE per feature (conflà class-mix + domain). Sanity normalizzazione prima.
Run: /opt/miniconda3/envs/tmd/bin/python research/e14_domain_shift.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    T = pd.read_parquet(ROOT / "data/v2/features_trento_full.parquet")
    S = pd.read_parquet(ROOT / "data/v2/features_shl_full.parquet")  # tmd-aligned (230)
    feat = sorted(set(c for c in T.columns if c[:2] in ("A_", "B_", "C_", "D_")) & set(S.columns))
    print("=" * 64); print("E14 — domain-shift Trento↔SHL (KS per feature comune)"); print("=" * 64)
    print(f"feature comuni Trento∩SHL: {len(feat)} (Trento {sum(1 for c in T.columns if c[:2] in ('A_','B_','C_','D_'))} / SHL {sum(1 for c in S.columns if c[:2] in ('A_','B_','C_','D_'))})")

    # sanity normalizzazione (A su scala comune?)
    print("\nsanity normalizzazione (mediana A_acc_*; atteso scale simili):")
    for f in ["A_acc_mag_mean", "A_acc_sma", "A_acc_mag_rms"]:
        if f in feat:
            print(f"  {f:16s} Trento {T[f].median():.2f} | SHL {S[f].median():.2f}")

    # KS per feature
    rows = []
    for f in feat:
        t, s = T[f].dropna().values, S[f].dropna().values
        if len(t) > 50 and len(s) > 50:
            rows.append({"feature": f, "grp": f[0], "ks": ks_2samp(t, s).statistic})
    d = pd.DataFrame(rows).set_index("feature").sort_values("ks", ascending=False)
    # FLAG transfer-risk da E6
    try:
        e6 = pd.read_csv(FIG / "e6_feature_audit.csv").set_index("feature")
        d["e6_flag"] = d.index.map(lambda f: e6.at[f, "action"] if f in e6.index else "-")
    except Exception:
        d["e6_flag"] = "-"
    d.to_csv(FIG / "e14_domain_shift.csv")

    print(f"\nKS mediano su {len(d)} feature: {d.ks.median():.2f} | feature con KS>0.5 (shift forte): {(d.ks>0.5).sum()}")
    print("\nTop-12 feature più SHIFTATE (rischio negative-transfer):")
    for f, r in d.head(12).iterrows():
        print(f"  {f:26s} KS {r.ks:.2f}  [{r.grp}]  e6={r.e6_flag}")
    print("\nBottom-8 (più TRASFERIBILI):")
    for f, r in d.tail(8).iterrows():
        print(f"  {f:26s} KS {r.ks:.2f}  [{r.grp}]")
    print("\nKS medio per GRUPPO:", d.groupby("grp").ks.mean().round(2).to_dict())
    flag = d[d.e6_flag == "FLAG"]
    if len(flag):
        print(f"\nFLAG transfer-risk (E6): KS medio {flag.ks.mean():.2f} vs resto {d[d.e6_flag!='FLAG'].ks.mean():.2f} "
              f"→ {'i FLAG shiftano di più (validati)' if flag.ks.mean() > d[d.e6_flag!='FLAG'].ks.mean() else 'i FLAG NON shiftano più del resto'}")

    # ── figure (EN) ──
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
    top = d.head(20)[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(range(len(top)), top.ks.values, color=[colors[g] for g in top.grp])
    plt.yticks(range(len(top)), top.index, fontsize=7)
    plt.xlabel("KS statistic (Trento vs SHL)"); plt.title("Top-20 domain-shifted features (transfer risk)")
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color=colors[g], label=g) for g in "ABCD"], fontsize=8)
    plt.grid(alpha=.3, axis="x"); savefig("e14_ks_top")

    by = d.groupby("grp").ks.mean().reindex(list("ABCD")).dropna()
    plt.figure(figsize=(6, 4))
    plt.bar(by.index, by.values, color=[colors[g] for g in by.index])
    plt.ylabel("mean KS (Trento vs SHL)"); plt.xlabel("feature group")
    plt.title("Mean domain-shift by feature group"); plt.grid(alpha=.3, axis="y")
    savefig("e14_ks_by_group")

    print(f"\ntabella → research/figures/e14_domain_shift.csv | figure → e14_{{ks_top,ks_by_group}}")


if __name__ == "__main__":
    main()
