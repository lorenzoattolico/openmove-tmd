"""
e6_feature_select.py — variable selection su OpenMove (Fase 1c · task E6).

Scopo:    ri-derivare la selezione feature SUI DATI OpenMove (non ereditare il DROP_LIST SHL).
          Criteri ROBUSTI (dati pochi → niente pruning fragile):
            - DROP costanti (n_unique<=1) e quasi-costanti (top-freq >= 99.5%);
            - DROP always-NaN (es. B_alt_* — OpenMove non ha altitudine);
            - DROP ridondanti: Spearman >= 0.95 su OpenMove (cluster → tieni 1 rappresentante);
            - FLAG (NON drop) rischio-transfer per PRINCIPIO (l'ablazione 1e decide):
              B_n_gps, D_has_reliable_gps (disponibilità-GPS), A_acc_d2v (qualità stradale),
              orientation (grav/gyr mean).
          Check stratificato: %NaN su GPS-present (B/C devono esserci quando il GPS c'è).
          Cross-check vs DROP_LIST SHL (accordo/disaccordo sul target).
Input:    data/v2/features_trento_full.parquet  (230 feature; offline)
Output:   research/figures/e6_feature_audit.csv        (tabella per-feature + azione)
          research/figures/e6_nan_per_feature.{png,pdf}
          research/figures/e6_maxcorr.{png,pdf}        (ridondanza: max |Spearman| per feature)
Alimenta: thesis/eda.md §3 (E6) → decide il set canonico (G-features)
Sez.tesi: 4.5 selezione variabili

Run: /opt/miniconda3/envs/tmd/bin/python research/e6_feature_select.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "v2" / "features_trento_full.parquet"
FIG = ROOT / "thesis" / "figures"
CORR_THRESH = 0.95
NEARCONST = 0.995

# rischio-transfer per PRINCIPIO (deployment/locale-specifiche, orientation-dipendenti)
TRANSFER_RISK = ("B_n_gps", "D_has_reliable_gps", "A_acc_d2v",
                 "A_grav_", "A_gyr_x_mean", "A_gyr_y_mean", "A_gyr_z_mean")


def is_transfer_risk(f: str) -> bool:
    return any(f == p or f.startswith(p) for p in TRANSFER_RISK)


def main() -> None:
    if not FULL.exists():
        sys.exit(f"Manca {FULL} — run_pipeline --all-features?")
    df = pd.read_parquet(FULL)
    feat = [c for c in df.columns if c[:2] in ("A_", "B_", "C_", "D_")]
    gpres = pd.to_numeric(df["gps_frac"], errors="coerce") > 0.5
    X = df[feat].apply(pd.to_numeric, errors="coerce")

    # ── metriche per-feature ──────────────────────────────────────────────────
    rows = []
    for f in feat:
        s = X[f]
        nona = s.dropna()
        nun = nona.nunique()
        topfreq = (nona.value_counts(normalize=True).iloc[0] if len(nona) else 1.0)
        rows.append({
            "feature": f, "group": f[0],
            "pct_nan": 100 * s.isna().mean(),
            "pct_nan_gpspres": 100 * s[gpres].isna().mean(),
            "n_unique": int(nun),
            "top_freq": 100 * topfreq,
            "transfer_risk": is_transfer_risk(f),
        })
    A = pd.DataFrame(rows).set_index("feature")

    # ── ridondanza: Spearman pairwise (pandas gestisce NaN pairwise-complete) ──
    corr = X.corr(method="spearman").abs()
    corr = corr.where(~np.eye(len(corr), dtype=bool), 0.0)   # azzera diagonale (no self-corr)
    A["max_corr"] = corr.max()
    A["corr_with"] = corr.idxmax()

    # cluster ridondanti (componenti connesse su |corr|>=THRESH) → tieni 1 rappresentante
    drop_const = set(A.index[(A.n_unique <= 1) | (A.top_freq >= 100 * NEARCONST)])
    drop_nan = set(A.index[A.pct_nan >= 99.9])
    # B/C quasi-sempre indefinite anche con GPS presente (es. stop_regularity senza fermate) → troppo sparse
    drop_sparse = set(A.index[A.group.isin(["B", "C"]) & (A.pct_nan_gpspres > 80)])
    hard_drop = drop_const | drop_nan | drop_sparse
    cand = [f for f in feat if f not in hard_drop]
    adj = {f: set() for f in cand}
    cm = corr.loc[cand, cand]
    for i, f in enumerate(cand):
        for g in cand[i + 1:]:
            if cm.at[f, g] >= CORR_THRESH:
                adj[f].add(g); adj[g].add(f)
    seen, redundant = set(), set()
    clusters = []
    for f in cand:
        if f in seen or not adj[f]:
            seen.add(f); continue
        comp, stack = [], [f]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.append(x)
            stack.extend(adj[x] - seen)
        if len(comp) > 1:
            # rappresentante = meno NaN, poi nome più corto
            rep = sorted(comp, key=lambda c: (A.at[c, "pct_nan"], len(c)))[0]
            redundant |= (set(comp) - {rep})
            clusters.append((rep, sorted(set(comp) - {rep})))

    def action(f):
        if f in drop_const: return ("DROP", "constant/near-constant")
        if f in drop_nan:   return ("DROP", "always-NaN")
        if f in drop_sparse: return ("DROP", "too-sparse (>80% NaN on GPS-present)")
        if f in redundant:  return ("DROP", f"redundant (Spearman≥{CORR_THRESH})")
        if A.at[f, "transfer_risk"]: return ("FLAG", "transfer-risk (ablation 1e)")
        return ("KEEP", "")
    A["action"] = [action(f)[0] for f in A.index]
    A["reason"] = [action(f)[1] for f in A.index]
    FIG.mkdir(parents=True, exist_ok=True)
    A.sort_values(["action", "group", "feature"]).to_csv(FIG / "e6_feature_audit.csv")

    keep = A[A.action.isin(["KEEP", "FLAG"])]
    drop = A[A.action == "DROP"]

    # ── cross-check vs DROP_LIST SHL ──────────────────────────────────────────
    from tmd.features import DROP_LIST
    shl_drop_present = set(DROP_LIST) & set(feat)          # SHL-drop che ESISTONO nel full OpenMove
    my_drop = set(drop.index)
    agree = shl_drop_present & my_drop
    shl_only = shl_drop_present - my_drop                  # SHL droppa, io tengo
    me_only = my_drop - set(DROP_LIST)                     # io droppo, SHL no

    # ── riepilogo (→ thesis/eda.md) ─────────────────────────────────────────────
    print("=" * 64); print("E6 — VARIABLE SELECTION su OpenMove (full 230 feat)"); print("=" * 64)
    print(f"feature totali: {len(feat)} | per gruppo: {A.group.value_counts().to_dict()}")
    print(f"\nAZIONI: KEEP {int((A.action=='KEEP').sum())} | FLAG-transfer {int((A.action=='FLAG').sum())} "
          f"| DROP {len(drop)}  →  set proposto (KEEP+FLAG) = {len(keep)}")
    print(f"  DROP costanti/near-const: {len(drop_const)} | always-NaN: {len(drop_nan)} "
          f"| sparse(>80%NaN-present): {len(drop_sparse)} {sorted(drop_sparse)} | ridondanti: {len(redundant)}")
    print(f"\nCheck NaN (verificato in E6-deep):")
    print(f"  A (IMU): %NaN uniforme ~{A[A.group=='A'].pct_nan.median():.1f}% = finestre n_imu==0 "
          f"(stretch GPS-only entro tracking, R1) → benigno, gestito da imputer")
    cond = A[A.group.isin(['B','C']) & (A.action != 'DROP') & (A.pct_nan_gpspres > 5)]
    print(f"  B/C tenute ma condizionali (NaN-su-present >5%, es. dwell senza fermate): "
          f"{ {f: round(A.at[f,'pct_nan_gpspres'],0) for f in cond.index} }")
    print(f"\nCluster ridondanti (Spearman≥{CORR_THRESH}): {len(clusters)}")
    for rep, drp in clusters[:8]:
        print(f"  tieni {rep}  ⟵ droppa {drp}")
    if len(clusters) > 8:
        print(f"  … (+{len(clusters)-8} cluster, vedi CSV)")
    print(f"\nFLAG transfer-risk: {sorted(keep.index[keep.transfer_risk])[:10]}{' …' if keep.transfer_risk.sum()>10 else ''}")
    print(f"\nCROSS-CHECK vs DROP_LIST SHL (presenti nel full: {len(shl_drop_present)}):")
    print(f"  accordo (entrambi droppano): {len(agree)}")
    print(f"  SHL droppa MA io tengo: {len(shl_only)} → {sorted(shl_only)[:8]}{' …' if len(shl_only)>8 else ''}")
    print(f"  io droppo MA SHL no:     {len(me_only)} → {sorted(me_only)[:8]}{' …' if len(me_only)>8 else ''}")

    # ── figure (EN) ───────────────────────────────────────────────────────────
    # 1) %NaN per feature (sorted), colore per gruppo
    s = A.sort_values("pct_nan", ascending=False)
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(s)), s.pct_nan.values, color=[colors[g] for g in s.group])
    plt.xlabel("feature (sorted by %NaN)"); plt.ylabel("% NaN (all windows)")
    plt.title("Feature missingness on OpenMove (B/C NaN when GPS absent)")
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color=colors[g], label=g) for g in "ABCD"], fontsize=8)
    plt.grid(alpha=.3, axis="y"); savefig_local("e6_nan_per_feature")

    # 2) ridondanza: max |Spearman| per feature
    plt.figure(figsize=(6.5, 4))
    plt.hist(A.max_corr.dropna(), bins=40)
    plt.axvline(CORR_THRESH, ls="--", c="red", label=f"≥{CORR_THRESH} → redundant")
    plt.xlabel("max |Spearman| with any other feature"); plt.ylabel("features")
    plt.title("Feature redundancy on OpenMove"); plt.legend(); plt.grid(alpha=.3)
    savefig_local("e6_maxcorr")

    print(f"\ntabella → research/figures/e6_feature_audit.csv | figure → e6_{{nan_per_feature,maxcorr}}")


def savefig_local(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
