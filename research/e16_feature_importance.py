"""
e16_feature_importance.py — feature importance model-based + ridondanza (Fase 1c · task E16).

Scopo:    complementa E12 (MI univariata) con la PERMUTATION IMPORTANCE (model-based, multivariata,
          robusta alla correlazione): quali feature il modello USA davvero? E quante feature ad alta
          MI sono RIDONDANTI (alta MI ma bassa importanza perché correlate)? → informa la parsimonia
          del set per il rebuild. Aggregato per gruppo A/B/C/D (triangola E12/E17).
Metodo:   RF 300 con split session-grouped (no-leakage E23) su GT GPS-present; permutation_importance
          (n_repeats=10) sul test. MI ricalcolata inline per il confronto MI↔perm (ridondanza).
Input:    data/v2/features_trento.parquet (label GT + 163 feat + session_id + gps_frac)
Output:   research/figures/e16_perm_importance.{png,pdf} · e16_mi_vs_perm.{png,pdf}
          research/figures/e16_feature_importance.csv
Alimenta: thesis/eda.md (E16)
Sez.tesi: 4.5 selezione variabili

Lettura: perm-imp dà il set EFFETTIVO (più piccolo della MI per ridondanza); per-gruppo conferma
         B/C dominanti per-feature (E12). MI-alta+perm-bassa = ridondante (potabile).
Run: /opt/miniconda3/envs/tmd/bin/python research/e16_feature_importance.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
SEED = 0
CLASSES = ["Still", "Walk", "Bus", "Car", "Train"]
COL = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    g = df[(df.gps_frac > 0.5) & df.label.notna() & df.label.isin(CLASSES)].copy().reset_index(drop=True)
    feat = [c for c in g.columns if c[:2] in ("A_", "B_", "C_", "D_")]
    X = SimpleImputer(strategy="median").fit_transform(g[feat])
    le = LabelEncoder(); y = le.fit_transform(g.label)
    groups = g.session_id.to_numpy()
    print("=" * 70); print("E16 — feature importance model-based + ridondanza"); print("=" * 70)
    print(f"finestre GT GPS-present (5 classi): {len(g)} | feature: {len(feat)} (session-grouped split)")

    # split session-grouped (1 fold come hold-out)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    tr, te = next(sgkf.split(X, y, groups))
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
    print(f"hold-out: train {len(tr)} / test {len(te)} | test acc {clf.score(X[te], y[te]):.3f}")

    # permutation importance sul test
    pi = permutation_importance(clf, X[te], y[te], n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=feat)
    # MI inline (per il confronto ridondanza)
    mi = pd.Series(mutual_info_classif(X[tr], y[tr], random_state=SEED), index=feat)
    out = pd.DataFrame({"perm_imp": imp, "mi": mi, "grp": [f[0] for f in feat]}).sort_values("perm_imp", ascending=False)
    out.round(4).to_csv(FIG / "e16_feature_importance.csv")

    print("\nTop-15 permutation importance (set EFFETTIVO del modello):")
    for f, r in out.head(15).iterrows():
        print(f"  {f:28s} perm {r.perm_imp:.4f}  [{r.grp}]")
    # concentrazione
    pos = out[out.perm_imp > 0].perm_imp
    print(f"\nconcentrazione: top-10 = {100*out.perm_imp.head(10).sum()/pos.sum():.0f}% dell'importanza positiva; "
          f"top-30 = {100*out.perm_imp.head(30).sum()/pos.sum():.0f}% | feature con perm-imp>0: {(out.perm_imp>1e-5).sum()}/{len(feat)}")

    # per gruppo
    grp = out.groupby("grp").perm_imp.agg(["sum", "mean", "count"]).reindex(list("ABCD"))
    print("\nimportanza per GRUPPO (somma / media-per-feature / n):")
    print(grp.round(4).to_string())

    # ridondanza: MI-alta ma perm-bassa
    out["mi_rank"] = out.mi.rank(ascending=False)
    out["perm_rank"] = out.perm_imp.rank(ascending=False)
    redund = out[(out.mi_rank <= 30) & (out.perm_rank > 60)].sort_values("mi", ascending=False)
    print(f"\nRIDONDANTI (MI top-30 ma perm-imp oltre-60°): {len(redund)} feature potabili — es.:")
    for f in redund.head(8).index:
        print(f"  {f:28s} MI-rank {int(out.mi_rank[f]):3d} → perm-rank {int(out.perm_rank[f]):3d}")

    # ── FIG 1: top-20 permutation importance ──
    top = out.head(20)[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(range(len(top)), top.perm_imp.values, color=[COL[gp] for gp in top.grp])
    plt.yticks(range(len(top)), top.index, fontsize=7)
    plt.xlabel("permutation importance (Δ accuracy)"); plt.title("Top-20 features the model actually uses")
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color=COL[k], label=k) for k in "ABCD"], fontsize=8)
    plt.grid(alpha=.3, axis="x"); savefig("e16_perm_importance")

    # ── FIG 2: MI vs perm (ridondanza) ──
    plt.figure(figsize=(6.5, 5))
    plt.scatter(out.mi, out.perm_imp, c=[COL[gp] for gp in out.grp], s=18, alpha=.7)
    plt.xlabel("mutual information (E12, univariate)"); plt.ylabel("permutation importance (multivariate)")
    plt.title("MI vs model importance: high-MI/low-perm = redundant")
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color=COL[k], label=k) for k in "ABCD"], fontsize=8)
    plt.grid(alpha=.3); savefig("e16_mi_vs_perm")

    print("\nfigure → e16_perm_importance · e16_mi_vs_perm | tabella → e16_feature_importance.csv")


if __name__ == "__main__":
    main()
