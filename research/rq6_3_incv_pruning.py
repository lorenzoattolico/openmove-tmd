"""
research/rq6_3_incv_pruning.py — feature_selection §6.3: potatura-ridondanza DENTRO i fold.

Scopo:    BLINDATURA del rigore. La potatura Tier-B (ridondanza Spearman≥0.95) di E6 è fatta su
          ALL-DATA → bias lieve (selezione fuori dai fold, §3.5). Qui si fa la potatura DENTRO ogni
          train-fold e si verifica: (1) F1 in-fold-pruned ≈ all-data-163 ≈ full-230 → la selezione
          all-data NON gonfia (nessun leakage); (2) il set selezionato è STABILE tra fold (Jaccard
          alto) → selezione riproducibile, non rumore-di-fold. È la difesa "selezione dentro CV"
          raccomandata in letteratura (de Jong, Lones 2021).
Metodo:   full-230 (effettivo, no all-NaN). Per ogni rolling fold: su TRAIN compute |Spearman|,
          greedy-drop ≥0.95 (tieni 1 rappresentante per cluster) → K_fold; train RF su K_fold,
          full, e fixed-163 (E6 all-data); eval GPS-present. Stabilità = Jaccard medio tra K_fold.
Input:    data/v2/features_trento_full.parquet · data/v2/features_trento.parquet (per il set 163)
Output:   riepilogo numerico stdout (nessuna figura).
Alimenta: thesis/feature_selection.md §6.3 + thesis/results.md. Sez.tesi: 4.5.

Run: python research/rq6_3_incv_pruning.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
from tmd.training.trainer import get_feature_cols, rolling_splits  # noqa: E402
from tmd.models.hierarchical import HierarchicalTMD  # noqa: E402

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
RHO = 0.95


def macro5(yt, yp):
    seen = [c for c in FIVE if (np.asarray(yt) == c).sum() > 0]
    return f1_score(yt, yp, labels=seen, average="macro", zero_division=0)


def matrix(df, feats):
    X = np.full((len(df), len(feats)), np.nan, np.float32)
    for i, c in enumerate(feats):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    return X


def greedy_prune(df_train, feats, rho=RHO):
    """Tieni 1 rappresentante per cluster |Spearman|>=rho (calcolato sul TRAIN fold)."""
    X = df_train[feats].apply(lambda c: c.fillna(c.median()))
    corr = X.corr(method="spearman").abs().fillna(0.0)
    kept = []
    for f in feats:                                   # ordine stabile = ordine colonne
        if all(corr.loc[f, k] < rho for k in kept):
            kept.append(f)
    return kept


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def main():
    full = pd.read_parquet(ROOT / "data/v2/features_trento_full.parquet")
    feats_full = get_feature_cols(full, ["A", "B", "C", "D"])
    feats_163 = get_feature_cols(pd.read_parquet(ROOT / "data/v2/features_trento.parquet"), ["A", "B", "C", "D"])
    silver = full[full.silver_label.notna()].copy()
    evalp = full[full.label.notna()].copy()

    print("=" * 66); print("§6.3 — potatura-ridondanza DENTRO i fold (blindatura)"); print("=" * 66)
    print(f"full effettivo={len(feats_full)} | fixed-163(E6 all-data)={len(feats_163)} | Spearman≥{RHO}\n")

    rows, kept_sets = [], []
    for name, dtr, dte in rolling_splits(silver, evalp):
        dte = dte[dte.label.notna() & (dte.gps_frac > 0.5)]
        if len(dte) < 10:
            continue
        yte = dte.label.values
        cls = [c for c in full.label.dropna().unique() if c in set(dtr.silver_label)]
        K = greedy_prune(dtr, feats_full)
        kept_sets.append(K)

        def fe(feats):
            m = HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(dtr, feats), dtr.silver_label.values)
            return macro5(yte, m.predict(matrix(dte, feats)))

        rows.append((fe(feats_full), fe(feats_163), fe(K), len(K)))
        print(f"  {name}: full {rows[-1][0]:.3f} | fixed-163 {rows[-1][1]:.3f} | "
              f"in-fold({len(K)}) {rows[-1][2]:.3f}")

    a = np.array([r[:3] for r in rows])
    ksz = [r[3] for r in rows]
    print(f"\n  MEDIA-fold: full-230 {a[:,0].mean():.3f} | fixed-163 {a[:,1].mean():.3f} | "
          f"in-fold-pruned {a[:,2].mean():.3f}")
    jacs = [jaccard(kept_sets[i], kept_sets[j]) for i in range(len(kept_sets)) for j in range(i + 1, len(kept_sets))]
    print(f"  in-fold kept-set: |K| medio {np.mean(ksz):.0f} | Jaccard medio tra fold {np.mean(jacs):.2f}")
    j163 = np.mean([jaccard(K, feats_163) for K in kept_sets])
    print(f"  Jaccard in-fold vs fixed-163(E6): {j163:.2f}")
    print(f"\nLettura: le 3 colonne ~uguali ⇒ selezione all-data NON gonfia (no leakage); "
          f"Jaccard alto ⇒ set stabile/riproducibile (la selezione è robusta, non rumore di fold).")


if __name__ == "__main__":
    main()
