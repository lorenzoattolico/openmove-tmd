"""
research/rq1_6_silver_vs_random.py — RQ1.6: silver vs random-label (isola QUALITÀ-LF).

Scopo:    dimostrare che il guadagno viene dalla QUALITÀ delle labeling-function fisiche, NON dalla
          quantità di dati. Per una tesi sul weak-supervision è quasi obbligatorio (atteso da referee).
          Confronto, a parità di finestre/feature/modello:
            - silver        : silver_label fisico (reale)
            - random        : silver_label PERMUTATO (stessa distribuzione marginale, zero relazione
                              feature↔label) → misura quanto "dà" la sola quantità-dati
            - majority       : predici sempre la classe maggioritaria (Still) → floor banale
Metodo:   test FISSO = split temporale, motiontag GT, GPS-present (operativo). RF gerarchico.
          Il random è mediato su più permutazioni (seed).
Input:    data/v2/features_trento.parquet
Output:   riepilogo numerico stdout (nessuna figura).
Alimenta: thesis/results.md (silver-vs-random 1.6). Sez.tesi: 5.x weak-supervision / 7.

Run: python research/rq1_6_silver_vs_random.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
from tmd.training.trainer import get_feature_cols, temporal_splits  # noqa: E402
from tmd.models.hierarchical import HierarchicalTMD  # noqa: E402

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
CITY = ["Still", "Walk", "Bike", "Car", "Bus", "Train"]
SEEDS = [0, 1, 2, 3, 4]


def macro5(yt, yp):
    seen = [c for c in FIVE if (np.asarray(yt) == c).sum() > 0]
    return f1_score(yt, yp, labels=seen, average="macro", zero_division=0)


def matrix(df, feats):
    X = np.full((len(df), len(feats)), np.nan, np.float32)
    for i, c in enumerate(feats):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    return X


def train_eval(Xtr, ytr, Xte, yte):
    cls = [c for c in CITY if c in set(ytr)]
    m = HierarchicalTMD(cls, [], clf_type="rf").fit(Xtr, ytr)
    yp = m.predict(Xte)
    return macro5(yte, yp), accuracy_score(yte, yp)


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    silver = df[df.silver_label.notna()].copy()
    evalp = df[df.label.notna()].copy()
    feats = get_feature_cols(silver, ["A", "B", "C", "D"])
    _, pool, test = next(iter(temporal_splits(silver, evalp)))
    test = test[test.gps_frac > 0.5]
    Xtr, ytr = matrix(pool, feats), pool.silver_label.values
    Xte, yte = matrix(test, feats), test.label.values

    print("=" * 60); print("RQ1.6 — silver vs random-label (isola qualità-LF)"); print("=" * 60)
    print(f"train pool = {len(pool)} | test GPS-present = {len(test)}\n")

    f_sil, a_sil = train_eval(Xtr, ytr, Xte, yte)
    print(f"  silver (LF fisico)   : macro-F1 {f_sil:.3f}  acc {a_sil:.3f}")

    rs = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        f, a = train_eval(Xtr, rng.permutation(ytr), Xte, yte)
        rs.append((f, a))
    rs = np.array(rs)
    print(f"  random (label permut): macro-F1 {rs[:,0].mean():.3f}±{rs[:,0].std():.3f}  "
          f"acc {rs[:,1].mean():.3f}±{rs[:,1].std():.3f}  (media {len(SEEDS)} perm)")

    # majority floor
    maj = pd.Series(ytr).value_counts().idxmax()
    f_maj = macro5(yte, np.full(len(yte), maj))
    a_maj = accuracy_score(yte, np.full(len(yte), maj))
    print(f"  majority ('{maj}')      : macro-F1 {f_maj:.3f}  acc {a_maj:.3f}")

    print(f"\n  → guadagno qualità-LF (silver − random): macro-F1 +{f_sil - rs[:,0].mean():.3f}")
    print(f"     il random ha la STESSA quantità-dati e distribuzione del silver → il salto è "
          f"TUTTO qualità delle LF fisiche (non quantità).")


if __name__ == "__main__":
    main()
