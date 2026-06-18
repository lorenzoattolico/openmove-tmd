"""
research/rq2_4_hier_vs_flat.py — RQ2.4: gerarchico (L1 Still/Moving + L2) vs flat.

Scopo:    chiudere il nodo "la gerarchia serve?". Single-split pre-freeze dava hier ≫ flat
          (contaminato); il rolling de-contaminato dava hier≈flat IN-DOMAIN Trento. Qui si
          conferma su SHL nativo (GT pulito) e su TRANSFER Trento→SHL — i due regimi non
          ancora testati. Se hier≈flat ovunque → si tiene per principio (interpretabilità),
          non per performance; se hier aiuta su un regime → lo si motiva.
Metodo:   stesso clf base (RF, canonico) e stesse feature; 5 classi comuni (Still/Walk/Car/
          Bus/Train). HIER = HierarchicalTMD; FLAT = singolo RF multiclasse (median-impute).
          (A) SHL in-domain: train SHL-train → eval SHL-validate (163 feat comuni).
          (B) Transfer: train Trento silver → eval SHL-validate (163 feat).
Input:    data/v2/features_shl_full.parquet · data/v2/features_trento.parquet
Output:   riepilogo numerico stdout (nessuna figura).
Alimenta: thesis/results.md (hier-vs-flat 2.4). Sez.tesi: 5.x architettura modello.

Run: python research/rq2_4_hier_vs_flat.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
from tmd.training.trainer import get_feature_cols  # noqa: E402
from tmd.models.hierarchical import HierarchicalTMD  # noqa: E402

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
RF = dict(n_estimators=400, max_depth=None, min_samples_leaf=5,
          n_jobs=-1, random_state=42, class_weight="balanced")


def macro(y, p):
    seen = [c for c in FIVE if (np.asarray(y) == c).sum() > 0]
    return f1_score(y, p, labels=seen, average="macro", zero_division=0)


class Flat:
    """RF multiclasse piatto (median-impute), stessi iperparametri del base RF gerarchico."""
    def __init__(self):
        self.imp = SimpleImputer(strategy="median")
        self.clf = RandomForestClassifier(**RF)

    def fit(self, X, y):
        self.clf.fit(self.imp.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.clf.predict(self.imp.transform(X))


def fit_eval(Xtr, ytr, Xte, yte, classes):
    h = HierarchicalTMD(classes, [], clf_type="rf").fit(Xtr, ytr)
    f = Flat().fit(Xtr, ytr)
    return macro(yte, h.predict(Xte)), macro(yte, f.predict(Xte))


def matrix(df, feats):
    X = np.full((len(df), len(feats)), np.nan, np.float32)
    for i, c in enumerate(feats):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    return X


def main():
    sh = pd.read_parquet(ROOT / "data/v2/features_shl_full.parquet")
    sh = sh[sh.label.isin(FIVE)]
    tr = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")

    print("=" * 60)
    print("RQ2.4 — gerarchico vs flat (RF, 5 classi comuni)")
    print("=" * 60)

    # ── (A) SHL in-domain (fixed split) ──
    feats_shl = get_feature_cols(sh, ["A", "B", "C", "D"])
    shtr, shva = sh[sh.split == "train"], sh[sh.split == "validate"]
    Xtr = matrix(shtr, feats_shl); Xva = matrix(shva, feats_shl)
    h, f = fit_eval(Xtr, shtr.label.values, Xva, shva.label.values,
                    [c for c in FIVE if c in set(shtr.label)])
    print(f"\n(A) SHL in-domain (train={len(shtr)} → validate={len(shva)}, {len(feats_shl)} feat):")
    print(f"    HIER {h:.3f}  |  FLAT {f:.3f}  |  Δ(hier−flat) {h-f:+.3f}")

    # ── (B) Transfer Trento silver → SHL validate (163 feat Trento) ──
    feats_tr = get_feature_cols(tr, ["A", "B", "C", "D"])
    sil = tr[tr.silver_label.notna()]
    Xs = matrix(sil, feats_tr); ys = sil.silver_label.values
    Xv = matrix(shva, feats_tr)
    cls = [c for c in FIVE if c in set(ys)]
    h2, f2 = fit_eval(Xs, ys, Xv, shva.label.values, cls)
    print(f"\n(B) Transfer Trento silver → SHL validate (train={len(sil)}, {len(feats_tr)} feat):")
    print(f"    HIER {h2:.3f}  |  FLAT {f2:.3f}  |  Δ(hier−flat) {h2-f2:+.3f}")

    print(f"\nLettura: |Δ|<0.02 ⇒ neutro (tieni la gerarchia per principio); "
          f"Δ>0 ⇒ la gerarchia aiuta su quel regime.")


if __name__ == "__main__":
    main()
