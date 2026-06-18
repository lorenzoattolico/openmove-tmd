"""
research/rq4_5_walk_universal.py — RQ4.x: figura "Walk universale" cross-dataset.

Scopo:    il volto del transfer (C4): Walk si trasferisce quasi-perfetto su 3 contesti (IT in-domain →
          UK transfer → Cina zero-shot) mentre i motorizzati sono feature-limited e calano cross-dataset.
          → la gait/velocità è un universale fisico; Car/Bus/Train dipendono da velocità/infrastruttura
          locali che trasferiscono meno.
Metodo:   per-classe macro-F1 dei 4 modi in movimento in 3 setting:
            (1) Trento in-domain GPS-present  — eval rolling-OOF canonico
            (2) Transfer Trento→SHL           — parquet transfer ML
            (3) GeoLife zero-shot (Pechino)   — Trento-silver su 27 feat B/C/D comuni, moving-forced
Input:    eval canonico · transfer ML parquet · features_trento · features_geolife
Output:   research/figures/rq4_5_walk_universal.{png,pdf}
Alimenta: thesis/results.md (RQ4, figura transfer). Sez.tesi: 6.3.

Run: python research/rq4_5_walk_universal.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/v2/processed"
from tmd.training.trainer import get_feature_cols  # noqa: E402
from tmd.models.hierarchical import HierarchicalTMD, STILL_CLASS  # noqa: E402

MOV = ["Walk", "Car", "Bus", "Train"]


def perclass(y, p):
    return {c: f1_score(y, p, labels=[c], average="macro", zero_division=0) for c in MOV}


def matrix(df, feats):
    X = np.full((len(df), len(feats)), np.nan, np.float32)
    for i, c in enumerate(feats):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    return X


def predict_moving(model, X):
    Xp = model._prep(X)
    classes_all = [STILL_CLASS] + model.l2_classes
    p_moving = model.l1.predict_proba(Xp)[:, 1]
    p_l2 = model.l2.predict_proba(Xp) * p_moving[:, None]
    Pr = np.zeros((len(X), len(classes_all)))
    Pr[:, 0] = 1 - p_moving
    for i, cls in enumerate(model.le_l2.classes_):
        Pr[:, classes_all.index(cls)] = p_l2[:, i]
    mov = [i for i, c in enumerate(classes_all) if c != STILL_CLASS]
    sub = Pr[:, mov]
    return np.array([classes_all[mov[j]] for j in sub.argmax(1)], object)


def main():
    # (1) Trento in-domain GPS-present
    ev = pd.read_parquet(P / "eval_trento_20260612_202507.parquet")
    ev = ev[(ev.gps_frac > 0.5) & ev.label.isin(MOV)]
    s1 = perclass(ev.label.values, ev.predicted_class.values)

    # (2) Transfer Trento→SHL
    ml = pd.read_parquet(next(P.glob("transfer_trento_20260612_202512_on_features_shl_full_*.parquet")))
    ml = ml[ml.label.isin(MOV)]
    s2 = perclass(ml.label.values, ml.predicted_class.values)

    # (3) GeoLife zero-shot
    g = pd.read_parquet(ROOT / "data/processed/features_geolife.parquet")
    g = g[g.label.isin(MOV) & g.in_china].reset_index(drop=True)   # solo-Cina
    tr = pd.read_parquet(ROOT / "data/v2/features_trento_full.parquet")  # FULL-230: tutte 27 le feat comuni
    feats = sorted(c for c in g.columns if c[:2] in ("B_", "C_", "D_"))
    sil = tr[tr.silver_label.notna()]
    cls = [c for c in ["Still"] + MOV if c in set(sil.silver_label)]
    m = HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(sil, feats), sil.silver_label.values)
    pg = predict_moving(m, matrix(g, feats))
    s3 = perclass(np.asarray(g.label, object), pg)

    settings = [("Trento in-domain", s1), ("Trento→SHL transfer", s2), ("GeoLife zero-shot", s3)]
    print("=" * 60); print("RQ4.x — Walk universale cross-dataset (per-classe F1)"); print("=" * 60)
    print(f"  {'class':<7}" + "".join(f"{nm:>22}" for nm, _ in settings))
    for c in MOV:
        print(f"  {c:<7}" + "".join(f"{s[c]:>22.2f}" for _, s in settings))

    # figura
    x = np.arange(len(MOV)); w = 0.26
    colors = ["tab:green", "tab:orange", "tab:red"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (nm, s) in enumerate(settings):
        ax.bar(x + (i - 1) * w, [s[c] for c in MOV], w, label=nm, color=colors[i])
    ax.axhspan(0.7, 1.0, color="green", alpha=0.05)
    ax.set_xticks(x); ax.set_xticklabels(MOV); ax.set_ylabel("F1 (per class)"); ax.set_ylim(0, 1)
    # niente title in-immagine (C8): la caption LaTeX racconta la figura (banda ombreggiata ≥0.7 = "universale")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"research/figures/rq4_5_walk_universal.{ext}", dpi=150, bbox_inches="tight")
    print("\nfigura → rq4_5_walk_universal.{png,pdf}")


if __name__ == "__main__":
    main()
