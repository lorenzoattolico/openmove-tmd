"""
geolife_full_matrix.py — matrice GeoLife COMPLETA: in-domain · silver(protocollo) · transfer.

China-only (flag in_china), 27 feat GPS+infra (no IMU), 4 classi moving, GroupKFold-5-utente.
Tre blocchi:
  IN-DOMAIN  ceiling per gruppo-feature (B / B+D / B+C / B+C+D), confusione, spread per-utente.
  SILVER     labeler universale -> agreement vs GT (per-classe, coverage/abstain); train locale su
             silver -> macro vs GT; costo-label-freeness in-domain (= supervised - silver).
  TRANSFER   Trento-silver-> · Trento-GT(SDK)-> [isola LF vs supervised, stessa sorgente] · SHL-GT-> ;
             leak-free drop-C sul transfer; rule-based (labeler-as-classifier).

Run: /opt/miniconda3/envs/tmd/bin/python research/geolife_full_matrix.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
from tmd.models.hierarchical import HierarchicalTMD, STILL_CLASS            # noqa: E402
from tmd.config import CityConfig                                          # noqa: E402
from tmd.labeling.window_labeler import label_windows_universal            # noqa: E402
from rq4_3_geolife import matrix, predict_moving           # noqa: E402

FOUR = ["Walk", "Car", "Bus", "Train"]
FIVE = ["Still"] + FOUR
RF = dict(n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight="balanced")


def macro4(yt, yp):
    return f1_score(yt, yp, labels=FOUR, average="macro", zero_division=0)


def perclass(yt, yp):
    return {c: round(v, 2) for c, v in zip(FOUR, f1_score(yt, yp, labels=FOUR, average=None, zero_division=0))}


def grpfeats(feats, groups):
    return [c for c in feats if c.split("_")[0] in groups]


def main():
    g = pd.read_parquet(ROOT / "data/processed/features_geolife.parquet")
    g = g[g.label.isin(FOUR) & g.in_china].reset_index(drop=True)
    tr = pd.read_parquet(ROOT / "data/v2/features_trento_full.parquet")  # FULL-230: tutte 27 le feat comuni (no transfer su set inconsistente)
    sh = pd.read_parquet(ROOT / "data/v2/features_shl_full.parquet")
    feats = sorted(c for c in g.columns if c[:2] in ("B_", "C_", "D_"))
    yg = np.asarray(g.label, object); grp = np.asarray(g.userId, object)
    print(f"GeoLife China-only: {len(g):,} finestre, {g.userId.nunique()} utenti, feat={len(feats)}\n")

    # ===================== IN-DOMAIN =====================
    print("=" * 70); print("IN-DOMAIN (flat RF, GroupKFold-5-utente, GT GeoLife)"); print("=" * 70)
    preds_full = None
    for groups in (["B"], ["B", "D"], ["B", "C"], ["B", "C", "D"]):
        f = grpfeats(feats, groups)
        X = matrix(g, f)
        yp = cross_val_predict(make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(**RF)),
                               X, yg, groups=grp, cv=GroupKFold(5), n_jobs=-1)
        print(f"  {'+'.join(groups):<7} ({len(f):2d} feat)  macro-F1 {macro4(yg, yp):.3f}   {perclass(yg, yp)}")
        if groups == ["B", "C", "D"]:
            preds_full = yp
    print("\n  Matrice di confusione (B+C+D, righe=GT, %):")
    cm = confusion_matrix(yg, preds_full, labels=FOUR)
    cmn = cm / cm.sum(1, keepdims=True)
    print("        " + "".join(f"{c:>8}" for c in FOUR))
    for i, c in enumerate(FOUR):
        print(f"  {c:<6}" + "".join(f"{cmn[i, j]:8.2f}" for j in range(4)))
    peruser = [macro4(yg[grp == u], preds_full[grp == u]) for u in np.unique(grp) if (grp == u).sum() >= 30]
    print(f"\n  Spread per-utente (n={len(peruser)} utenti ≥30 win): macro-F1 {np.mean(peruser):.3f} ± {np.std(peruser):.3f}")

    # ===================== SILVER (protocollo) =====================
    print("\n" + "=" * 70); print("SILVER — labeler universale + train locale label-free"); print("=" * 70)
    cfg = CityConfig.from_yaml(ROOT / "tmd/configs/cities/trento.yaml")
    silver = np.array([s if s is not None else None for s in label_windows_universal(g, cfg)[0]], object)
    cov = silver != None  # noqa: E711
    print(f"  Coverage labeler: {cov.mean():.1%} etichettate, {(~cov).mean():.1%} ABSTAIN")
    lab_dist = pd.Series(silver[cov]).value_counts()
    print(f"  Distribuzione silver: {lab_dist.to_dict()}")
    # agreement per-classe (precision: di chi il labeler chiama X, quanti sono davvero X in GT)
    print("  Labeler→GT agreement (precision sul coperto, per classe silver):")
    for c in FOUR:
        m = silver == c
        if m.sum():
            print(f"    silver={c:<6} n={m.sum():>6}  precision-vs-GT {(yg[m] == c).mean():.2f}")
    still_m = silver == STILL_CLASS
    if still_m.sum():
        print(f"    silver=Still  n={still_m.sum():>6}  (spurio: GeoLife non ha Still) — vero GT: {pd.Series(yg[still_m]).value_counts().to_dict()}")

    # train locale su silver vs supervised (stesso flat RF, GroupKFold, eval su GT)
    # AGGREGATO out-of-fold (coerente col ceiling, NON media-per-fold) + per-classe
    Xall = matrix(g, feats)
    osup = np.empty(len(g), object); osil = np.full(len(g), None, object)
    for tri, tei in GroupKFold(5).split(Xall, yg, grp):
        imp = SimpleImputer(strategy="median").fit(Xall[tri])
        Xtr, Xte = imp.transform(Xall[tri]), imp.transform(Xall[tei])
        osup[tei] = RandomForestClassifier(**RF).fit(Xtr, yg[tri]).predict(Xte)  # supervised: GT[tri]
        sm = np.isin(silver[tri], FOUR)                                          # silver: silver∈FOUR[tri] (label-free)
        osil[tei] = RandomForestClassifier(**RF).fit(Xtr[sm], silver[tri][sm].astype(str)).predict(Xte)
    sup_m, sil_m = macro4(yg, osup), macro4(yg, osil)
    spc, lpc = perclass(yg, osup), perclass(yg, osil)
    print(f"\n  In-domain SUPERVISED (GT-trained)  macro-F1 {sup_m:.3f}   {spc}")
    print(f"  In-domain SILVER     (label-free)  macro-F1 {sil_m:.3f}   {lpc}")
    print(f"  → COSTO label-freeness in-domain = {sup_m - sil_m:+.3f}  (per-cl: " +
          " ".join(f"{c}{spc[c] - lpc[c]:+.2f}" for c in FOUR) + ")   [rif. Trento +0.04 · SHL +0.08]")

    # ===================== TRANSFER =====================
    print("\n" + "=" * 70); print("TRANSFER (predict_moving su GeoLife)"); print("=" * 70)
    Xg = matrix(g, feats)

    def fit_src(df, ycol, f):
        sub = df[df[ycol].isin(FIVE)]
        cls = [c for c in FIVE if c in set(sub[ycol])]
        return HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(sub, f), sub[ycol].values)

    trsil = tr[tr.silver_label.notna()]; trgt = tr[tr.label.notna()]
    shtr = sh[(sh.split == "train") & sh.label.isin(FIVE)]
    for name, src, ycol in [("Trento-silver (label-free)", trsil, "silver_label"),
                            ("Trento-GT (SDK supervised)", trgt, "label"),
                            ("SHL-GT (supervised, UK)", shtr, "label")]:
        yp = predict_moving(fit_src(src, ycol, feats), Xg)
        print(f"  {name:<28} macro-F1 {macro4(yg, yp):.3f}   {perclass(yg, yp)}")

    # leak-free drop-C sul transfer (Trento-silver)
    print("\n  Leak-free (Trento-silver → GeoLife): il gruppo C aiuta il transfer?")
    for label, groups in [("B+C+D (full)", ["B", "C", "D"]), ("B+D (no C)", ["B", "D"])]:
        f = grpfeats(feats, groups)
        yp = predict_moving(fit_src(trsil, "silver_label", f), matrix(g, f))
        print(f"    {label:<14} ({len(f):2d} feat)  transfer macro-F1 {macro4(yg, yp):.3f}   Train={perclass(yg, yp)['Train']}")

    # rule-based: labeler-as-classifier (forza una moving sui coperti)
    rb = silver.copy()
    rb_m = np.isin(rb, FOUR)
    print(f"\n  Rule-based (labeler diretto, solo coperti n={rb_m.sum():,}): "
          f"macro-F1 {macro4(yg[rb_m], rb[rb_m].astype(str)):.3f}   (vs ML transfer 0.581)")


if __name__ == "__main__":
    main()
