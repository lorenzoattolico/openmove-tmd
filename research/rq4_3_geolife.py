"""
research/rq4_3_geolife.py — RQ4.3: zero-shot Trento/SHL → GeoLife (Pechino).

Scopo:    validità esterna N=3 (IT → UK → Cina) col solo cuore cinematico+infra-GPS (B/C/D):
          GeoLife è GPS-only (niente IMU). Testa se velocità/struttura GPS trasferiscono su un
          terzo continente SENZA adattamento. GeoLife non ha "Still" → si forza la decisione tra
          le 4 classi moving (Walk/Car/Bus/Train) via il ramo L2 (bypass del gate Still/moving).
Metodo:   train Trento-silver e SHL-GT sulle **27 feature B/C/D comuni a GeoLife** (no IMU);
          predizione moving-forced su GeoLife; macro-F1(4cl) + per-classe. Ceiling in-domain =
          flat RF GroupKFold per-utente sulle stesse 27 feature (limite raggiungibile).
Input:    data/processed/features_geolife.parquet · data/v2/features_trento.parquet
          · data/v2/features_shl_full.parquet
Output:   riepilogo numerico stdout (nessuna figura).
Alimenta: thesis/results.md (GeoLife 4.3). Sez.tesi: 6.3 transfer / validità esterna.

Run: python research/rq4_3_geolife.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
from tmd.models.hierarchical import HierarchicalTMD, STILL_CLASS  # noqa: E402

FOUR = ["Walk", "Car", "Bus", "Train"]
RF = dict(n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight="balanced")


def macro4(yt, yp):
    return f1_score(yt, yp, labels=FOUR, average="macro", zero_division=0)


def perclass(yt, yp):
    return dict(zip(FOUR, f1_score(yt, yp, labels=FOUR, average=None, zero_division=0)))


def matrix(df, feats):
    X = np.full((len(df), len(feats)), np.nan, np.float32)
    for i, c in enumerate(feats):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    return X


def predict_moving(model, X):
    """Forza la classe tra le sole moving (GeoLife non ha Still)."""
    Xp = model._prep(X)
    classes_all = [STILL_CLASS] + model.l2_classes
    p_moving = model.l1.predict_proba(Xp)[:, 1]
    p_l2 = model.l2.predict_proba(Xp) * p_moving[:, None]
    P = np.zeros((len(X), len(classes_all)))
    P[:, 0] = 1 - p_moving
    for i, cls in enumerate(model.le_l2.classes_):
        P[:, classes_all.index(cls)] = p_l2[:, i]
    mov = [i for i, c in enumerate(classes_all) if c != STILL_CLASS]
    sub = P[:, mov]
    return np.array([classes_all[mov[j]] for j in sub.argmax(1)], object)


def report(name, yt, yp):
    print(f"\n── {name}: macro-F1(4cl) = {macro4(yt, yp):.3f} ──")
    print("   " + "  ".join(f"{c}={v:.2f}" for c, v in perclass(yt, yp).items()))


def main():
    g = pd.read_parquet(ROOT / "data/processed/features_geolife.parquet")
    g = g[g.label.isin(FOUR) & g.in_china].reset_index(drop=True)   # solo-Cina (esclusa coda 0.6% Nord America)
    tr = pd.read_parquet(ROOT / "data/v2/features_trento_full.parquet")  # FULL-230: ha tutte 27 le feat comuni (il 163-selezionato ne droppa 12 → set inconsistente nel transfer)
    sh = pd.read_parquet(ROOT / "data/v2/features_shl_full.parquet")

    feats = sorted(c for c in g.columns if c[:2] in ("B_", "C_", "D_"))
    print("=" * 64); print("RQ4.3 — zero-shot → GeoLife (27 feat B/C/D comuni, no IMU)"); print("=" * 64)
    print(f"GeoLife: {len(g)} finestre, {g.userId.nunique()} utenti, classi {FOUR}; feat={len(feats)}")
    Xg, yg = matrix(g, feats), np.asarray(g.label, dtype=object)
    groups = np.asarray(g.userId, dtype=object)

    # ── source Trento silver ──
    sil = tr[tr.silver_label.notna()]
    cls_t = [c for c in ["Still", "Walk", "Car", "Bus", "Train"] if c in set(sil.silver_label)]
    mt = HierarchicalTMD(cls_t, [], clf_type="rf").fit(matrix(sil, feats), sil.silver_label.values)
    report("Trento silver → GeoLife (zero-shot)", yg, predict_moving(mt, Xg))

    # ── source SHL GT ──
    shtr = sh[(sh.split == "train") & sh.label.isin(["Still", "Walk", "Car", "Bus", "Train"])]
    cls_s = sorted(set(shtr.label))
    ms = HierarchicalTMD(cls_s, [], clf_type="rf").fit(matrix(shtr, feats), shtr.label.values)
    report("SHL GT → GeoLife (zero-shot)", yg, predict_moving(ms, Xg))

    # ── ceiling in-domain GeoLife (flat RF, GroupKFold per utente) ──
    Xi = SimpleImputer(strategy="median").fit_transform(Xg)
    yp = cross_val_predict(RandomForestClassifier(**RF), Xi, yg,
                           groups=groups, cv=GroupKFold(5), n_jobs=-1)
    report("GeoLife in-domain ceiling (flat RF, GroupKFold-user)", yg, yp)

    print("\nLettura: Walk universale (0.78); i motorizzati TRASFERISCONO date le mappe OSM China-wide (rail+bus): "
          "Train 0.60 (la rotaia cinese da OSM, prima cieca fuori-Pechino), Car 0.53, Bus 0.38. Residuo vs ceiling "
          "0.68 = sensing GPS-only. [China-only via flag in_china; infra completa da load_geolife_infra_china.py.]")


if __name__ == "__main__":
    main()
