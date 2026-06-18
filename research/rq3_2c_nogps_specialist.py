"""
research/rq3_2c_nogps_specialist.py — RQ3.2c: un classificatore DEDICATO allo zero-GPS aiuta?

Scopo:    rispondere alla domanda "alleniamo un modello apposta SOLO sulle finestre senza-GPS (da usare
          su quelle) e fa meglio del modello generale?". Serve la GT (motiontag): il silver non ha label
          MOVING nello zero-GPS (il labeler astiene) → un dedicato-silver è impossibile. Feature = A (IMU):
          sullo zero-GPS B/C sono tutte NaN, quindi conta solo l'IMU.
Metodo:   GroupKFold(5) per sessione. Test = finestre zero-GPS delle sessioni held-out. Due training,
          stesso test:
            - DEDICATO: A-only su finestre zero-GPS delle sessioni di train.
            - GENERALE: A-only su TUTTE le finestre delle sessioni di train (impara i moving dalle
                        finestre GPS-present e ne trasferisce la firma-IMU).
          Ipotesi: lo zero-GPS è ~91% Still → il dedicato ha pochissimi moving da cui imparare → il
          generale (moving abbondanti su GPS-present) regge meglio. Se così, specializzare NON aiuta:
          la cura è insegnare al generale a USARE l'IMU (gps-dropout), non specializzarlo.
Input:    data/v2/features_trento.parquet
Output:   riepilogo numerico stdout.
Alimenta: thesis/results.md §RQ3 (3.2c). Sez.tesi: 6.5.

Run: python research/rq3_2c_nogps_specialist.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
from tmd.training.trainer import get_feature_cols  # noqa: E402
from tmd.models.hierarchical import HierarchicalTMD  # noqa: E402

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]


def macro(y, p):
    seen = [c for c in FIVE if (np.asarray(y) == c).sum() > 0]
    return f1_score(y, p, labels=seen, average="macro", zero_division=0)


def matrix(df, feats):
    return df[feats].values.astype(np.float32)


def fit_pred(dtr, dte, feats):
    cls = [c for c in FIVE if c in set(dtr.label)]
    if len(cls) < 2:
        return None
    m = HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(dtr, feats), dtr.label.values)
    return m.predict(matrix(dte, feats))


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    df = df[df.label.isin(FIVE)].copy().reset_index(drop=True)
    feats = get_feature_cols(df, ["A"])
    print("=" * 64); print("RQ3.2c — modello DEDICATO allo zero-GPS vs GENERALE (GT, A-only)"); print("=" * 64)
    nz = df[df.gps_frac == 0]
    print(f"finestre zero-GPS GT = {len(nz)} | dist: {nz.label.value_counts().reindex(FIVE).fillna(0).astype(int).to_dict()}\n")

    gkf = GroupKFold(5)
    ded, gen = {c: [] for c in FIVE}, {c: [] for c in FIVE}
    ded_m, gen_m = [], []
    sess = df.session_id.values
    for tr_idx, te_idx in gkf.split(df, groups=sess):
        tr_sess = set(sess[tr_idx])
        test = df.iloc[te_idx]; test = test[test.gps_frac == 0]
        if len(test) < 10:
            continue
        train_all = df[df.session_id.isin(tr_sess)]
        train_ded = train_all[train_all.gps_frac == 0]
        p_ded = fit_pred(train_ded, test, feats)
        p_gen = fit_pred(train_all, test, feats)
        if p_ded is None or p_gen is None:
            continue
        ded_m.append(macro(test.label.values, p_ded)); gen_m.append(macro(test.label.values, p_gen))
        for c in FIVE:
            ded[c].append(f1_score(test.label.values, p_ded, labels=[c], average="macro", zero_division=0))
            gen[c].append(f1_score(test.label.values, p_gen, labels=[c], average="macro", zero_division=0))

    print(f"macro-F1 su zero-GPS (media {len(ded_m)} fold):")
    print(f"  DEDICATO (train solo zero-GPS): {np.mean(ded_m):.3f} ± {np.std(ded_m):.3f}")
    print(f"  GENERALE (train tutte le fin.): {np.mean(gen_m):.3f} ± {np.std(gen_m):.3f}")
    print(f"  → Δ (generale − dedicato): {np.mean(gen_m) - np.mean(ded_m):+.3f}\n")
    print(f"  per-classe:  {'class':<7}{'dedicato':>10}{'generale':>10}")
    for c in FIVE:
        if ded[c]:
            print(f"  {'':13}{c:<7}{np.mean(ded[c]):>10.2f}{np.mean(gen[c]):>10.2f}")
    print("\nLettura: se GENERALE ≥ DEDICATO ⇒ specializzare NON aiuta (lo zero-GPS ha troppi pochi moving "
          "da cui imparare; il generale impara la firma-IMU dei moving sulle finestre GPS-present e la "
          "trasferisce). La cura giusta è insegnare al modello UNICO a usare l'IMU (gps-dropout), non un router specializzato.")


if __name__ == "__main__":
    main()
