"""
research/rq1_3_self_correction.py — RQ1.3: il modello generalizza oltre il labeler.

Scopo:    cuore del weak-supervision — il classificatore allenato su silver-label rumorose
          (a) batte il labeler da cui ha imparato, (b) è accurato dove il labeler ABSTAIN.
          Decompone "self-correction" in COPERTURA (predice dove il labeler tace) vs
          CORREZIONE (corregge dove il labeler copre-ma-sbaglia).
Input:    data/v2/processed/eval_trento_20260612_202507.parquet (rolling-OOF canonico, onesto)
          + data/v2/features_trento.parquet (per il silver_label). Bike esclusa (mai nel silver).
Output:   riepilogo numerico stdout (nessuna figura). Strat GPS-present (>0.5) + ALL.
Alimenta: thesis/results.md (Self-correction 1.3)
Sez.tesi: 5.x label-free / 7 — coerente con E19 (abstain decomposition).

Run: python research/rq1_3_self_correction.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
EVAL = ROOT / "data/v2/processed/eval_trento_20260612_202507.parquet"   # canonico silver rolling
FEAT = ROOT / "data/v2/features_trento.parquet"


def macro(y, p):
    seen = [c for c in FIVE if (np.asarray(y) == c).sum() > 0]
    return f1_score(y, p, labels=seen, average="macro", zero_division=0)


def report(m: pd.DataFrame, tag: str):
    gt = m["label"].values
    mp_raw = m["predicted_class"].values
    mp = m["predicted_class_smooth"].values if "predicted_class_smooth" in m else mp_raw
    sl = m["silver_label"]
    lab = sl.where(sl.isin(FIVE), "Unknown").values         # labeler-as-predictor (ABSTAIN=errore)

    f_lab, f_raw, f_sm = macro(gt, lab), macro(gt, mp_raw), macro(gt, mp)
    print(f"\n=== {tag} ===  ({len(m)} finestre GT 5cl)")
    print(f"  Labeler (silver, ABSTAIN=errore)  F1: {f_lab:.3f}")
    print(f"  Modello raw                        F1: {f_raw:.3f}   (Δ vs labeler +{f_raw - f_lab:.3f})")
    print(f"  Modello smooth                     F1: {f_sm:.3f}   (Δ vs labeler +{f_sm - f_lab:.3f})")

    abst = ~sl.isin(FIVE)
    cov = sl.isin(FIVE)
    acc_abst = (mp[abst.values] == gt[abst.values]).mean() if abst.any() else float("nan")
    acc_cov = (mp[cov.values] == gt[cov.values]).mean() if cov.any() else float("nan")
    # CORREZIONE: tra le finestre coperte ma SBAGLIATE dal labeler, quante il modello corregge
    wrong = cov.values & (sl.values != gt)
    corrected = (mp[wrong] == gt[wrong]).mean() if wrong.any() else float("nan")
    print(f"  COPERTURA  — ABSTAIN labeler {abst.sum():>4} ({abst.mean():.0%}): acc modello = {acc_abst:.3f}")
    print(f"  (riferim.) — coperte         {cov.sum():>4} ({cov.mean():.0%}): acc modello = {acc_cov:.3f}")
    print(f"  CORREZIONE — labeler coperto-ma-SBAGLIATO {wrong.sum():>4}: il modello ne corregge {corrected:.0%}")


def main():
    ev = pd.read_parquet(EVAL)
    ft = pd.read_parquet(FEAT)[["userId", "ts_start", "silver_label"]]
    m = ev.merge(ft, on=["userId", "ts_start"], how="left")
    m = m[m["label"].isin(FIVE)].copy()

    report(m, "ALL windows")
    report(m[m.gps_frac > 0.5], "GPS-present (>0.5)")

    # per-classe: silver-precision (qualità label train) vs model-F1 (test, smooth)
    print(f"\n  {'classe':<7}{'silver-prec(train)':>20}{'model-F1(test,GPS-pres)':>26}")
    full = pd.read_parquet(FEAT)
    sv, g = full["silver_label"], full["label"]
    gp = m[m.gps_frac > 0.5]
    gt_gp = gp["label"].values
    mp_gp = gp["predicted_class_smooth"].values if "predicted_class_smooth" in gp else gp["predicted_class"].values
    for c in FIVE:
        tp = ((sv == c) & (g == c)).sum()
        fp = ((sv == c) & (g != c) & g.notna()).sum()
        sp = tp / (tp + fp) if tp + fp else float("nan")
        mf = f1_score(gt_gp, mp_gp, labels=[c], average="macro", zero_division=0)
        print(f"  {c:<7}{sp:>20.3f}{mf:>26.3f}")


if __name__ == "__main__":
    main()
