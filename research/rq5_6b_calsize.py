"""
research/rq5_6b_calsize.py — RQ5.6b-bis: quanto piccolo può essere il set di calibrazione?

Scopo:    la quantification (5.6b) "richiede un piccolo set di calibrazione etichettato" — qui si misura
          QUANTO piccolo: sweep della taglia del campione usato per stimare M, a parità di protocollo.
Metodo:   come 5.6b (split 50/50, GPS-present moving, M -> M^T p = q_pred) ma con CAL sotto-campionato
          a n ∈ {50,100,200,400,800,1600,half}; 20 seed. TVD corretto vs taglia.
Input:    data/v2/processed/eval_trento_20260612_202507.parquet (canonico silver rolling-OOF)
Output:   riepilogo numerico stdout.
Alimenta: thesis/results.md §RQ5 (modal-split 5.6b — taglia calibrazione). Sez.tesi: 6.6 use-case.

Run: python research/rq5_6b_calsize.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data/v2/processed/eval_trento_20260612_202507.parquet"
MOV = ["Walk", "Bus", "Car", "Train"]
SIZES = [50, 100, 200, 400, 800, 1600, None]  # None = metà piena (come 5.6b)


def shares(labels):
    s = pd.Series(labels)
    v = s[s.isin(MOV)].value_counts().reindex(MOV).fillna(0)
    return (v / v.sum()).values


def tvd(p, q):
    return 0.5 * np.abs(p - q).sum() * 100


def correct(q_pred, M):
    p, *_ = np.linalg.lstsq(M.T, q_pred, rcond=None)
    p = np.clip(p, 0, None)
    return p / p.sum() if p.sum() > 0 else q_pred


def main():
    ev = pd.read_parquet(EVAL)
    pc = "predicted_class_smooth" if "predicted_class_smooth" in ev else "predicted_class"
    g = ev[(ev.gps_frac > 0.5) & ev.label.isin(MOV)].copy()
    print("=" * 60)
    print("RQ5.6b-bis — taglia del set di calibrazione (quantification)")
    print("=" * 60)
    print(f"finestre moving GT GPS-present = {len(g)} (finestre 120s, stride 60s)\n")

    for n_cal in SIZES:
        naive, corr = [], []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(g)); h = len(g) // 2
            calpool, tst = g.iloc[idx[:h]], g.iloc[idx[h:]]
            cal = calpool if n_cal is None or n_cal >= len(calpool) else calpool.sample(n=n_cal, random_state=seed)
            M = np.zeros((4, 4))
            for i, ci in enumerate(MOV):
                sub = cal[(cal.label == ci) & cal[pc].isin(MOV)]
                if len(sub):
                    M[i] = shares(sub[pc].values)
                else:
                    M[i, i] = 1.0
            q_true = shares(tst.label.values)
            q_pred = shares(tst[tst[pc].isin(MOV)][pc].values)
            naive.append(tvd(q_pred, q_true))
            corr.append(tvd(correct(q_pred, M), q_true))
        k = n_cal if n_cal else f"half({len(g)//2})"
        print(f"  cal={str(k):>11}: TVD naive {np.mean(naive):4.1f} -> corretto {np.mean(corr):4.1f} ± {np.std(corr):3.1f}")

    print("\nLettura: sotto ~200 finestre la M è mal stimata e la correzione può PEGGIORARE il naive; "
          "il plateau (~4.2-4.6) si raggiunge da ~400 finestre (~6-7h di moving etichettato).")


if __name__ == "__main__":
    main()
