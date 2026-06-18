"""
research/rq5_6b_quantification.py — RQ5.6b: de-bias del modal-split (quantification).

Scopo:    il modal-split del modello reale (5.6) ha un BIAS sistematico (Car/Train sovra-, Bus/Walk
          sotto-contati) → TVD ~6.7% finestra. La stima per-conteggio ("classify & count") eredita il
          bias del classificatore. La cura standard è la **quantification** (Saerens 2002 / Forman 2008
          "adjusted count"): stimare la matrice di confusione M su un campione di calibrazione e
          **invertirla** per correggere le quote aggregate. Verifica: riduce il TVD?
Metodo:   split onesto 50/50 (più seed) su GPS-present moving (Walk/Bus/Car/Train). Su CAL: M[i,j]=
          P(pred=j|true=i). Su TEST: quota naive = distribuzione predetta; quota corretta = soluzione di
          M^T p = q_pred (lstsq, clip simplex). Confronto TVD(naive) vs TVD(corretta) vs vero (TEST).
Input:    data/v2/processed/eval_trento_20260612_202507.parquet (canonico silver rolling-OOF)
Output:   riepilogo numerico stdout.
Alimenta: thesis/results.md §RQ5 (modal-split 5.6b). Sez.tesi: 6.6 use-case.

Run: python research/rq5_6b_quantification.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data/v2/processed/eval_trento_20260612_202507.parquet"
MOV = ["Walk", "Bus", "Car", "Train"]


def shares(labels, classes):
    s = pd.Series(labels)
    v = s[s.isin(classes)].value_counts().reindex(classes).fillna(0)
    return (v / v.sum()).values


def tvd(p, q):
    return 0.5 * np.abs(p - q).sum() * 100


def correct(q_pred, M):
    """Quantification: risolvi M^T p = q_pred, clip al simplesso."""
    p, *_ = np.linalg.lstsq(M.T, q_pred, rcond=None)
    p = np.clip(p, 0, None)
    return p / p.sum() if p.sum() > 0 else q_pred


def main():
    ev = pd.read_parquet(EVAL)
    pc = "predicted_class_smooth" if "predicted_class_smooth" in ev else "predicted_class"
    g = ev[(ev.gps_frac > 0.5) & ev.label.isin(MOV) & ev[pc].isin(MOV + ["Still"])].copy()
    # tieni solo righe con GT moving; pred può essere qualsiasi → mappiamo pred fuori-MOV come "altro" escluso
    g = g[g.label.isin(MOV)]
    print("=" * 60); print("RQ5.6b — quantification (de-bias modal-split, GPS-present)"); print("=" * 60)
    print(f"finestre moving GT = {len(g)} | classi {MOV}\n")

    naive_tvds, corr_tvds = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(g)); h = len(g) // 2
        cal, tst = g.iloc[idx[:h]], g.iloc[idx[h:]]
        # M[i,j] = P(pred=j | true=i) sul CAL (solo pred in MOV)
        M = np.zeros((len(MOV), len(MOV)))
        for i, ci in enumerate(MOV):
            sub = cal[cal.label == ci]
            sub = sub[sub[pc].isin(MOV)]
            if len(sub):
                M[i] = shares(sub[pc].values, MOV)
        # evita righe nulle
        for i in range(len(MOV)):
            if M[i].sum() == 0:
                M[i, i] = 1.0
        q_true = shares(tst.label.values, MOV)
        q_pred = shares(tst[tst[pc].isin(MOV)][pc].values, MOV)
        q_corr = correct(q_pred, M)
        naive_tvds.append(tvd(q_pred, q_true)); corr_tvds.append(tvd(q_corr, q_true))

    print(f"  TVD naive  (classify & count): {np.mean(naive_tvds):.1f}% ± {np.std(naive_tvds):.1f}")
    print(f"  TVD corretto (quantification): {np.mean(corr_tvds):.1f}% ± {np.std(corr_tvds):.1f}")
    print(f"  → riduzione: {np.mean(naive_tvds) - np.mean(corr_tvds):+.1f} pt")

    # quote medie per-modo (full set, M su tutto come stima deployment) per mostrare il bias residuo
    M = np.array([shares(g[(g.label == ci) & g[pc].isin(MOV)][pc].values, MOV) for ci in MOV])
    for i in range(len(MOV)):
        if M[i].sum() == 0:
            M[i, i] = 1.0
    q_true = shares(g.label.values, MOV); q_pred = shares(g[g[pc].isin(MOV)][pc].values, MOV)
    q_corr = correct(q_pred, M)
    print(f"\n  {'modo':<7}{'vero%':>8}{'naive%':>8}{'corr%':>8}")
    for i, c in enumerate(MOV):
        print(f"  {c:<7}{100*q_true[i]:>8.1f}{100*q_pred[i]:>8.1f}{100*q_corr[i]:>8.1f}")

    print("\nLettura: se TVD-corretto < TVD-naive ⇒ la quantification de-biasa il modal-split aggregato "
          "(cura standard, serve un piccolo set di calibrazione etichettato in deployment).")


if __name__ == "__main__":
    main()
