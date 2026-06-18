"""
research/strat_eval.py — HELPER (RQ3.1): GPS-stratified macro-F1 da un eval parquet.

Riusabile: prende uno o più eval_*.parquet (output di tmd.cli.train) e stampa la
macro-F1 (raw + smooth) per strato GPS, sulle 5 classi in movimento (Bike esclusa:
mai nel silver, GT rumorosa E18/E20). Definizione 'seen' = classi con supporto nel test.

Stratificazione 3-way ALLINEATA a E5 (`research/e5_windows.py`), NON binaria:
  - GPS-absent  (=0)        : nessun fix GPS → IMU-only puro (floor strutturale, E5)
  - GPS-sparse  (0<·<=0.5)  : GPS rado/degradato (<0.5 fix/s)
  - GPS-present (>0.5)       : dominio operativo (headline)
Il binario present-vs-absent(≤0.5) MASCHERAVA il floor reale: "absent" lumpava lo zero-GPS
(molto basso) con lo sparse (decente). gps_frac = #fix GPS / win_s (densità ~Hz, cap 1.5).

Alimenta: thesis/results.md (headline 3-way · GPS-dropout 3.2). Sez.tesi: 6.x confine operativo.
Uso:  python research/strat_eval.py data/v2/processed/eval_trento_XXXX.parquet [...]
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]


def macro(y, p):
    seen = [c for c in FIVE if (y == c).sum() > 0]
    return f1_score(y, p, labels=seen, average="macro", zero_division=0)


def strat(ev: pd.DataFrame, name: str):
    ev = ev[ev["label"].isin(FIVE)].copy()
    gf = ev.gps_frac
    print(f"\n=== {name} ===  ({len(ev)} finestre 5cl etichettate)")
    print(f"{'strato':<28}{'n':>7}{'F1 raw':>9}{'F1 smooth':>11}")
    for lab, m in [("GPS-absent  (=0)",      gf == 0),
                   ("GPS-sparse  (0<.<=0.5)", (gf > 0) & (gf <= 0.5)),
                   ("GPS-present (>0.5)",     gf > 0.5),
                   ("ALL",                    ev.index == ev.index)]:
        s = ev[m]
        if len(s) == 0:
            continue
        fr = macro(s.label.values, s.predicted_class.values)
        fs = (macro(s.label.values, s.predicted_class_smooth.values)
              if "predicted_class_smooth" in s.columns else float("nan"))
        print(f"{lab:<28}{len(s):>7}{fr:>9.3f}{fs:>11.3f}")


def main():
    for p in sys.argv[1:]:
        strat(pd.read_parquet(p), Path(p).name)


if __name__ == "__main__":
    main()
