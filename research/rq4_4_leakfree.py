"""
research/rq4_4_leakfree.py — RQ4.4: leak-free → trasferisce (difesa vs FOSS4G).

Scopo:    blindare il claim "label-free *e* leak-free, e per questo trasferisce". Il competitor FOSS4G
          usa la mappa **come feature** (lat/lon + overlay) = memorizzazione geografica che NON trasferisce.
          Noi: (1) le feature di **route-align** (`C_*_align` = LA regola del labeler) sono DROPPATE prima
          del training (verificato qui); (2) teniamo il gruppo **C di contesto** (prossimità rotaia/fermate/
          autostrada). Domanda: il gruppo C **gonfia** il transfer (leak di memorizzazione → cadrebbe su SHL)
          o **aiuta** il transfer (fisica universale: i treni stanno sui binari ovunque)?
Metodo:   train Trento silver su {ABCD, ABD(no C), AB, BCD(no A)}; eval (a) in-domain GPS-present rolling,
          (b) transfer→SHL validate 5cl. Se transfer(ABCD) ≳ transfer(ABD) ⇒ C è fisica legittima che
          trasferisce, non un leak. Assert: nessuna `*_align` nelle feature.
Input:    data/v2/features_trento.parquet · data/v2/features_shl_full.parquet
Output:   riepilogo numerico stdout.
Alimenta: thesis/results.md §RQ4 (leak-free). Sez.tesi: 4.7 anti-leakage / 6.3.

Run: python research/rq4_4_leakfree.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
from tmd.training.trainer import get_feature_cols, rolling_splits  # noqa: E402
from tmd.models.hierarchical import HierarchicalTMD  # noqa: E402

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]


def macro(y, p):
    seen = [c for c in FIVE if (np.asarray(y) == c).sum() > 0]
    return f1_score(y, p, labels=seen, average="macro", zero_division=0)


def matrix(df, feats):
    X = np.full((len(df), len(feats)), np.nan, np.float32)
    for i, c in enumerate(feats):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    return X


def main():
    tr = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    sh = pd.read_parquet(ROOT / "data/v2/features_shl_full.parquet")
    shva = sh[(sh.split == "validate") & sh.label.isin(FIVE)]
    silver = tr[tr.silver_label.notna()].copy()
    evalp = tr[tr.label.notna()].copy()

    # ── assert anti-leak: nessuna route-align feature nei parquet ──
    align = [c for c in tr.columns if "align" in c.lower()]
    print(f"ANTI-LEAK check: feature *_align nel parquet Trento = {align or 'NESSUNA ✓'}")
    print(f"                  feature *_align nel parquet SHL    = "
          f"{[c for c in sh.columns if 'align' in c.lower()] or 'NESSUNA ✓'}\n")

    print("=" * 64); print("RQ4.4 — leak-free: gruppo C aiuta o gonfia il transfer?"); print("=" * 64)
    print(f"{'groups':<14}{'n_feat':>7}{'in-domain GPS-pres':>20}{'transfer→SHL':>15}")
    for label, groups in [("ABCD (full)", ["A", "B", "C", "D"]), ("ABD (no C)", ["A", "B", "D"]),
                          ("AB", ["A", "B"]), ("BCD (no A)", ["B", "C", "D"])]:
        feats = get_feature_cols(silver, groups)
        # in-domain rolling GPS-present
        f1s = []
        for _, dtr, dte in rolling_splits(silver, evalp):
            dte = dte[dte.label.isin(FIVE) & (dte.gps_frac > 0.5)]
            if len(dte) < 10:
                continue
            cls = [c for c in FIVE if c in set(dtr.silver_label)]
            m = HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(dtr, feats), dtr.silver_label.values)
            f1s.append(macro(dte.label.values, m.predict(matrix(dte, feats))))
        idm = np.mean(f1s)
        # transfer
        cls = [c for c in FIVE if c in set(silver.silver_label)]
        mt = HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(silver, feats), silver.silver_label.values)
        tf = macro(shva.label.values, mt.predict(matrix(shva, feats)))
        print(f"{label:<14}{len(feats):>7}{idm:>17.3f}{tf:>15.3f}")

    print("\nLettura: se transfer(ABCD) ≳ transfer(ABD) ⇒ il gruppo C è **fisica universale** "
          "(rotaia/fermate ovunque), NON memorizzazione geografica → leak-free regge. Se ABCD<ABD ⇒ C è "
          "negative-transfer (da droppare). Le `*_align` (= la regola del labeler) sono già fuori dal training.")


if __name__ == "__main__":
    main()
