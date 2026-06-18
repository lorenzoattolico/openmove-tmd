"""
research/rq2_2_drop_high_ks.py — RQ2.2 (feature_selection §6.2): drop-high-KS transfer.

Scopo:    le feature più domain-shiftate Trento↔SHL (KS alto, E14) sono a rischio negative-transfer.
          Droppandole: atteso ~neutro IN-DOMAIN (RF robusto, E6/full≈163) ma ↑ TRANSFER (si toglie
          rumore non trasferibile). Verifica diretta del valore del FLAG transfer-risk di E6.
Metodo:   KS da E14 (research/figures/e14_domain_shift.csv) sui 163 feat del modello. Set: full-163 +
          drop KS>{0.5,0.4,0.3}. Per ogni set: (a) IN-DOMAIN = rolling-origin GPS-present (stesso
          protocollo headline), media-fold; (b) TRANSFER = train su tutto il silver Trento → SHL
          validate 5cl. Clf = RF gerarchico (canonico).
Input:    data/v2/features_trento.parquet · data/v2/features_shl_full.parquet · e14_domain_shift.csv
Output:   riepilogo numerico stdout (nessuna figura).
Alimenta: thesis/results.md + thesis/feature_selection.md §6.2. Sez.tesi: 4.5 / 6.3.

Run: python research/rq2_2_drop_high_ks.py
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


def indomain_rolling(tr, feats):
    """macro-F1 GPS-present media sui fold rolling (silver-train, motiontag-test)."""
    silver = tr[tr.silver_label.notna()].copy()
    evalp = tr[tr.label.notna()].copy()
    f1s = []
    for _, dtr, dte in rolling_splits(silver, evalp):
        dte = dte[dte.label.notna() & (dte.gps_frac > 0.5)]
        if len(dte) < 10:
            continue
        cls = [c for c in tr.label.dropna().unique() if c in set(dtr.silver_label)]
        m = HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(dtr, feats), dtr.silver_label.values)
        f1s.append(macro(dte.label.values, m.predict(matrix(dte, feats))))
    return np.mean(f1s), np.std(f1s)


def transfer(tr, shva, feats):
    sil = tr[tr.silver_label.notna()]
    cls = [c for c in FIVE if c in set(sil.silver_label)]
    m = HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(sil, feats), sil.silver_label.values)
    return macro(shva.label.values, m.predict(matrix(shva, feats)))


def main():
    tr = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    sh = pd.read_parquet(ROOT / "data/v2/features_shl_full.parquet")
    shva = sh[(sh.split == "validate") & sh.label.isin(FIVE)]
    feats163 = get_feature_cols(tr, ["A", "B", "C", "D"])
    ks = pd.read_csv(ROOT / "research/figures/e14_domain_shift.csv").set_index("feature").ks

    print("=" * 70)
    print("RQ2.2 — drop-high-KS (transfer-risk) — in-domain vs transfer")
    print("=" * 70)
    print(f"{'set':<18}{'n_feat':>7}{'in-domain GPS-pres':>22}{'transfer→SHL':>16}")
    sets = [("full-163", None)] + [(f"drop KS>{t}", t) for t in (0.5, 0.4, 0.3)]
    for name, thr in sets:
        if thr is None:
            feats = feats163
        else:
            drop = {f for f in feats163 if f in ks.index and ks[f] > thr}
            feats = [f for f in feats163 if f not in drop]
        idm, idstd = indomain_rolling(tr, feats)
        tf = transfer(tr, shva, feats)
        print(f"{name:<18}{len(feats):>7}{idm:>15.3f}±{idstd:.3f}{tf:>16.3f}")

    print("\nLettura: se transfer ↑ mentre in-domain ~stabile ⇒ le feature high-KS portano "
          "negative-transfer (FLAG E6 validato). Se entrambi piatti ⇒ RF già robusto, neutro.")


if __name__ == "__main__":
    main()
