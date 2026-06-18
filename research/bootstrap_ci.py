"""
research/bootstrap_ci.py — HELPER: intervalli di confidenza bootstrap sui delta headline.

Scopo:    dare le BARRE D'ERRORE ai claim-chiave (un delta è significativo se il CI esclude 0). Bootstrap
          a livello SESSIONE (cluster bootstrap) — onesto: le finestre nella stessa sessione sono correlate
          (overlap 60s<120s, E23), quindi si ricampionano le sessioni, non le finestre.
Metodo:   per ogni confronto appaiato (stesse finestre, due predizioni A/B): ricampiona le sessioni con
          rimpiazzo (B=5000), ricalcola macro-F1(A)−macro-F1(B), riporta media + CI 95% (perc. 2.5/97.5).
Delta:    Δ1 costo-LF GPS-present (GT−silver) · Δ2 costo-LF no-GPS · Δ3 GPS-dropout-0.7−baseline (no-GPS)
          · Δ4 transfer ML−rule-based (SHL) · Δ5 self-correction (modello−labeler).
Input:    eval parquets canonico/GT/dropout + transfer ML/rule-based (data/v2/processed) + features (silver).
Output:   riepilogo numerico stdout.
Alimenta: thesis/results.md (CI sui delta headline). Sez.tesi: 5/6.

Run: python research/bootstrap_ci.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/v2/processed"
FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
NCLS = 6  # 5 classi + 1 sentinella "Unknown/altro" (ABSTAIN)
B = 3000


def _code(arr):
    """codifica intera: Still0..Train4, tutto il resto (Unknown/ABSTAIN) → 5."""
    mp = {c: i for i, c in enumerate(FIVE)}
    return np.array([mp.get(x, 5) for x in arr], dtype=np.int64)


def fast_macro(y, p):
    """macro-F1 sulle classi VISTE (solo 0..4; la sentinella 5 non è una classe target)."""
    f1s = []
    for c in range(5):
        ypos = y == c
        if not ypos.any():
            continue
        ppos = p == c
        tp = np.count_nonzero(ppos & ypos)
        fp = np.count_nonzero(ppos & ~ypos)
        fn = np.count_nonzero(~ppos & ypos)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def boot_delta(df, group, ytrue, predA, predB, seed=0):
    """CI 95% di macro-F1(A)-macro-F1(B), cluster-bootstrap sulle sessioni (numpy veloce)."""
    y = _code(df[ytrue].values); a = _code(df[predA].values); bb = _code(df[predB].values)
    groups = df[group].values
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    obs = fast_macro(y, a) - fast_macro(y, bb)
    deltas = np.empty(B)
    for b in range(B):
        samp = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([idx_by_g[k] for k in samp])
        deltas[b] = fast_macro(y[rows], a[rows]) - fast_macro(y[rows], bb[rows])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return obs, lo, hi


def main():
    print("=" * 70)
    print("CI bootstrap (cluster su sessione, B=%d) sui delta headline" % B)
    print("=" * 70)
    sv = pd.read_parquet(P / "eval_trento_20260612_202507.parquet")          # silver baseline
    gt = pd.read_parquet(P / "eval_trento_20260612_202826.parquet")          # GT (motiontag)
    dr = pd.read_parquet(P / "eval_trento_20260612_202633.parquet")          # gps-dropout 0.7
    key = ["session_id", "ts_start"]

    def align(a, b, sa, sb):
        m = a[key + ["label", "gps_frac", "predicted_class"]].rename(columns={"predicted_class": sa}).merge(
            b[key + ["predicted_class"]].rename(columns={"predicted_class": sb}), on=key, how="inner")
        return m[m.label.isin(FIVE)]

    # Δ1/Δ2 costo label-free (GT − silver)
    m = align(sv, gt, "silver", "gt")
    for nm, mask in [("Δ1 costo-LF GPS-present (GT−silver)", m.gps_frac > 0.5),
                     ("Δ2 costo-LF no-GPS (GT−silver)", m.gps_frac == 0)]:
        d = m[mask]
        o, lo, hi = boot_delta(d, "session_id", "label", "gt", "silver")
        sig = "SIGNIFICATIVO (CI esclude 0)" if lo > 0 else "n.s."
        print(f"\n{nm}: +{o:.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]  → {sig}")

    # Δ3 GPS-dropout-0.7 − baseline (no-GPS)
    m = align(sv, dr, "base", "drop")
    d = m[m.gps_frac == 0]
    o, lo, hi = boot_delta(d, "session_id", "label", "drop", "base")
    print(f"\nΔ3 GPS-dropout-0.7 − baseline, no-GPS: +{o:.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]  → "
          f"{'SIGNIFICATIVO' if lo > 0 else 'n.s.'}")

    # Δ4 transfer ML − rule-based (SHL validate)
    ml = pd.read_parquet(next(P.glob("transfer_trento_20260612_202512_on_features_shl_full_*.parquet")))
    rb = pd.read_parquet(next(P.glob("transfer_rule_based_trento_on_features_shl_full_*.parquet")))
    grp = "session_id" if "session_id" in ml.columns else ("block_id" if "block_id" in ml.columns else None)
    print(f"\n[transfer] colonne ML: chiave-gruppo = {grp or 'finestra (no session)'}")
    if grp is None:
        ml = ml.reset_index().rename(columns={"index": "_g"}); grp = "_g"; rb = rb.reset_index(drop=True)
    t = ml[[grp, "label", "predicted_class"]].rename(columns={"predicted_class": "ml"}).copy()
    t["rb"] = rb["predicted_class"].values
    t = t[t.label.isin(FIVE)]
    t["ml"] = t["ml"].where(t["ml"].isin(FIVE), "Unknown")
    t["rb"] = t["rb"].where(t["rb"].isin(FIVE), "Unknown")   # ABSTAIN = errore
    o, lo, hi = boot_delta(t, grp, "label", "ml", "rb")
    print(f"Δ4 transfer ML − rule-based (SHL): +{o:.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]  → "
          f"{'SIGNIFICATIVO' if lo > 0 else 'n.s.'}")

    # Δ5 self-correction (modello − labeler), GPS-present
    ft = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")[["session_id", "ts_start", "silver_label"]]
    sc = sv.merge(ft, on=key, how="left")
    sc = sc[sc.label.isin(FIVE) & (sc.gps_frac > 0.5)].copy()
    pc = "predicted_class_smooth" if "predicted_class_smooth" in sc else "predicted_class"
    sc["model"] = sc[pc]
    sc["labeler"] = sc.silver_label.where(sc.silver_label.isin(FIVE), "Unknown")
    o, lo, hi = boot_delta(sc, "session_id", "label", "model", "labeler")
    print(f"Δ5 self-correction modello − labeler (GPS-present): +{o:.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]  → "
          f"{'SIGNIFICATIVO' if lo > 0 else 'n.s.'}")


if __name__ == "__main__":
    main()
