"""
research/rq5_4_calibration.py — RQ5.4: calibrazione della confidenza top-label.

Scopo:    `HierarchicalTMD.predict_proba` ritorna (classe, max_prob); max_prob è una confidenza
          NON calibrata (prodotto L1×L2). Qui: (1) ECE/MCE/Brier + reliability su max_prob vs
          correttezza; (2) l'isotonic la rende affidabile (split ONESTO: fit su metà, ECE sull'altra).
          Due regimi: in-domain GPS-present (rolling OOF) e transfer SHL validate.
          Prerequisito per abstain in produzione e per il ranking AL (6.2): uncertainty = 1−conf
          ha senso come soglia solo se conf è calibrata.
Metodo:   in-domain = eval rolling-OOF canonico (max_prob, correct), GPS-present, 5cl.
          transfer = modello canonico predetto su SHL validate (held-out), 5cl.
Input:    data/v2/processed/eval_trento_20260612_202507.parquet · data/v2/models/trento_20260612_202512.pkl
          · data/v2/features_shl_full.parquet
Output:   research/figures/rq5_4_reliability.{png,pdf}
Alimenta: thesis/results.md (calibrazione 5.4). Sez.tesi: 4 / 7 — prerequisito 6.2.

Run: python research/rq5_4_calibration.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
from tmd.models.registry import load_model  # noqa: E402

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
MODEL = ROOT / "data/v2/models/trento_20260612_202512.pkl"
EVAL = ROOT / "data/v2/processed/eval_trento_20260612_202507.parquet"
FIG = ROOT / "research/figures"
N_BINS, SEED = 15, 42


def ece_mce(conf, correct, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = mce = 0.0
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append((0.5 * (bins[b] + bins[b + 1]), np.nan, 0.0, 0)); continue
        acc, cf, w = correct[m].mean(), conf[m].mean(), m.mean()
        ece += w * abs(acc - cf); mce = max(mce, abs(acc - cf))
        rows.append((cf, acc, w, int(m.sum())))
    return ece, mce, rows


def honest_calib(conf, correct, seed=SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(conf)); half = len(conf) // 2
    cal, tst = perm[:half], perm[half:]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(conf[cal], correct[cal])
    cc = iso.predict(conf[tst])
    return {"ece_before": ece_mce(conf[tst], correct[tst])[0], "ece_after": ece_mce(cc, correct[tst])[0],
            "brier_before": brier_score_loss(correct[tst], conf[tst]),
            "brier_after": brier_score_loss(correct[tst], cc), "n_test": len(tst)}


def get_indomain():
    ev = pd.read_parquet(EVAL)
    ev = ev[(ev.gps_frac > 0.5) & ev.label.isin(FIVE)]
    return ev["max_prob"].to_numpy(float), ev["correct"].astype(int).to_numpy()


def get_transfer():
    b = load_model(MODEL); model, feat = b["model"], b["feature_cols"]
    sh = pd.read_parquet(ROOT / "data/v2/features_shl_full.parquet")
    sh = sh[(sh.split == "validate") & sh.label.isin(FIVE)].reset_index(drop=True)
    X = np.full((len(sh), len(feat)), np.nan, np.float32)
    for i, c in enumerate(feat):
        if c in sh.columns:
            X[:, i] = sh[c].values.astype(np.float32)
    pred, conf = model.predict_proba(X)
    return np.asarray(conf, float), (np.asarray(pred, object) == sh.label.values).astype(int)


def main():
    data = {"in-domain GPS-present": get_indomain(), "transfer SHL validate": get_transfer()}
    print("=" * 64); print("RQ5.4 — calibrazione confidenza top-label (isotonic, split onesto)"); print("=" * 64)
    for name, (conf, corr) in data.items():
        ece, mce, _ = ece_mce(conf, corr)
        res = honest_calib(conf, corr)
        gap = conf.mean() - corr.mean()
        print(f"\n=== {name} ===")
        print(f"  n={len(conf)}  acc={corr.mean():.3f}  mean_conf={conf.mean():.3f}  "
              f"gap={gap:+.3f} → {'OVER' if gap > 0 else 'under'}confident")
        print(f"  ECE(full)={ece:.4f}  MCE={mce:.3f}")
        print(f"  isotonic (n_test={res['n_test']}): ECE {res['ece_before']:.4f} → {res['ece_after']:.4f}"
              f"   Brier {res['brier_before']:.4f} → {res['brier_after']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (name, (conf, corr)) in zip(axes, data.items()):
        ece, _, rows = ece_mce(conf, corr)
        cf = [r[0] for r in rows if r[3] > 0]; ac = [r[1] for r in rows if r[3] > 0]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(cf, ac, "o-", color="#c77757", label="model")
        ax.set_title(f"{name}\nECE={ece:.3f}"); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    # niente suptitle in-immagine (C8): la caption LaTeX dice che sono le curve raw/uncalibrated
    # (i titoli-pannello restano: distinguono i due regimi e ne danno l'ECE)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"rq5_4_reliability.{ext}", dpi=150, bbox_inches="tight")
    print(f"\nReliability diagram → rq5_4_reliability.{{png,pdf}}")


if __name__ == "__main__":
    main()
