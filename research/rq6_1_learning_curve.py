"""
research/rq6_1_learning_curve.py — RQ6.1: learning curve (cold-start → migliora coi dati).

Scopo:    storia deployment. A inizio vita un'app non ha dati locali → cold-start = modello SHL
          trasferito. Man mano che i dati locali si accumulano, il silver model fisico (label-free)
          addestrato sul locale migliora. Misura F1 in funzione di (a) VOLUME finestre e
          (b) DIVERSITÀ utenti accumulati; il crossover col cold-start = da quanti dati il locale
          batte il transfer. Tetto = limite del labeler/feature (oltre serve label vere → AL, 6.2).
Metodo:   test FISSO = split temporale (ultime sessioni), motiontag GT, **GPS-present** (operativo).
          Train pool = silver delle sessioni precedenti. RF gerarchico (canonico). Media su seed.
          Cold-start = modello SHL nativo valutato sullo stesso test Trento.
Input:    data/v2/features_trento.parquet · data/v2/models/shl_20260613_002207.pkl
Output:   research/figures/rq6_1_learning_curve.{png,pdf}
Alimenta: thesis/results.md (learning-curve 6.1). Sez.tesi: 7 / 8 deployment.

Run: python research/rq6_1_learning_curve.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
from tmd.training.trainer import get_feature_cols, temporal_splits  # noqa: E402
from tmd.models.hierarchical import HierarchicalTMD  # noqa: E402
from tmd.models.registry import load_model  # noqa: E402

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
CITY = ["Still", "Walk", "Bike", "Car", "Bus", "Train"]
TRENTO = ROOT / "data/v2/features_trento.parquet"
SHL_MODEL = ROOT / "data/v2/models/shl_20260613_002207.pkl"
FIG = ROOT / "research/figures"
FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]
SEEDS = [0, 1, 2]


def macro5(yt, yp):
    seen = [c for c in FIVE if (np.asarray(yt) == c).sum() > 0]
    return f1_score(yt, yp, labels=seen, average="macro", zero_division=0)


def matrix(df, feats):
    X = np.full((len(df), len(feats)), np.nan, np.float32)
    for i, c in enumerate(feats):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    return X


def main():
    df = pd.read_parquet(TRENTO)
    silver = df[df.silver_label.notna()].copy()
    evalp = df[df.label.notna()].copy()
    feats = get_feature_cols(silver, ["A", "B", "C", "D"])
    _, pool, test = next(iter(temporal_splits(silver, evalp)))
    test = test[test.gps_frac > 0.5]                       # dominio operativo
    Xte, yte = matrix(test, feats), test.label.values

    # cold-start: modello SHL nativo → stesso test Trento (GPS-present)
    shl = load_model(SHL_MODEL)
    cold = macro5(yte, shl["model"].predict_proba(matrix(test, shl["feature_cols"]))[0])
    print("=" * 64); print("RQ6.1 — learning curve (test Trento GPS-present, motiontag)"); print("=" * 64)
    print(f"cold-start (SHL nativo → test Trento): F1 = {cold:.3f}")
    print(f"train pool silver = {len(pool)} | test GPS-present = {len(test)}\n")

    # ── (a) curva per VOLUME (finestre random) ──
    n = len(pool)
    print(f"{'frac':>6}{'n_train':>9}{'TrentoF1(GPS-pres)':>22}")
    vol = []
    for f in FRACTIONS:
        sc = []
        for s in SEEDS:
            rng = np.random.default_rng(s)
            k = min(max(50, int(f * n)), n)
            sub = pool.iloc[rng.choice(n, k, replace=False)]
            cls = [c for c in CITY if c in set(sub.silver_label)]
            m = HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(sub, feats), sub.silver_label.values)
            sc.append(macro5(yte, m.predict(Xte)))
        vol.append((min(int(f * n), n), np.mean(sc), np.std(sc)))
        print(f"{f:>6.2f}{vol[-1][0]:>9}{np.mean(sc):>15.3f}±{np.std(sc):.3f}")

    # ── (b) curva per UTENTI accumulati (diversità) ──
    users = sorted(pool.userId.unique())
    print(f"\nper UTENTI accumulati (diversità) — {len(users)} utenti:")
    print(f"{'n_user':>7}{'macro':>9} | " + " ".join(f"{c:>6}" for c in FIVE))
    usr = []
    for k in [2, 3, 5, 8, 15, 25, len(users)]:
        if k > len(users):
            continue
        mac, pc = [], {c: [] for c in FIVE}
        for s in range(3):
            rng = np.random.default_rng(s)
            sub = pool[pool.userId.isin(rng.choice(users, k, replace=False))]
            cls = [c for c in CITY if c in set(sub.silver_label)]
            if len(cls) < 2:
                continue
            m = HierarchicalTMD(cls, [], clf_type="rf").fit(matrix(sub, feats), sub.silver_label.values)
            yp = m.predict(Xte)
            mac.append(macro5(yte, yp))
            for c in FIVE:
                pc[c].append(f1_score(yte, yp, labels=[c], average="macro", zero_division=0))
        if mac:
            usr.append((k, np.mean(mac)))
            print(f"{k:>7}{np.mean(mac):>9.3f} | " + " ".join(f"{np.mean(pc[c]):>6.3f}" for c in FIVE))

    # ── figura ──
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    vx = [v[0] for v in vol]; vy = [v[1] for v in vol]; ve = [v[2] for v in vol]
    ax[0].errorbar(vx, vy, yerr=ve, fmt="o-", color="tab:blue", label="silver (local)")
    ax[0].axhline(cold, ls="--", color="tab:red", label=f"cold-start SHL ({cold:.2f})")
    ax[0].set_xlabel("# training windows (random)"); ax[0].set_ylabel("macro-F1 (Trento GPS-present)")
    ax[0].set_title("Learning curve by VOLUME"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ux = [u[0] for u in usr]; uy = [u[1] for u in usr]
    ax[1].plot(ux, uy, "o-", color="tab:green", label="silver (local)")
    ax[1].axhline(cold, ls="--", color="tab:red", label=f"cold-start SHL ({cold:.2f})")
    ax[1].set_xlabel("# accumulated users (diversity)"); ax[1].set_ylabel("macro-F1 (Trento GPS-present)")
    ax[1].set_title("Learning curve by USER DIVERSITY"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    # niente suptitle in-immagine (C8): la caption LaTeX racconta la figura (i titoli-pannello restano)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"rq6_1_learning_curve.{ext}", bbox_inches="tight", dpi=150)
    print(f"\nfigura → rq6_1_learning_curve.{{png,pdf}}")
    print(f"\nLettura: VOLUME piatto (poche centinaia di finestre bastano) · UTENTI sale "
          f"({uy[0]:.2f}→{uy[-1]:.2f}) → conta la DIVERSITÀ utenti (ciò che si accumula nel tempo).")


if __name__ == "__main__":
    main()
