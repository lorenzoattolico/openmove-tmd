"""
research/rq6_2_active_learning.py — RQ6.2: pool-based active learning (testbed SHL pulito).

Scopo:    dato un budget di label, quali finestre etichettare per massimizzare la macro-F1?
          Confronto strategie con ORACOLO affidabile (GT SHL pulito): random · least-confidence
          (1−maxP) · entropy · margin (top1−top2). Quantifica il risparmio-label di AL vs random
          → percorso "deployabile" (poche query incerte per viaggio bastano). Prerequisito = la
          confidenza calibrata (5.4); il ranking AL è invariante alla calibrazione monotona →
          si usa l'uncertainty grezza.
Metodo:   pool = SHL train 5cl, eval = SHL validate 5cl, modello = HierarchicalTMD (RF). Tutte le
          strategie partono dallo STESSO seed set (confronto equo). Media su SEEDS.
Input:    data/v2/features_shl_full.parquet
Output:   research/figures/rq6_2_active_learning.{png,pdf}
Alimenta: thesis/results.md (active-learning 6.2). Sez.tesi: 7 / 8 deployment.

Run: python research/rq6_2_active_learning.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
from tmd.models.hierarchical import HierarchicalTMD, STILL_CLASS  # noqa: E402
from tmd.training.trainer import get_feature_cols  # noqa: E402

FIVE = ["Still", "Walk", "Car", "Bus", "Train"]
SEED_N, K, ROUNDS, POOL = 100, 50, 8, 6000
SEEDS = [0, 1, 2, 3]
STRATS = ["random", "least_conf", "entropy", "margin"]
FIG = ROOT / "research/figures"


def macro5(yt, yp):
    return f1_score(yt, yp, labels=FIVE, average="macro", zero_division=0)


def proba_matrix(model, X):
    Xp = model._prep(X)
    classes_all = [STILL_CLASS] + model.l2_classes
    p_moving = model.l1.predict_proba(Xp)[:, 1]
    p_l2 = model.l2.predict_proba(Xp) * p_moving[:, None]
    P = np.zeros((len(X), len(classes_all)))
    P[:, 0] = 1 - p_moving
    for i, cls in enumerate(model.le_l2.classes_):
        P[:, classes_all.index(cls)] = p_l2[:, i]
    return P


def query(strat, model, Xu, k, rng):
    if strat == "random":
        return rng.choice(len(Xu), size=min(k, len(Xu)), replace=False)
    P = proba_matrix(model, Xu); Ps = np.clip(P, 1e-9, 1)
    if strat == "least_conf":
        score = 1 - P.max(1)
    elif strat == "entropy":
        score = -(Ps * np.log(Ps)).sum(1)
    else:  # margin
        s = np.sort(P, 1); score = -(s[:, -1] - s[:, -2])
    return np.argsort(-score)[:k]


def fit(Xl, yl):
    return HierarchicalTMD([c for c in FIVE if c in set(yl)], [], clf_type="rf").fit(Xl, yl)


def main():
    sh = pd.read_parquet(ROOT / "data/v2/features_shl_full.parquet")
    feat = get_feature_cols(sh, ["A", "B", "C", "D"])
    tr = sh[(sh.split == "train") & sh.label.isin(FIVE)].reset_index(drop=True)
    va = sh[(sh.split == "validate") & sh.label.isin(FIVE)].reset_index(drop=True)
    Xva, yva = va[feat].values.astype(np.float32), va.label.values
    print(f"pool train={len(tr)}  eval validate={len(va)}  feat={len(feat)}")

    budgets = [SEED_N + K * r for r in range(ROUNDS + 1)]
    curves = {s: [] for s in STRATS}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        pool = rng.choice(len(tr), size=min(POOL, len(tr)), replace=False)
        Xp = tr.iloc[pool][feat].values.astype(np.float32)
        yp = tr.iloc[pool].label.values
        seed_lab = rng.choice(len(pool), size=SEED_N, replace=False)
        for strat in STRATS:
            lab = list(seed_lab); unlab = list(set(range(len(pool))) - set(lab)); f1s = []
            for r in range(ROUNDS + 1):
                m = fit(Xp[lab], yp[lab]); f1s.append(macro5(yva, m.predict(Xva)))
                if r == ROUNDS:
                    break
                qi = query(strat, m, Xp[unlab], K, np.random.default_rng(1000 * sd + r))
                picked = [unlab[i] for i in qi]; lab += picked; unlab = list(set(unlab) - set(picked))
            curves[strat].append(f1s)
        print(f"  seed {sd} done")

    avg = {s: np.mean(curves[s], 0) for s in STRATS}
    std = {s: np.std(curves[s], 0) for s in STRATS}
    print("\n=== macro-F1 (validate) per budget — media±std su", len(SEEDS), "seed ===")
    print("budget   " + "  ".join(f"{s:>12}" for s in STRATS))
    for j, b in enumerate(budgets):
        print(f"{b:>6}   " + "  ".join(f"{avg[s][j]:.3f}±{std[s][j]:.02f}" for s in STRATS))

    target = avg["random"][-1]
    print(f"\nTarget = F1 finale random ({target:.3f}) @ budget {budgets[-1]}:")
    for s in STRATS:
        hit = next((budgets[j] for j in range(len(budgets)) if avg[s][j] >= target), None)
        save = f"  (risparmio {budgets[-1]-hit} label = {100*(budgets[-1]-hit)/budgets[-1]:.0f}%)" if hit and hit < budgets[-1] else ""
        print(f"  {s:12}: raggiunto a budget {hit}{save}")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s in STRATS:
        ax.plot(budgets, avg[s], "o-", label=s)
        ax.fill_between(budgets, avg[s] - std[s], avg[s] + std[s], alpha=0.12)
    ax.set_xlabel("# human labels (budget)"); ax.set_ylabel("macro-F1 (SHL validate)")
    # niente title in-immagine (C8): la caption LaTeX racconta la figura
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    # C8: annotare il vantaggio entropy (raggiunge la F1-finale-random con −20% label)
    ax.axhline(target, ls=":", color="tab:blue", lw=1)
    ax.annotate("entropy reaches random@500\nwith 400 labels (−20%)",
                xy=(398, target + 0.001), xytext=(165, 0.715),
                fontsize=8, color="dimgray", ha="left",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8,
                                connectionstyle="arc3,rad=0.15"))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"rq6_2_active_learning.{ext}", dpi=150, bbox_inches="tight")
    print(f"\nCurva → rq6_2_active_learning.{{png,pdf}}")


if __name__ == "__main__":
    main()
