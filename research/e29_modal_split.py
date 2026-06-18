"""
e29_modal_split.py — errore di modal-split aggregato (Fase 1c · task E29).

Scopo:    la metrica USE-CASE di OpenMove non è l'accuratezza per-finestra ma il MODAL-SPLIT
          aggregato (quote per modo a livello popolazione, per indicatori di sostenibilità/SUMI).
          Domanda chiave: gli errori per-finestra si CANCELLANO nell'aggregato (buono per il
          use-case) o c'è un BIAS sistematico? Misura share-error per modo + TVD e lo confronta
          con l'errore per-finestra (1−acc).
Metodo:   pred out-of-sample (RF 200 cross-val 5-fold, come E23) su GT GPS-present; modal-split
          a livello FINESTRA e TRIP (session-majority), su modi in movimento (Walk/Bus/Car/Train;
          Bike caveata: rumorosa E20 + persa nel silver E18). Numeri ceiling (RF su GT).
Input:    data/v2/features_trento.parquet (label GT + 163 feat + session_id + gps_frac)
Output:   research/figures/e29_modal_split.{png,pdf} · e29_modal_split.csv
Alimenta: thesis/eda.md (E29)
Sez.tesi: 1.x use-case / 6.x / 7

Lettura: se TVD-aggregato << errore-per-finestra → gli errori si cancellano → il sistema è
         fit-for-purpose per il modal-split anche con accuratezza per-finestra imperfetta.
Run: /opt/miniconda3/envs/tmd/bin/python research/e29_modal_split.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
SEED = 0
MOVING = ["Walk", "Bus", "Car", "Train"]   # Bike caveata (E18/E20)


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def shares(series, modes):
    s = series[series.isin(modes)].value_counts()
    return (s.reindex(modes).fillna(0) / s.sum() * 100)


def tvd(p, q):
    return 0.5 * np.abs(p.values - q.values).sum()


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    g = df[(df.gps_frac > 0.5) & df.label.notna()].copy().reset_index(drop=True)
    feat = [c for c in g.columns if c[:2] in ("A_", "B_", "C_", "D_")]
    X = SimpleImputer(strategy="median").fit_transform(g[feat])
    le = LabelEncoder(); y = le.fit_transform(g.label)
    print("=" * 70); print("E29 — errore di modal-split aggregato (use-case OpenMove)"); print("=" * 70)
    print(f"finestre GT GPS-present: {len(g)} | modi in movimento: {MOVING} (Bike caveata)")

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pp = cross_val_predict(RandomForestClassifier(n_estimators=200, class_weight="balanced",
                           random_state=SEED, n_jobs=-1), X, y, cv=skf, method="predict_proba", n_jobs=-1)
    g["pred"] = le.inverse_transform(pp.argmax(1))
    acc = accuracy_score(g.label, g.pred)
    acc_mov = accuracy_score(g[g.label.isin(MOVING)].label, g[g.label.isin(MOVING)].pred)
    print(f"\nper-window accuracy: all {acc:.3f} | moving-only {acc_mov:.3f} → errore per-finestra ~{100*(1-acc_mov):.0f}%")

    # ── modal-split a livello FINESTRA (moving) ──
    gt_w = shares(g.label, MOVING)
    pr_w = shares(g.pred, MOVING)
    # ── modal-split a livello TRIP (session-majority) ──
    maj_gt = g.groupby("session_id").label.agg(lambda s: s.value_counts().idxmax())
    maj_pr = g.groupby("session_id").pred.agg(lambda s: s.value_counts().idxmax())
    gt_t = shares(maj_gt, MOVING)
    pr_t = shares(maj_pr, MOVING)

    out = pd.DataFrame({"GT_window%": gt_w, "pred_window%": pr_w,
                        "GT_trip%": gt_t, "pred_trip%": pr_t})
    out["win_err_pt"] = (out["pred_window%"] - out["GT_window%"])
    out["trip_err_pt"] = (out["pred_trip%"] - out["GT_trip%"])
    out.round(1).to_csv(FIG / "e29_modal_split.csv")
    print("\nModal-split (quote % sui modi in movimento):")
    print(out.round(1).to_string())

    tvd_w, tvd_t = tvd(pr_w, gt_w), tvd(pr_t, gt_t)
    print(f"\n🔑 errore per-finestra (moving) ~{100*(1-acc_mov):.0f}%  vs  modal-split TVD: "
          f"window {tvd_w:.1f}% | trip {tvd_t:.1f}%")
    if tvd_w < 100 * (1 - acc_mov):
        print(f"   → gli errori si CANCELLANO in aggregato (TVD-window {tvd_w:.1f}% << errore-finestra {100*(1-acc_mov):.0f}%) "
              f"→ fit-for-purpose per il modal-split.")
    print(f"   max share-error per modo: window {out.win_err_pt.abs().max():.1f}pt ({out.win_err_pt.abs().idxmax()}) | "
          f"trip {out.trip_err_pt.abs().max():.1f}pt ({out.trip_err_pt.abs().idxmax()})")

    # ── figura (window + trip) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (lvl, gtv, prv, t) in zip(axes, [("window", gt_w, pr_w, tvd_w), ("trip", gt_t, pr_t, tvd_t)]):
        x = np.arange(len(MOVING)); w = 0.38
        ax.bar(x - w/2, gtv.values, w, label="GT", color="tab:green")
        ax.bar(x + w/2, prv.values, w, label="predicted", color="tab:blue")
        for i, m in enumerate(MOVING):
            d = prv.values[i] - gtv.values[i]
            ax.text(i, max(gtv.values[i], prv.values[i]) + 1, f"{d:+.0f}", ha="center", fontsize=8,
                    color="tab:red" if abs(d) > 3 else "gray")
        ax.set_xticks(x); ax.set_xticklabels(MOVING); ax.set_ylabel("modal share %")
        ax.set_title(f"{lvl}-level modal split (TVD {t:.1f}%)"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle(f"Aggregate modal-split error vs per-window error ({100*(1-acc_mov):.0f}%): do errors cancel?", fontsize=11)
    fig.tight_layout(); savefig("e29_modal_split")
    print("\nfigura → e29_modal_split | tabella → e29_modal_split.csv")


if __name__ == "__main__":
    main()
