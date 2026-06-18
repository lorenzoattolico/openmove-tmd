"""
research/rq5_6_modal_split.py — RQ5.6: errore di modal-split del MODELLO REALE.

Scopo:    E29 misura il modal-split sul CEILING (RF cross-val su GT). Qui si usa il MODELLO
          CANONICO DEPLOYABILE (silver label-free, rolling-OOF → onesto, test=motiontag GT)
          per stimare l'errore di share aggregato che vedrebbe davvero un deployment OpenMove.
          Domanda use-case: gli errori per-finestra si CANCELLANO nell'aggregato (modal-split
          per indicatori di sostenibilità, riferimento ~5% SUMI) o c'è un BIAS sistematico?
Metodo:   predizioni = predicted_class_smooth dell'eval rolling-OOF; modal-split su modi in
          movimento Walk/Bus/Car/Train (Bike caveata E18/E20), livelli FINESTRA + TRIP
          (session-majority). Strato operativo = GPS-present (>0.5); riportato anche ALL.
Input:    data/v2/processed/eval_trento_20260612_202507.parquet (canonico silver rolling)
Output:   research/figures/rq5_6_modal_split_model.{png,pdf} · rq5_6_modal_split_model.csv
Alimenta: thesis/results.md (Modal-split 5.6). Sez.tesi: 1.x use-case / 6 / 7.

Lettura: se TVD-aggregato << errore-per-finestra → errori si cancellano → fit-for-purpose.
Run: python research/rq5_6_modal_split.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
EVAL = ROOT / "data/v2/processed/eval_trento_20260612_202507.parquet"
MOVING = ["Walk", "Bus", "Car", "Train"]


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def shares(series):
    s = series[series.isin(MOVING)].value_counts()
    return (s.reindex(MOVING).fillna(0) / s.sum() * 100)


def tvd(p, q):
    return 0.5 * np.abs(p.values - q.values).sum()


def run(ev, tag, pred_col):
    g = ev[ev.label.isin(MOVING) | ev[pred_col].isin(MOVING)].copy()
    gmov = g[g.label.isin(MOVING)]
    err = 1 - accuracy_score(gmov.label, gmov[pred_col])
    gt_w, pr_w = shares(g.label), shares(g[pred_col])
    maj_gt = g.groupby("session_id").label.agg(lambda s: s.value_counts().idxmax())
    maj_pr = g.groupby("session_id")[pred_col].agg(lambda s: s.value_counts().idxmax())
    gt_t, pr_t = shares(maj_gt), shares(maj_pr)
    out = pd.DataFrame({"GT_win%": gt_w, "pred_win%": pr_w, "GT_trip%": gt_t, "pred_trip%": pr_t})
    out["win_err_pt"] = out["pred_win%"] - out["GT_win%"]
    out["trip_err_pt"] = out["pred_trip%"] - out["GT_trip%"]
    tw, tt = tvd(pr_w, gt_w), tvd(pr_t, gt_t)
    print(f"\n=== {tag} ===  finestre moving-GT={len(gmov)} | errore per-finestra (moving) ~{100*err:.0f}%")
    print(out.round(1).to_string())
    print(f"  🔑 errore per-finestra ~{100*err:.0f}%  vs  modal-split TVD: window {tw:.1f}% | trip {tt:.1f}%")
    print(f"     max share-error per modo: window {out.win_err_pt.abs().max():.1f}pt "
          f"({out.win_err_pt.abs().idxmax()}) | trip {out.trip_err_pt.abs().max():.1f}pt ({out.trip_err_pt.abs().idxmax()})")
    return out, gt_w, pr_w, gt_t, pr_t, tw, tt, 100 * err


def main():
    ev = pd.read_parquet(EVAL)
    pred_col = "predicted_class_smooth" if "predicted_class_smooth" in ev else "predicted_class"
    gp = ev[ev.gps_frac > 0.5]
    out, gt_w, pr_w, gt_t, pr_t, tw, tt, errpct = run(gp, "GPS-present (dominio operativo)", pred_col)
    out.round(1).to_csv(FIG / "rq5_6_modal_split_model.csv")
    run(ev, "ALL windows", pred_col)

    # ── figura (window + trip), GPS-present ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (lvl, gtv, prv, t) in zip(axes, [("window", gt_w, pr_w, tw), ("trip", gt_t, pr_t, tt)]):
        x = np.arange(len(MOVING)); w = 0.38
        ax.bar(x - w/2, gtv.values, w, label="GT (MotionTag)", color="tab:green")
        ax.bar(x + w/2, prv.values, w, label="model (silver, deployable)", color="tab:blue")
        for i in range(len(MOVING)):
            d = prv.values[i] - gtv.values[i]
            ax.text(i, max(gtv.values[i], prv.values[i]) + 1, f"{d:+.0f}", ha="center", fontsize=8,
                    color="tab:red" if abs(d) > 3 else "gray")
        ax.set_xticks(x); ax.set_xticklabels(MOVING); ax.set_ylabel("modal share %")
        ax.set_title(f"{lvl}-level modal split (TVD {t:.1f}%)"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    # niente suptitle in-immagine (C8): la caption LaTeX racconta la figura (i titoli-pannello danno il TVD)
    fig.tight_layout(); savefig("rq5_6_modal_split_model")
    print("\nfigura → rq5_6_modal_split_model | tabella → rq5_6_modal_split_model.csv")


if __name__ == "__main__":
    main()
