"""
e9_class_dist.py — distribuzione classi GT + sbilanciamento + rappresentatività (Fase 1c · task E9).

Scopo:    - distribuzione classi GT (finestre + viaggi) e RAPPORTO DI SBILANCIAMENTO (→ macro-F1, support);
          - concentrazione per-utente (pochi power-user dominano? → varianza LOUO);
          - #utenti per classe (classi rare in pochi utenti → transfer/LOUO fragili);
          - RAPPRESENTATIVITÀ: funnel registro→IMU→GT, State (dropout) e device dei nostri utenti.
Input:    data/v2/features_trento_full.parquet (label GT, userId) · data/raw_freeze/labels.parquet
          data/raw_freeze/users_2026-06-09.csv (State/Platform) · data/v2/device_map_trento.csv
Output:   research/figures/e9_class_dist.{png,pdf} · e9_per_user_class.{png,pdf}
Alimenta: thesis/eda.md (E9)
Sez.tesi: 3.3 classi / 3.1+7 rappresentatività

Run: /opt/miniconda3/envs/tmd/bin/python research/e9_class_dist.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
ORDER = ["Still", "Walk", "Bike", "Bus", "Car", "Train"]


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum()) - (n + 1) / n) if x.sum() else 0.0


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento_full.parquet", columns=["label", "userId"])
    gt = df[df.label.notna()]
    lab = pd.read_parquet(ROOT / "data/raw_freeze/labels.parquet")

    # ── distribuzione classi ──
    win = gt.label.value_counts().reindex(ORDER).fillna(0).astype(int)
    trip = lab.mode_tmd.value_counts().reindex(ORDER).fillna(0).astype(int)
    print("=" * 64); print("E9 — distribuzione classi GT + rappresentatività"); print("=" * 64)
    print(f"\n{'classe':7s} {'finestre':>9s} {'%':>6s} {'viaggi':>7s} {'%':>6s}")
    for c in ORDER:
        print(f"{c:7s} {win[c]:9d} {100*win[c]/win.sum():6.1f} {trip[c]:7d} {100*trip[c]/trip.sum():6.1f}")
    print(f"sbilanciamento finestre (max/min, escl.0): {win[win>0].max()/win[win>0].min():.0f}:1  "
          f"| viaggi: {trip[trip>0].max()/trip[trip>0].min():.0f}:1")

    # ── #utenti per classe (classi rare in pochi utenti) ──
    upc = gt.groupby("label").userId.nunique().reindex(ORDER).fillna(0).astype(int)
    print(f"\n#utenti con ≥1 finestra GT per classe: {upc.to_dict()}  (tot utenti GT: {gt.userId.nunique()})")

    # ── concentrazione per-utente ──
    pu = gt.groupby("userId").size().sort_values(ascending=False)
    top5 = 100 * pu.head(5).sum() / pu.sum()
    print(f"\nconcentrazione: top-5 utenti = {top5:.0f}% delle finestre GT | Gini = {gini(pu.values):.2f} "
          f"(0=uniforme,1=concentrato)")

    # ── rappresentatività: funnel + State + device ──
    reg = pd.read_csv(ROOT / "data/raw_freeze/users_2026-06-09.csv")
    dev = pd.read_csv(ROOT / "data/v2/device_map_trento.csv")
    gt_users = set(gt.userId.unique())
    reg_gt = reg[reg.User.isin(gt_users)].copy()
    reg_gt["stato"] = reg_gt.State.fillna("?").apply(lambda s: "dropout" if "dropout" in str(s).lower() else "attivo")
    print(f"\nFUNNEL: registro {len(reg)} utenti → IMU {len(dev)} → con GT {len(gt_users)}")
    print(f"State dei nostri utenti-GT (nel registro {len(reg_gt)}/{len(gt_users)}): {reg_gt.stato.value_counts().to_dict()}")
    print(f"  (dettaglio State: {reg_gt.State.value_counts().to_dict()})")
    print(f"device utenti-GT (device_map): {dev[dev.userId.isin(gt_users)].device.value_counts().to_dict()}")

    # ── figure: distribuzione (sx) + durata mediana segmento (dx) = perché window↔segment divergono ──
    dur = (lab.assign(d=(lab.finished_at - lab.started_at) / 60000)  # ms → min
              .groupby("mode_tmd").d.median().reindex(ORDER))
    x = np.arange(len(ORDER)); w = 0.4
    pw, pt = 100*win/win.sum(), 100*trip/trip.sum()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    # sinistra: quote per finestra vs per segmento (MotionTag = trips + stays, Cap.3 R3)
    b1 = axL.bar(x - w/2, pw, w, label="windows")
    b2 = axL.bar(x + w/2, pt, w, label="segments")
    for bars, vals in ((b1, pw), (b2, pt)):          # C8: % annotate (log poco leggibile da sola)
        for r, v in zip(bars, vals):
            axL.text(r.get_x()+r.get_width()/2, r.get_height()*1.07, f"{v:.1f}", ha="center", fontsize=7)
    axL.set_yscale("log"); axL.set_xticks(x); axL.set_xticklabels(ORDER)
    axL.set_ylabel("% of labeled data (log)"); axL.set_xlabel("Reference class")
    axL.set_ylim(top=axL.get_ylim()[1]*1.4); axL.legend(); axL.grid(alpha=.3, axis="y")
    # destra: durata mediana del segmento (spiega la divergenza: corti = poche finestre)
    bb = axR.bar(x, dur.values, w*2, color="tab:green")
    for r, v in zip(bb, dur.values):
        axR.text(r.get_x()+r.get_width()/2, r.get_height()+0.3, f"{v:.1f}", ha="center", fontsize=7.5)
    axR.set_xticks(x); axR.set_xticklabels(ORDER)
    axR.set_ylabel("median segment duration (min)"); axR.set_xlabel("Reference class")
    axR.grid(alpha=.3, axis="y")
    fig.tight_layout()
    savefig("e9_class_dist")

    piv = (gt.assign(v=1).pivot_table(index="userId", columns="label", values="v", aggfunc="sum")
           .reindex(columns=ORDER).fillna(0))
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(8, 4)); bottom = np.zeros(len(piv))
    for c in ORDER:
        plt.bar(range(len(piv)), piv[c].values, bottom=bottom, label=c); bottom += piv[c].values
    plt.xlabel("user (sorted by #GT windows)"); plt.ylabel("GT windows")
    plt.legend(fontsize=7, ncol=3); plt.grid(alpha=.3, axis="y"); savefig("e9_per_user_class")

    print(f"\nfigure → research/figures/e9_{{class_dist,per_user_class}}.png|pdf")


if __name__ == "__main__":
    main()
