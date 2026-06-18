"""
e19_abstain.py — analisi dell'ABSTAIN del silver-labeler (Fase 1c · task E19).

Scopo:    DOVE e PERCHÉ il labeler si astiene (37% delle finestre, silver_weight==0), e che
          COSTO ha per il training (copertura persa + bias di distribuzione). Completa E18:
          E18 = accuratezza dove committa; E19 = dove NON committa e con quali conseguenze.
Driver:   (1) assenza GPS (niente cinematica/infra → niente regole); (2) banda di velocità
          ambigua (Walk/Bike/slow-vehicle si sovrappongono fisicamente).
Input:    data/v2/features_trento_full.parquet (label GT + silver_label + silver_weight + B_speed_mean + gps_frac)
Output:   research/figures/e19_abstain_drivers.{png,pdf} (by-class + vs-speed)
          research/figures/e19_training_skew.{png,pdf} (silver-train vs GT) · e19_abstain.csv
Alimenta: thesis/eda.md (E19)
Sez.tesi: 4.4 weak-supervision / 6.4

Lettura: ABSTAIN è principiato (preserva la precisione, E18) ma ha un COSTO = bias verso Still
         e perdita di Walk/Bus/Bike nel set di training → il modello ne deve tener conto.
Run: /opt/miniconda3/envs/tmd/bin/python research/e19_abstain.py
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


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento_full.parquet")
    abst = df.silver_label.isna()
    print("=" * 70); print("E19 — analisi ABSTAIN del silver-labeler"); print("=" * 70)
    print(f"finestre {len(df)} | ABSTAIN {abst.sum()} ({100*abst.mean():.0f}%) | "
          f"== silver_weight==0: {bool((abst == (df.silver_weight == 0)).all())} (decisione netta a peso 0)")

    # ── driver 1: GPS ──
    print("\nDriver 1 — disponibilità GPS:")
    for nm, m in [("GPS-present(>0.5)", df.gps_frac > 0.5), ("GPS-absent(<=0.5)", df.gps_frac <= 0.5)]:
        print(f"  {nm:18s} n={m.sum():6d}  abstain={100*df[m].silver_label.isna().mean():.0f}%")

    # ── by GT class (overall + within GPS-present) ──
    g = df[df.label.notna()]
    gp = g[g.gps_frac > 0.5]
    classes = [c for c in ORDER if c in g.label.unique()]
    rows = []
    print("\nABSTAIN per classe GT (overall | entro GPS-present):")
    for c in classes:
        a_all = g[g.label == c].silver_label.isna().mean()
        a_gp = gp[gp.label == c].silver_label.isna().mean() if (gp.label == c).any() else np.nan
        rows.append({"class": c, "abstain_all": a_all, "abstain_gps_present": a_gp,
                     "n_all": int((g.label == c).sum())})
        print(f"  {c:6s} {100*a_all:3.0f}% | {100*a_gp:3.0f}%")
    met = pd.DataFrame(rows).set_index("class")

    # ── driver 2: banda di velocità (TUTTE le finestre GPS-present: l'abstain è scelta del
    #    labeler data la velocità, indipendente dalla GT → più rappresentativo e più dati) ──
    sp = df[df.gps_frac > 0.5].dropna(subset=["B_speed_mean"]).copy()
    bins = [-.1, 0.5, 2, 5, 10, 100]
    labs = ["~0 (still)", "0.5-2 (walk)", "2-5 (slow)", "5-10 (mid)", ">10 (fast)"]
    sp["spd_bin"] = pd.cut(sp.B_speed_mean, bins, labels=labs)
    spd = sp.groupby("spd_bin", observed=True).agg(n=("silver_label", "size"),
                                                    abstain=("silver_label", lambda s: s.isna().mean()))
    print("\nDriver 2 — banda di velocità (GPS-present): ABSTAIN a U, picco nella banda ambigua")
    print(spd.assign(abstain=(spd.abstain * 100).round(0)).to_string())

    # ── costo: bias del set di training ──
    tr = df[df.silver_label.notna()].silver_label.value_counts(normalize=True).mul(100)
    gt = df[df.label.notna()].label.value_counts(normalize=True).mul(100)
    skew = pd.DataFrame({"GT_%": gt, "silver_train_%": tr}).reindex(ORDER).fillna(0).round(0)
    skew.to_csv(FIG / "e19_abstain.csv")
    print("\nCOSTO — distribuzione classi: silver-train vs GT (l'ABSTAIN distorce):")
    print(skew.to_string())
    print("  → over-Still, under-Walk/Bus, Bike=0 → il modello su silver eredita questo bias.")

    # ── FIG 1: drivers (1x2) ──
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    x = np.arange(len(classes)); w = 0.38
    ax[0].bar(x - w/2, met.abstain_all * 100, w, label="all windows", color="tab:gray")
    ax[0].bar(x + w/2, met.abstain_gps_present * 100, w, label="GPS-present only", color="tab:blue")
    ax[0].set_xticks(x); ax[0].set_xticklabels(classes); ax[0].set_ylabel("ABSTAIN %")
    ax[0].set_title("ABSTAIN by GT class"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    ax[1].bar(range(len(spd)), spd.abstain * 100, color="tab:orange")
    ax[1].set_xticks(range(len(spd))); ax[1].set_xticklabels(spd.index, rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("ABSTAIN %"); ax[1].set_xlabel("mean GPS speed band")
    ax[1].set_title("ABSTAIN vs speed (GPS-present)\npeak in the ambiguous slow band"); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); savefig("e19_abstain_drivers")

    # ── FIG 2: training skew ──
    x = np.arange(len(ORDER)); w = 0.38
    plt.figure(figsize=(7, 4.2))
    plt.bar(x - w/2, skew["GT_%"], w, label="GT distribution", color="tab:green")
    plt.bar(x + w/2, skew["silver_train_%"], w, label="silver-train distribution", color="tab:red")
    plt.xticks(x, ORDER); plt.ylabel("% of labeled windows")
    # niente title in-immagine (C8): la caption LaTeX racconta la figura (over-Still, under-Walk/Bus, Bike=0)
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y"); savefig("e19_training_skew")

    print("\nfigure → e19_abstain_drivers · e19_training_skew | tabella → e19_abstain.csv")


if __name__ == "__main__":
    main()
