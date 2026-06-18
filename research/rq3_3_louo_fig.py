"""
research/rq3_3_louo_fig.py — RQ3.3: figura LOUO (varianza cross-utente).

Scopo:    figura della generalizzazione cross-utente (LOUO): distribuzione delle macro-F1 per-utente
          (leave-one-user-out). Il messaggio onesto NON è il mean (gonfiato da utenti mono-classe) ma
          la **varianza altissima** → forte dipendenza-utente = limite cold-start da dichiarare.
Metodo:   legge i `fold_results` (32 utenti) dal meta del modello LOUO già addestrato (nessun re-run);
          boxplot + strip dei per-user F1, mediana + mean±std annotati.
Input:    data/v2/models/trento_20260612_203303_meta.json (eval_strategy=louo, source=silver)
Output:   research/figures/rq3_3_louo.{png,pdf}
Alimenta: thesis/results.md §RQ3 (LOUO 3.3). Sez.tesi: 6.x dipendenza-utente / 7 cold-start.

Run: python research/rq3_3_louo_fig.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data/v2/models/trento_20260612_203303_meta.json"
FIG = ROOT / "research/figures"


def main():
    m = json.load(open(META))
    assert m["eval_strategy"] == "louo", "non è un modello LOUO"
    f1 = np.array([r["f1_macro_seen"] for r in m["fold_results"]])
    print(f"LOUO: {len(f1)} utenti | mean {f1.mean():.3f} ± {f1.std():.3f} | "
          f"mediana {np.median(f1):.3f} | min {f1.min():.3f} max {f1.max():.3f} | "
          f"IQR [{np.percentile(f1,25):.2f}, {np.percentile(f1,75):.2f}]")

    fig, ax = plt.subplots(figsize=(4.6, 5))
    ax.boxplot(f1, widths=0.5, showmeans=True,
               medianprops=dict(color="tab:blue", lw=2),
               meanprops=dict(marker="D", markerfacecolor="tab:red", markeredgecolor="tab:red"))
    x = np.random.default_rng(0).normal(1, 0.04, len(f1))
    ax.scatter(x, f1, alpha=0.5, s=22, color="grey", zorder=3)
    ax.set_xticks([1]); ax.set_xticklabels([f"LOUO\n({len(f1)} users)"])
    ax.set_ylabel("macro-F1 (per held-out user)"); ax.set_ylim(0, 1.02)
    # niente title in-immagine (C8): la caption LaTeX porta media/mediana/IQR
    ax.grid(alpha=.3, axis="y")
    ax.text(0.97, 0.04, "high variance =\nstrong user-dependence\n(cold-start limit)",
            transform=ax.transAxes, fontsize=8, color="tab:red",
            ha="right", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"rq3_3_louo.{ext}", dpi=150, bbox_inches="tight")
    print("figura → rq3_3_louo.{png,pdf}")


if __name__ == "__main__":
    main()
