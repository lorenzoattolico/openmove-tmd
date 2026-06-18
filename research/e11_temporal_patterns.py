"""
e11_temporal_patterns.py — pattern ora/giorno per modo (Fase 1c · task E11).

Scopo:    caratterizzazione temporale dei viaggi (Cap.3): a che ORA e in che GIORNO si usa
          ciascun modo? (ore di punta, feriale vs weekend). Descrittivo — arricchisce il capitolo
          dati. Nota: il modello NON usa il tempo (evita overfit al periodo di raccolta).
Input:    data/v2/features_trento.parquet (label GT + ts_start epoch-ms + gps_frac)
Output:   research/figures/e11_hour_by_mode.{png,pdf} (heatmap ora×modo) · e11_weekday_weekend.{png,pdf}
          research/figures/e11_temporal.csv
Alimenta: thesis/eda.md (E11)
Sez.tesi: 3.x caratterizzazione dati

⚠ ts_start = epoch-ms (UTC); convertito in ORA LOCALE Europe/Rome (come il freeze).
⚠ Campione piccolo (3 settimane, ~30 utenti beta) → pattern indicativi, NON generalizzabili.
Run: /opt/miniconda3/envs/tmd/bin/python research/e11_temporal_patterns.py
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
MOVING = ["Walk", "Bus", "Car", "Train"]   # Bike caveata (E22)
ALLM = ["Still", "Walk", "Bus", "Car", "Train"]


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    g = df[df.label.notna()].copy()
    # epoch-ms (UTC) → ora locale Europe/Rome
    t = pd.to_datetime(g.ts_start, unit="ms", utc=True).dt.tz_convert("Europe/Rome")
    g["hour"] = t.dt.hour
    g["weekend"] = t.dt.dayofweek >= 5
    print("=" * 70); print("E11 — pattern ora/giorno per modo (ora locale Europe/Rome)"); print("=" * 70)
    print(f"finestre GT: {len(g)} | range: {t.min():%Y-%m-%d %H:%M} → {t.max():%Y-%m-%d %H:%M} (IT)")
    print(f"weekend: {100*g.weekend.mean():.0f}% delle finestre")

    # ── ora×modo (normalizzato per modo: 'a che ora si usa il modo X') ──
    ct = pd.crosstab(g.label, g.hour).reindex(ALLM).fillna(0)
    ct = ct.reindex(columns=range(24), fill_value=0)
    norm = ct.div(ct.sum(axis=1), axis=0)   # ogni riga somma 1
    ct.to_csv(FIG / "e11_temporal.csv")
    # ore di picco per modo (movimento)
    print("\nOre di picco (top-3) per modo di viaggio (ora locale):")
    for m in MOVING:
        top = norm.loc[m].nlargest(3)
        print(f"  {m:6s}: " + ", ".join(f"{h:02d}h({100*v:.0f}%)" for h, v in top.items()))

    # ── feriale vs weekend (modal split sui modi in movimento) ──
    wk = g[g.label.isin(MOVING)]
    split = pd.DataFrame({
        "weekday": wk[~wk.weekend].label.value_counts(normalize=True).reindex(MOVING).fillna(0) * 100,
        "weekend": wk[wk.weekend].label.value_counts(normalize=True).reindex(MOVING).fillna(0) * 100,
    })
    print("\nModal split feriale vs weekend (modi in movimento, %):")
    print(split.round(1).to_string())

    # ── FIG 1: heatmap ora×modo ──
    plt.figure(figsize=(10, 3.6))
    im = plt.imshow(norm.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="fraction of mode's windows")
    plt.yticks(range(len(ALLM)), ALLM); plt.xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    plt.xlabel("hour of day (Europe/Rome)"); plt.title("When is each mode used? (row-normalized hour-of-day profile)")
    savefig("e11_hour_by_mode")

    # ── FIG 2: feriale vs weekend ──
    x = np.arange(len(MOVING)); w = 0.38
    plt.figure(figsize=(7, 4.2))
    plt.bar(x - w/2, split["weekday"].values, w, label="weekday", color="tab:blue")
    plt.bar(x + w/2, split["weekend"].values, w, label="weekend", color="tab:orange")
    plt.xticks(x, MOVING); plt.ylabel("modal share %")
    plt.title("Weekday vs weekend modal split (moving modes)")
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y")
    savefig("e11_weekday_weekend")

    print("\n⚠ campione piccolo (3 sett, ~30 utenti beta) → pattern indicativi, non generalizzabili.")
    print("figure → e11_hour_by_mode · e11_weekday_weekend | tabella → e11_temporal.csv")


if __name__ == "__main__":
    main()
