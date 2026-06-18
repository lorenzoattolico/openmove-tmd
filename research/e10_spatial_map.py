"""
e10_spatial_map.py — mappa spaziale dei modi (Fase 1c · task E10).

Scopo:    figura descrittiva Cap.3: i tracciati GPS colorati per modo mostrano la geografia del
          segnale C (infrastruttura, E13/E16) — Car/Bus seguono le strade (incl. corridoio A22),
          Train segue la ferrovia. Rende visibile perché l'infrastruttura (C) discrimina.
Input:    data/v2/gps_sessions_trento.parquet (punti GPS: lat/lon/session_id)
          data/v2/features_trento.parquet (label GT per finestra → modo per sessione via maggioranza)
Output:   research/figures/e10_spatial_map.{png,pdf} (4 modi) · e10_spatial_rail_road.{png,pdf} (Train vs Car)
Alimenta: thesis/eda.md (E10)
Sez.tesi: 3.x caratterizzazione dati / 4.5

Nota: modo per-sessione = maggioranza GT delle sue finestre (approssimazione; sessioni multimodali
      collassate). Bbox focalizzato sulla regione operativa (lat 2–99.5° = taglia la coda sud
      extra-regione tenendo il corridoio Brennero→Verona; lon 1–99°). NESSUN userId mostrato (privacy).
Run: /opt/miniconda3/envs/tmd/bin/python research/e10_spatial_map.py
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
COL = {"Walk": "tab:green", "Bus": "tab:orange", "Car": "tab:blue", "Train": "tab:red"}
MODES = ["Walk", "Bus", "Car", "Train"]
SEED = 0


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    pts = pd.read_parquet(ROOT / "data/v2/gps_sessions_trento.parquet")
    win = pd.read_parquet(ROOT / "data/v2/features_trento.parquet", columns=["session_id", "label"])
    win = win[win.label.notna()]
    print("=" * 70); print("E10 — mappa spaziale dei modi"); print("=" * 70)
    print(f"punti GPS: {len(pts):,} | sessioni con GT: {win.session_id.nunique()}")

    # modo per sessione = maggioranza GT
    sess_mode = win.groupby("session_id").label.agg(lambda s: s.value_counts().idxmax())
    pts = pts.copy()
    pts["mode"] = pts.session_id.map(sess_mode)
    p = pts[pts["mode"].isin(MODES)].dropna(subset=["latitude", "longitude"])
    # bbox focalizzato sulla regione operativa: lat asimmetrica (taglia la coda SUD
    # — viaggi extra-regione oltre Verona, ~2% — tenendo il corridoio Brennero→Verona),
    # lon 1–99° (toglie outlier ovest/Milano ed est/Venezia). Tiene ~96% dei punti.
    lo_la, hi_la = p.latitude.quantile([.02, .995])
    lo_lo, hi_lo = p.longitude.quantile([.01, .99])
    p = p[(p.latitude.between(lo_la, hi_la)) & (p.longitude.between(lo_lo, hi_lo))]
    lat0 = float(p.latitude.mean())
    aspect = 1.0 / np.cos(np.radians(lat0))
    print(f"punti mappati (4 modi, dentro bbox): {len(p):,}")
    print("conteggio punti per modo:", {m: int((p["mode"] == m).sum()) for m in MODES})

    # subsample per leggibilità (deterministico)
    def sub(d, n):
        return d.sample(min(n, len(d)), random_state=SEED)

    # ── FIG 1: tutti i modi sovrapposti (Train/Bus/Car/Walk) ──
    plt.figure(figsize=(7.5, 8))
    for m in ["Walk", "Car", "Bus", "Train"]:   # Train sopra
        d = sub(p[p["mode"] == m], 25000)
        plt.scatter(d.longitude, d.latitude, s=1.5, c=COL[m], alpha=.25, label=m, linewidths=0)
    plt.gca().set_aspect(aspect)
    plt.xlabel("longitude"); plt.ylabel("latitude")
    # niente title in-immagine (C8): la caption LaTeX racconta la figura
    lg = plt.legend(markerscale=6, fontsize=9)
    for h in lg.legend_handles:
        h.set_alpha(1)
    plt.grid(alpha=.2); savefig("e10_spatial_map")

    # ── FIG 2: Train vs Car (rail vs strade/A22) — pannelli affiancati ──
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    for a, m in zip(ax, ["Train", "Car"]):
        d = sub(p[p["mode"] == m], 40000)
        a.scatter(d.longitude, d.latitude, s=1.5, c=COL[m], alpha=.3, linewidths=0)
        a.set_aspect(aspect); a.set_title(f"{m} traces ({'rail corridors' if m=='Train' else 'road network incl. A22'})")
        a.set_xlabel("longitude"); a.grid(alpha=.2)
    ax[0].set_ylabel("latitude")
    fig.suptitle("Train follows rail, Car follows roads — basis of the C (infrastructure) features", fontsize=12)
    fig.tight_layout(); savefig("e10_spatial_rail_road")

    print("\nfigure → e10_spatial_map · e10_spatial_rail_road")


if __name__ == "__main__":
    main()
