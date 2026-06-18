"""Boxplot + quantili per-classe di C_bus_stops_prop e C_osm_rail_prop su GeoLife (China-only)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
from load_geolife import read_user_points, MODE_MAP, WIN_MS, MIN_PTS  # noqa: E402

GEOLIFE = ROOT / "data/external/raw/geolife/Data"
FOUR = ["Walk", "Car", "Bus", "Train"]
g = pd.read_parquet(ROOT / "data/processed/features_geolife.parquet")
g = g[g.label.isin(FOUR) & g.in_china].reset_index(drop=True)

# raw_mode per separare subway (per la nota text)
rows = []
for ud in sorted([d for d in GEOLIFE.iterdir() if d.is_dir() and (d / "labels.txt").exists()]):
    pts = read_user_points(ud)
    if pts is None or len(pts) < MIN_PTS:
        continue
    lab = pd.read_csv(ud / "labels.txt", sep="\t", skiprows=1, header=None, names=["s", "e", "m"])
    seg = 0
    for _, r in lab.iterrows():
        raw = str(r.m).strip().lower()
        if MODE_MAP.get(raw) not in FOUR:
            continue
        t0 = pd.to_datetime(r.s, format="%Y/%m/%d %H:%M:%S").value // 1_000_000
        t1 = pd.to_datetime(r.e, format="%Y/%m/%d %H:%M:%S").value // 1_000_000
        lo, hi = int(np.searchsorted(pts[:, 0], t0, "left")), int(np.searchsorted(pts[:, 0], t1, "right"))
        if hi - lo < MIN_PTS:
            continue
        sub = pts[lo:hi]; seg += 1; w0 = sub[0, 0]
        while w0 + WIN_MS <= sub[-1, 0] + WIN_MS:
            wl = int(np.searchsorted(sub[:, 0], w0, "left")); wh = int(np.searchsorted(sub[:, 0], w0 + WIN_MS, "left")); w0 += WIN_MS
            if wh - wl >= MIN_PTS:
                rows.append({"session_id": f"geolife_{ud.name}_s{seg}", "ts_start": int(sub[wl, 0]), "raw_mode": raw})
g = g.merge(pd.DataFrame(rows), on=["session_id", "ts_start"], how="left")

FEATS = [("C_bus_stops_prop", "bus-stop proximity"), ("C_osm_rail_prop", "rail proximity")]
print("Per-classe: n · %prop>0 · mediana · p75 · p90\n")
for col, title in FEATS:
    print(f"== {col} ({title}) ==")
    for c in FOUR:
        s = g[g.label == c][col].dropna()
        print(f"  {c:<6} n={len(s):>6}  >0:{(s>0).mean():5.1%}  med {s.median():.3f}  p75 {s.quantile(.75):.3f}  p90 {s.quantile(.9):.3f}")
    print()
# nota subway vs train sul rail
for col in ["C_osm_rail_prop", "C_bus_stops_prop"]:
    sub = g[g.raw_mode == "subway"][col].dropna(); trn = g[g.raw_mode == "train"][col].dropna()
    print(f"  [{col}] subway med {sub.median():.3f} (>0 {(sub>0).mean():.0%}) | train med {trn.median():.3f} (>0 {(trn>0).mean():.0%})")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (col, title) in zip(axes, FEATS):
    ax.boxplot([g[g.label == c][col].dropna().values for c in FOUR], tick_labels=FOUR, showfliers=False, widths=0.6)
    ax.set_title(f"{title}  ({col})"); ax.set_ylabel("fraction of fixes within threshold"); ax.grid(axis="y", alpha=.3)
fig.suptitle("GeoLife (Beijing): infrastructure-proximity features by class")
fig.tight_layout()
out = ROOT / "research/figures/geolife_infra_separation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nfigura → {out.relative_to(ROOT)}")
