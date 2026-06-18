"""
feature_separation_boxplots.py — boxplot per-classe delle feature che il labeler usa, per
mostrare DOVE sta la separazione: nelle quantita' FISICHE universali (cinematica) prima ancora
che nell'infrastruttura. Se la separazione e' nella fisica, il protocollo trasferisce per quella.

Trento, GT MotionTag. Feature GPS su GPS-present (B_n_gps>0). Output: research/figures/feature_separation.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
f = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
f = f[f["label_source"] == "motiontag"]          # solo finestre con GT
present = f["B_n_gps"] > 0

CLASSES = ["Still", "Walk", "Bike", "Car", "Bus", "Train"]
PANELS = [
    ("B_speed_p95", "peak speed p95 (m/s)", True),
    ("B_stop_frac", "stop fraction", True),
    ("B_path_efficiency", "path efficiency", True),
    ("B_speed_std", "speed variability (m/s)", True),
    ("C_osm_rail_prop", "rail proximity", True),
    ("C_bus_stops_prop", "bus-stop proximity", True),
    ("A_lin_mag_iqr", "motion variability (lin. acc.)", False),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.ravel()
for ax, (col, title, gps_only) in zip(axes, PANELS):
    sub = f[present] if gps_only else f
    data = [sub[sub["label"] == c][col].dropna().values for c in CLASSES]
    ax.boxplot(data, labels=CLASSES, showfliers=False, widths=0.6)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    if col == "B_speed_p95":
        ax.axhline(5.0, color="red", ls="--", lw=1, alpha=0.7)   # ceiling locomozione
axes[-1].axis("off")
fig.suptitle("Per-class distributions of the labeler's discriminative features "
             "(Trento, GT; GPS features on GPS-present windows)", fontsize=12)
fig.tight_layout()
out = ROOT / "research/figures/feature_separation.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)

# quantili per lettura testuale
print("\nmediana per classe (GPS-present per le feature GPS):")
hdr = "feature".ljust(20) + "".join(c.rjust(8) for c in CLASSES)
print(hdr)
for col, title, gps_only in PANELS:
    sub = f[present] if gps_only else f
    meds = [sub[sub["label"] == c][col].median() for c in CLASSES]
    print(col.ljust(20) + "".join(f"{m:8.2f}" for m in meds))
