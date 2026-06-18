"""
protocol_on_shl.py — DIMOSTRAZIONE del transfer-di-PROTOCOLLO (non del modello).

Si cala l'intero protocollo (labeler fisico universale -> silver -> train) su SHL/UK, SENZA alcun
tuning di Trento, e si confronta col supervisionato addestrato sulla GT-SHL: STESSO RF, STESSE
feature, STESSO split — l'unica variabile e' la fonte delle etichette (silver vs GT). Se silver
≈ GT, il protocollo trasferisce (label-free ≈ supervised anche su un altro Paese).

Output: macro-F1 5-classi su validate (GPS-present), per-classe, e il "costo label-free su SHL".
Boxplot SHL: research/figures/feature_separation_shl.png
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from tmd.config import CityConfig
from tmd.labeling.window_labeler import label_windows_universal
from tmd.training.trainer import get_feature_cols

ROOT = Path(__file__).resolve().parents[1]
FIVE = ["Still", "Walk", "Car", "Bus", "Train"]

cfg = CityConfig.from_yaml(ROOT / "tmd/configs/cities/trento.yaml")  # window_labeler vuoto = default universali
fs = pd.read_parquet(ROOT / "data/v2/features_shl_full.parquet")
fs["silver"] = pd.Series(label_windows_universal(fs, cfg)[0], index=fs.index)
feat = get_feature_cols(fs, ["A", "B", "C", "D"])

tr = fs[fs["split"] == "train"]
va = fs[(fs["split"] == "validate") & (fs["label"].isin(FIVE))]
med = tr[feat].median()

def train_eval(ycol, name):
    sub = tr[tr[ycol].isin(FIVE)]
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=0, n_jobs=-1)
    rf.fit(sub[feat].fillna(med), sub[ycol])
    res = {}
    for cond, lbl in [(va["B_n_gps"] > 0, "GPS-present"), (va["B_n_gps"] >= 0, "all")]:
        v = va[cond]
        pred = rf.predict(v[feat].fillna(med))
        f1 = f1_score(v["label"], pred, labels=FIVE, average="macro")
        perc = f1_score(v["label"], pred, labels=FIVE, average=None)
        res[lbl] = f1
        print(f"  {name:<11}[{lbl:<11}] macro-F1 {f1:.3f}   per-cl "
              f"{dict(zip(FIVE, np.round(perc, 2)))}")
    print(f"             (n_train={len(sub)})")
    return res

print("=== PROTOCOLLO-SU-SHL: silver-trained vs GT-trained (stesso RF/feature/split, 5 classi) ===\n")
rs = train_eval("silver", "SILVER")
rg = train_eval("label", "SUPERVISED")
print(f"\nCOSTO label-free su SHL (GT - silver): "
      f"GPS-present {rg['GPS-present'] - rs['GPS-present']:+.3f} | all {rg['all'] - rs['all']:+.3f}")
print(f"   (Trento, riferimento: costo-LF GPS-present +0.039)")

# ── boxplot SHL (la fisica si separa anche in UK?) ──
present = fs["B_n_gps"] > 0
PANELS = [("B_speed_p95", "peak speed (m/s)", True), ("B_stop_frac", "stop fraction", True),
          ("B_path_efficiency", "path efficiency", True), ("C_osm_rail_prop", "rail proximity", True),
          ("C_bus_stops_prop", "bus-stop proximity", True), ("A_lin_mag_iqr", "motion variability", False)]
CL = ["Still", "Walk", "Bike", "Car", "Bus", "Train", "Subway"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.ravel()
print("\nSHL — mediana per classe (GPS-present per feature GPS):")
print("feature".ljust(20) + "".join(c[:5].rjust(8) for c in CL))
for ax, (col, title, gps) in zip(axes, PANELS):
    sub = fs[present] if gps else fs
    data = [sub[sub["label"] == c][col].dropna().values for c in CL]
    ax.boxplot(data, tick_labels=CL, showfliers=False, widths=0.6)
    ax.set_title(title); ax.tick_params(axis="x", rotation=45, labelsize=8); ax.grid(axis="y", alpha=0.3)
    print(col.ljust(20) + "".join(f"{sub[sub['label']==c][col].median():8.2f}" for c in CL))
fig.suptitle("SHL/UK: per-class distributions of the labeler's features (same physics as Trento?)")
fig.tight_layout()
fig.savefig(ROOT / "research/figures/feature_separation_shl.png", dpi=140, bbox_inches="tight")
print("\nsaved research/figures/feature_separation_shl.png")
