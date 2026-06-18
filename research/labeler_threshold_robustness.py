"""
labeler_threshold_robustness.py — robustezza del silver-labeler alle sue soglie.

Scopo:    rispondere alla critica "le soglie del labeler (es. rail 0.35, route-align 0.40)
          sono arbitrarie / fittate su Trento". Due esperimenti, entrambi vs la GT MotionTag:
          (A) DECOMPOSIZIONE prossimità vs allineamento — riesegue il labeler con la sola
              prossimità, il solo allineamento, o entrambi (canonico) → mostra che
              l'allineamento porta la precisione del Bus, la prossimità è quasi ridondante.
          (B) SWEEP delle 4 soglie infrastruttura, una alla volta attorno al valore canonico
              → la PRECISIONE silver e' PIATTA su bande larghe (= soglie NON fittate al target;
              se fossero fittate, la precisione sarebbe un picco). Solo bus_route_align ha un
              "ginocchio" a 0.40 (precisione a plateau da li' in su).
          → fonda l'affermazione di replicabilita'-per-costruzione (Cap.4): i parametri sono
            a priori e l'output e' misurabilmente robusto ad essi.
Metodo:   labeler universale (label_windows_universal) su features_trento_full (25.989 finestre),
          align ricalcolato (infra_align: GPS 623k punti + spatial index OSM + shapes GTFS),
          cache in data/v2/analysis/. Precisione = P(GT==c | silver==c); accordo = silver==GT
          su finestre co-etichettate, ex-Bike.
Input:    data/v2/features_trento_full.parquet, gps_sessions_trento.parquet,
          data/processed/spatial_index_trento.pkl, data/gtfs/trento/urbano/shapes.txt
Output:   research/figures/labeler_threshold_robustness.{png,pdf} + riepilogo stdout (→ results.md)
Alimenta: thesis/results.md (RQ1 — robustezza labeler). Sez.tesi: 4.4 (soglie) / 4.6 (replicabilita').

Run: PYTHONPATH=. python research/labeler_threshold_robustness.py
"""
from __future__ import annotations
import copy, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tmd.config import CityConfig
from tmd.labeling.infra_align import add_infra_features
from tmd.labeling.window_labeler import label_windows_universal

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
CACHE = ROOT / "data" / "v2" / "analysis" / "labeler_align_cache_trento.parquet"
CANON = {"train_rail_min": 0.35, "train_rail_align_min": 0.15,
         "bus_prop_min": 0.20, "bus_route_align_min": 0.40}


def load_with_align() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(ROOT / "data" / "v2" / "features_trento_full.parquet")
    if CACHE.exists():
        dfa = pd.read_parquet(CACHE)
        print(f"align: cache {CACHE.name}")
    else:
        gps = pd.read_parquet(ROOT / "data" / "v2" / "gps_sessions_trento.parquet")
        idx = pickle.load(open(ROOT / "data" / "processed" / "spatial_index_trento.pkl", "rb"))
        shapes = str(ROOT / "data" / "gtfs" / "trento" / "urbano" / "shapes.txt")
        print("align: ricalcolo (BallTree su 623k punti, ~5 min)...")
        dfa = add_infra_features(df, gps, idx, shapes)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        dfa.to_parquet(CACHE, index=False)
    return df, dfa


def main() -> None:
    df, dfa = load_with_align()
    cfg = CityConfig.from_yaml(ROOT / "tmd" / "configs" / "cities" / "trento.yaml")
    gt = df["label"].values
    has_gt = pd.notna(gt)

    def run(ov):
        c = copy.deepcopy(cfg)
        wl = dict(c.window_labeler or {}); wl.update(ov)
        object.__setattr__(c, "window_labeler", wl)
        labs, _ = label_windows_universal(dfa, c)
        return np.array(labs, dtype=object)

    def measure(s, cls):
        m = (s == cls) & has_gt
        n = int(m.sum())
        prec = float((gt[m] == cls).mean()) if n else 0.0
        gm = (gt == cls)
        rec = float((s[gm] == cls).mean()) if gm.sum() else 0.0
        return n, prec, rec

    def agreement(s):
        co = has_gt & pd.notna(s) & (gt != "Bike")
        return float((s[co] == gt[co]).mean()) if co.sum() else 0.0

    # ── (A) decomposizione prossimita' vs allineamento ────────────────────────
    print("\n== (A) DECOMPOSIZIONE prossimita' vs allineamento ==")
    variants = {
        "both (canonico)":  {},
        "solo-prossimita":  {"train_rail_align_min": -1.0, "bus_route_align_min": -1.0},
        "solo-allineamento": {"train_rail_min": -1.0, "bus_prop_min": -1.0},
    }
    for name, ov in variants.items():
        s = run(ov)
        (_, pT, _), (_, pB, _) = measure(s, "Train"), measure(s, "Bus")
        print(f"  {name:18s} | Train prec {pT:.3f} | Bus prec {pB:.3f} | agree {agreement(s):.3f}")

    # ── (B) sweep soglie ──────────────────────────────────────────────────────
    sweeps = {
        "train_rail_min":       ("Train", np.round(np.arange(0.20, 0.501, 0.025), 3)),
        "train_rail_align_min": ("Train", np.round(np.arange(0.05, 0.301, 0.025), 3)),
        "bus_prop_min":         ("Bus",   np.round(np.arange(0.05, 0.351, 0.025), 3)),
        "bus_route_align_min":  ("Bus",   np.round(np.arange(0.20, 0.601, 0.05), 3)),
    }
    titles = {
        "train_rail_min": "rail proximity (Train)",
        "train_rail_align_min": "rail-following (Train)",
        "bus_prop_min": "bus-stop proximity (Bus)",
        "bus_route_align_min": "route-following (Bus)",
    }
    results = {}
    print("\n== (B) SWEEP soglie (precisione vs GT) ==")
    for key, (cls, vals) in sweeps.items():
        rows = []
        for v in vals:
            s = run({key: float(v)})
            _, prec, rec = measure(s, cls)
            rows.append((float(v), prec, rec, agreement(s)))
        results[key] = (cls, np.array(rows))
        pr = results[key][1][:, 1]
        print(f"  {key:22s} | {cls} prec range [{pr.min():.3f}, {pr.max():.3f}] "
              f"(span {pr.max()-pr.min():.3f}) su {vals[0]}..{vals[-1]}")

    # ── figura ────────────────────────────────────────────────────────────────
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, key in zip(axes.flat, sweeps):
        cls, arr = results[key]
        x, prec, rec, agr = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        ax.plot(x, prec, "-o", color="tab:blue", lw=2, ms=3, label=f"{cls} precision")
        ax.plot(x, rec, ":s", color="tab:orange", lw=1.4, ms=2.5, label=f"{cls} recall")
        ax.plot(x, agr, "--", color="gray", lw=1.2, label="overall agreement")
        ax.axvline(CANON[key], color="black", ls="-", lw=0.8, alpha=0.6)
        ax.annotate(f"canonical\n{CANON[key]:g}", xy=(CANON[key], 0.06),
                    fontsize=7, ha="center", color="black")
        ax.set_ylim(0, 1); ax.set_xlabel(f"{key}  ({titles[key]})", fontsize=8)
        ax.set_ylabel("score vs MotionTag GT", fontsize=8)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle("Silver-label precision is flat across each threshold band "
                 "(parameters are not tuned to the target)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"labeler_threshold_robustness.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigura -> research/figures/labeler_threshold_robustness.{{png,pdf}}")


if __name__ == "__main__":
    main()
