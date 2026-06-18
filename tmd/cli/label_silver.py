"""
tmd.cli.label_silver — aggiunge silver_label / silver_weight al parquet feature.

Default: window + infra (canonico). --no-infra → window puro label-free.
Per l'infra serve: gps_sessions_{city}.parquet (prodotto dall'orchestratore Step 5),
spatial_index_{city}.pkl, e shapes.txt GTFS.

Uso:
    python -m tmd.cli.label_silver --city trento [--features PATH] [--out PATH] [--no-infra]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from tmd.config import CityConfig
from tmd.labeling.silver import label_silver

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="trento")
    ap.add_argument("--features", default=None,
                    help="parquet feature (default data/v2/features_{city}.parquet)")
    ap.add_argument("--out", default=None, help="default: stesso path (in-place)")
    ap.add_argument("--no-infra", action="store_true", help="window puro (no route/rail align)")
    a = ap.parse_args()

    cfg = CityConfig.from_yaml(PROJECT_ROOT / "tmd" / "configs" / "cities" / f"{a.city}.yaml")
    feat_path = Path(a.features) if a.features else PROJECT_ROOT / "data" / "v2" / f"features_{a.city}.parquet"
    out = Path(a.out) if a.out else feat_path
    df = pd.read_parquet(feat_path)

    use_infra = not a.no_infra
    gps = spatial_index = shapes = None
    if use_infra:
        gps_path = feat_path.parent / f"gps_sessions_{a.city}.parquet"
        idx_path = PROJECT_ROOT / "data" / "processed" / f"spatial_index_{a.city}.pkl"
        shapes_p = PROJECT_ROOT / "data" / "gtfs" / a.city / "urbano" / "shapes.txt"
        if gps_path.exists() and idx_path.exists() and shapes_p.exists():
            gps = pd.read_parquet(gps_path)
            spatial_index = pickle.load(open(idx_path, "rb"))
            shapes = str(shapes_p)
        else:
            print("[!] infra non disponibile (gps/index/shapes mancanti) → window puro")
            use_infra = False

    out_df = label_silver(df, cfg, use_infra=use_infra, gps=gps,
                          spatial_index=spatial_index, shapes_path=shapes)
    out_df.to_parquet(out, index=False)

    n = int(out_df["silver_label"].notna().sum())
    print(f"silver: {n:,}/{len(out_df):,} ({100*n/max(len(out_df),1):.0f}%) "
          f"| infra={'ON' if use_infra else 'OFF'} → {out}")
    print("distribuzione:", out_df["silver_label"].value_counts().to_dict())
    leak = [c for c in ("C_bus_route_align", "C_rail_align") if c in out_df.columns]
    print("align nel file (deve essere vuoto):", leak or "nessuna ✓")


if __name__ == "__main__":
    main()
