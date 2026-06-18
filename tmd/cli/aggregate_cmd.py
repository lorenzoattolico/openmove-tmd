"""tmd.cli.aggregate_cmd — predizioni per-finestra -> modal-split + CO2.

Wrapper sottile su tmd.aggregate. Senza un set di calibrazione produce le quote
naive; la correzione (quantification) richiede etichette vere (vedi production/DEPLOY).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tmd import aggregate as agg

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=None,
                    help="parquet con la colonna predetta (default: data/v2/predictions.parquet)")
    ap.add_argument("--pred-col", default="predicted_class_smooth")
    ap.add_argument("--km-col", default="B_dist_total_m",
                    help="colonna distanza in metri per la CO2 ('' per saltarla)")
    a = ap.parse_args()

    path = a.predictions or str(PROJECT_ROOT / "data" / "v2" / "predictions.parquet")
    df = pd.read_parquet(path)
    pc = a.pred_col if a.pred_col in df.columns else "predicted_class"
    km = (df[a.km_col] / 1000.0).to_numpy() if a.km_col and a.km_col in df.columns else None

    res = agg.aggregate(df[pc].to_numpy(), km=km)
    print("Modal-split (quote dei modi):")
    for m, v in res["modal_split"].items():
        print(f"  {m:<7} {100 * v:5.1f}%")
    if "co2_g" in res:
        print(f"CO2 aggregata: {res['co2_g'] / 1000:.1f} kgCO2e  "
              "(fattori di riferimento, indicatore d'uso non certificato)")


if __name__ == "__main__":
    main()
