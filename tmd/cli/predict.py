"""
tmd.cli.predict — gira un modello su un parquet feature → predictions parquet.

Wrapper sottile su predict_parquet (carica modello, predice, smooth + coherence).

Uso:
    python -m tmd.cli.predict --model data/models/trento_X.pkl \
        --features data/v2/features_trento.parquet [--out PATH] [--city trento]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tmd.config import CityConfig
from tmd.inference.predictor import predict_parquet

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--city", default="trento")
    a = ap.parse_args()

    cfg = CityConfig.from_yaml(PROJECT_ROOT / "tmd" / "configs" / "cities" / f"{a.city}.yaml")
    out = a.out or str(PROJECT_ROOT / "data" / "v2" / "predictions.parquet")
    predict_parquet(a.model, a.features, output_path=out, city_cfg=cfg)


if __name__ == "__main__":
    main()
