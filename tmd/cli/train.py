"""
scripts/train.py — CLI wrapper per tmd.training.trainer

Eseguire da project root:
  python scripts/train.py --parquet data/processed/features_shl_bootstrap.parquet \
      --city shl --groups A,B,C,D --eval-strategy fixed --split-col split \
      --specialists S1 S2

  python scripts/train.py --parquet data/processed/features_trento.parquet \
      --city trento --eval-strategy louo --specialists S1 --cleanlab
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # tmd/cli/ → repo root
sys.path.insert(0, str(PROJECT_ROOT))

from tmd.training.trainer import run_training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",       required=True)
    p.add_argument("--city",          default="shl")
    p.add_argument("--groups",        default="A,B,D")
    p.add_argument("--eval-strategy", dest="eval_strategy",
                   default="fixed", choices=["fixed", "louo", "session", "temporal", "rolling"])
    p.add_argument("--split-col",     dest="split_col", default="split")
    p.add_argument("--specialists",   nargs="*", default=["S1", "S2"],
                   choices=["S1", "S2"])
    p.add_argument("--no-specialists", dest="no_specialists",
                   action="store_true")
    p.add_argument("--clf",            default="xgboost",
                   choices=["xgboost", "rf", "lgbm"],
                   help="Classificatore base (default: xgboost)")
    p.add_argument("--cleanlab",      action="store_true")
    p.add_argument("--registry",      default="data/models")
    p.add_argument("--source",        default="motiontag",
                   choices=["motiontag", "silver"],
                   help="Label source per training. "
                        "motiontag=label ground truth, "
                        "silver=silver_label da assign_silver_labels.py")
    p.add_argument("--win-s",             dest="win_s", type=float, default=None,
                   help="Lunghezza finestra in secondi. None = inferita dal parquet.")
    p.add_argument("--no-exclude-users",  dest="no_exclude_users", action="store_true",
                   help="Ignora exclude_users dalla city config (per ablation).")
    p.add_argument("--gps-dropout",       dest="gps_dropout", type=float, default=0.0,
                   help="Prob. di mascherare B_/C_ a NaN durante training (0=off, es. 0.3).")
    return p.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
