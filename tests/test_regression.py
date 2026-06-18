"""Regressione: il modello spedito carica, e il canonico riproduce 0.6298."""
import glob
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODEL = next(iter(glob.glob(str(REPO / "models" / "trento_*.pkl"))), None)
FEATURES = REPO / "data" / "v2" / "features_trento.parquet"


@pytest.mark.skipif(MODEL is None, reason="modello spedito assente")
def test_shipped_model_loads():
    from tmd.models.registry import load_model
    m = load_model(MODEL)
    assert "model" in m and len(m["feature_cols"]) == 163
    assert type(m["model"]).__name__ == "HierarchicalTMD"


@pytest.mark.slow
@pytest.mark.skipif(not FEATURES.exists(), reason="dati congelati non disponibili")
def test_canonical_reproduces_0_6298():
    reg = str(REPO / "data" / "v2" / "_test_regen")
    subprocess.run(
        [sys.executable, "-m", "tmd.cli.train", "--parquet", str(FEATURES),
         "--city", "trento", "--groups", "A,B,C,D", "--source", "silver",
         "--eval-strategy", "rolling", "--clf", "rf", "--no-specialists",
         "--registry", reg],
        check=True, capture_output=True, cwd=REPO,
    )
    meta = json.load(open(sorted(glob.glob(reg + "/*_meta.json"))[-1]))
    assert round(meta["f1_macro_mean"], 4) == 0.6298
