"""Test del labeler fisico universale su finestre sintetiche (data-free)."""
from pathlib import Path

import pandas as pd

import tmd
from tmd.config import CityConfig
from tmd.labeling.window_labeler import label_windows_universal

CFG = CityConfig.from_yaml(Path(tmd.__file__).resolve().parent / "configs" / "cities" / "trento.yaml")


def test_cascade_assigns_expected_modes():
    df = pd.DataFrame([
        {"B_speed_mean": 0.10, "B_speed_max": 0.30, "B_stop_frac": 0.95,
         "B_path_efficiency": 0.30, "B_n_gps": 10},                          # Still
        {"B_speed_mean": 1.20, "B_speed_max": 2.00, "B_stop_frac": 0.30,
         "B_path_efficiency": 0.80, "B_n_gps": 10},                          # Walk
        {"B_speed_mean": 15.0, "B_speed_max": 20.0, "B_stop_frac": 0.05,
         "B_path_efficiency": 0.90, "B_n_gps": 10,
         "C_osm_rail_prop": 0.0, "C_bus_stops_prop": 0.0},                   # Car
    ])
    labels, weights = label_windows_universal(df, CFG)
    assert labels[0] == "Still"
    assert labels[1] == "Walk"
    assert labels[2] == "Car"
    assert len(weights) == len(df)


def test_abstain_on_ambiguous():
    # velocita' nella banda ambigua, nessuna infrastruttura -> ABSTAIN (None)
    df = pd.DataFrame([{"B_speed_mean": 3.0, "B_speed_max": 4.5, "B_stop_frac": 0.4,
                        "B_path_efficiency": 0.6, "B_n_gps": 10}])
    labels, _ = label_windows_universal(df, CFG)
    assert labels[0] is None
