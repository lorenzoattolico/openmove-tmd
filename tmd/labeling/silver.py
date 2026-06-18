"""
tmd.labeling.silver — produzione di silver_label / silver_weight (Step 6).

Consolida i 3 entry-point del vecchio (`assign_silver_labels` window/hybrid +
`label_silver_infra`) in UNA funzione con un flag:
  - use_infra=True  → window + infrastructure-following (CANONICO, = label_silver_infra)
  - use_infra=False → window puro, label-free (= assign_silver_labels --mode window)

Il path hybrid/segment-LF (calibrato su motiontag) è NON-canonico e NON portato qui.

ANTI-LEAK: le feature di infrastructure-following (C_bus_route_align / C_rail_align,
mappe LOCALI) servono SOLO al labeler per pulire Bus/Train; vengono SEMPRE droppate
prima di restituire → il modello non le vede mai (sennò il transfer crolla).
"""
from __future__ import annotations

import pandas as pd

from tmd.config import CityConfig
from tmd.labeling.window_labeler import label_windows_universal
from tmd.labeling.infra_align import add_infra_features

ALIGN_COLS = ["C_bus_route_align", "C_rail_align"]


def label_silver(df: pd.DataFrame, cfg: CityConfig, use_infra: bool = True,
                 gps: pd.DataFrame | None = None, spatial_index: dict | None = None,
                 shapes_path=None) -> pd.DataFrame:
    """
    Aggiunge silver_label / silver_weight a un parquet feature (window-level).

    use_infra=True richiede: gps (punti con session_id), spatial_index, shapes_path.
    Se mancano, ricade su window puro.
    """
    df = df.copy()
    if use_infra and gps is not None and spatial_index and shapes_path:
        df = add_infra_features(df, gps, spatial_index, shapes_path)

    labels, weights = label_windows_universal(df, cfg)

    df = df.drop(columns=ALIGN_COLS, errors="ignore")   # ANTI-LEAK: align solo al labeler
    df["silver_label"]  = labels
    df["silver_weight"] = weights
    return df
