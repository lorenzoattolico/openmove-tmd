"""
tmd/labeling/infra_align.py — feature di "infrastructure-following" a livello sessione.

Per il labeler ad alta precisione locale (window+infra). Distingue i motorizzati
guardando se la traiettoria SEGUE l'infrastruttura (non solo se le è vicina):
  - C_bus_route_align : miglior allineamento a UNA linea bus (frazione punti < 50 m
                        dalla polilinea della singola linea best). Bus ≈ 0.74, Car ≈ 0.04.
  - C_rail_align      : frazione punti sessione < 20 m da rotaia OSM (stretto).
                        Train ≈ 0.35, Car ≈ 0.00 (la prossimità a ~30m fa passare le auto).

NON trasferibile: usa le mappe LOCALI (GTFS shapes + OSM rail) della città. È il punto:
ogni deployment usa le sue mappe → migliora il labeler locale. Vedi docs/thesis.md (tensione
locale-vs-trasferibile) e thesis/results.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_R_M = 6_371_000.0


def _route_trees(shapes_path) -> list[BallTree]:
    sh = pd.read_csv(shapes_path)
    return [BallTree(np.radians(g[["shape_pt_lat", "shape_pt_lon"]].values), metric="haversine")
            for _, g in sh.groupby("shape_id")]


def add_infra_features(df_windows: pd.DataFrame, gps: pd.DataFrame, spatial_index: dict,
                       shapes_path, route_thr_m: float = 50.0, rail_thr_m: float = 20.0
                       ) -> pd.DataFrame:
    """Aggiunge C_bus_route_align e C_rail_align (session-level, propagate alle finestre)."""
    rt, rl = route_thr_m / EARTH_R_M, rail_thr_m / EARTH_R_M

    # rail: una query per tutti i punti, poi frazione per sessione
    rad = np.radians(gps[["latitude", "longitude"]].values)
    drail = spatial_index["osm_rail"]["balltree"].query(rad, k=1)[0].flatten()
    rail_by = gps.assign(_d=drail).groupby("session_id")["_d"].apply(lambda d: float((d < rl).mean()))

    # route: per sessione, miglior allineamento a UNA linea
    trees = _route_trees(shapes_path)
    route = {}
    for sid, g in gps.groupby("session_id"):
        if len(g) < 2:
            route[sid] = np.nan
            continue
        r = np.radians(g[["latitude", "longitude"]].values)
        route[sid] = max(((t.query(r, k=1)[0].flatten() < rt).mean() for t in trees), default=0.0)

    out = df_windows.copy()
    out["C_bus_route_align"] = out["session_id"].map(route)
    out["C_rail_align"] = out["session_id"].map(rail_by)
    return out
