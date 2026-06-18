"""
tmd/features/gtfs_spatial.py — Lorenzo Attolico, OpenMove / UniTN, Maggio 2026

Gruppo C: feature spaziali GTFS via BallTree.
Invariato da SHL v3/v4 nella logica; interfaccia standardizzata.
"""

from __future__ import annotations
import numpy as np


EARTH_R_M = 6_371_000.0

LAYER_NAMES = [
    'bus_stops', 'rail_stations', 'subway_stations',
    'osm_rail', 'osm_subway', 'osm_motorway', 'osm_cycleway',
]

PROP_THRESHOLDS = {
    'bus_stops': 50, 'rail_stations': 500, 'subway_stations': 200,
    'osm_rail': 30, 'osm_subway': 200, 'osm_motorway': 50, 'osm_cycleway': 10,
}

# selezione B7 Spearman 0.85 (da 35 → 15 feature, SHL v3)
_C_KEEP = {
    'bus_stops':       [('prop', lambda d, t: float((d < t).mean())),
                        ('mean', lambda d, t: float(np.mean(d))),
                        ('flag', lambda d, t: float(np.min(d) < t))],
    'rail_stations':   [('min',  lambda d, t: float(np.min(d))),
                        ('flag', lambda d, t: float(np.min(d) < t))],
    'subway_stations': [('mean', lambda d, t: float(np.mean(d))),
                        ('flag', lambda d, t: float(np.min(d) < t))],
    'osm_rail':        [('prop', lambda d, t: float((d < t).mean())),
                        ('p10',  lambda d, t: float(np.percentile(d, 10)))],
    'osm_subway':      [('prop', lambda d, t: float((d < t).mean()))],
    'osm_motorway':    [('flag', lambda d, t: float(np.min(d) < t)),
                        ('min',  lambda d, t: float(np.min(d)))],
    'osm_cycleway':    [('flag', lambda d, t: float(np.min(d) < t)),
                        ('mean', lambda d, t: float(np.mean(d))),
                        ('min',  lambda d, t: float(np.min(d)))],
}

_C_NAN = {f'C_{layer}_{suf}': np.nan
          for layer, specs in _C_KEEP.items()
          for suf, _ in specs}


def precompute_session(lat: np.ndarray, lon: np.ndarray,
                       idx_spatial: dict) -> dict[str, np.ndarray]:
    """
    Pre-calcola distanze BallTree per tutti i punti GPS della sessione.
    Chiamata una volta per sessione, non per finestra.
    idx_spatial: dict layer_name → {'balltree': BallTree}
    """
    if len(lat) == 0:
        return {name: np.array([]) for name in LAYER_NAMES}
    coords_rad = np.radians(np.column_stack([lat, lon]))
    result = {}
    for name in LAYER_NAMES:
        if name not in idx_spatial:
            result[name] = np.full(len(lat), np.nan)
        else:
            dist_rad, _ = idx_spatial[name]['balltree'].query(coords_rad, k=1)
            result[name] = dist_rad.flatten() * EARTH_R_M
    return result


def compute(win_idx: np.ndarray, gtfs_dists: dict) -> dict:
    """
    Feature GTFS per una finestra.
    win_idx: indici dei punti GPS della finestra rispetto alla sessione.
    gtfs_dists: output di precompute_session.
    """
    feats = dict(_C_NAN)
    if len(win_idx) == 0:
        return feats
    for layer, specs in _C_KEEP.items():
        dist_arr = gtfs_dists.get(layer, np.array([]))
        if len(dist_arr) == 0:
            continue
        dw     = dist_arr[win_idx]
        thresh = PROP_THRESHOLDS[layer]
        for suf, fn in specs:
            try:
                feats[f'C_{layer}_{suf}'] = fn(dw, thresh)
            except Exception:
                feats[f'C_{layer}_{suf}'] = np.nan
    return feats
