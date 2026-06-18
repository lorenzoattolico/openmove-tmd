"""Labeler gerarchico a livello finestra (120s) su features_{city}.parquet.

Alta precisione > copertura: ABSTAIN sui casi ambigui, il modello apprende dai
casi certi. Cascata Still -> Walk -> Train -> Bus -> Car -> ABSTAIN.

Segnale primario = feature B (GPS, universali); C (infrastruttura) come booster
opzionale; A (IMU) solo come fallback GPS-absent. Soglie fisiche, non calibrate
sul singolo dataset.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from tmd.config import CityConfig

# Pesi non usati: ogni finestra silver pesa 1.0 (dict tenuto per la firma).
_WEIGHTS_DEFAULT: dict[str, float] = {
    "Still": 1.0,
    "Walk":  1.0,
    "Train": 1.0,
    "Car":   1.0,
    "Bus":   1.0,
}


def _weights(cfg) -> dict[str, float]:
    w = cfg.window_labeler_weights
    return {k: float(w.get(k, _WEIGHTS_DEFAULT.get(k, 1.0)))
            for k in _WEIGHTS_DEFAULT}


def _t(cfg, key: str, default: float) -> float:
    return float((cfg.window_labeler or {}).get(key, default))


def _col(df: pd.DataFrame, name: str, fill: float) -> np.ndarray:
    """Legge la colonna dal df, riempie i NaN con fill."""
    if name in df.columns:
        return df[name].to_numpy(dtype=float, na_value=fill)
    return np.full(len(df), fill, dtype=float)


def label_windows_universal(
    df: pd.DataFrame,
    city_cfg,
) -> tuple[list[str | None], list[float]]:
    """Labeler universale, identico su ogni contesto (Trento, SHL, ...).

    df       : features parquet (almeno B_speed_mean, B_speed_max, B_stop_frac, B_n_gps).
    city_cfg : CityConfig; legge window_labeler.* per gli override soglie.
    Ritorna (labels, weights) paralleli a df, None dove ABSTAIN.
    """
    if not isinstance(city_cfg, CityConfig):
        raise TypeError(
            f"city_cfg deve essere una CityConfig, non {type(city_cfg).__name__} "
            "(usa CityConfig.from_yaml).")
    c = city_cfg
    n = len(df)

    # Feature arrays: GPS cinematiche (primarie)
    spd_mean  = _col(df, "B_speed_mean",      0.0)
    spd_max   = _col(df, "B_speed_max",       0.0)
    stop_frac = _col(df, "B_stop_frac",       0.5)   # 0.5 = neutro
    path_eff  = _col(df, "B_path_efficiency", 0.5)   # 0.5 = neutro
    n_gps     = _col(df, "B_n_gps",           0.0)

    # Infrastruttura (opzionali; assente -> 0.0 = ABSTAIN conservativo)
    rail_prop = _col(df, "C_osm_rail_prop",    0.0)
    bus_prop  = _col(df, "C_bus_stops_prop",   0.0)
    # Allineamento all'infrastruttura: assente -> 1.0 (nessun vincolo, modalità window pura).
    route_align = _col(df, "C_bus_route_align", 1.0)
    rail_align  = _col(df, "C_rail_align",      1.0)

    # IMU (solo fallback GPS-absent)
    lin_mag   = _col(df, "A_lin_mag_mean",     0.0)
    psd_2_4   = _col(df, "A_acc_x_psd_2_4",   0.0)

    gps_present = n_gps > 0
    gps_absent  = ~gps_present

    # 1. STILL — GPS-present: lento e quasi fermo. GPS-absent: IMU sotto il noise-floor MEMS.
    still_spd_strict = _t(c, "still_spd_strict", 0.20)  # m/s
    still_spd_max    = _t(c, "still_spd_max",    0.50)  # m/s, con conferma stop_frac
    still_stop_min   = _t(c, "still_stop_min",   0.85)
    still_lin_nogps  = _t(c, "still_lin_nogps",  0.015) # noise-floor MEMS (GPS-absent)

    mask_still = (
        (gps_present & (
            (spd_mean < still_spd_strict) |
            ((spd_mean < still_spd_max) & (stop_frac > still_stop_min))
        )) |
        (gps_absent & (lin_mag < still_lin_nogps))
    )

    # 2. WALK (GPS-present) — B_speed_max = ceiling della locomozione umana (Bohannon 1997).
    walk_spd_lo   = _t(c, "walk_spd_lo",    0.5)   # m/s
    walk_spd_hi   = _t(c, "walk_spd_hi",    2.5)   # m/s, walking comodo
    walk_max_spd  = _t(c, "walk_max_spd",   5.0)   # m/s, ceiling locomozione
    walk_stop_max = _t(c, "walk_stop_max",  0.50)  # max stop_frac (midpoint neutro)

    mask_walk_gps = (
        gps_present & ~mask_still &
        (spd_mean  > walk_spd_lo)  &
        (spd_mean  < walk_spd_hi)  &
        (spd_max   < walk_max_spd) &
        (stop_frac < walk_stop_max)
    )

    # WALK GPS-absent: IMU + banda passi PSD 2-4 Hz.
    nogps_lin = _t(c, "walk_nogps_lin_min", 0.35)
    nogps_psd = _t(c, "walk_nogps_psd_min", 0.25)
    nogps_w   = _t(c, "walk_nogps_weight",  0.89)

    mask_walk_nogps = (
        gps_absent & ~mask_still &
        (lin_mag > nogps_lin) &
        (psd_2_4 > nogps_psd)
    )

    # 3. TRAIN — copertura binario OSM + velocità; senza spatial index (rail_prop=0) -> ABSTAIN.
    train_rail = _t(c, "train_rail_min",  0.35)   # C_osm_rail_prop
    train_spd  = _t(c, "train_spd_min",   8.0)    # m/s (29 km/h)
    train_rail_align = _t(c, "train_rail_align_min", 0.15)  # segue la rotaia

    mask_train = (
        gps_present & ~mask_still & ~mask_walk_gps &
        (rail_prop  > train_rail) &
        (spd_mean   > train_spd)  &
        (rail_align > train_rail_align)
    )

    # 4. BUS — richiede fermate (C_bus_stops_prop) + stop-and-go; senza index -> ABSTAIN.
    bus_prop_min = _t(c, "bus_prop_min",   0.20)   # C_bus_stops_prop
    bus_spd_lo   = _t(c, "bus_spd_lo",    1.5)    # m/s
    bus_spd_hi   = _t(c, "bus_spd_hi",   14.0)    # m/s (esclude intercity veloci)
    bus_stop_min = _t(c, "bus_stop_min",  0.10)   # stop_frac
    bus_route_align = _t(c, "bus_route_align_min", 0.40)  # segue una linea

    mask_bus = (
        gps_present & ~mask_still & ~mask_walk_gps & ~mask_train &
        (bus_prop    > bus_prop_min) &
        (spd_mean    > bus_spd_lo)   &
        (spd_mean    < bus_spd_hi)   &
        (stop_frac   > bus_stop_min) &
        (route_align > bus_route_align)
    )

    # 5. CAR — veloce, non su binario, traiettoria efficiente.
    car_spd_lo   = _t(c, "car_spd_lo",    5.0)    # m/s (18 km/h)
    car_rail_max = _t(c, "car_rail_max",   0.15)   # C_osm_rail_prop
    car_path_min = _t(c, "car_path_min",   0.40)   # path_efficiency

    mask_car = (
        gps_present & ~mask_still & ~mask_walk_gps & ~mask_train & ~mask_bus &
        (spd_mean  > car_spd_lo)   &
        (rail_prop < car_rail_max) &
        (path_eff  > car_path_min)
    )

    # Output
    W = _weights(c)
    out_labels  = np.full(n, None, dtype=object)
    out_weights = np.zeros(n, dtype=float)

    for mask, lbl, w in (
        (mask_still,      "Still", W["Still"]),
        (mask_walk_gps,   "Walk",  W["Walk"]),
        (mask_walk_nogps, "Walk",  nogps_w),
        (mask_train,      "Train", W["Train"]),
        (mask_bus,        "Bus",   W["Bus"]),
        (mask_car,        "Car",   W["Car"]),
    ):
        out_labels[mask]  = lbl
        out_weights[mask] = w

    return out_labels.tolist(), out_weights.tolist()
