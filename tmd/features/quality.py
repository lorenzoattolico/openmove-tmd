"""
tmd.features.quality — pulizia/normalizzazione a livello di punto.
Port da tmd/data/quality_filter.py; rimosse le funzioni legacy non usate
(normalize_imu_units, filter_imu_spikes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tmd.config import CityConfig


def filter_gps(df: pd.DataFrame, cfg: CityConfig) -> tuple[pd.DataFrame, dict]:
    """Rimuove punti GPS con accuracy > soglia (la sessione resta)."""
    if df.empty:
        return df, {"n_in": 0, "n_out": 0, "n_removed": 0}
    n_in = len(df)
    acc_max = cfg.quality["gps_accuracy_max_m"]
    mask_bad = df["accuracy"].notna() & (df["accuracy"] > acc_max)
    df_clean = df[~mask_bad].reset_index(drop=True)
    return df_clean, {"n_in": n_in, "n_out": len(df_clean), "n_removed": int(mask_bad.sum())}


def filter_imu(df: pd.DataFrame, cfg: CityConfig) -> tuple[pd.DataFrame, dict]:
    """Rimuove campioni IMU con timestamp fuori dall'epoca valida (uptime pre-fix)."""
    if df.empty:
        return df, {"n_in": 0, "n_out": 0, "n_removed": 0}
    n_in = len(df)
    ts_min = cfg.timestamp_epoch_range["min_ms"]
    ts_max = cfg.timestamp_epoch_range["max_ms"]
    mask_bad = (df["timestamp"] < ts_min) | (df["timestamp"] > ts_max)
    df_clean = df[~mask_bad].reset_index(drop=True)
    return df_clean, {"n_in": n_in, "n_out": len(df_clean), "n_removed": int(mask_bad.sum())}


def apply_all(df_gps: pd.DataFrame, df_imu: pd.DataFrame,
              cfg: CityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_gps_clean, _ = filter_gps(df_gps, cfg)
    df_imu_clean, _ = filter_imu(df_imu, cfg)
    return df_gps_clean, df_imu_clean


filter_all = apply_all   # alias storico


# ── Normalizzazione IMU (usata da pipeline.extract_session) ───────────────────

G_UNIT_THRESHOLD = 5.0    # acc_mag mediana > soglia → assume m/s², converti in g
ACC_SPIKE_G      = 4.0    # soglia spike accelerometro in g


def normalize_and_filter_imu(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Normalizza unità IMU (m/s²→g se mediana>5) e rimuove spike (acc_mag>4g),
    in una sola passata (acc_mag calcolata una volta).
    """
    if df.empty:
        return df
    ax = df["acc_x"].values.astype(np.float64)
    ay = df["acc_y"].values.astype(np.float64)
    az = df["acc_z"].values.astype(np.float64)
    acc_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    if float(np.median(acc_mag)) > G_UNIT_THRESHOLD:
        ax = ax / 9.80665; ay = ay / 9.80665; az = az / 9.80665
        acc_mag = acc_mag / 9.80665
        df = df.copy()
        df["acc_x"] = ax; df["acc_y"] = ay; df["acc_z"] = az

    mask_ok = acc_mag <= ACC_SPIKE_G
    if int((~mask_ok).sum()):
        if verbose:
            print(f"Filter IMU spikes: rimossi {int((~mask_ok).sum()):,} campioni > {ACC_SPIKE_G}g")
        df = df[mask_ok].reset_index(drop=True)
    return df
