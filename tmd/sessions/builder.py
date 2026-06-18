"""
tmd.sessions.builder — sessionizzazione R1.

Una sessione = blocco continuo di dati sull'UNIONE GPS∪IMU: si taglia SOLO dove
mancano ENTRAMBI i sensori per > gap_s. Sostituisce il vecchio session_builder
GPS-primario ("spezza il GPS, poi attacca l'IMU orfano"), che spezzava un viaggio
a ogni dropout GPS e generava sessioni 'imu_only' fantasma.

Stesse 9 colonne di prima → a valle lo schema non cambia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tmd.config import CityConfig

COLUMNS = ["session_id", "userId", "t0_ms", "t1_ms", "dur_s",
           "n_gps", "n_imu", "imu_hz", "type"]


def _split_by_gap(ts_sorted: np.ndarray, gap_s: float) -> list[tuple[int, int]]:
    """Indici (start, end) di segmenti contigui separati da gap > gap_s secondi."""
    if len(ts_sorted) == 0:
        return []
    gaps   = np.diff(ts_sorted) / 1000.0
    splits = np.where(gaps > gap_s)[0] + 1
    starts = np.concatenate([[0], splits])
    ends   = np.concatenate([splits - 1, [len(ts_sorted) - 1]])
    return list(zip(starts.tolist(), ends.tolist()))


def _estimate_imu_hz(ts_in_session: np.ndarray, default: float = 50.0) -> float:
    """Frequenza IMU dalla mediana degli intervalli (robusta a gap), clamp [10,250]."""
    if len(ts_in_session) < 2:
        return default
    isi_ms = np.diff(np.sort(ts_in_session))
    isi_ms = isi_ms[isi_ms < 1000]          # esclude le pause (gap > 1s)
    if len(isi_ms) == 0:
        return default
    median_isi = float(np.median(isi_ms))
    if median_isi <= 0:
        return default
    return float(np.clip(1000.0 / median_isi, 10.0, 250.0))


def _session_id(uid: str, t0_ms: int) -> str:
    return f"{uid[:8]}_{t0_ms}"


def build_sessions_for_user(uid: str, ts_gps: np.ndarray, ts_imu: np.ndarray,
                            gap_s: float, min_s: float) -> list[dict]:
    """
    R1: spezza l'UNIONE GPS∪IMU sui gap. Una sessione = blocco continuo di dati
    (qualsiasi sensore); i confini sono del sensore presente → un dropout GPS a
    metà NON spezza più il viaggio.
    """
    ts_gps = np.asarray(ts_gps, dtype=np.float64)
    ts_imu = np.asarray(ts_imu, dtype=np.float64)
    if len(ts_gps) == 0 and len(ts_imu) == 0:
        return []
    ts_all = np.union1d(ts_gps, ts_imu)          # ordinato + unico = la timeline

    sessions: list[dict] = []
    for s, e in _split_by_gap(ts_all, gap_s):
        t0, t1 = int(ts_all[s]), int(ts_all[e])
        dur_s  = (t1 - t0) / 1000.0
        if dur_s < min_s:
            continue
        n_gps  = int(((ts_gps >= t0) & (ts_gps <= t1)).sum()) if len(ts_gps) else 0
        imu_in = ts_imu[(ts_imu >= t0) & (ts_imu <= t1)] if len(ts_imu) else np.array([])
        n_imu  = int(len(imu_in))
        sess_type = ("tracking" if (n_gps and n_imu)
                     else "imu_only" if n_gps == 0 else "gps_only")
        sessions.append({
            "session_id": _session_id(uid, t0), "userId": uid,
            "t0_ms": t0, "t1_ms": t1, "dur_s": dur_s,
            "n_gps": n_gps, "n_imu": n_imu,
            "imu_hz": _estimate_imu_hz(imu_in), "type": sess_type,
        })
    return sessions


def build_sessions(df_gps: pd.DataFrame, df_imu: pd.DataFrame,
                   cfg: CityConfig) -> pd.DataFrame:
    """Wrapper su tutti gli utenti. Legge gap_s/min_duration_s da cfg.session."""
    gap_s = cfg.session["gap_s"]
    min_s = cfg.session["min_duration_s"]
    users = sorted(
        set(df_gps["userId"].unique() if not df_gps.empty else []) |
        set(df_imu["userId"].unique() if not df_imu.empty else [])
    )
    rows: list[dict] = []
    for uid in users:
        ts_gps = (df_gps.loc[df_gps["userId"] == uid, "timestamp"].to_numpy(np.float64)
                  if not df_gps.empty else np.array([]))
        ts_imu = (df_imu.loc[df_imu["userId"] == uid, "timestamp"].to_numpy(np.float64)
                  if not df_imu.empty else np.array([]))
        rows += build_sessions_for_user(uid, ts_gps, ts_imu, gap_s, min_s)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows).sort_values(["userId", "t0_ms"]).reset_index(drop=True)


def filter_sessions(df: pd.DataFrame, remove_gps_only: bool = True) -> pd.DataFrame:
    """Rimuove le sessioni non processabili (gps_only = senza IMU)."""
    if remove_gps_only and not df.empty:
        df = df[df["type"] != "gps_only"]
    return df.reset_index(drop=True)
