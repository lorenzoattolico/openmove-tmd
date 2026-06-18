"""
tmd/features/pipeline.py — Lorenzo Attolico, OpenMove / UniTN, Maggio 2026

Orchestrator: adatta colonne sorgente → arrays standard, windowing, assembly.

SOURCE_SCHEMA definisce la mappatura colonne per ogni sorgente.
Aggiungere una nuova sorgente = aggiungere una entry al dict.
"""

from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd

from . import imu as feat_imu
from . import gps_kinematic as feat_gps_kin
from . import gtfs_spatial as feat_gtfs
from . import gps_structure as feat_gps_str

# ── Colonne attese dai moduli feature ─────────────────────────────────────────
# Ogni sorgente mappa i suoi nomi → nomi standard interni.
# None = campo non disponibile in quella sorgente.

SOURCE_SCHEMA = {
    'openmove': {
        'imu': {
            'acc_x': 'acc_x', 'acc_y': 'acc_y', 'acc_z': 'acc_z',
            'gyr_x': 'gyr_x', 'gyr_y': 'gyr_y', 'gyr_z': 'gyr_z',
            'lin_acc_x': None, 'lin_acc_y': None, 'lin_acc_z': None,
            'grav_x': None,    'grav_y': None,    'grav_z': None,
            'mag_x': None,     'mag_y': None,     'mag_z': None,
        },
        'gps': {
            'lat': 'latitude', 'lon': 'longitude', 'ts_ms': 'timestamp',
            'speed': 'speed', 'accuracy': 'accuracy', 'altitude': None,
        },
        'fs': 50.0,
    },
    'shl': {
        'imu': {
            'acc_x': 'ax', 'acc_y': 'ay', 'acc_z': 'az',
            'gyr_x': 'gx', 'gyr_y': 'gy', 'gyr_z': 'gz',
            'lin_acc_x': 'lin_ax', 'lin_acc_y': 'lin_ay', 'lin_acc_z': 'lin_az',
            'grav_x': 'grav_x',   'grav_y': 'grav_y',    'grav_z': 'grav_z',
            'mag_x': 'mx',        'mag_y': 'my',          'mag_z': 'mz',
        },
        'gps': {
            'lat': 'lat', 'lon': 'lon', 'ts_ms': 'ts',
            'speed': None, 'accuracy': 'accuracy', 'altitude': None,
        },
        'fs': 100.0,
    },
}

# ── Drop list feature ridondanti ───────────────────────────────────────────────
# Tre categorie (da feature_selection_v4.md):
#   redundant_b7: Spearman >= 0.85 sul train split
#   orientation:  non generalizzano in LOUO
#   override:     sostituiti da feature migliori

DROP_LIST: dict[str, str] = {
    'A_acc_mag_kurt':        'redundant_b7',
    'A_acc_mag_mean':        'redundant_b7',
    'A_acc_mag_p25':         'redundant_b7',
    'A_acc_mag_rms':         'redundant_b7',
    'A_acc_x_spec_entropy':  'redundant_b7',
    'A_acc_z_zcr':           'redundant_b7',
    'A_gyr_mag_iqr':         'redundant_b7',
    'A_gyr_mag_kurt':        'redundant_b7',
    'A_gyr_mag_rms':         'redundant_b7',
    'A_gyr_mag_skew':        'redundant_b7',
    'A_gyr_mag_spec_entropy':'redundant_b7',
    # A_gyr_mag_std: tenuto (commento originale) ✓
    'A_gyr_x_rms':           'redundant_b7',
    'A_gyr_x_spec_entropy':  'redundant_b7',
    'A_gyr_x_std':           'redundant_b7',
    'A_gyr_y_iqr':           'redundant_b7',
    'A_gyr_y_rms':           'redundant_b7',
    'A_gyr_y_std':           'redundant_b7',
    'A_gyr_z_iqr':           'redundant_b7',
    'A_gyr_z_kurt':          'redundant_b7',
    'A_gyr_z_rms':           'redundant_b7',
    'A_gyr_z_spec_entropy':  'redundant_b7',
    'A_gyr_z_std':           'redundant_b7',
    'A_jerk_kurt':           'redundant_b7',
    # A_jerk_mean: tenuto (formula corretta vettoriale) ✓
    'A_jerk_rms':            'redundant_b7',
    'A_lin_mag_iqr':         'redundant_b7',
    'A_lin_mag_kurt':        'redundant_b7',
    'A_lin_mag_psd_0_2':     'redundant_b7',
    # A_lin_mag_std: tenuto ✓
    'A_lin_mag_rms':         'redundant_b7',
    'A_grav_x_mean':         'orientation',
    'A_grav_y_mean':         'orientation',
    'A_grav_z_mean':         'orientation',
    'A_gyr_x_mean':          'orientation',
    'A_gyr_y_mean':          'orientation',
    'B_acc_mean':            'redundant_b7',
    'B_n_stops':             'redundant_b7',
    'B_speed_p50':           'redundant_b7',
    'B_speed_p95':           'redundant_b7',
    'B_dist_total_m':        'override',
    'A_acc_mag_zcr':         'uninformative_magnitude',
    'A_gyr_mag_zcr':         'uninformative_magnitude',
    'A_lin_mag_zcr':         'uninformative_magnitude',
    'A_jerk_zcr':            'uninformative_magnitude',
    'B_alt_mean':            'always_nan',
    'B_alt_std':             'always_nan',
    'B_alt_range':           'always_nan',
}


# ── Adattatori colonne ─────────────────────────────────────────────────────────

def _get_arr(df: pd.DataFrame, col_std: str,
             schema: dict, required: bool = True) -> np.ndarray | None:
    """Legge colonna da df usando il nome mappato dallo schema."""
    col_src = schema.get(col_std)
    if col_src is None:
        return None
    if col_src not in df.columns:
        return None
    vals = df[col_src].values.astype(np.float64)
    return vals


def _imu_arrays(df_imu: pd.DataFrame, schema: dict) \
        -> tuple[np.ndarray, np.ndarray,
                 np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Estrae (acc, gyr, lin_acc, grav, mag) dallo schema sorgente."""
    def _arr3(x, y, z):
        ax = _get_arr(df_imu, x, schema)
        ay = _get_arr(df_imu, y, schema)
        az = _get_arr(df_imu, z, schema)
        if ax is None or ay is None or az is None:
            return None
        return np.column_stack([ax, ay, az])

    acc     = _arr3('acc_x', 'acc_y', 'acc_z')
    gyr     = _arr3('gyr_x', 'gyr_y', 'gyr_z')
    lin_acc = _arr3('lin_acc_x', 'lin_acc_y', 'lin_acc_z')
    grav    = _arr3('grav_x', 'grav_y', 'grav_z')
    mag     = _arr3('mag_x', 'mag_y', 'mag_z')
    return acc, gyr, lin_acc, grav, mag


# ── Windowing ─────────────────────────────────────────────────────────────────

def _make_windows(ts_ms: np.ndarray, win_ms: int, step_ms: int) -> list[tuple[int, int]]:
    """Finestre sliding su ts_ms. Ritorna lista di (t0_ms, t1_ms)."""
    if len(ts_ms) == 0:
        return []
    t_start = ts_ms[0]
    t_end   = ts_ms[-1]
    windows = []
    t0 = t_start
    while t0 + win_ms <= t_end + step_ms:
        windows.append((int(t0), int(t0 + win_ms)))
        t0 += step_ms
    return windows


def _imu_indices_for_window(imu_ts: np.ndarray, t0: int, t1: int,
                              fs: float, win_s: float) -> np.ndarray:
    """
    Indici IMU nella finestra [t0, t1).
    Usa ricerca temporale: campioni con timestamp nel range.
    """
    mask = (imu_ts >= t0) & (imu_ts < t1)
    return np.where(mask)[0]




def _extract_session_arrays(
    df_imu_s: pd.DataFrame,
    schema:   dict,
) -> tuple:
    """
    Pre-estrae array numpy completi a livello di sessione.
    Chiamata UNA VOLTA per sessione invece di _imu_arrays() per ogni finestra.
    Evita 720 accessi colonna pandas per sessione da 120 finestre.
    Ritorna (acc, gyr, lin_acc, grav, mag) — None per campi non disponibili.
    """
    def _col(std_name):
        src_col = schema.get(std_name)
        if src_col is None or src_col not in df_imu_s.columns:
            return None
        return df_imu_s[src_col].values.astype(np.float64)

    ax  = _col('acc_x');    ay  = _col('acc_y');    az  = _col('acc_z')
    gx  = _col('gyr_x');    gy  = _col('gyr_y');    gz  = _col('gyr_z')
    lx  = _col('lin_acc_x'); ly = _col('lin_acc_y'); lz = _col('lin_acc_z')
    gvx = _col('grav_x');   gvy = _col('grav_y');   gvz = _col('grav_z')
    mx  = _col('mag_x');    my  = _col('mag_y');    mz  = _col('mag_z')

    acc     = np.column_stack([ax, ay, az])   if ax  is not None else None
    gyr     = np.column_stack([gx, gy, gz])   if gx  is not None else None
    lin_acc = np.column_stack([lx, ly, lz])   if lx  is not None else None
    grav    = np.column_stack([gvx, gvy, gvz]) if gvx is not None else None
    mag     = np.column_stack([mx, my, mz])   if mx  is not None else None

    return acc, gyr, lin_acc, grav, mag

# ── Entry point principale ─────────────────────────────────────────────────────

def add_rolling_context_features(df: pd.DataFrame, windows: int = 5) -> pd.DataFrame:
    """
    Aggiunge feature di contesto temporale calcolate su finestre adiacenti nella stessa sessione.

    B_stop_rate_5win: rolling mean di B_stop_frac sulle ultime `windows` finestre (default 5 = 10 min).
    Utile per Bus: una finestra bus ad alta velocità senza stop ha stop_rate_5win alto
    se le finestre precedenti nella stessa corsa si sono fermate.

    Chiamare DOPO pd.read_parquet e PRIMA di get_feature_cols.
    """
    if "B_stop_frac" not in df.columns or "session_id" not in df.columns:
        return df
    df = df.copy()
    df["B_stop_rate_5win"] = (
        df.sort_values(["session_id", "ts_start"])
        .groupby("session_id")["B_stop_frac"]
        .transform(lambda x: x.rolling(windows, min_periods=1).mean())
    )
    return df


def extract_session(
    df_imu:      pd.DataFrame,
    df_gps:      pd.DataFrame,
    t0_ms:       int,
    t1_ms:       int,
    idx_spatial: dict,
    source:      Literal['openmove', 'shl'] = 'openmove',
    fs:          float | None = None,
    win_s:       float = 120.0,
    step_s:      float = 60.0,
    min_gps_pts: int   = 3,
    gps_frac_max: float = 1.5,
    groups:      list[str] = ['A', 'B', 'C', 'D'],
    meta:        dict | None = None,
    apply_drop_list: bool = True,
    gap_thresh_s: float = 30.0,
) -> pd.DataFrame:
    """
    Estrae feature per tutte le finestre di una sessione.

    Parametri
    ---------
    df_imu, df_gps : DataFrame con colonne secondo SOURCE_SCHEMA[source]
    t0_ms, t1_ms   : boundary sessione in epoch ms
    idx_spatial    : dict layer → {'balltree': BallTree} (può essere {})
    groups         : subset di gruppi da calcolare ['A','B','C','D']
    meta           : dict di colonne extra da aggiungere a ogni riga
                     (es. {'session_id': '...', 'userId': '...'})
    """
    schema  = SOURCE_SCHEMA[source]
    fs      = fs if fs is not None else schema['fs']
    win_ms  = int(win_s * 1000)
    step_ms = int(step_s * 1000)

    imu_schema = schema['imu']
    gps_schema = schema['gps']

    # filtra finestra temporale sessione
    imu_ts_col = imu_schema.get('ts_ms', 'timestamp') or 'timestamp'
    # per shl il campo timestamp dell'IMU non è in schema['imu'] ma viene passato
    # come colonna 'ts' nel DataFrame — gestiamo entrambi i casi
    if 'timestamp' in df_imu.columns:
        imu_ts = df_imu['timestamp'].values.astype(np.int64)
    elif 'ts' in df_imu.columns:
        imu_ts = df_imu['ts'].values.astype(np.int64)
    else:
        return pd.DataFrame()

    gps_ts_col = gps_schema['ts_ms']
    if gps_ts_col not in df_gps.columns:
        return pd.DataFrame()
    gps_ts = df_gps[gps_ts_col].values.astype(np.int64)

    # subset sessione
    imu_mask = (imu_ts >= t0_ms) & (imu_ts < t1_ms)
    gps_mask = (gps_ts >= t0_ms) & (gps_ts < t1_ms)
    from tmd.features.quality import normalize_and_filter_imu

    df_imu_s = df_imu[imu_mask].reset_index(drop=True)
    df_gps_s = df_gps[gps_mask].reset_index(drop=True)

    # normalizzazione per-sessione: iOS=g, Android=m/s², utente può cambiare device
    # normalize_and_filter_imu calcola acc_mag una sola volta (vs due .apply() separati)
    if source == "openmove":
        df_imu_s = normalize_and_filter_imu(df_imu_s)

    # imu_ts_s sempre assegnata dopo eventuale filtering
    imu_ts_s = df_imu_s["timestamp"].values.astype(np.int64) \
            if "timestamp" in df_imu_s.columns \
            else df_imu_s["ts"].values.astype(np.int64)
    gps_ts_s = gps_ts[gps_mask]

    # pre-calcola distanze GTFS per tutta la sessione (ottimizzazione)
    if 'C' in groups and len(df_gps_s) > 0 and idx_spatial:
        lat_s = df_gps_s[gps_schema['lat']].values
        lon_s = df_gps_s[gps_schema['lon']].values
        gtfs_dists = feat_gtfs.precompute_session(lat_s, lon_s, idx_spatial)
    else:
        gtfs_dists = {}

    # ── Ordina timestamp una volta per sessione → abilita searchsorted O(log n) ──
    # IMU: già ordinato da MongoDB (indice userId_1_timestamp_1); sort stabile per sicurezza
    # GPS: load_gps non garantisce ordinamento → sort esplicito necessario
    if len(imu_ts_s) > 1 and not bool(np.all(imu_ts_s[:-1] <= imu_ts_s[1:])):
        imu_order = np.argsort(imu_ts_s, kind='stable')
        df_imu_s  = df_imu_s.iloc[imu_order].reset_index(drop=True)
        imu_ts_s  = imu_ts_s[imu_order]

    if len(gps_ts_s) > 1 and not bool(np.all(gps_ts_s[:-1] <= gps_ts_s[1:])):
        gps_order = np.argsort(gps_ts_s, kind='stable')
        df_gps_s  = df_gps_s.iloc[gps_order].reset_index(drop=True)
        gps_ts_s  = gps_ts_s[gps_order]
        # ricalcola distanze GTFS con ordine corretto
        if gtfs_dists and len(df_gps_s) > 0:
            lat_s = df_gps_s[gps_schema['lat']].values
            lon_s = df_gps_s[gps_schema['lon']].values
            gtfs_dists = feat_gtfs.precompute_session(lat_s, lon_s, idx_spatial)

    _gps_empty = df_gps_s.iloc[0:0]   # DataFrame vuoto con stesse colonne, senza riallocare

    # ── Pre-estrazione array numpy (una volta per sessione) ───────────────────
    # Evita _imu_arrays() per ogni finestra (720 pandas col-access su 120 finestre).
    # acc_full[imu_lo:imu_hi] è una slice numpy O(1) vs DataFrame column access.
    if 'A' in groups and len(df_imu_s) > 0:
        acc_full, gyr_full, lin_full, grav_full, mag_full = _extract_session_arrays(
            df_imu_s, imu_schema)
    else:
        acc_full = gyr_full = lin_full = grav_full = mag_full = None

    windows = _make_windows(imu_ts_s, win_ms, step_ms)
    records = []

    for t0_w, t1_w in windows:
        # ── slicing O(log n) via searchsorted su array già ordinati ──────────
        imu_lo   = int(np.searchsorted(imu_ts_s, t0_w, side='left'))
        imu_hi   = int(np.searchsorted(imu_ts_s, t1_w, side='right'))
        df_imu_w = df_imu_s.iloc[imu_lo:imu_hi]
        n_imu_w  = imu_hi - imu_lo   # conta senza .sum() su boolean mask

        gps_lo    = int(np.searchsorted(gps_ts_s, t0_w, side='left'))
        gps_hi    = int(np.searchsorted(gps_ts_s, t1_w, side='right'))
        df_gps_w  = df_gps_s.iloc[gps_lo:gps_hi]
        gps_w_idx = np.arange(gps_lo, gps_hi, dtype=np.intp)

        # GPS fraction — safety net
        gps_frac = len(df_gps_w) / win_s
        if gps_frac > gps_frac_max:
            gps_w_idx = np.array([], dtype=np.intp)
            df_gps_w  = _gps_empty

        feats: dict = {}

        # Gruppo A — IMU
        if 'A' in groups:
            if acc_full is not None and gyr_full is not None and n_imu_w > 0:
                # Slice numpy diretto — O(1) vs accesso colonne DataFrame per finestra
                acc_w  = acc_full[imu_lo:imu_hi]
                gyr_w  = gyr_full[imu_lo:imu_hi]
                lin_w  = lin_full[imu_lo:imu_hi]  if lin_full  is not None else None
                grav_w = grav_full[imu_lo:imu_hi] if grav_full is not None else None
                mag_w  = mag_full[imu_lo:imu_hi]  if mag_full  is not None else None
                feats.update(feat_imu.compute(acc_w, gyr_w, lin_w, grav_w, mag_w, fs=fs))
            else:
                feats.update(feat_imu._empty_features())

        # Gruppo B — GPS cinematica
        if 'B' in groups:
            if len(df_gps_w) >= 3:
                lat  = df_gps_w[gps_schema['lat']].values.astype(np.float64)
                lon  = df_gps_w[gps_schema['lon']].values.astype(np.float64)
                ts   = df_gps_w[gps_schema['ts_ms']].values.astype(np.int64)
                spd  = df_gps_w[gps_schema['speed']].values.astype(np.float64) \
                       if gps_schema.get('speed') and gps_schema['speed'] in df_gps_w.columns \
                       else None
                alt  = df_gps_w[gps_schema['altitude']].values.astype(np.float64) \
                       if gps_schema.get('altitude') and gps_schema['altitude'] in df_gps_w.columns \
                       else None
                feats.update(feat_gps_kin.compute(lat, lon, ts, speed=spd, altitude=alt))
            else:
                feats.update({'B_n_gps': len(df_gps_w),
                              **{k: np.nan for k in feat_gps_kin._NAN_KEYS}})

        # Gruppo C — GTFS spaziale
        if 'C' in groups:
            feats.update(feat_gtfs.compute(gps_w_idx, gtfs_dists))

        # Gruppo D — GPS struttura
        if 'D' in groups:
            if len(df_gps_w) > 0:
                ts_d  = df_gps_w[gps_schema['ts_ms']].values.astype(np.int64)
                acc_d = df_gps_w[gps_schema['accuracy']].values.astype(np.float64) \
                        if gps_schema.get('accuracy') and gps_schema['accuracy'] in df_gps_w.columns \
                        else None
                feats.update(feat_gps_str.compute(ts_d, acc_d, win_s=win_s,gap_thresh_s=gap_thresh_s))
            else:
                feats.update(feat_gps_str.compute(np.array([]), None, win_s=win_s))

        # rimuovi feature ridondanti (saltabile: --all-features per la variable-selection su OpenMove)
        if apply_drop_list:
            for k in DROP_LIST:
                feats.pop(k, None)

        feats['ts_start'] = int(t0_w)
        feats['ts_end']   = int(t1_w)
        feats['gps_frac'] = gps_frac
        feats['n_imu']    = n_imu_w

        if meta:
            feats.update(meta)

        records.append(feats)

    return pd.DataFrame(records)
