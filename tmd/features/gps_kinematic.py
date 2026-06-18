"""
tmd/features/gps_kinematic.py — Lorenzo Attolico, OpenMove / UniTN, Maggio 2026

Gruppo B: feature cinematiche GPS.
Input: array lat, lon, ts_ms (obbligatori) + speed, altitude opzionali.
"""

from __future__ import annotations
import numpy as np


EARTH_R_M        = 6_371_000.0
GPS_SPEED_CAP    = 60.0
SPEED_CHANGE_THR = 2.0     # m/s — soglia VCR (Zheng 2008)
STOP_SPEED_THR   = 1.0     # m/s
STOP_MIN_DUR_S   = 3.0     # s

_NAN_KEYS = [
    'B_speed_mean', 'B_speed_std', 'B_speed_max',
    'B_speed_p25',  'B_speed_p50', 'B_speed_p95',
    'B_dist_total_m', 'B_acc_mean', 'B_acc_std',
    'B_n_stops', 'B_stop_frac', 'B_bearing_change_mean',
    'B_speed_change_rate', 'B_path_efficiency',
    'B_stop_regularity', 'B_dwell_mean',
    'B_speed_trend',
    'B_alt_mean', 'B_alt_std', 'B_alt_range',
    'B_n_valid_pairs',
]


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _detect_stops(sv: np.ndarray, ts_mid: np.ndarray) -> list[tuple[float, float]]:
    stops, in_stop, t_start = [], False, None
    for s, t in zip(sv, ts_mid):
        if s < STOP_SPEED_THR:
            if not in_stop:
                in_stop, t_start = True, t
        elif in_stop:
            if (t - t_start) >= STOP_MIN_DUR_S:
                stops.append((t_start, t - t_start))
            in_stop = False
    if in_stop and t_start is not None:
        dur = ts_mid[-1] - t_start
        if dur >= STOP_MIN_DUR_S:
            stops.append((t_start, dur))
    return stops


def compute(
    lat:      np.ndarray,            # gradi decimali WGS84
    lon:      np.ndarray,
    ts_ms:    np.ndarray,            # timestamp epoch ms
    speed:    np.ndarray | None = None,   # m/s dal chip GPS (opzionale)
    altitude: np.ndarray | None = None,   # metri (opzionale)
) -> dict:
    """
    Feature cinematiche GPS.
    speed dal chip GPS usata se disponibile per statistiche veloc; altrimenti
    calcolata da differenze haversine (come in SHL).
    altitude opzionale: B_alt_* = NaN se assente.
    """
    feats = {'B_n_gps': len(lat), **{k: np.nan for k in _NAN_KEYS}}
    if len(lat) < 3:
        return feats

    order = np.argsort(ts_ms)
    lat, lon, ts_ms = lat[order], lon[order], ts_ms[order]
    ts_s = ts_ms / 1000.0

    dt   = np.diff(ts_s)
    dist = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])

    # Coppie con gap <= 30s (gap_thresh_s da Test 1A): proxy qualità GPS per stima velocità
    feats['B_n_valid_pairs'] = int((dt <= 30.0).sum())

    # speed per statistiche cinematiche: chip se disponibile, else haversine
    if speed is not None and len(speed) == len(lat):
        speed = speed[order]
        sv_chip = np.where(speed >= 0, speed, np.nan)
        sv_chip = np.where(sv_chip > GPS_SPEED_CAP, np.nan, sv_chip)
        # usa haversine solo dove chip non disponibile
        with np.errstate(divide='ignore', invalid='ignore'):
            sv_hav = np.where((dt > 0) & (dt < 60), dist / dt, np.nan)
        sv_hav  = np.where(sv_hav > GPS_SPEED_CAP, np.nan, sv_hav)
        # mid-point per chip → media punti adiacenti
        sv_mid  = (sv_chip[:-1] + sv_chip[1:]) / 2.0
        sv_full = np.where(~np.isnan(sv_mid), sv_mid, sv_hav)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            sv_full = np.where((dt > 0) & (dt < 60), dist / dt, np.nan)
        sv_full = np.where(sv_full > GPS_SPEED_CAP, np.nan, sv_full)

    valid = ~np.isnan(sv_full)
    if valid.sum() < 2:
        return feats

    sv   = sv_full[valid]
    dv   = dist[valid]
    acc_v = np.abs(np.diff(sv)) if len(sv) > 1 else np.array([])
    ts_mid = ((ts_s[:-1] + ts_s[1:]) / 2.0)[valid]

    lat1r  = np.radians(lat[:-1])
    lat2r  = np.radians(lat[1:])
    dlon_r = np.radians(lon[1:] - lon[:-1])
    bearing = np.degrees(np.arctan2(
        np.sin(dlon_r) * np.cos(lat2r),
        np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon_r),
    ))
    bc = np.abs(np.diff(bearing[valid]))
    bc = np.where(bc > 180, 360 - bc, bc)
    stops_flag = sv < STOP_SPEED_THR

    feats.update({
        'B_speed_mean':          float(np.mean(sv)),
        'B_speed_std':           float(np.std(sv)),
        'B_speed_max':           float(np.max(sv)),
        'B_speed_p25':           float(np.percentile(sv, 25)),
        'B_speed_p50':           float(np.median(sv)),
        'B_speed_p95':           float(np.percentile(sv, 95)),
        'B_dist_total_m':        float(dv.sum()),
        'B_acc_mean':            float(np.mean(acc_v))  if len(acc_v) else np.nan,
        'B_acc_std':             float(np.std(acc_v))   if len(acc_v) else np.nan,
        'B_n_stops':             int(stops_flag.sum()),
        'B_stop_frac':           float(stops_flag.mean()),
        'B_bearing_change_mean': float(np.mean(bc))     if len(bc) else np.nan,
        'B_speed_change_rate':   float((acc_v > SPEED_CHANGE_THR).mean())
                                 if len(acc_v) > 0 else np.nan,
    })

    if len(sv) >= 5:
        feats['B_speed_trend'] = float(np.polyfit(np.linspace(0, 1, len(sv)), sv, 1)[0])

    total_path = float(dv.sum())
    if total_path > 1.0:
        displacement = float(_haversine_m(lat[0], lon[0], lat[-1], lon[-1]))
        feats['B_path_efficiency'] = min(1.0, displacement / total_path)

    detected = _detect_stops(sv, ts_mid)
    if len(detected) >= 3:
        starts = np.array([s[0] for s in detected])
        feats['B_stop_regularity'] = float(np.std(np.diff(starts)))
    if detected:
        feats['B_dwell_mean'] = float(np.mean([s[1] for s in detected]))

    # altitudine — opzionale
    if altitude is not None and len(altitude) == len(lat):
        alt = altitude[order].astype(np.float64)
        alt_valid = alt[alt > -500]  # filtra sentinel (-777 in alcuni provider)
        if len(alt_valid) >= 3:
            feats['B_alt_mean']  = float(np.mean(alt_valid))
            feats['B_alt_std']   = float(np.std(alt_valid))
            feats['B_alt_range'] = float(alt_valid.max() - alt_valid.min())

    return feats
