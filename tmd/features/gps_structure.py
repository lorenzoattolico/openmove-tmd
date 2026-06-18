"""
tmd/features/gps_structure.py — Lorenzo Attolico, OpenMove / UniTN, Maggio 2026

Gruppo D: feature strutturali GPS (qualità, gap, copertura).
"""

from __future__ import annotations
import numpy as np


GAP_THRESH_S = 10.0   # gap GPS > 10s = galleria o segnale perso


def compute(
    ts_ms:    np.ndarray,         # timestamp epoch ms dei fix GPS
    accuracy: np.ndarray | None,  # m (opzionale)
    win_s:    float = 120.0,
    gap_thresh_s: float = 30.0,
) -> dict:
    feats = {
        'D_acc_mean':         np.nan,
        'D_acc_max':          np.nan,
        'D_has_reliable_gps': 0.0,
        'D_n_gaps':           0,
        'D_gap_fraction':     0.0,
    }
    if len(ts_ms) == 0:
        return feats

    if accuracy is not None and len(accuracy) == len(ts_ms):
        feats['D_acc_mean']         = float(np.mean(accuracy))
        feats['D_acc_max']          = float(np.max(accuracy))
        feats['D_has_reliable_gps'] = float(np.median(accuracy) <= 20)

    if len(ts_ms) >= 2:
        ts_s  = np.sort(ts_ms) / 1000.0
        diffs = np.diff(ts_s)
        large = diffs[diffs > gap_thresh_s]
        feats['D_n_gaps']       = int(len(large))
        feats['D_gap_fraction'] = float(min(1.0, large.sum() / win_s))

    return feats
