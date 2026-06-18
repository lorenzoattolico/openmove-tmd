"""
tmd.features.gt_labels — attacca l'etichetta GT motiontag a ogni finestra.

Port da tmd/data/labeler.py::assign_labels_for_session (vettorizzata).
FIX: il vecchio crashava (IndexError) con labels senza colonne mode_key/
detected_mode_key — run_pipeline lo aggirava iniettando None a mano. Qui il
"source-tagging" (morto: la funzione ritornava solo le label) è rimosso →
nessun crash, e l'output `label` è IDENTICO al vecchio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_MIN_OVERLAP  = 0.8   # frazione minima finestra coperta dal segmento (strict)
LABEL_MIN_MAJORITY = 0.5   # soglia modalità majority (più recall)


def _overlap_matrix(ts0: np.ndarray, ts1: np.ndarray,
                    lbl_start: np.ndarray, lbl_end: np.ndarray) -> np.ndarray:
    """Matrice overlap [n_windows × n_labels] = frazione di finestra coperta."""
    win_dur = (ts1 - ts0).astype(np.float64)
    inter = np.maximum(
        0,
        np.minimum(ts1[:, None], lbl_end[None, :]) -
        np.maximum(ts0[:, None], lbl_start[None, :])
    ).astype(np.float64)
    win_dur_safe = np.where(win_dur > 0, win_dur, 1.0)
    return inter / win_dur_safe[:, None]


def assign_labels_for_session(df_feat: pd.DataFrame, df_labels: pd.DataFrame, uid: str,
                              min_overlap: float = LABEL_MIN_OVERLAP,
                              mode: str = "strict") -> pd.Series:
    """
    Assegna la label motiontag a ogni finestra per overlap temporale.
    mode='strict'   → label solo se overlap >= min_overlap (0.8).
    mode='majority' → label dominante se copre >= 0.5.
    Ritorna una Series allineata a df_feat (None dove nessuna label supera la soglia).
    """
    labels = pd.Series([None] * len(df_feat), dtype=object, index=df_feat.index)
    if df_labels.empty or df_feat.empty:
        return labels
    user_labels = (df_labels[df_labels["userId"] == uid]
                   if "userId" in df_labels.columns else df_labels)
    if user_labels.empty:
        return labels

    ts0 = df_feat["ts_start"].values.astype(np.int64)
    ts1 = df_feat["ts_end"].values.astype(np.int64)
    lbl_start = user_labels["started_at_ms"].values.astype(np.int64)
    lbl_end   = user_labels["finished_at_ms"].values.astype(np.int64)
    lbl_modes = user_labels["mode_tmd"].values.astype(object)

    frac = _overlap_matrix(ts0, ts1, lbl_start, lbl_end)
    threshold = LABEL_MIN_MAJORITY if mode == "majority" else min_overlap
    best_idx  = np.argmax(frac, axis=1)
    best_frac = frac[np.arange(frac.shape[0]), best_idx]
    mask      = best_frac >= threshold
    result    = np.where(mask, lbl_modes[best_idx], None)
    return pd.Series(result, dtype=object, index=df_feat.index)
