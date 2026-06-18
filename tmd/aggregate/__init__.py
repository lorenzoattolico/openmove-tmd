"""tmd/aggregate — dal per-finestra all'aggregato corretto (modal-split + CO2).

E' il deliverable d'uso: la classificazione per-finestra e' il mezzo, l'aggregato il fine.
La quantification (Saerens 2002 / Forman 2008) de-biasa le quote invertendo la matrice di
confusione stimata su un piccolo set di calibrazione etichettato.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOVING_MODES = ["Walk", "Bus", "Car", "Train"]

# Fattori di emissione gCO2e/passenger-km (riferimento EEA/DEFRA): indicatore d'uso, non misura locale.
EMISSION_FACTORS = {"Still": 0.0, "Walk": 0.0, "Bus": 100.0, "Car": 170.0, "Train": 40.0}


def modal_split(labels, modes=MOVING_MODES) -> dict[str, float]:
    """Quote dei modi (somma 1) sulle etichette appartenenti a `modes`."""
    v = pd.Series(labels)
    v = v[v.isin(modes)].value_counts().reindex(modes).fillna(0.0)
    tot = float(v.sum())
    return {m: float(v[m]) / tot if tot else 0.0 for m in modes}


def confusion_matrix(true, pred, modes=MOVING_MODES, weights=None) -> np.ndarray:
    """M[i, j] = P(pred=j | true=i) sul set di calibrazione.

    weights: pesi per-campione (es. km) per la versione distanza-pesata; None = conteggi.
    """
    true, pred = np.asarray(true), np.asarray(pred)
    w = np.ones(len(true)) if weights is None else np.asarray(weights, dtype=float)
    M = np.zeros((len(modes), len(modes)))
    for i, ci in enumerate(modes):
        mi = true == ci
        tot = w[mi].sum()
        if tot > 0:
            for j, cj in enumerate(modes):
                M[i, j] = w[mi & (pred == cj)].sum() / tot
        else:
            M[i, i] = 1.0
    return M


def _adjust(y, M) -> np.ndarray:
    """Inverte la confusione: risolve M^T x = y, clip a >= 0 (Saerens/Forman)."""
    x, *_ = np.linalg.lstsq(M.T, np.asarray(y, dtype=float), rcond=None)
    return np.clip(x, 0.0, None)


def quantify(q_pred, M) -> np.ndarray:
    """Quote predette de-biasate (somma 1)."""
    x = _adjust(q_pred, M)
    return x / x.sum() if x.sum() > 0 else np.asarray(q_pred, dtype=float)


def co2(km_by_mode: dict[str, float], ef=EMISSION_FACTORS) -> float:
    """Emissioni aggregate (gCO2e) = somma_modo km_modo * fattore_modo."""
    return float(sum(km_by_mode.get(m, 0.0) * ef.get(m, 0.0) for m in km_by_mode))


def aggregate(pred, km=None, calibration=None, modes=MOVING_MODES, ef=EMISSION_FACTORS) -> dict:
    """Aggregato d'uso a partire dalle predizioni per-finestra.

    pred        : etichette predette (iterabile).
    km          : km per-finestra allineati a `pred` (per la CO2); None = niente CO2.
    calibration : (true, pred) etichettati per de-biasare le quote; None = solo naive.
                  Per la CO2 km-pesata, aggiungere i km: (true, pred, km).
    """
    pred = np.asarray(pred)
    out: dict = {"modal_split": modal_split(pred, modes)}

    if calibration is not None:
        ct, cp = calibration[0], calibration[1]
        M = confusion_matrix(ct, cp, modes)
        q = quantify([out["modal_split"][m] for m in modes], M)
        out["modal_split_corrected"] = dict(zip(modes, q))

    if km is not None:
        km = np.asarray(km, dtype=float)
        km_by_mode = {m: float(km[pred == m].sum()) for m in modes}
        out["co2_g"] = co2(km_by_mode, ef)
        if calibration is not None and len(calibration) > 2:
            M_km = confusion_matrix(calibration[0], calibration[1], modes, weights=calibration[2])
            km_corr = _adjust([km_by_mode[m] for m in modes], M_km)
            out["co2_g_corrected"] = float(km_corr @ np.array([ef[m] for m in modes]))
    return out
