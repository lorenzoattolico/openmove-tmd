"""
tmd/features/imu.py — Lorenzo Attolico, OpenMove / UniTN, Maggio 2026

Gruppo A: feature IMU statistico-spettrali + rotation-invariant.

Cambiamenti rispetto alla versione precedente:
  - Jerk CORRETTO: da |Δ||v||·fs (scalare) a ||Δv||·fs (vettoriale, Kunze 2017)
  - Eigenvalori covarianza (G): rotation-invariant per proprietà algebrica (Van Der Donckt 2024)
  - d2v = ||Δ²v|| (C2): norma derivata seconda vettoriale
  - Angoli tra campioni consecutivi (C2): rotation-invariant per norma/dot product
"""

from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.signal import welch, butter, filtfilt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*invalid value.*divide.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*Degrees of freedom.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*Precision loss.*")

FS_DEFAULT = 50.0


# ── Stima gravity / linear acc ────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _butter_coeffs(fs: float):
    """
    Coefficienti filtro Butterworth — invarianti per fs costante.
    Cache @lru_cache: calcolati UNA sola volta per valore di fs invece che
    ad ogni finestra (default 50Hz → un solo calcolo per tutta la run).
    """
    b_low,  a_low  = butter(2, 0.3 / (fs / 2), btype='low')
    b_high, a_high = butter(2, 0.3 / (fs / 2), btype='high')
    return (b_low, a_low), (b_high, a_high)


def _estimate_gravity(acc: np.ndarray, fs: float) -> np.ndarray:
    if len(acc) < 15:
        return np.tile(np.mean(acc, axis=0), (len(acc), 1))
    (b, a), _ = _butter_coeffs(fs)
    return filtfilt(b, a, acc, axis=0)


def _estimate_linear_acc(acc: np.ndarray, fs: float) -> np.ndarray:
    if len(acc) < 15:
        return acc - np.mean(acc, axis=0)
    _, (b, a) = _butter_coeffs(fs)
    return filtfilt(b, a, acc, axis=0)


# ── Helper statistiche ────────────────────────────────────────────────────────

def _skew_fast(x: np.ndarray) -> float:
    """Skewness senza overhead scipy (no nan-check, no axis handling)."""
    s = float(x.std())
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 3))


def _kurt_fast(x: np.ndarray) -> float:
    """Excess kurtosis senza overhead scipy."""
    s = float(x.std())
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 4) - 3.0)


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Correlazione di Pearson diretta senza np.corrcoef.
    np.corrcoef costruisce matrice 2x2 e alloca memoria — qui serve solo un elemento.
    ~3x più veloce per array 1D.
    """
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.linalg.norm(xm) * np.linalg.norm(ym)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(xm, ym) / denom)

def _stat(x: np.ndarray, prefix: str, skip_trivial: bool = False) -> dict:
    if len(x) == 0:
        keys = ['skew', 'kurt', 'zcr', 'iqr']
        if not skip_trivial:
            keys += ['mean', 'std', 'rms']
        return {f'{prefix}_{k}': np.nan for k in keys}
    zcr = float(np.sum(np.diff(np.sign(x)) != 0) / len(x))
    out = {
        f'{prefix}_skew': _skew_fast(x),
        f'{prefix}_kurt': _kurt_fast(x),
        f'{prefix}_zcr':  zcr,
        f'{prefix}_iqr':  float(np.percentile(x, 75) - np.percentile(x, 25)),
    }
    if not skip_trivial:
        out[f'{prefix}_mean'] = float(np.mean(x))
        out[f'{prefix}_std']  = float(np.std(x))
        out[f'{prefix}_rms']  = float(np.sqrt(np.mean(x ** 2)))
    return out


def _spettrale(x: np.ndarray, prefix: str, fs: float = FS_DEFAULT) -> dict:
    nperseg = min(len(x), 512)
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    total = psd.sum()
    if total < 1e-12:
        return {f'{prefix}_dom_freq': 0.0, f'{prefix}_spec_entropy': 0.0,
                f'{prefix}_psd_0_2': 0.0, f'{prefix}_psd_2_4': 0.0,
                f'{prefix}_psd_4_8': 0.0, f'{prefix}_psd_8_20': 0.0}
    pn = psd / total
    out = {
        f'{prefix}_dom_freq':     float(freqs[psd.argmax()]),
        f'{prefix}_spec_entropy': float(-np.sum(pn * np.log2(pn + 1e-12)) / np.log2(len(pn))),
    }
    for tag, (lo, hi) in [('psd_0_2', (0, 2)), ('psd_2_4', (2, 4)),
                           ('psd_4_8', (4, 8)), ('psd_8_20', (8, 20))]:
        out[f'{prefix}_{tag}'] = float(psd[(freqs >= lo) & (freqs < hi)].sum() / total)
    return out


def _value_entropy(x: np.ndarray, n_bins: int = 10) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    hist, _ = np.histogram(x, bins=n_bins)
    s = hist.sum()
    if s == 0:
        return 0.0
    p = hist / s
    return float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(n_bins))


def _time_entropy(x: np.ndarray, K: int = 10) -> float:
    x = x[np.isfinite(x)]
    if len(x) < K:
        return 0.0
    chunks = np.array_split(x, K)
    means  = np.array([np.abs(np.mean(c)) + 1e-12 for c in chunks if len(c) > 0])
    p = means / means.sum()
    return float(-np.sum(p * np.log2(p)) / np.log2(len(means)))


# ── Feature rotation-invariant (G + C2) ──────────────────────────────────────

def _rotation_invariant(mat: np.ndarray, prefix: str) -> dict:
    """
    Eigenvalori covarianza (G, Van Der Donckt 2024) +
    d2v = ||Δ²v|| (C2) + angoli tra campioni (C2, Kunze 2017).
    mat: (N, 3) — z-scored per acc, raw per gyr.
    Rotation-invariant per proprietà algebrica (norme e autovalori).
    """
    feats = {}
    n = len(mat)

    # G — eigenvalori covarianza 3x3
    if n >= 3:
        cov  = np.cov(mat.T)
        try:
            eigs = np.sort(np.linalg.eigvalsh(cov))[::-1].clip(0)
        except np.linalg.LinAlgError:
            eigs = np.zeros(3)
        λ1, λ2, λ3 = eigs
        eigs_sum = eigs.sum() + 1e-12
        feats.update({
            f'{prefix}_eig1':        float(λ1),
            f'{prefix}_eig2':        float(λ2),
            f'{prefix}_eig3':        float(λ3),
            f'{prefix}_eig_ratio':   float(λ1 / (λ2 + 1e-9)),
            f'{prefix}_eig_planar':  float((λ2 - λ3) / (λ1 + 1e-9)),
            f'{prefix}_eig_spheric': float(λ3 / (λ1 + 1e-9)),
            f'{prefix}_eig_entropy': float(
                -np.sum((eigs / eigs_sum) * np.log(eigs / eigs_sum + 1e-12))),
        })
    else:
        for k in ['eig1','eig2','eig3','eig_ratio','eig_planar','eig_spheric','eig_entropy']:
            feats[f'{prefix}_{k}'] = np.nan

    # C2 — d2v: norma derivata seconda vettoriale
    if n >= 4:
        d2v = np.linalg.norm(np.diff(mat, n=2, axis=0), axis=1)
        feats.update({
            f'{prefix}_d2v_mean': float(np.mean(d2v)),
            f'{prefix}_d2v_std':  float(np.std(d2v)),
            f'{prefix}_d2v_iqr':  float(np.percentile(d2v, 75) - np.percentile(d2v, 25)),
            f'{prefix}_d2v_p95':  float(np.percentile(d2v, 95)),
        })
    else:
        for k in ['d2v_mean','d2v_std','d2v_iqr','d2v_p95']:
            feats[f'{prefix}_{k}'] = np.nan

    # C2 — angoli tra campioni consecutivi
    if n >= 3:
        v0    = mat[:-1]
        v1    = mat[1:]
        dots  = np.einsum('ij,ij->i', v0, v1)
        norms = (np.linalg.norm(v0, axis=1) *
                 np.linalg.norm(v1, axis=1) + 1e-9)
        ang   = np.arccos(np.clip(dots / norms, -1, 1))
        feats.update({
            f'{prefix}_angle_mean': float(np.mean(ang)),
            f'{prefix}_angle_std':  float(np.std(ang)),
            f'{prefix}_angle_iqr':  float(np.percentile(ang, 75) - np.percentile(ang, 25)),
            f'{prefix}_angle_p95':  float(np.percentile(ang, 95)),
        })
    else:
        for k in ['angle_mean','angle_std','angle_iqr','angle_p95']:
            feats[f'{prefix}_{k}'] = np.nan

    return feats


# ── Entry point ────────────────────────────────────────────────────────────────

def compute(
    acc:     np.ndarray,
    gyr:     np.ndarray,
    lin_acc: np.ndarray | None = None,
    grav:    np.ndarray | None = None,
    mag:     np.ndarray | None = None,
    fs:      float = FS_DEFAULT,
) -> dict:
    if len(acc) == 0:
        return _empty_features()

    if lin_acc is None:
        lin_acc = _estimate_linear_acc(acc, fs)
    if grav is None:
        grav = _estimate_gravity(acc, fs)

    ax, ay, az = acc[:, 0].copy(), acc[:, 1].copy(), acc[:, 2].copy()
    gx, gy, gz = gyr[:, 0], gyr[:, 1], gyr[:, 2]
    lx, ly, lz = lin_acc[:, 0], lin_acc[:, 1], lin_acc[:, 2]
    grav_x, grav_y, grav_z = grav[:, 0], grav[:, 1], grav[:, 2]

    # z-score per-asse sull'accelerometro (Van Der Donckt 2023)
    for arr in (ax, ay, az):
        s = arr.std()
        if s > 1e-6:
            arr -= arr.mean()
            arr /= s

    acc_mat  = np.column_stack([ax, ay, az])   # z-scored, per jerk vettoriale e G/C2
    gyr_mat  = np.column_stack([gx, gy, gz])   # raw rad/s, per G/C2 giroscopio

    acc_mag  = np.linalg.norm(acc_mat, axis=1)
    gyr_mag  = np.linalg.norm(gyr_mat, axis=1)
    lin_mag  = np.sqrt(lx**2 + ly**2 + lz**2)
    grav_mag = np.sqrt(grav_x**2 + grav_y**2 + grav_z**2)

    # jerk CORRETTO: ||Δv||·fs vettoriale (Kunze 2017)
    # sostituisce il precedente |Δacc_mag|·fs che mancava i cambi di direzione
    jerk = np.linalg.norm(np.diff(acc_mat, axis=0), axis=1) * fs

    feats: dict = {}

    # acc_mag
    feats.update(_stat(acc_mag, 'A_acc_mag'))
    feats.update(_spettrale(acc_mag, 'A_acc_mag', fs))
    feats['A_acc_mag_p25']      = float(np.percentile(acc_mag, 25))
    feats['A_acc_mag_p75']      = float(np.percentile(acc_mag, 75))
    feats['A_acc_mag_ptp']      = float(acc_mag.max() - acc_mag.min())
    feats['A_acc_mag_val_ent']  = _value_entropy(acc_mag)
    feats['A_acc_mag_time_ent'] = _time_entropy(acc_mag)
    feats['A_acc_sma']          = float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
    feats['A_acc_corr_xy']      = _pearsonr(ax, ay)
    feats['A_acc_corr_xz']      = _pearsonr(ax, az)
    feats['A_acc_corr_yz']      = _pearsonr(ay, az)

    for nome, sig in [('A_acc_x', ax), ('A_acc_y', ay), ('A_acc_z', az)]:
        feats.update(_stat(sig, nome, skip_trivial=True))
        feats.update(_spettrale(sig, nome, fs))

    # giroscopio
    feats.update(_stat(gyr_mag, 'A_gyr_mag'))
    feats.update(_spettrale(gyr_mag, 'A_gyr_mag', fs))
    feats['A_gyr_sma']          = float(np.mean(np.abs(gx) + np.abs(gy) + np.abs(gz)))
    feats['A_gyr_mag_val_ent']  = _value_entropy(gyr_mag)
    feats['A_gyr_mag_time_ent'] = _time_entropy(gyr_mag)
    feats['A_gyr_corr_xy']      = _pearsonr(gx, gy)
    feats['A_gyr_corr_xz']      = _pearsonr(gx, gz)
    feats['A_gyr_corr_yz']      = _pearsonr(gy, gz)
    for nome, sig in [('A_gyr_x', gx), ('A_gyr_y', gy), ('A_gyr_z', gz)]:
        feats.update(_stat(sig, nome))
        feats.update(_spettrale(sig, nome, fs))

    # linear acc
    feats.update(_stat(lin_mag, 'A_lin_mag'))
    feats.update(_spettrale(lin_mag, 'A_lin_mag', fs))
    feats['A_lin_mag_val_ent']  = _value_entropy(lin_mag)
    feats['A_lin_mag_time_ent'] = _time_entropy(lin_mag)

    # gravity
    for nome, sig in [('A_grav_x', grav_x), ('A_grav_y', grav_y),
                      ('A_grav_z', grav_z), ('A_grav_mag', grav_mag)]:
        feats[f'{nome}_mean'] = float(np.mean(sig))
        feats[f'{nome}_std']  = float(np.std(sig))

    feats['A_gyr_acc_ratio'] = float(np.mean(gyr_mag) / (np.mean(acc_mag) + 1e-6))

    # jerk (vettoriale corretto)
    feats.update(_stat(jerk, 'A_jerk'))
    feats.update(_spettrale(jerk, 'A_jerk', fs))

    # autocorrelazione acc_mag
    if len(acc_mag) > 100:
        xc  = acc_mag - acc_mag.mean()
        var = np.dot(xc, xc)
        if var > 1e-12:
            nn  = len(xc)
            X   = np.fft.rfft(xc, n=2 * nn - 1)
            acf = np.fft.irfft(X * np.conj(X))[:nn] / var
            idx_05s = int(0.5 * fs)
            idx_1s  = int(1.0 * fs)
            feats['A_acc_autocorr_05s'] = float(acf[idx_05s]) if idx_05s < nn else np.nan
            feats['A_acc_autocorr_1s']  = float(acf[idx_1s])  if idx_1s  < nn else np.nan
        else:
            feats['A_acc_autocorr_05s'] = 0.0
            feats['A_acc_autocorr_1s']  = 0.0
    else:
        feats['A_acc_autocorr_05s'] = np.nan
        feats['A_acc_autocorr_1s']  = np.nan

    # G + C2 rotation-invariant — acc (z-scored) e gyr (raw)
    feats.update(_rotation_invariant(acc_mat, 'A_acc'))
    feats.update(_rotation_invariant(gyr_mat, 'A_gyr'))

    # magnetometro
    if mag is not None and len(mag) > 0:
        mx, my, mz = mag[:, 0], mag[:, 1], mag[:, 2]
        mag_mag = np.sqrt(mx**2 + my**2 + mz**2)
        sp = _spettrale(mag_mag, 'A_mag_mag', fs)
        feats['A_mag_mag_mean']    = float(np.mean(mag_mag))
        feats['A_mag_mag_std']     = float(np.std(mag_mag))
        feats['A_mag_mag_val_ent'] = _value_entropy(mag_mag)
        feats['A_mag_mag_psd_0_2'] = sp['A_mag_mag_psd_0_2']
        feats['A_mag_mag_psd_2_4'] = sp['A_mag_mag_psd_2_4']
        feats['A_mag_mag_psd_4_8'] = sp['A_mag_mag_psd_4_8']
        feats['A_mag_corr_xy']     = _pearsonr(mx, my)
        feats['A_mag_corr_xz']     = _pearsonr(mx, mz)
        feats['A_mag_corr_yz']     = _pearsonr(my, mz)
    else:
        for k in ['A_mag_mag_mean', 'A_mag_mag_std', 'A_mag_mag_val_ent',
                  'A_mag_mag_psd_0_2', 'A_mag_mag_psd_2_4', 'A_mag_mag_psd_4_8',
                  'A_mag_corr_xy', 'A_mag_corr_xz', 'A_mag_corr_yz']:
            feats[k] = np.nan

    return feats


def _empty_features() -> dict:
    dummy_acc = np.ones((25, 3)) * 0.01
    dummy_gyr = np.zeros((25, 3))
    f = compute(dummy_acc, dummy_gyr)
    return {k: np.nan for k in f}
