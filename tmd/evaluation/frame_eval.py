"""
tmd/evaluation/frame_eval.py — Lorenzo Attolico, OpenMove / UniTN, Maggio 2026

Valutazione frame-level per predizioni a granularità segmento.

Problema: segmenti predetti e segmenti motiontag hanno boundary diversi.
Soluzione: proiettare tutto su una timeline a risoluzione fissa (default 1s),
assegnare a ogni frame la label del segmento che lo contiene, poi calcolare
le metriche standard su quei frame.

Vantaggi rispetto al matching segmento-a-segmento:
  - Gestisce N-a-M senza matching euristico
  - Pesatura naturale per durata (un segmento lungo conta più di uno corto)
  - Metrica comparabile con la window accuracy (che è anch'essa frame-level)
  - Identico a come si valutano i sistemi HAR nel benchmark SHL

Entry points:
  frames = build_frame_index(pred_segments, gt_segments)
  metrics = compute_metrics(frames)
  report  = full_report(pred_segments, gt_segments)   ← tutto in uno

Formato atteso pred_segments:
  DataFrame con colonne: userId, t0_ms, t1_ms, predicted_label
  (output di scripts/run_segment_pipeline.py + predizione modello)

Formato atteso gt_segments (motiontag):
  DataFrame con colonne: userId, started_at_ms, finished_at_ms, mode_tmd
  (output di load_labels in mongo_reader)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ── Assegnazione label ai frame ───────────────────────────────────────────────

def _assign_labels_to_frames(
    frame_ts:  np.ndarray,
    t0_arr:    np.ndarray,
    t1_arr:    np.ndarray,
    labels:    np.ndarray,
) -> np.ndarray:
    """
    Assegna a ogni frame la label del segmento che lo contiene.
    Efficiente: O(F log S) con searchsorted su segmenti ordinati per t0.

    frame_ts: array di timestamp in ms (frame da valutare)
    t0_arr, t1_arr, labels: segmenti ordinati per t0
    """
    out = np.full(len(frame_ts), None, dtype=object)
    if len(t0_arr) == 0:
        return out

    order  = np.argsort(t0_arr)
    t0s    = t0_arr[order]
    t1s    = t1_arr[order]
    lbls   = labels[order]

    # Per ogni frame: indice dell'ultimo segmento con t0 <= frame_ts
    idx = np.searchsorted(t0s, frame_ts, side="right") - 1

    # Valido solo se il frame cade dentro il segmento trovato (frame_ts <= t1)
    in_range = idx >= 0
    clipped  = np.clip(idx, 0, len(t1s) - 1)
    in_range &= frame_ts <= t1s[clipped]

    out[in_range] = lbls[clipped[in_range]]
    return out


# ── Costruzione frame index ───────────────────────────────────────────────────

def build_frame_index(
    pred_segments: pd.DataFrame,
    gt_segments:   pd.DataFrame,
    resolution_s:  float = 1.0,
    user_col:      str   = "userId",
) -> pd.DataFrame:
    """
    Costruisce una serie temporale frame-by-frame confrontando predizioni e GT.

    Ogni riga del risultato rappresenta un intervallo di `resolution_s` secondi
    con: userId, ts_ms, gt_label, pred_label.

    Solo i frame coperti da almeno una delle due sorgenti vengono inclusi.
    Frame senza GT → gt_label = None (non usati nel calcolo metriche).
    Frame senza pred → pred_label = None (contati come errore nel recall GT).

    Parametri:
        pred_segments  colonne: userId, t0_ms, t1_ms, predicted_label
        gt_segments    colonne: userId, started_at_ms, finished_at_ms, mode_tmd
        resolution_s   risoluzione temporale in secondi (default 1.0)
    """
    resolution_ms = int(resolution_s * 1000)
    rows = []

    users = set(pred_segments[user_col].unique()) | set(gt_segments[user_col].unique())

    for uid in users:
        pred_u = pred_segments[pred_segments[user_col] == uid]
        gt_u   = gt_segments[gt_segments[user_col]    == uid]

        if pred_u.empty and gt_u.empty:
            continue

        # Range temporale comune
        t_min = min(
            pred_u["t0_ms"].min() if not pred_u.empty else np.inf,
            gt_u["started_at_ms"].min() if not gt_u.empty else np.inf,
        )
        t_max = max(
            pred_u["t1_ms"].max() if not pred_u.empty else -np.inf,
            gt_u["finished_at_ms"].max() if not gt_u.empty else -np.inf,
        )
        if not np.isfinite(t_min) or not np.isfinite(t_max):
            continue

        frame_ts = np.arange(int(t_min), int(t_max) + 1, resolution_ms, dtype=np.int64)
        if len(frame_ts) == 0:
            continue

        # GT labels
        gt_t0  = gt_u["started_at_ms"].values.astype(np.int64)
        gt_t1  = gt_u["finished_at_ms"].values.astype(np.int64)
        gt_lbl = gt_u["mode_tmd"].values

        # Pred labels
        pr_t0  = pred_u["t0_ms"].values.astype(np.int64)
        pr_t1  = pred_u["t1_ms"].values.astype(np.int64)
        pr_lbl = pred_u["predicted_label"].values

        gt_assigned   = _assign_labels_to_frames(frame_ts, gt_t0,  gt_t1,  gt_lbl)
        pred_assigned = _assign_labels_to_frames(frame_ts, pr_t0,  pr_t1,  pr_lbl)

        # Tieni solo i frame dove almeno GT o pred è noto
        keep = (gt_assigned != None) | (pred_assigned != None)  # noqa: E711
        if not keep.any():
            continue

        for ts, gt, pr in zip(frame_ts[keep], gt_assigned[keep], pred_assigned[keep]):
            rows.append({
                user_col:      uid,
                "ts_ms":       int(ts),
                "gt_label":    gt,
                "pred_label":  pr,
            })

    if not rows:
        return pd.DataFrame(columns=[user_col, "ts_ms", "gt_label", "pred_label"])
    return pd.DataFrame(rows)


# ── Metriche ──────────────────────────────────────────────────────────────────

def compute_metrics(frames: pd.DataFrame) -> dict:
    """
    Calcola metriche frame-level su frame con entrambe le label.

    Ritorna dict con:
        n_frames_total, n_frames_comparable, n_frames_no_pred, n_frames_no_gt
        accuracy, f1_macro, precision_macro, recall_macro, kappa
        per_class: dict {classe: {precision, recall, f1, support}}
        confusion: dict {true_class: {pred_class: count}}
    """
    # Solo frame con entrambe le label
    both = frames[frames["gt_label"].notna() & frames["pred_label"].notna()].copy()
    n_no_pred = int((frames["gt_label"].notna() & frames["pred_label"].isna()).sum())
    n_no_gt   = int((frames["gt_label"].isna()  & frames["pred_label"].notna()).sum())

    if both.empty:
        return {
            "n_frames_total": len(frames),
            "n_frames_comparable": 0,
            "n_frames_no_pred": n_no_pred,
            "n_frames_no_gt": n_no_gt,
            "accuracy": np.nan, "f1_macro": np.nan,
            "precision_macro": np.nan, "recall_macro": np.nan,
            "kappa": np.nan,
            "per_class": {}, "confusion": {},
        }

    y_true  = both["gt_label"].values
    y_pred  = both["pred_label"].values
    classes = sorted(set(y_true) | set(y_pred))

    acc  = float(accuracy_score(y_true, y_pred))
    f1m  = float(f1_score(y_true, y_pred, labels=classes, average="macro",  zero_division=0))
    prec = float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    rec  = float(recall_score(y_true, y_pred, labels=classes, average="macro",  zero_division=0))
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = np.nan

    # Per classe
    pc_p = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    pc_r = recall_score(   y_true, y_pred, labels=classes, average=None, zero_division=0)
    pc_f = f1_score(       y_true, y_pred, labels=classes, average=None, zero_division=0)
    per_class = {
        c: {
            "precision": float(p),
            "recall":    float(r),
            "f1":        float(f),
            "support":   int((y_true == c).sum()),
        }
        for c, p, r, f in zip(classes, pc_p, pc_r, pc_f)
    }

    # Confusion matrix come dict annidato
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    confusion = {
        true_c: {pred_c: int(cm[i, j]) for j, pred_c in enumerate(classes)}
        for i, true_c in enumerate(classes)
    }

    return {
        "n_frames_total":      len(frames),
        "n_frames_comparable": len(both),
        "n_frames_no_pred":    n_no_pred,
        "n_frames_no_gt":      n_no_gt,
        "accuracy":            acc,
        "f1_macro":            f1m,
        "precision_macro":     prec,
        "recall_macro":        rec,
        "kappa":               kappa,
        "per_class":           per_class,
        "confusion":           confusion,
    }


# ── Report completo ───────────────────────────────────────────────────────────

def full_report(
    pred_segments: pd.DataFrame,
    gt_segments:   pd.DataFrame,
    resolution_s:  float = 1.0,
    print_report:  bool  = True,
) -> dict:
    """
    Tutto in uno: costruisce frame index, calcola metriche, stampa report.

    Ritorna il dict di compute_metrics.
    """
    frames  = build_frame_index(pred_segments, gt_segments, resolution_s)
    metrics = compute_metrics(frames)

    if print_report:
        _print_metrics(metrics)

    return metrics, frames


def _print_metrics(m: dict):
    print(f"\n── Frame-level eval (1s) ──")
    print(f"  Frame totali:      {m['n_frames_total']:,}")
    print(f"  Confrontabili:     {m['n_frames_comparable']:,} "
          f"(GT + pred entrambi)")
    print(f"  Solo GT (no pred): {m['n_frames_no_pred']:,}")
    print(f"  Solo pred (no GT): {m['n_frames_no_gt']:,}")
    print(f"  Accuracy:          {m['accuracy']*100:.1f}%")
    print(f"  F1 macro:          {m['f1_macro']:.3f}")
    print(f"  Precision macro:   {m['precision_macro']:.3f}")
    print(f"  Recall macro:      {m['recall_macro']:.3f}")
    print(f"  Cohen κ:           {m['kappa']:.3f}" if not np.isnan(m["kappa"])
          else "  Cohen κ:           N/A")
    if m["per_class"]:
        print(f"\n  classe       precision  recall    F1    support")
        for cls, v in sorted(m["per_class"].items()):
            print(f"  {cls:12s}  {v['precision']:.3f}      "
                  f"{v['recall']:.3f}    {v['f1']:.3f}    {v['support']}")
