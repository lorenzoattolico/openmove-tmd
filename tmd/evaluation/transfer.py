"""
tmd/evaluation/transfer.py — Valutazione cross-dataset (transfer).

Carica un modello dal registry, lo applica a un parquet target con labels GT,
e calcola F1 macro + per classe sulle classi comuni.

Feature alignment automatico:
  - feature nel modello ma assenti nel target → NaN (XGBoost le gestisce)
  - feature nel target non usate dal modello  → ignorate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tmd.models.registry import load_model, load_meta


EXTRA_COLS = ["B_speed_mean", "B_speed_max", "B_stop_frac",
              "D_gap_fraction", "D_has_reliable_gps"]


def _align_features(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Costruisce X con le esatte feature_cols del modello.
    Colonne mancanti riempite con NaN; colonne extra ignorate.
    """
    X = np.full((len(df), len(feature_cols)), np.nan, dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if col in df.columns:
            X[:, i] = df[col].values.astype(np.float32)
    return X


def _build_eval_frame(df: pd.DataFrame, y_pred: np.ndarray,
                      max_prob: np.ndarray, model_tag: str) -> pd.DataFrame:
    meta_cols = ["session_id", "ts_start", "ts_end", "userId", "label"]
    out = df[[c for c in meta_cols if c in df.columns]].copy()
    out["predicted_class"] = y_pred
    out["max_prob"]        = max_prob
    out["correct"]         = y_pred == df["label"].values
    out["split"]           = "transfer"
    out["fold"]            = model_tag
    for col in EXTRA_COLS:
        if col in df.columns:
            out[col] = df[col].values
    return out


def evaluate_transfer(
    model_path: str | Path,
    df_target: pd.DataFrame,
    classes: list[str] | None = None,
    filter_split: str | None = None,
    smooth: bool = False,
    city_cfg: dict | None = None,
    win_s: float = 120.0,
    model_tag: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Valuta un modello Trento sul parquet target (es. SHL).

    Parametri
    ---------
    model_path   : path al .pkl o stringa restituita da registry
    df_target    : DataFrame con feature + colonna 'label' GT
    classes      : classi su cui calcolare le metriche (None = intersezione automatica)
    filter_split : se presente, filtra df_target a df['split'] == filter_split
    smooth       : applica smoothing post-inferenza per sessione
    city_cfg     : config città per parametri smoothing
    win_s        : durata finestra in secondi
    model_tag    : etichetta per output (default: stem del pkl)

    Ritorna
    -------
    metrics : dict con f1_macro, f1_per_class, n_windows, n_missing_feats, ecc.
    eval_df : DataFrame nello stesso formato degli eval parquet di trainer.py
    """
    loaded      = load_model(model_path)
    model       = loaded["model"]
    feature_cols = loaded["feature_cols"]

    if model_tag is None:
        model_tag = Path(str(model_path)).stem

    meta = load_meta(model_path)

    df = df_target.copy()
    if filter_split and "split" in df.columns:
        df = df[df["split"] == filter_split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Nessuna riga con split='{filter_split}' nel parquet target")

    df = df[df["label"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError("Nessuna riga con label nel parquet target")

    # Classi: intersezione tra modello e target, filtrate a quelle richieste
    model_classes  = set(meta.get("classes", []))
    target_classes = set(df["label"].unique())
    common         = model_classes & target_classes
    if classes is not None:
        common = common & set(classes)
    eval_classes = sorted(common)

    if not eval_classes:
        raise ValueError(
            f"Nessuna classe in comune. Modello: {sorted(model_classes)}  "
            f"Target: {sorted(target_classes)}  Richieste: {classes}"
        )

    discarded = target_classes - set(eval_classes)
    if discarded:
        print(f"  [{model_tag}] classi target non nel modello, scartate: {sorted(discarded)}")

    df = df[df["label"].isin(eval_classes)].reset_index(drop=True)

    # Feature alignment
    missing_feats = [c for c in feature_cols if c not in df.columns]
    n_missing     = len(missing_feats)
    if missing_feats:
        print(f"  [{model_tag}] {n_missing} feature mancanti nel target → NaN: "
              f"{missing_feats[:5]}{'...' if n_missing > 5 else ''}")

    X = _align_features(df, feature_cols)
    y_true = df["label"].values

    y_pred, max_prob = model.predict_proba(X)

    # Smoothing opzionale per sessione
    eval_df = _build_eval_frame(df, y_pred, max_prob, model_tag)

    if smooth:
        from tmd.inference.predictor import (
            smooth_predictions, segment_coherence_filter, _smooth_window_n,
        )
        sw = _smooth_window_n(city_cfg, win_s)
        eval_df["predicted_class_smooth"] = smooth_predictions(eval_df, window=sw)
        eval_df["predicted_class_smooth"] = segment_coherence_filter(
            eval_df, pred_col="predicted_class_smooth", city_cfg=city_cfg, win_s=win_s)
        # plausibility_filter non usato (stack canonico = smooth + coherence).
        eval_df["correct_smooth"] = eval_df["predicted_class_smooth"] == eval_df["label"]
        y_pred_eval = eval_df["predicted_class_smooth"].values
    else:
        y_pred_eval = y_pred

    seen   = [c for c in eval_classes if (y_true == c).sum() > 0]
    f1_seen = f1_score(y_true, y_pred_eval, labels=seen, average="macro", zero_division=0)
    f1_per  = f1_score(y_true, y_pred_eval, labels=eval_classes, average=None, zero_division=0)

    metrics = {
        "model_tag":      model_tag,
        "source":         meta.get("source", "?"),
        "groups":         ",".join(meta.get("groups", [])),
        "trento_f1":      meta.get("f1_macro_mean"),
        "transfer_f1":    round(float(f1_seen), 4),
        "f1_per_class":   {c: round(float(f), 4)
                           for c, f in zip(eval_classes, f1_per)},
        "eval_classes":   eval_classes,
        "n_windows":      len(df),
        "n_missing_feats": n_missing,
        "smooth":         smooth,
    }

    return metrics, eval_df


def evaluate_rule_based(
    df_target: pd.DataFrame,
    city_cfg: dict,
    classes: list[str] | None = None,
    filter_split: str | None = None,
    model_tag: str = "rule_based",
) -> tuple[dict, pd.DataFrame]:
    """
    Valuta il window_labeler (rule-based) sul parquet target.

    ABSTAIN (None) viene trattato come predizione errata ('Unknown'),
    così il confronto con i modelli ML è onesto (entrambi coprono tutte
    le finestre). La abstain_rate è riportata separatamente nelle metriche.
    """
    from tmd.labeling.window_labeler import label_windows_universal
    _labeler = label_windows_universal

    df = df_target.copy()
    if filter_split and "split" in df.columns:
        df = df[df["split"] == filter_split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Nessuna riga con split='{filter_split}' nel parquet target")

    df = df[df["label"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError("Nessuna riga con label nel parquet target")

    target_classes = set(df["label"].unique())
    if classes is not None:
        eval_classes = sorted(set(classes) & target_classes)
    else:
        eval_classes = sorted(target_classes)

    df = df[df["label"].isin(eval_classes)].reset_index(drop=True)

    labels, _ = _labeler(df, city_cfg)
    y_pred = np.array(["Unknown" if l is None else l for l in labels], dtype=object)
    y_true = df["label"].values

    abstain_mask = y_pred == "Unknown"
    abstain_rate = float(abstain_mask.mean())

    seen    = [c for c in eval_classes if (y_true == c).sum() > 0]
    f1_seen = f1_score(y_true, y_pred, labels=seen, average="macro", zero_division=0)
    f1_per  = f1_score(y_true, y_pred, labels=eval_classes, average=None, zero_division=0)

    eval_df = _build_eval_frame(df, y_pred, np.zeros(len(df)), model_tag)

    metrics = {
        "model_tag":       model_tag,
        "source":          "rule_based",
        "groups":          "A,B,C",
        "trento_f1":       None,
        "transfer_f1":     round(float(f1_seen), 4),
        "f1_per_class":    {c: round(float(f), 4)
                            for c, f in zip(eval_classes, f1_per)},
        "eval_classes":    eval_classes,
        "n_windows":       len(df),
        "abstain_rate":    round(abstain_rate, 4),
        "n_missing_feats": 0,
        "smooth":          False,
    }

    print(f"  [{model_tag}] abstain: {abstain_rate:.1%}  "
          f"({abstain_mask.sum()} finestre su {len(df)})")

    return metrics, eval_df
