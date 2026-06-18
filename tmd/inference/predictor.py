"""
tmd/inference/predictor.py — Lorenzo Attolico, OpenMove / UniTN, Maggio 2026

Inferenza batch su Parquet features → predictions.parquet.
Smoothing majority vote + filtro plausibilità fisica velocità-classe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tmd.models.registry import load_model
from tmd.config import CityConfig

MOTORIZED = {"Car", "Bus", "Train", "Subway"}

# Walk/Run/Bike isolati tra due finestre motorizzate identiche → probabilmente rumore.
SANDWICH_CORRECTABLE: set[str] = {"Walk", "Run", "Bike"}

# Still viene sandwiched solo se i vicini sono Bus o Train.
# Motivazione: uno stop di 120s in un viaggio Bus/Train è quasi certamente
# GPS drift a una fermata (rumore). Lo stesso stop in un viaggio Car è
# uno stato fisico legittimo (semaforo, parcheggio, attesa).
# L'EDA su Trento mostra: sandwiching generico su Still → +8 worsened, +2 improved (Bus),
# quindi si preserva solo il caso Bus/Train dove il segnale è più affidabile.
SANDWICH_STILL_ALLOWED: set[str] = {"Bus", "Train", "Subway"}

# vincoli fisici velocità media su finestra 120s (m/s) — default
# Overridabili via city_cfg['post_processing']['speed_bounds']
# Train/Subway: nessun lower bound (si fermano in stazione).
SPEED_BOUNDS_DEFAULT: dict[str, tuple[float | None, float | None]] = {
    "Still":  (None, 0.8),
    "Walk":   (0.2,  3.5),
    "Run":    (1.5,  7.0),
    "Bike":   (1.5, 14.0),
    "Bus":    (None, 30.0),
    "Car":    (None, 55.0),
    "Train":  (None, None),
    "Subway": (None, None),
}
# Mantenuto per retrocompatibilità con codice che importa SPEED_BOUNDS direttamente
SPEED_BOUNDS = SPEED_BOUNDS_DEFAULT


def smooth_predictions(df: pd.DataFrame,
                       window: int = 5,
                       motorized: set[str] = MOTORIZED) -> pd.Series:
    """
    Majority vote con finestra scorrevole.

    Logica doppia:
    1. Finestre motorizzate: majority vote tra i motorizzati vicini
    2. Finestre non-motorizzate isolate tra motorizzati uguali:
       Bus Walk Bus → Bus Bus Bus  (Walk in mezzo → corretto a Bus)
    """
    pred = df["predicted_class"].copy()

    for sid, grp in df.groupby("session_id"):
        grp = grp.sort_values("ts_start")
        idx = grp.index.tolist()
        labels = pred[idx].tolist()
        n = len(labels)
        smoothed = labels.copy()

        for i in range(n):
            half  = window // 2
            start = max(0, i - half)
            end   = min(n, i + half + 1)
            ctx   = labels[start:end]

            if labels[i] in motorized:
                # majority vote tra motorizzati nel contesto
                motor = [l for l in ctx if l in motorized]
                if len(motor) >= window // 2 + 1:
                    counts = {}
                    for l in motor:
                        counts[l] = counts.get(l, 0) + 1
                    smoothed[i] = max(counts, key=counts.get)

            elif labels[i] in SANDWICH_CORRECTABLE:
                # Walk/Run/Bike isolati tra due motorizzati identici → quasi certamente rumore
                neighbors = [labels[j] for j in [i - 1, i + 1]
                             if 0 <= j < n and labels[j] in motorized]
                if len(neighbors) == 2 and len(set(neighbors)) == 1:
                    smoothed[i] = neighbors[0]
            elif labels[i] == "Still":
                # Still corretto solo tra Bus/Train identici (GPS drift a una fermata)
                neighbors = [labels[j] for j in [i - 1, i + 1]
                             if 0 <= j < n and labels[j] in SANDWICH_STILL_ALLOWED]
                if len(neighbors) == 2 and len(set(neighbors)) == 1:
                    smoothed[i] = neighbors[0]

        for i, ix in enumerate(idx):
            pred[ix] = smoothed[i]

    return pred


# Coppie correggibili: dominant → correggi questi run.
# Restringe la correzione a confusioni note tra motorizzati,
# evitando di toccare Walk/Still/Bike e di invertire l'errore
# (es. non corregge Car runs in sessioni Car-dominanti → Train).
SESSION_CORRECTION_PAIRS: dict[str, set[str]] = {
    "Train": {"Car"},   # treno che rallenta → patch Car → Train
    "Bus":   {"Car"},   # autobus nel traffico → patch Car → Bus
}


# Coppie soggette a oscillazione rapida (bidirezionale).
# Es: Bus→Car→Bus→Car in breve tempo → quasi certamente rumore del modello.
OSCILLATION_PAIRS: set[frozenset] = {
    frozenset({"Bus", "Car"}),    # confusione più comune in tratti urbani
    frozenset({"Still", "Walk"}), # fermo vs passo lento
}


# Durata minima per classe — in MINUTI (indipendente dalla lunghezza della finestra).
# Overridabile via city_cfg['post_processing']['min_segment_min'].
# Retrocompat: se il YAML usa ancora 'min_segment_windows' (conteggio finestre),
# viene letto direttamente senza conversione.
MIN_SEGMENT_MIN: dict[str, float] = {
    "Train": 10.0,
    "Bus":    8.0,
    "Bike":   4.0,
    "Car":    2.0,
    "Walk":   2.0,
    "Still":  0.0,
}
# Alias retrocompat per codice esterno che importa MIN_SEGMENT_WINDOWS
MIN_SEGMENT_WINDOWS = {k: max(0, round(v * 60 / 120)) for k, v in MIN_SEGMENT_MIN.items()}

# Durata finestra scorrevole smooth — in MINUTI (default 10 min).
SMOOTH_MIN_DEFAULT: float = 10.0


def _post_processing(city_cfg) -> dict:
    """post_processing dalla CityConfig (None = default). Errore CHIARO se è un dict raw
    (evita il fallback silenzioso ai default)."""
    if city_cfg is None:
        return {}
    if not isinstance(city_cfg, CityConfig):
        raise TypeError(
            f"city_cfg deve essere una CityConfig o None, non {type(city_cfg).__name__} "
            "(probabilmente un dict raw — usa CityConfig.from_yaml).")
    return city_cfg.post_processing or {}


def _min_segment_windows(city_cfg, win_s: float = 120.0) -> dict:
    """Converte durate minime in numero di finestre per win_s dato."""
    pp = _post_processing(city_cfg)
    # Formato nuovo: minuti → converte in finestre
    if "min_segment_min" in pp:
        mins = dict(MIN_SEGMENT_MIN)
        mins.update({k: float(v) for k, v in pp["min_segment_min"].items()})
        return {k: max(0, round(v * 60 / win_s)) for k, v in mins.items()}
    # backward-compat: min_segment_windows (conteggio a 120s), scala con win_s
    if "min_segment_windows" in pp:
        base = {k: max(0, round(v * 60 / win_s)) for k, v in MIN_SEGMENT_MIN.items()}
        base.update({k: max(0, round(int(v) * 120 / win_s))
                     for k, v in pp["min_segment_windows"].items()})
        return base
    return {k: max(0, round(v * 60 / win_s)) for k, v in MIN_SEGMENT_MIN.items()}


def _smooth_window_n(city_cfg, win_s: float = 120.0) -> int:
    """Numero di finestre per lo smooth majority vote, dato win_s."""
    pp = _post_processing(city_cfg)
    if "smooth_min" in pp:
        return max(1, round(float(pp["smooth_min"]) * 60 / win_s))
    if "smooth_window" in pp:  # retrocompat: era conteggio a 120s
        return max(1, round(int(pp["smooth_window"]) * 120 / win_s))
    return max(1, round(SMOOTH_MIN_DEFAULT * 60 / win_s))

# Transizioni fisicamente impossibili senza modo intermedio.
# Se A → B senza Walk/Still in mezzo, il segmento più corto viene corretto.
IMPOSSIBLE_DIRECT: set[frozenset] = {
    frozenset({"Train", "Car"}),
    frozenset({"Train", "Bus"}),
    frozenset({"Train", "Bike"}),
}


def segment_coherence_filter(df: pd.DataFrame,
                              pred_col: str = "predicted_class_smooth",
                              min_windows: dict | None = None,
                              impossible_direct: set | None = None,
                              max_passes: int = 10,
                              city_cfg: dict | None = None,
                              win_s: float = 120.0) -> pd.Series:
    """
    Healing algorithm: rende le predizioni temporalmente coerenti.

    Passaggio A — minimum duration:
      Segmenti più corti della durata minima realistica per quel modo
      vengono assorbiti dal segmento adiacente più lungo.
      Es: [Train×1, Car×20] → [Car×21]  (Train di 2 min = impossibile)

    Passaggio B — impossible transitions:
      Se due modi in IMPOSSIBLE_DIRECT si susseguono direttamente senza
      un Walk/Still intermedio, il segmento più corto viene assorbito.
      Es: [Train×40, Car×3, Train×10] → [Train×53]

    Entrambi i passaggi vengono ripetuti fino a convergenza (max_passes).
    Non tocca transizioni legittime con Walk/Still in mezzo.

    Ref: Guvensan et al. (2017), "A Novel Segment-Based Approach for
    Improving Classification Performance of Transport Mode Detection",
    Sensors 18(1):87.
    """
    if min_windows is None:
        min_windows = _min_segment_windows(city_cfg, win_s)
    if impossible_direct is None:
        impossible_direct = IMPOSSIBLE_DIRECT

    WALK_STILL = {"Walk", "Still"}

    result    = df[pred_col].copy()
    n_fixed_a = 0
    n_fixed_b = 0

    for sid, grp in df.groupby("session_id"):
        grp   = grp.sort_values("ts_start")
        idx   = grp.index.tolist()
        preds = result[idx].tolist()
        n     = len(preds)

        def build_segments(p):
            """Lista di (mode, start, end) con end esclusivo."""
            segs = []
            i = 0
            while i < n:
                j = i
                while j < n and p[j] == p[i]:
                    j += 1
                segs.append([p[i], i, j])
                i = j
            return segs

        def apply_segments(segs, p):
            out = p.copy()
            for mode, s, e in segs:
                for k in range(s, e):
                    out[k] = mode
            return out

        for _pass in range(max_passes):
            changed = False
            segs = build_segments(preds)

            # ── Passaggio A: minimum duration ────────────────────────────────
            new_segs = []
            i = 0
            while i < len(segs):
                mode, s, e = segs[i]
                run_len = e - s
                min_req = min_windows.get(mode, 0)
                if run_len < min_req:
                    # trova il vicino più lungo (prima o dopo)
                    prev_len = segs[i-1][2] - segs[i-1][1] if i > 0 else 0
                    next_len = segs[i+1][2] - segs[i+1][1] if i < len(segs)-1 else 0
                    if prev_len == 0 and next_len == 0:
                        new_segs.append(segs[i])
                    elif prev_len >= next_len and i > 0 and new_segs:
                        # assorbi nel precedente (se il precedente è stato già
                        # assorbito in avanti, new_segs è vuoto → ramo sotto)
                        new_segs[-1][2] = e
                        n_fixed_a += run_len
                        changed = True
                    elif i < len(segs) - 1:
                        # assorbi nel successivo
                        segs[i+1][1] = s
                        segs[i+1][0] = segs[i+1][0]  # mantieni modo successivo
                        n_fixed_a += run_len
                        changed = True
                    else:
                        new_segs.append(segs[i])
                else:
                    new_segs.append(segs[i])
                i += 1
            segs = new_segs

            # ── Passaggio B: impossible transitions ───────────────────────────
            new_segs = []
            i = 0
            while i < len(segs):
                mode, s, e = segs[i]
                if i > 0:
                    prev_mode = new_segs[-1][0]
                    pair = frozenset({mode, prev_mode})
                    if pair in impossible_direct:
                        # controlla se c'è Walk/Still tra i due
                        # (non può esserci se sono adiacenti per costruzione)
                        prev_len = new_segs[-1][2] - new_segs[-1][1]
                        cur_len  = e - s
                        if cur_len <= prev_len:
                            # assorbi corrente nel precedente
                            new_segs[-1][2] = e
                            n_fixed_b += cur_len
                            changed = True
                            i += 1
                            continue
                        else:
                            # assorbi precedente nel corrente
                            new_segs[-1][0] = mode
                            n_fixed_b += prev_len
                            changed = True
                new_segs.append([mode, s, e])
                i += 1
            segs = new_segs

            preds = apply_segments(segs, preds)
            if not changed:
                break

        for k, ix in enumerate(idx):
            result[ix] = preds[k]

    total = n_fixed_a + n_fixed_b
    if total > 0:
        print(f"  Segment coherence: {n_fixed_a} min-duration + "
              f"{n_fixed_b} impossible-transition = {total} finestre corrette")
    return result


def predict_parquet(model_path: str | Path,
                    parquet_path: str | Path,
                    output_path: str | Path | None = None,
                    smooth_window: int | None = None,
                    city_class_map: dict | None = None,
                    city_cfg: dict | None = None,
                    win_s: float | None = None) -> pd.DataFrame:
    """
    Gira il modello su un Parquet features e salva predictions.parquet.

    win_s: dimensione finestra in secondi. None = inferita dal parquet.
    smooth_window: numero finestre per majority vote. None = calcolato da
                   city_cfg['post_processing']['smooth_min'] e win_s.

    Output columns:
      session_id, ts_start, ts_end, userId,
      predicted_class, max_prob,
      predicted_class_smooth (se smooth_window > 1),
      label (se presente nel Parquet input)
    """
    bundle    = load_model(model_path)
    model     = bundle["model"]
    feat_cols = bundle["feature_cols"]

    df = pd.read_parquet(parquet_path)
    print(f"Parquet: {len(df):,} finestre")

    # Inferisci win_s dal parquet se non specificato
    if win_s is None and "ts_start" in df.columns and "ts_end" in df.columns and len(df) > 0:
        win_s = float(np.median((df["ts_end"] - df["ts_start"]).values / 1000))
    win_s = win_s or 120.0

    # Calcola smooth_window da city_cfg + win_s se non passato esplicitamente
    if smooth_window is None:
        smooth_window = _smooth_window_n(city_cfg, win_s)
    print(f"win_s={win_s:.0f}s  smooth_window={smooth_window} finestre ({smooth_window * win_s / 60:.0f} min)")

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] {len(missing)} feature mancanti (saranno NaN)")

    X = np.full((len(df), len(feat_cols)), np.nan, dtype=np.float32)
    for i, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, i] = df[col].values.astype(np.float32)

    pred_cls, max_prob = model.predict_proba(X)

    if city_class_map:
        pred_cls = np.array([city_class_map.get(c, c) for c in pred_cls])

    df_out = df[["session_id", "ts_start", "ts_end", "userId"]].copy()
    df_out["predicted_class"] = pred_cls
    df_out["max_prob"]        = max_prob

    # copia feature necessarie ai filtri post-inferenza
    if "B_speed_mean" in df.columns:
        df_out["B_speed_mean"] = df["B_speed_mean"]
    if "D_has_reliable_gps" in df.columns:
        df_out["D_has_reliable_gps"] = df["D_has_reliable_gps"]

    if smooth_window > 1:
        df_out["predicted_class_smooth"] = smooth_predictions(
            df_out, window=smooth_window
        )
        df_out["predicted_class_smooth"] = segment_coherence_filter(
            df_out, pred_col="predicted_class_smooth",
            city_cfg=city_cfg, win_s=win_s,
        )

    if "label" in df.columns:
        df_out["label"] = df["label"]

    # ── colonne analisi qualitativa ───────────────────────────────────────────
    # split: train / test / unlabeled  (basato sui test_session_ids in meta)
    meta = {}
    try:
        from tmd.models.registry import load_meta
        meta = load_meta(model_path)
    except Exception:
        pass

    test_ids = set(meta.get("test_session_ids", []))
    if test_ids:
        # Il modello conosce il suo test set → assegna direttamente
        df_out["split"] = df_out["session_id"].apply(
            lambda s: "test" if s in test_ids else "train"
        )
        df_out.loc[df["label"].isna(), "split"] = "unlabeled"
    elif "split" in df.columns:
        # Colonna split già presente nel Parquet features (es. SHL con --split-col split)
        # → propagala direttamente, è la fonte più affidabile
        df_out["split"] = df["split"].values
        print("  Split propagato dal Parquet features (colonna 'split' trovata)")
    else:
        df_out["split"] = "unlabeled"

    # correct: True / False / None (solo per finestre con label)
    pred_col_final = ("predicted_class_smooth"
                      if "predicted_class_smooth" in df_out.columns
                      else "predicted_class")
    if "label" in df_out.columns:
        labeled_mask = df_out["label"].notna()
        df_out["correct"] = None
        df_out.loc[labeled_mask, "correct"] = (
            df_out.loc[labeled_mask, pred_col_final] ==
            df_out.loc[labeled_mask, "label"]
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(output_path, index=False)
        print(f"Salvato: {output_path}")

    col = "predicted_class_smooth" if smooth_window > 1 else "predicted_class"
    print(f"\nDistribuzione predizioni ({col}):")
    print(df_out[col].value_counts().to_string())

    return df_out
