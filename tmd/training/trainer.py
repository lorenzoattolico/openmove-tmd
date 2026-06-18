"""
tmd/training/trainer.py — Lorenzo Attolico, OpenMove / UniTN, Maggio 2026

Logica di training: eval strategies, Cleanlab, metriche, training loop.
Usato da scripts/train.py come thin wrapper CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tmd.models.hierarchical import HierarchicalTMD
from tmd.models.registry import save_model
from tmd.config import CityConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Feature selection ─────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame, groups: list[str]) -> list[str]:
    """Colonne feature per i gruppi specificati, esclude quelle sempre-NaN."""
    cols = [c for c in df.columns
            if any(c.startswith(g + "_") for g in groups)]
    return [c for c in cols if df[c].notna().any()]


# ── Eval strategies ───────────────────────────────────────────────────────────

def fixed_splits(df: pd.DataFrame, split_col: str):
    train = df[df[split_col] == "train"]
    val   = df[df[split_col] == "validate"]
    yield split_col, train, val


def louo_splits(df: pd.DataFrame, user_col: str = "userId"):
    for u in df[user_col].unique():
        yield u, df[df[user_col] != u], df[df[user_col] == u]


def session_splits(df: pd.DataFrame,
                   test_frac: float = 0.2, seed: int = 42):
    sessions = df["session_id"].unique()
    rng  = np.random.default_rng(seed)
    test = set(rng.choice(sessions,
                           size=max(1, int(len(sessions) * test_frac)),
                           replace=False))
    mask = df["session_id"].isin(test)
    yield "session_split", df[~mask], df[mask]


def temporal_splits(df_train_pool: pd.DataFrame,
                    df_eval_pool: pd.DataFrame | None = None,
                    test_frac: float = 0.2):
    """
    Split temporale su sessioni ordinate per ts_start.

    df_train_pool: finestre da cui estrarre il training set (silver o motiontag)
    df_eval_pool:  finestre da cui estrarre il test set (sempre motiontag).
                   Se None, usa df_train_pool (backward compat per louo/session).

    Il cutoff viene calcolato su df_eval_pool → stesso cutoff per silver e
    motiontag → le metriche sono confrontabili su test set identico.
    """
    ref = df_eval_pool if df_eval_pool is not None else df_train_pool
    sess_times = (ref.groupby("session_id")["ts_start"].min()
                     .sort_values())
    n_test        = max(1, int(len(sess_times) * test_frac))
    test_sessions = set(sess_times.index[-n_test:])

    t_cutoff_ms = sess_times.iloc[-n_test]
    t_cutoff    = pd.Timestamp(t_cutoff_ms, unit="ms", tz="UTC").tz_convert("Europe/Rome")
    print(f"  Temporal split: cutoff {t_cutoff.strftime('%Y-%m-%d %H:%M')}  "
          f"({n_test} sessioni test, {len(sess_times)-n_test} train)")

    df_train = df_train_pool[~df_train_pool["session_id"].isin(test_sessions)]
    df_test  = ref[ref["session_id"].isin(test_sessions)]

    yield "temporal_split", df_train, df_test


def rolling_splits(df_train_pool: pd.DataFrame,
                   df_eval_pool: pd.DataFrame | None = None,
                   n_folds: int = 5,
                   last_frac: float = 0.5):
    """
    Rolling-origin (forward-chaining) temporale: n_folds blocchi di test
    SUCCESSIVI che coprono l'ultimo `last_frac` delle sessioni motiontag.
    Per ogni blocco: train = silver delle sessioni che iniziano STRETTAMENTE
    prima del blocco (no leakage temporale, anche per le sessioni senza motiontag).

    Robusto al singolo taglio: il trainer aggrega media ± std sui fold.
    Vedi thesis/results.md (Headline Trento / robustezza split).
    """
    ref = df_eval_pool if df_eval_pool is not None else df_train_pool
    sess_start = ref.groupby("session_id")["ts_start"].min().sort_values()
    N = len(sess_start)
    tp_start = df_train_pool.groupby("session_id")["ts_start"].min()
    edges = np.linspace(1.0 - last_frac, 1.0, n_folds + 1)
    for k in range(n_folds):
        i_lo, i_hi = int(N * edges[k]), int(N * edges[k + 1])
        test_sessions = set(sess_start.index[i_lo:i_hi])
        block_start = sess_start.iloc[i_lo]
        train_sessions = set(tp_start.index[tp_start < block_start]) - test_sessions
        df_train = df_train_pool[df_train_pool["session_id"].isin(train_sessions)]
        df_test = ref[ref["session_id"].isin(test_sessions)]
        if len(df_test) < 10 or df_train.empty:
            continue
        print(f"  Rolling fold {k+1}/{n_folds}: train={len(df_train):,} test={len(df_test):,} "
              f"(blocco da {pd.Timestamp(block_start, unit='ms').strftime('%m-%d %H:%M')})")
        yield f"rolling_{k+1}", df_train, df_test


# ── Cleanlab ──────────────────────────────────────────────────────────────────

def cleanlab_weights(X: np.ndarray, y: np.ndarray,
                     n_classes: int) -> np.ndarray:
    """
    Stima sample weights con Cleanlab confident learning.

    Flusso:
      1. Cross-validation (5 fold) con XGBoost leggero
      2. OOF probabilities → find_label_issues
      3. Label rumorose → peso 0.1, pulite → peso 1.0
    """
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        print("  [WARN] cleanlab non installato: pip install cleanlab")
        return np.ones(len(y))

    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from tmd.models.hierarchical import XGB_PARAMS

    oof_probs = np.zeros((len(y), n_classes))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val      = X[val_idx]

        clf = xgb.XGBClassifier(**{**XGB_PARAMS, "n_estimators": 200})
        clf.fit(X_tr, y_tr)

        oof_probs[val_idx] = clf.predict_proba(X_val)

    issues = find_label_issues(
        y, oof_probs,
        return_indices_ranked_by="self_confidence",
    )
    weights = np.ones(len(y))
    weights[issues] = 0.1
    print(f"  Cleanlab: {len(issues):,}/{len(y):,} rumorosi ({len(issues)/len(y)*100:.1f}%)")
    return weights


# ── Metriche ──────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             classes: list[str]) -> dict:
    """
    f1_macro_seen:  macro F1 calcolato SOLO sulle classi con supporto > 0
                    nel test set. Metrica principale.
    f1_macro_all:   macro F1 su tutte le classi (penalizza classi assenti).
    """
    seen   = [c for c in classes if (y_true == c).sum() > 0]
    absent = [c for c in classes if c not in seen]

    f1_seen = f1_score(y_true, y_pred, labels=seen,
                       average="macro", zero_division=0)
    f1_all  = f1_score(y_true, y_pred, labels=classes,
                       average="macro", zero_division=0)
    f1_per  = f1_score(y_true, y_pred, labels=classes,
                       average=None,   zero_division=0)

    if absent:
        print(f"    [classi assenti nel test: {absent} → escluse da f1_macro_seen]")

    return {
        "f1_macro":       round(float(f1_seen), 4),
        "f1_macro_seen":  round(float(f1_seen), 4),
        "f1_macro_all":   round(float(f1_all),  4),
        "n_seen":         len(seen),
        "absent_classes": absent,
        "f1_per_class":   {c: round(float(f), 4)
                           for c, f in zip(classes, f1_per)},
    }


# ── Eval parquet ──────────────────────────────────────────────────────────────

def _build_eval_frame(df_test: pd.DataFrame,
                      y_pred: np.ndarray,
                      max_prob: np.ndarray,
                      fold_name: str) -> pd.DataFrame:
    meta_cols  = ["session_id", "ts_start", "ts_end", "userId", "label"]
    extra_cols = ["B_speed_mean", "B_speed_max", "B_stop_frac",
                  "D_gap_fraction", "D_has_reliable_gps",
                  "gps_frac", "n_imu"]   # gps_frac/n_imu → eval GPS-stratificato self-contained

    df_eval = df_test[[c for c in meta_cols if c in df_test.columns]].copy()
    df_eval["predicted_class"] = y_pred
    df_eval["max_prob"]        = max_prob
    df_eval["correct"]         = y_pred == df_test["label"].values
    df_eval["split"]           = "test"
    df_eval["fold"]            = str(fold_name)

    for col in extra_cols:
        if col in df_test.columns:
            df_eval[col] = df_test[col].values

    return df_eval


def _plot_confusion_matrices(
    df: pd.DataFrame,
    classes: list[str],
    city: str,
    ts: str,
    analysis_dir: Path,
):
    """Salva una figura con CM raw e CM smooth affiancate in data/analysis/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix as _cm

        labeled = df[df["label"].notna()]
        if labeled.empty:
            return

        y_true = labeled["label"].values
        present = [c for c in classes if c in set(y_true)]
        if not present:
            return

        pairs = []
        if "predicted_class" in labeled.columns:
            pairs.append(("predicted_class", "Raw"))
        if "predicted_class_smooth" in labeled.columns:
            pairs.append(("predicted_class_smooth", "Smooth (post-inferenza)"))

        if not pairs:
            return

        analysis_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
        if len(pairs) == 1:
            axes = [axes]

        for ax, (pred_col, title) in zip(axes, pairs):
            y_pred  = labeled[pred_col].values
            cm      = _cm(y_true, y_pred, labels=present)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(present)))
            ax.set_yticks(range(len(present)))
            ax.set_xticklabels(present, rotation=45, ha="right")
            ax.set_yticklabels(present)
            ax.set_xlabel("Predetto")
            ax.set_ylabel("Vero")
            ax.set_title(f"Confusion matrix — {title}")
            for i in range(len(present)):
                for j in range(len(present)):
                    v = cm_norm[i, j]
                    ax.text(j, i, f"{v:.2f}\n({cm[i,j]})",
                            ha="center", va="center", fontsize=8,
                            color="white" if v > 0.55 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"TMD — {city}  |  {ts}", fontsize=11)
        fig.tight_layout()
        out_cm = analysis_dir / f"cm_{city}_{ts}.png"
        fig.savefig(out_cm, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Confusion matrix salvata: {out_cm.name}")
    except Exception as e:
        print(f"  [WARN] confusion matrix non salvata: {e}")


def _save_eval_parquet(df_eval_combined: pd.DataFrame,
                       city: str,
                       registry_dir: Path,
                       train_classes: list | None = None,
                       city_cfg: dict | None = None,
                       win_s: float = 120.0) -> tuple[Path, dict]:
    """
    Applica smoothing + session correction al DataFrame OOF combinato,
    lo salva in data/processed/eval_{city}_{ts}.parquet e restituisce
    le metriche smooth per la stampa nel riepilogo.

    city_cfg: YAML completo (incluso post_processing) per smooth_min e min_segment_min.
    win_s:    lunghezza finestra in secondi — necessario per scalare i parametri.
    train_classes: classi su cui il modello è stato addestrato — usato per
    calcolare F1 smooth con la stessa definizione di 'seen' del trainer
    (esclude classi come Bike mai viste in training).
    """
    from sklearn.metrics import f1_score as _f1
    from tmd.inference.predictor import (
        smooth_predictions, segment_coherence_filter, _smooth_window_n,
    )

    df = df_eval_combined.copy()
    smooth_metrics: dict = {}
    sw = _smooth_window_n(city_cfg, win_s)

    try:
        df["predicted_class_smooth"] = smooth_predictions(df, window=sw)
        df["predicted_class_smooth"] = segment_coherence_filter(
            df, pred_col="predicted_class_smooth",
            city_cfg=city_cfg, win_s=win_s,
        )
        # Stack canonico = smooth + segment_coherence (no plausibility_filter).
        df["correct_smooth"] = df["predicted_class_smooth"] == df["label"]

        labeled = df[df["label"].notna()]
        if len(labeled) > 0:
            y_true   = labeled["label"].values
            y_smooth = labeled["predicted_class_smooth"].values
            # Usa train_classes se disponibile per escludere classi mai viste
            # in training (es. Bike), allineando la definizione di 'seen' al trainer.
            if train_classes is not None:
                seen = [c for c in train_classes if c in set(y_true)]
            else:
                seen = [c for c in sorted(set(y_true))
                        if (y_true == c).sum() > 0]
            f1_macro = float(_f1(y_true, y_smooth, labels=seen,
                                 average="macro", zero_division=0))
            f1_per   = {c: float(_f1(y_true, y_smooth, labels=[c],
                                     average="macro", zero_division=0))
                        for c in seen}
            smooth_metrics = {"f1_macro": f1_macro, "f1_per_class": f1_per}

    except Exception as e:
        print(f"  [WARN] smoothing/session correction non applicato: {e}")

    ts  = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = registry_dir.parent / "processed" / f"eval_{city}_{ts}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    n_folds = df["fold"].nunique() if "fold" in df.columns else 1
    print(f"  Eval parquet salvato: {out.name}  "
          f"({len(df):,} finestre, {n_folds} fold)")

    analysis_dir = registry_dir.parent / "analysis"
    _plot_confusion_matrices(
        df=df,
        classes=train_classes or sorted(df["label"].dropna().unique().tolist()),
        city=city,
        ts=ts,
        analysis_dir=analysis_dir,
    )

    return out, smooth_metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(args) -> str:
    """
    Addestra il modello gerarchico sul Parquet specificato.
    Ritorna il nome del modello salvato nel registry.

    Separazione train/test:
      source=silver:    train = silver_labeled ∩ primo 80% sessioni (per data)
                        test  = motiontag_labeled ∩ ultimo 20% sessioni
      source=motiontag: train = motiontag_labeled ∩ primo 80% sessioni
                        test  = motiontag_labeled ∩ ultimo 20% sessioni

    Il test set è sempre il pool motiontag → confronto diretto tra le due source.
    """
    from sklearn.preprocessing import LabelEncoder
    import yaml as _yaml_full

    groups = [g.strip() for g in args.groups.split(",")]
    specs  = [] if args.no_specialists else (args.specialists or [])
    source = getattr(args, "source", "motiontag")

    df_full = pd.read_parquet(args.parquet)

    # Inferisci win_s dal parquet (mediana durata finestre)
    if "ts_start" in df_full.columns and "ts_end" in df_full.columns and len(df_full) > 0:
        win_s = float(np.median((df_full["ts_end"] - df_full["ts_start"]).values / 1000))
    else:
        win_s = float(getattr(args, "win_s", None) or 120.0)

    # Carica city_cfg completo (incluso post_processing) per smooth_min + min_segment_min
    cfg = None
    if not args.city.startswith("shl"):
        _cfg_path = PROJECT_ROOT / "tmd" / "configs" / "cities" / f"{args.city}.yaml"
        if _cfg_path.exists():
            cfg = CityConfig.from_yaml(_cfg_path)

    # Esclude utenti con problemi strutturali di raccolta dati (da city YAML).
    # Filtro applicato sia a train pool che a eval pool per non distorcere le metriche.
    # Disabilitabile con --no-exclude-users per ablation.
    if not args.city.startswith("shl") and not getattr(args, "no_exclude_users", False):
        exclude_users = (cfg.quality.get("exclude_users", []) if cfg else [])
        if exclude_users and "userId" in df_full.columns:
            before = len(df_full)
            df_full = df_full[~df_full["userId"].isin(exclude_users)]
            n_excl = before - len(df_full)
            if n_excl:
                print(f"  Esclusi {n_excl:,} finestre da {len(exclude_users)} utente/i (exclude_users)")
    elif getattr(args, "no_exclude_users", False):
        print("  [--no-exclude-users] exclude_users ignorato")

    # Pool di valutazione: sempre motiontag, usato come test set
    df_eval_pool = df_full[df_full["label"].notna()].copy()
    df_eval_pool["_train_label"] = df_eval_pool["label"]

    # Pool di training: dipende dalla source
    if source == "silver":
        if "silver_label" not in df_full.columns:
            raise ValueError(
                "silver_label non trovato nel parquet — "
                "esegui assign_silver_labels.py prima"
            )
        df = df_full[df_full["silver_label"].notna()].copy()
        df["_train_label"] = df["silver_label"]
        # silver_weight non usato: ogni finestra silver conta uguale.
        weight_col = None
        print(f"Source: silver  |  {len(df):,} finestre con silver label  [no sample-weight]")
    else:
        df = df_eval_pool.copy()
        weight_col = None
        print(f"Source: motiontag  |  {len(df):,} finestre etichettate")

    df = df.reset_index(drop=True)
    print(f"Train pool: {len(df):,}  |  Eval pool (motiontag): {len(df_eval_pool):,}")

    if len(df) == 0:
        raise ValueError("Nessuna finestra etichettata nel training pool")

    if args.city.startswith("shl"):
        classes = sorted(df_eval_pool["label"].unique().tolist())
    else:
        classes = cfg.classes

    print(f"Classi ({len(classes)}): {classes}")
    print(f"Specialisti: {specs or 'nessuno'}  |  Eval: {args.eval_strategy}")

    feat_cols = get_feature_cols(df, groups)
    print(f"Feature: {len(feat_cols)}")

    # ── Eval folds ────────────────────────────────────────────────────────────
    # Per temporal: df_eval_pool come riferimento → test set identico per silver e
    # motiontag (confronto diretto). Altre strategie: test estratto dal pool di training.
    if args.eval_strategy == "fixed":
        splits = list(fixed_splits(df, args.split_col))
    elif args.eval_strategy == "louo":
        splits = list(louo_splits(df))
    elif args.eval_strategy == "temporal":
        splits = list(temporal_splits(df, df_eval_pool))
    elif args.eval_strategy == "rolling":
        splits = list(rolling_splits(df, df_eval_pool))
    else:
        splits = list(session_splits(df))

    all_results  = []
    eval_frames  = []
    last_train_classes: list = []

    for fold_name, df_train, df_test in splits:
        print(f"\n── Fold: {fold_name}  train={len(df_train):,}  test={len(df_test):,} ──")

        # Test set contiene sempre label motiontag (viene da df_eval_pool o df)
        # Nessun filtro aggiuntivo necessario
        if df_test["label"].isna().any():
            n_before = len(df_test)
            df_test  = df_test[df_test["label"].notna()]
            print(f"  [INFO] Rimosse {n_before - len(df_test)} finestre senza label motiontag dal test set")

        if len(df_test) < 10:
            print(f"  [SKIP] test set troppo piccolo ({len(df_test)} finestre)")
            continue

        X_train     = df_train[feat_cols].values.astype(np.float32)
        y_train     = df_train["_train_label"].values
        X_test      = df_test[feat_cols].values.astype(np.float32)
        y_test_eval = df_test["label"].values   # sempre motiontag

        # GPS dropout augmentation: maschera B_ e C_ a NaN su p% delle finestre
        # GPS-presenti → XGBoost impara branch NaN utili anche senza dati GPS-assenti
        gps_dropout = getattr(args, "gps_dropout", 0.0)
        if gps_dropout > 0.0:
            b_c_idx = [i for i, c in enumerate(feat_cols)
                       if c.startswith("B_") or c.startswith("C_")]
            if b_c_idx:
                rng = np.random.default_rng(42)
                # Usa B_n_gps come proxy GPS-present: semanticamente corretto
                # e più robusto del primo B_/C_ che potrebbe avere NaN per altri motivi.
                n_gps_proxy = (feat_cols.index("B_n_gps")
                               if "B_n_gps" in feat_cols else b_c_idx[0])
                gps_ok_mask = np.isfinite(X_train[:, n_gps_proxy]) & (X_train[:, n_gps_proxy] > 0)
                drop_mask   = gps_ok_mask & (rng.random(len(X_train)) < gps_dropout)
                X_train = X_train.copy()
                X_train[np.ix_(drop_mask, b_c_idx)] = np.nan
                print(f"  GPS dropout p={gps_dropout:.0%}: {drop_mask.sum()} finestre mascherate "
                      f"({drop_mask.sum()/gps_ok_mask.sum()*100:.0f}% delle GPS-ok)")

        train_classes      = [c for c in classes if c in np.unique(y_train)]
        last_train_classes = train_classes

        sw = None
        if weight_col and weight_col in df_train.columns:
            sw = df_train[weight_col].values.astype(np.float32)
            sw = np.where(np.isnan(sw), 1.0, sw)
        elif args.cleanlab:
            le_tmp = LabelEncoder().fit(train_classes)
            sw     = cleanlab_weights(X_train, le_tmp.transform(y_train),
                                       len(train_classes))

        model = HierarchicalTMD(train_classes, specs, clf_type=getattr(args, "clf", "xgboost"))
        model.fit(X_train, y_train, sample_weight=sw)

        y_pred, max_prob = model.predict_proba(X_test)
        results = evaluate(y_test_eval, y_pred, train_classes)
        results.update({"fold": str(fold_name),
                        "n_train": len(df_train), "n_test": len(df_test)})
        all_results.append(results)

        print(f"  F1 macro: {results['f1_macro']:.4f}")
        for cls, f1 in results["f1_per_class"].items():
            print(f"    {cls:<12}: {f1:.4f}")

        fold_frame = _build_eval_frame(df_test, y_pred, max_prob, fold_name)
        eval_frames.append(fold_frame)

    f1_seen_scores = [r["f1_macro_seen"] for r in all_results]
    f1_all_scores  = [r["f1_macro_all"]  for r in all_results]

    print(f"\n── Riepilogo ──")
    print(f"  F1 macro (seen classes only): "
          f"{np.mean(f1_seen_scores):.4f} ± {np.std(f1_seen_scores):.4f}  ← metrica principale")
    print(f"  F1 macro (all classes):       "
          f"{np.mean(f1_all_scores):.4f} ± {np.std(f1_all_scores):.4f}  (include classi assenti)")

    class_f1s: dict[str, list[float]] = {c: [] for c in classes}
    for r in all_results:
        for c, f in r["f1_per_class"].items():
            if c not in r.get("absent_classes", []):
                class_f1s[c].append(f)
    print(f"  F1 per classe (media sui fold con supporto):")
    for cls in classes:
        vals = class_f1s[cls]
        if vals:
            print(f"    {cls:<8}: {np.mean(vals):.4f}  (n_fold={len(vals)})")
        else:
            print(f"    {cls:<8}: N/A (mai nel test set)")

    if eval_frames:
        df_eval_combined = pd.concat(eval_frames, ignore_index=True)
        _, smooth_metrics = _save_eval_parquet(
            df_eval_combined=df_eval_combined,
            city=args.city,
            registry_dir=PROJECT_ROOT / args.registry,
            train_classes=last_train_classes or None,
            city_cfg=cfg,
            win_s=win_s,
        )
        if smooth_metrics:
            f1_sm = smooth_metrics["f1_macro"]
            f1_raw_mean = np.mean(f1_seen_scores)
            print(f"  F1 macro (smooth, seen):      {f1_sm:.4f}"
                  f"  (+{f1_sm - f1_raw_mean:.4f} vs raw)  ← metrica produzione")
            print(f"  F1 per classe (smooth):")
            for cls in classes:
                f = smooth_metrics["f1_per_class"].get(cls)
                if f is not None:
                    f_raw = np.mean(class_f1s[cls]) if class_f1s.get(cls) else None
                    delta = f" (+{f - f_raw:.4f})" if f_raw is not None else ""
                    print(f"    {cls:<8}: {f:.4f}{delta}")

    _last_test_sids = [str(s) for _, _, df_t in splits
                       for s in df_t["session_id"].unique()]

    # ── Modello finale su tutti i dati ────────────────────────────────────────
    print(f"\n── Training modello finale (tutti i dati) ──")
    X_all = df[feat_cols].values.astype(np.float32)
    y_all = df["_train_label"].values
    final_classes = [c for c in classes if c in np.unique(y_all)]

    sw_all = None
    if weight_col and weight_col in df.columns:
        sw_all = df[weight_col].values.astype(np.float32)
        sw_all = np.where(np.isnan(sw_all), 1.0, sw_all)
    elif args.cleanlab:
        le_tmp = LabelEncoder().fit(final_classes)
        sw_all = cleanlab_weights(X_all, le_tmp.transform(y_all), len(final_classes))

    # GPS dropout augmentation: applicato anche al modello finale per coerenza
    # con i fold di valutazione. Stesso seed + stessa logica.
    gps_dropout_final = getattr(args, "gps_dropout", 0.0)
    if gps_dropout_final > 0.0:
        b_c_idx_f = [i for i, c in enumerate(feat_cols)
                     if c.startswith("B_") or c.startswith("C_")]
        if b_c_idx_f:
            rng_f = np.random.default_rng(43)  # seed diverso per no overlap con fold
            n_gps_proxy_f = (feat_cols.index("B_n_gps")
                             if "B_n_gps" in feat_cols else b_c_idx_f[0])
            gps_ok_f  = np.isfinite(X_all[:, n_gps_proxy_f]) & (X_all[:, n_gps_proxy_f] > 0)
            drop_f    = gps_ok_f & (rng_f.random(len(X_all)) < gps_dropout_final)
            X_all     = X_all.copy()
            X_all[np.ix_(drop_f, b_c_idx_f)] = np.nan
            print(f"  GPS dropout finale p={gps_dropout_final:.0%}: {drop_f.sum()} finestre mascherate")

    final_model = HierarchicalTMD(final_classes, specs, clf_type=getattr(args, "clf", "xgboost"))
    final_model.fit(X_all, y_all, sample_weight=sw_all)

    metadata = {
        "city":          args.city,
        "classes":       classes,
        "specialists":   specs,
        "groups":        groups,
        "n_features":    len(feat_cols),
        "n_windows":     len(df),
        "eval_strategy": args.eval_strategy,
        "source":        source,
        "f1_macro_mean":      round(float(np.mean(f1_seen_scores)), 4),
        "f1_macro_std":       round(float(np.std(f1_seen_scores)),  4),
        "f1_macro_all_mean":  round(float(np.mean(f1_all_scores)),  4),
        "f1_per_class_agg":   {c: round(float(np.mean(v)), 4) if v else None
                               for c, v in class_f1s.items()},
        "fold_results":  all_results,
        "trained_at":    pd.Timestamp.now().isoformat(),
        "cleanlab":      args.cleanlab,
        "test_session_ids": _last_test_sids,
    }

    registry_dir = PROJECT_ROOT / args.registry
    name = save_model(final_model, feat_cols, metadata, args.city, registry_dir)
    print(f"Modello salvato: {name}")
    return name
