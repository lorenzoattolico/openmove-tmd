"""
tmd/models/hierarchical.py — Lorenzo Attolico, OpenMove / UniTN, Maggio 2026

Classificatore gerarchico XGBoost: L1 (Still/Moving) + L2 (N-1 classi flat)
+ specialisti opzionali S1 (Car/Bus) e S2 (Train/Subway).
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

STILL_CLASS = "Still"

XGB_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "gamma":            0.0,
    "objective":        "multi:softprob",
    "eval_metric":      "mlogloss",
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

XGB_PARAMS_BINARY = {
    **XGB_PARAMS,
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "n_estimators":     400,
    "min_child_weight": 3,
}


def _make_clf(clf_type: str, binary: bool = False):
    """Factory per il classificatore base — xgboost (default), rf, lgbm."""
    if clf_type == "xgboost":
        params = XGB_PARAMS_BINARY if binary else {**XGB_PARAMS}
        return xgb.XGBClassifier(**params)
    if clf_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=42, class_weight="balanced")
    if clf_type == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbose=-1, class_weight="balanced")
    raise ValueError(f"clf_type sconosciuto: {clf_type!r}  (validi: xgboost, rf, lgbm)")


class HierarchicalTMD:
    """
    L1 (Still/Moving) + L2 (flat N-1 classi) + specialisti opzionali.
    Specialisti: S1=Car/Bus, S2=Train/Subway.

    clf_type: 'xgboost' (default), 'rf', 'lgbm'.
    RF usa median imputation per NaN (nessun supporto nativo).
    XGBoost e LightGBM gestiscono NaN nativamente.
    """

    def __init__(self, classes: list[str], specialists: list[str],
                 clf_type: str = "xgboost"):
        self.classes     = classes
        self.specialists = specialists
        self.clf_type    = clf_type
        self.l1          = None
        self.l2          = None
        self.s1          = None
        self.s2          = None
        self.le_l2: LabelEncoder | None = None
        self.l2_classes: list[str] = []
        # RF non gestisce NaN — imputer addestrato su fit(), applicato in predict()
        self.imputer: SimpleImputer | None = (
            SimpleImputer(strategy="median") if clf_type == "rf" else None
        )

    def _prep(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if self.imputer is None:
            return X
        return self.imputer.fit_transform(X) if fit else self.imputer.transform(X)

    def fit(self, X: np.ndarray, y_str: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "HierarchicalTMD":

        Xp = self._prep(X, fit=True)

        # L1: Still=0 vs Moving=1
        y_l1    = (y_str != STILL_CLASS).astype(int)
        self.l1 = _make_clf(self.clf_type, binary=True)
        self.l1.fit(Xp, y_l1, sample_weight=sample_weight)

        # L2: solo campioni Moving
        moving_mask        = y_str != STILL_CLASS
        self.l2_classes    = [c for c in self.classes if c != STILL_CLASS]
        self.le_l2         = LabelEncoder().fit(np.array(self.l2_classes))

        X_l2  = Xp[moving_mask]
        y_l2  = self.le_l2.transform(y_str[moving_mask])
        sw_l2 = sample_weight[moving_mask] if sample_weight is not None else None

        self.l2 = _make_clf(self.clf_type, binary=False)
        if self.clf_type == "xgboost":
            self.l2.set_params(objective="multi:softprob",
                               num_class=len(self.l2_classes))
        self.l2.fit(X_l2, y_l2, sample_weight=sw_l2)

        # S1: Car vs Bus
        if "S1" in self.specialists:
            mask = np.isin(y_str, ["Car", "Bus"])
            if mask.sum() > 20:
                sw      = sample_weight[mask] if sample_weight is not None else None
                self.s1 = _make_clf(self.clf_type, binary=True)
                self.s1.fit(Xp[mask], (y_str[mask] == "Bus").astype(int),
                            sample_weight=sw)

        # S2: Train vs Subway
        if "S2" in self.specialists:
            mask = np.isin(y_str, ["Train", "Subway"])
            if mask.sum() > 20:
                sw      = sample_weight[mask] if sample_weight is not None else None
                self.s2 = _make_clf(self.clf_type, binary=True)
                self.s2.fit(Xp[mask], (y_str[mask] == "Subway").astype(int),
                            sample_weight=sw)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xp        = self._prep(X)
        pred      = np.full(len(X), STILL_CLASS, dtype=object)
        is_moving = self.l1.predict(Xp).astype(bool)

        if is_moving.sum() > 0:
            l2_idx            = self.l2.predict(Xp[is_moving])
            pred[is_moving]   = self.le_l2.inverse_transform(l2_idx)

        if self.s1 is not None:
            mask = np.isin(pred, ["Car", "Bus"])
            if mask.sum() > 0:
                pred[mask] = np.where(self.s1.predict(Xp[mask]) == 1, "Bus", "Car")

        if self.s2 is not None:
            mask = np.isin(pred, ["Train", "Subway"])
            if mask.sum() > 0:
                pred[mask] = np.where(self.s2.predict(Xp[mask]) == 1, "Subway", "Train")

        return pred

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ritorna (predicted_class, max_prob) per ogni campione."""
        Xp          = self._prep(X)
        classes_all = [STILL_CLASS] + self.l2_classes

        # probabilità Still da L1
        p_moving = self.l1.predict_proba(Xp)[:, 1]
        p_still  = 1 - p_moving

        # probabilità classi Moving da L2
        p_l2 = self.l2.predict_proba(Xp) * p_moving[:, None]

        # assembla matrice n×n_classes
        # NOTA: p_l2 ha colonne nell'ordine di le_l2.classes_ (che potrebbe essere
        # diverso dall'ordine di l2_classes se LabelEncoder ha ordinato alfabeticamente).
        # Iteriamo su le_l2.classes_ per garantire il mapping corretto.
        n = len(X)
        proba = np.zeros((n, len(classes_all)))
        proba[:, 0] = p_still
        for i, cls in enumerate(self.le_l2.classes_):
            proba[:, classes_all.index(cls)] = p_l2[:, i]

        pred_idx  = proba.argmax(axis=1)
        pred_cls  = np.array([classes_all[i] for i in pred_idx])
        max_proba = proba.max(axis=1)

        # raffina con specialisti
        if self.s1 is not None:
            mask = np.isin(pred_cls, ["Car", "Bus"])
            if mask.sum() > 0:
                p_s1 = self.s1.predict_proba(Xp[mask])
                pred_cls[mask]  = np.where(p_s1[:, 1] > 0.5, "Bus", "Car")
                max_proba[mask] = p_s1.max(axis=1)

        if self.s2 is not None:
            mask = np.isin(pred_cls, ["Train", "Subway"])
            if mask.sum() > 0:
                p_s2 = self.s2.predict_proba(Xp[mask])
                pred_cls[mask]  = np.where(p_s2[:, 1] > 0.5, "Subway", "Train")
                max_proba[mask] = p_s2.max(axis=1)

        return pred_cls, max_proba
