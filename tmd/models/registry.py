"""
tmd/models/registry.py — Lorenzo Attolico, OpenMove / UniTN, Maggio 2026

Salvataggio, caricamento e listing modelli addestrati.
Ogni modello = .pkl (model + feature_cols) + _meta.json (metriche + config).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


def save_model(model, feature_cols: list[str],
               metadata: dict, city: str,
               registry_dir: Path) -> str:
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    ts   = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    name = f"{city}_{ts}"
    with open(registry_dir / f"{name}.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    with open(registry_dir / f"{name}_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # aggiorna puntatore latest per questa città
    latest_path = registry_dir / f"latest_{city}.json"
    latest_path.write_text(json.dumps({
        "name":       name,
        "path":       str(registry_dir / f"{name}.pkl"),
        "trained_at": metadata.get("trained_at", ""),
        "f1_macro":   metadata.get("f1_macro_mean"),
    }, indent=2))
    return name


def load_model(model_path: str | Path) -> dict:
    """Carica modello dal .pkl. Ritorna {'model': ..., 'feature_cols': [...]}."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "HierarchicalTMD":
                # rimappa per NOME → carica sia i pkl vecchi (tmd.*) sia i nuovi (tmd.*)
                from tmd.models.hierarchical import HierarchicalTMD
                return HierarchicalTMD
            return super().find_class(module, name)

    with open(model_path, "rb") as f:
        return _Unpickler(f).load()


def load_meta(model_path: str | Path) -> dict:
    """Carica metadata JSON associato al modello."""
    meta_path = Path(str(model_path).replace(".pkl", "_meta.json"))
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())

def load_latest(city: str, registry_dir: str | Path) -> dict:
    """Carica l'ultimo modello salvato per questa città."""
    p = Path(registry_dir) / f"latest_{city}.json"
    if not p.exists():
        raise FileNotFoundError(f"Nessun modello latest per {city} in {registry_dir}")
    info = json.loads(p.read_text())
    return load_model(info["path"]), info


def list_models(registry_dir: str | Path) -> pd.DataFrame:
    """Lista tutti i modelli nel registry con le loro metriche principali."""
    registry_dir = Path(registry_dir)
    if not registry_dir.exists():
        return pd.DataFrame()

    rows = []
    for meta_file in sorted(registry_dir.glob("*_meta.json")):
        meta = json.loads(meta_file.read_text())
        pkl  = meta_file.with_name(meta_file.name.replace("_meta.json", ".pkl"))
        rows.append({
            "name":          meta_file.stem.replace("_meta", ""),
            "city":          meta.get("city", ""),
            "f1_macro":      meta.get("f1_macro_mean", None),
            "eval_strategy": meta.get("eval_strategy", ""),
            "n_windows":     meta.get("n_windows", None),
            "n_features":    meta.get("n_features", None),
            "specialists":   ",".join(meta.get("specialists", [])),
            "cleanlab":      meta.get("cleanlab", False),
            "trained_at":    meta.get("trained_at", ""),
            "pkl_exists":    pkl.exists(),
        })

    return pd.DataFrame(rows)
