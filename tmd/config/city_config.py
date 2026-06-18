"""
tmd.config.city_config — configurazione città/dataset.

UN solo oggetto per TUTTA la config (struttura + labeling + post-processing).
Nessun I/O, nessuna dipendenza da pymongo: importarlo è offline e leggero.

Sostituisce la vecchia CityConfig (che viveva in tmd/data/mongo_reader.py e
scartava le 5 sezioni di labeling, costringendo a un secondo yaml.safe_load).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CityConfig:
    # ── struttura / dati (obbligatorie) ───────────────────────────────
    city:                  str
    db_name:               str
    collections:           dict
    classes:               list
    label_map:             dict
    bounds:                dict
    timestamp_epoch_range: dict
    session:               dict
    quality:               dict
    # ── labeling / post-processing (opzionali → default {}) ───────────
    lf_thresholds:          dict = field(default_factory=dict)
    window_labeler:         dict = field(default_factory=dict)
    window_labeler_weights: dict = field(default_factory=dict)
    lf_accuracy:            dict = field(default_factory=dict)
    post_processing:        dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CityConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "CityConfig":
        """Costruisce da dict. Errore chiaro su sezioni sconosciute (typo)."""
        known   = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"CityConfig: sezioni sconosciute {sorted(unknown)}. "
                f"Sezioni valide: {sorted(known)}."
            )
        return cls(**d)
