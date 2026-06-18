"""tmd.features — finestre + feature A/B/C/D (Step 5)."""
from .pipeline import (
    extract_session, add_rolling_context_features, SOURCE_SCHEMA, DROP_LIST,
)

# Colonne NON-feature (metadati + label) prodotte da extract_session/orchestratore.
META_COLS = [
    "session_id", "userId", "sess_type", "imu_hz", "city",
    "ts_start", "ts_end", "gps_frac", "n_imu", "label", "label_source",
]

__all__ = [
    "extract_session", "add_rolling_context_features",
    "SOURCE_SCHEMA", "DROP_LIST", "META_COLS",
]
