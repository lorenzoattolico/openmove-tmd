"""tmd.inference — post-processing predizioni (Step 9): smooth + coherence. No filtri morti."""
from .predictor import (
    smooth_predictions, segment_coherence_filter, predict_parquet,
    _smooth_window_n, _min_segment_windows,
)

__all__ = ["smooth_predictions", "segment_coherence_filter", "predict_parquet",
           "_smooth_window_n", "_min_segment_windows"]
