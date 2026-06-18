"""
tmd.datasets.openmove_raw — lettura del raw OpenMove (data/raw/{gps,imu,labels}).

Unifica i loader duplicati di run_pipeline.py + run_segment_pipeline.py in un'unica
classe. Solo parquet locale: niente MongoDB, niente pymongo, niente hack di cache.
(La sorgente live Mongo → data/raw/ resta nel DUMP, vedi tmd/ingest/.)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


class OpenMoveRaw:
    """Lettore del raw OpenMove già scaricato in data/raw/. Solo lettura locale."""

    GPS_COLS = ["userId", "timestamp", "latitude", "longitude",
                "accuracy", "speed", "bearing"]
    sensors  = {"gps": True, "imu": True}   # descrittore disponibilità sensori

    def __init__(self, raw_dir: str | Path = "data/raw"):
        self.raw = Path(raw_dir)

    @staticmethod
    def _ts_filters(since_ms: int | None, until_ms: int | None, margin_ms: int = 0):
        f = []
        if since_ms is not None:
            f.append(("timestamp", ">=", since_ms - margin_ms))
        if until_ms is not None:
            f.append(("timestamp", "<=", until_ms + margin_ms))
        return f or None

    def gps(self, since_ms: int | None = None,
            until_ms: int | None = None) -> pd.DataFrame:
        df = pq.read_table(str(self.raw / "gps"),
                           filters=self._ts_filters(since_ms, until_ms)).to_pandas()
        for c in self.GPS_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return (df.astype({"timestamp": "int64"})
                  .sort_values(["userId", "timestamp"]).reset_index(drop=True))

    def imu_user_ids(self) -> list[str]:
        d = self.raw / "imu"
        return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []

    def imu(self, uid: str, t0_ms: int | None = None, t1_ms: int | None = None,
            margin_ms: int = 3_600_000) -> pd.DataFrame:
        path = self.raw / "imu" / f"{uid}.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pq.read_table(str(path),
                           filters=self._ts_filters(t0_ms, t1_ms, margin_ms)).to_pandas()
        return (df.astype({"timestamp": "int64"})
                  .sort_values("timestamp").reset_index(drop=True))

    def labels(self, since_ms: int | None = None,
               until_ms: int | None = None) -> pd.DataFrame:
        path = self.raw / "labels.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["userId", "started_at_ms",
                                         "finished_at_ms", "mode_tmd"])
        df = pd.read_parquet(path).rename(columns={"started_at": "started_at_ms",
                                                   "finished_at": "finished_at_ms"})
        if since_ms is not None:
            df = df[df["finished_at_ms"] >= since_ms]
        if until_ms is not None:
            df = df[df["started_at_ms"] <= until_ms]
        return df.reset_index(drop=True)
