"""
device_map.py — mappa stabile userId → device (iOS/Android). Helper riusabile (Fase 1c).

Sorgente: **unità accelerometro** (|acc| mediana < 5 → iOS [g] · ≥ 5 → Android [m/s²]).
GOLD-VALIDATA: combacia **44/44** col registro utenti `data/raw_freeze/users_2026-06-09.csv`
(campo `Platform`); copre **tutti i 52 utenti IMU** (il registro ne copre 44). Offline:
NON richiede Mongo né il CSV (PII, gitignored). Output cache rigenerabile in `data/v2/`.

Uso modulo:  from device_map import device_map ; m = device_map()   # {userId: 'iOS'|'Android'}
Uso CLI:     python research/device_map.py [--refresh]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
IMU = ROOT / "data" / "raw_freeze" / "imu"
CACHE = ROOT / "data" / "v2" / "device_map_trento.csv"
SPLIT = 5.0


def device_map(refresh: bool = False) -> dict[str, str]:
    if CACHE.exists() and not refresh:
        d = pd.read_csv(CACHE)
        return dict(zip(d.userId, d.device))
    out = {}
    for f in sorted(IMU.glob("*.parquet")):
        b = pq.read_table(f, columns=["acc_x", "acc_y", "acc_z"]).to_pandas()
        am = np.sqrt(b.acc_x.astype("f8") ** 2 + b.acc_y.astype("f8") ** 2
                     + b.acc_z.astype("f8") ** 2).median()
        out[f.stem] = "iOS" if am < SPLIT else "Android"
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"userId": list(out), "device": list(out.values())}).to_csv(CACHE, index=False)
    return out


if __name__ == "__main__":
    m = device_map(refresh="--refresh" in sys.argv)
    vc = pd.Series(list(m.values())).value_counts().to_dict()
    print(f"{len(m)} utenti | {vc} → {CACHE.relative_to(ROOT)}")
