"""
rq2.6 — Window-size su Trento, confronto FRAME-LEVEL (onesto, comparabile tra granularità).

Il macro-F1 window-level NON è comparabile tra finestre 30/60/120s (finestre più corte =
meno segnale per finestra). Il confronto corretto riporta tutto alla stessa risoluzione
temporale: si assegna a ogni frame (1 s) la label del segmento GT MotionTag che lo contiene
e la predizione della finestra che lo contiene, poi si calcola il macro-F1 sui frame.

GT a risoluzione fine = segmenti MotionTag (data/raw_freeze/labels.parquet: started_at/
finished_at/mode_tmd per userId). Predizioni = eval parquet (split=test, ts_start/ts_end/
predicted_class/userId/gps_frac), allenati senza sample-weight a 30/60/120 s sul set 230.

Uso: python research/rq2_6_window_size_frame.py <eval_120> <eval_60> <eval_30>
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Still", "Walk", "Car", "Bus", "Train"]
RES_MS = 1000  # frame a 1 s


def _assign(frame_ts, starts, ends, vals):
    """Per ogni frame_ts, il valore del segmento [start,end] che lo contiene (None se nessuno)."""
    order = np.argsort(starts)
    s, e, v = starts[order], ends[order], np.asarray(vals, dtype=object)[order]
    idx = np.searchsorted(s, frame_ts, side="right") - 1
    out = np.full(len(frame_ts), None, dtype=object)
    ok = idx >= 0
    cl = np.clip(idx, 0, len(e) - 1)
    ok &= frame_ts <= e[cl]
    out[ok] = v[cl][ok]
    return out


def frame_eval(eval_path: str, gt: pd.DataFrame):
    df = pd.read_parquet(eval_path)
    df = df[df["split"] == "test"]
    gt_all, pr_all, gf_all = [], [], []
    for uid, g in df.groupby("userId"):
        gt_u = gt[gt["userId"] == uid]
        if len(gt_u) == 0 or len(g) == 0:
            continue
        t0 = int(max(g["ts_start"].min(), gt_u["started_at"].min()))
        t1 = int(min(g["ts_end"].max(), gt_u["finished_at"].max()))
        if t1 <= t0:
            continue
        frames = np.arange(t0, t1, RES_MS, dtype=np.int64)
        gt_f = _assign(frames, gt_u["started_at"].values, gt_u["finished_at"].values, gt_u["mode_tmd"].values)
        pr_f = _assign(frames, g["ts_start"].values, g["ts_end"].values, g["predicted_class"].values)
        gf_f = _assign(frames, g["ts_start"].values, g["ts_end"].values, g["gps_frac"].values)
        m = (gt_f != None) & (pr_f != None)  # noqa: E711
        gt_all.append(gt_f[m]); pr_all.append(pr_f[m]); gf_all.append(gf_f[m])
    gt_a = np.concatenate(gt_all); pr_a = np.concatenate(pr_all); gf_a = np.concatenate(gf_all)
    # restringi alle 5 classi (Bike/altro fuori dall'intersezione)
    keep = np.isin(gt_a, CLASSES)
    gt_a, pr_a, gf_a = gt_a[keep], pr_a[keep], gf_a[keep]

    def f1(g, p):
        return f1_score(g, p, labels=CLASSES, average="macro", zero_division=0)

    pres = gf_a.astype(float) > 0.5
    return {
        "frames": len(gt_a),
        "f1_all": f1(gt_a, pr_a),
        "acc_all": float((gt_a == pr_a).mean()),
        "frames_present": int(pres.sum()),
        "f1_present": f1(gt_a[pres], pr_a[pres]) if pres.sum() else float("nan"),
    }


def main():
    gt = pd.read_parquet(ROOT / "data/raw_freeze/labels.parquet")
    gt = gt[gt["mode_tmd"].isin(CLASSES)].copy()
    evals = list(zip(["120s", "60s", "30s"], sys.argv[1:4]))
    print(f"GT MotionTag: {len(gt):,} segmenti (5 classi) | frame 1 s\n")
    print(f"{'win':>5} {'frames':>9} {'frame-F1 ALL':>13} {'acc ALL':>9} {'frame-F1 GPS-pres':>18}")
    for win, path in evals:
        r = frame_eval(path, gt)
        print(f"{win:>5} {r['frames']:>9,} {r['f1_all']:>13.4f} {r['acc_all']:>9.4f} {r['f1_present']:>18.4f}")


if __name__ == "__main__":
    main()
