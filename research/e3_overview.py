"""
e3_overview.py — EDA overview dati + lag GT↔sensori sul freeze (Fase 1a · task E3).

Scopo:    quadro d'insieme per-utente (GPS+IMU+label+device) e coorte usabile per l'eval GT;
          + DENSITA' TEMPORALE: label e attivita' GPS per giorno nella finestra -> dimensiona l'eval
          e mostra dove i dati sono densi (verifica empirica: lag GT vs declino di partecipazione).
Input:    data/raw_freeze/{gps/, imu/<uid>.parquet, labels.parquet}   (offline, no Mongo)
Output:   research/figures/e3_daily_activity.{png,pdf}  (label/giorno vs utenti-con-GPS/giorno)
          research/figures/e3_cohort.{png,pdf}      (coorti: GPS / IMU / label / tutti e 3)
          research/figures/e3_user_overview.csv     (tabella per-utente, appendice)
Alimenta: thesis/eda.md §1 (E3)
Sez.tesi: 3.1 Deployment (base utenti) / 3.4 riferimento MotionTag (lag GT)

IMU: conteggi via metadata parquet (no load 111M righe); device via primo batch (campione).
Run: /opt/miniconda3/envs/tmd/bin/python research/e3_overview.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_freeze"
FIG = ROOT / "thesis" / "figures"
TZ = "Europe/Rome"
WIN = pd.date_range("2026-05-19", "2026-06-08", freq="D", tz=TZ)   # giorni finestra freeze (IT)
SPLIT = 5.0


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def load_gps() -> pd.DataFrame:
    parts = sorted((RAW / "gps").glob("year_month=*/part.parquet"))
    df = pd.concat([pd.read_parquet(p, columns=["userId", "timestamp"]) for p in parts],
                   ignore_index=True)
    df["userId"] = df["userId"].astype(str)
    df["date"] = pd.to_datetime(df.timestamp, unit="ms", utc=True).dt.tz_convert(TZ).dt.normalize()
    return df


def imu_per_user() -> pd.DataFrame:
    rows = []
    for f in sorted((RAW / "imu").glob("*.parquet")):
        n = pq.read_metadata(f).num_rows                       # niente load
        batch = next(pq.ParquetFile(f).iter_batches(batch_size=5000,
                     columns=["acc_x", "acc_y", "acc_z"]))     # campione per device
        b = batch.to_pandas()
        accmag = np.sqrt(b.acc_x ** 2 + b.acc_y ** 2 + b.acc_z ** 2).median()
        rows.append({"userId": f.stem, "n_imu": n,
                     "device": "iOS" if accmag < SPLIT else "Android"})
    return pd.DataFrame(rows)


def main() -> None:
    gps = load_gps()
    lab = pd.read_parquet(RAW / "labels.parquet")
    lab["userId"] = lab["userId"].astype(str)
    lab["date"] = pd.to_datetime(lab.started_at, unit="ms", utc=True).dt.tz_convert(TZ).dt.normalize()
    imu = imu_per_user()

    # ── tabella per-utente ────────────────────────────────────────────────────
    g_cnt = gps.groupby("userId").size().rename("n_gps")
    g_days = gps.groupby("userId").date.nunique().rename("gps_days")
    l_cnt = lab.groupby("userId").size().rename("n_lab")
    ov = (imu.set_index("userId")
          .join(g_cnt, how="outer").join(g_days, how="outer").join(l_cnt, how="outer")
          .reset_index())
    for c in ["n_gps", "gps_days", "n_lab", "n_imu"]:
        ov[c] = ov[c].fillna(0).astype(int)
    ov["device"] = ov["device"].fillna("?")
    ov = ov.sort_values("n_imu", ascending=False)
    FIG.mkdir(parents=True, exist_ok=True)
    ov.to_csv(FIG / "e3_user_overview.csv", index=False)

    # ── coorti ────────────────────────────────────────────────────────────────
    U_gps = set(g_cnt.index); U_imu = set(imu.userId); U_lab = set(l_cnt.index)
    U_all = U_gps & U_imu & U_lab
    cohorts = {
        "IMU": len(U_imu), "GPS": len(U_gps), "label (GT)": len(U_lab),
        "GPS∩IMU∩GT": len(U_all),
    }

    # ── riepilogo (→ thesis/eda.md) + SANITY vs E1/E2/verifica ──────────────────
    print("=" * 64); print("E3 — OVERVIEW DATI + DENSITA' TEMPORALE (freeze)"); print("=" * 64)
    print(f"utenti unici (qualsiasi canale): {len(set(ov.userId))}")
    print(f"  con IMU: {len(U_imu)} (atteso 52) | con GPS: {len(U_gps)} (atteso 45) | con label: {len(U_lab)} (atteso 38)")
    print(f"  coorte eval-GT (GPS∩IMU∩GT): {len(U_all)} | IMU-ma-no-GPS: {len(U_imu-U_gps)} | IMU-ma-no-label: {len(U_imu-U_lab)}")
    print(f"device: iOS {int((imu.device=='iOS').sum())} / Android {int((imu.device=='Android').sum())} (atteso 25/27)")
    print(f"label totali: {len(lab)} (atteso 1083) | range started_at: {lab.date.min().date()} → {lab.date.max().date()}")
    # lag: ultimi 3 giorni finestra
    lab_day = lab.groupby("date").size().reindex(WIN, fill_value=0)
    gpsU_day = gps.groupby("date").userId.nunique().reindex(WIN, fill_value=0)
    half = 14  # 19 mag–1 giu (14 gg) vs 2–8 giu (7 gg)
    print(f"densità temporale: label/gg mediana {int(lab_day.median())}, utenti-GPS/gg mediana {int(gpsU_day.median())}")
    print(f"  label: prime 2 sett {int(lab_day.iloc[:half].sum())} vs ultima sett {int(lab_day.iloc[half:].sum())}; "
          f"ultimi 3 gg {[int(v) for v in lab_day.iloc[-3:]]}")
    print(f"  utenti-GPS attivi: prime 2 sett {int((gpsU_day.iloc[:half] > 0).sum())}/14 gg vs ultima {int((gpsU_day.iloc[half:] > 0).sum())}/7 gg")
    print("  → label e attività calano INSIEME = declino di partecipazione (NON puro lag GT); coda 6–8 giu sparsa")

    # ── figure (EN) ───────────────────────────────────────────────────────────
    # 1) lag GT: label/giorno (barre) vs utenti-con-GPS/giorno (linea, asse dx)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(WIN))
    ax1.bar(x, lab_day.values, color="tab:blue", label="GT label segments")
    ax1.set_ylabel("GT label segments / day", color="tab:blue")
    ax1.set_xticks(x[::2]); ax1.set_xticklabels([d.strftime("%m-%d") for d in WIN[::2]], rotation=45, fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, gpsU_day.values, color="tab:red", marker="o", ms=3, label="users with GPS")
    ax2.set_ylabel("users with GPS / day", color="tab:red")
    ax1.set_xlabel("day (freeze window, Europe/Rome)")
    # niente titolo in-immagine (C8): lo porta la caption LaTeX in Data.tex (fig:daily-activity)
    ax1.grid(alpha=.3)
    savefig("e3_daily_activity")

    # 2) coorti
    plt.figure(figsize=(6, 4))
    k = list(cohorts.keys()); v = [cohorts[i] for i in k]
    plt.bar(k, v, color=["tab:gray", "tab:green", "tab:blue", "tab:purple"])
    for i, val in enumerate(v):
        plt.text(i, val + 0.3, str(val), ha="center")
    plt.ylabel("users"); plt.title("User cohorts by available channel")
    plt.xticks(rotation=15, fontsize=9); plt.grid(alpha=.3, axis="y")
    savefig("e3_cohort")

    print(f"\ntabella → research/figures/e3_user_overview.csv")
    print(f"figure  → research/figures/e3_{{daily_activity,cohort}}.png|pdf")


if __name__ == "__main__":
    main()
