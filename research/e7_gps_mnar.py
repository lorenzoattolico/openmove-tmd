"""
e7_gps_mnar.py — il guasto-GPS è MNAR? (Fase 1c · task E7).

Scopo:    stabilire se l'assenza GPS è SISTEMATICA (legata a modo/device/ora/intensità-moto)
          e non casuale → Missing-Not-At-Random (Barnett&Onnela): non eliminabile per imputazione,
          va CARATTERIZZATA (giustifica il framing "confine operativo" + reporting GPS-stratificato).
Input:    data/v2/features_trento_full.parquet  +  research/figures/e3_user_overview.csv (device/utente)
Output:   research/figures/e7_absent_by_mode.{png,pdf}   (GPS-assente % per modo GT)
          research/figures/e7_absent_by_hour.{png,pdf}   (pattern orario)
Alimenta: thesis/eda.md §2/§3 (E7)
Sez.tesi: 3.2 guasto GPS (MNAR)

Run: /opt/miniconda3/envs/tmd/bin/python research/e7_gps_mnar.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "v2" / "features_trento_full.parquet"
OV = ROOT / "thesis" / "figures" / "e3_user_overview.csv"
FIG = ROOT / "thesis" / "figures"
MOTION = "A_lin_mag_mean"   # intensità moto IMU, disponibile anche senza GPS


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    cols = ["gps_frac", "label", "userId", "ts_start"] + ([MOTION] if MOTION else [])
    df = pd.read_parquet(FULL, columns=cols)
    df["absent"] = pd.to_numeric(df.gps_frac, errors="coerce") <= 0.5
    dev = pd.read_csv(OV)[["userId", "device"]]
    df = df.merge(dev, on="userId", how="left")
    df["hour"] = pd.to_datetime(df.ts_start, unit="ms", utc=True).dt.tz_convert("Europe/Rome").dt.hour

    base = 100 * df.absent.mean()
    print("=" * 64); print("E7 — GPS-assente: MNAR?  (assente = gps_frac<=0.5)"); print("=" * 64)
    print(f"assente complessivo: {base:.1f}%  (n={len(df):,})")

    # 1) per MODO GT
    gt = df[df.label.notna()]
    by_mode = (100 * gt.groupby("label").absent.mean()).sort_values(ascending=False)
    n_mode = gt.groupby("label").size()
    print("\nassente % per MODO GT (n):")
    for m in by_mode.index:
        print(f"  {m:7s} {by_mode[m]:5.1f}%  (n={n_mode[m]:,})")

    # 2) per DEVICE
    by_dev = 100 * df.groupby("device").absent.mean()
    n_dev = df.groupby("device").size()
    print("\nassente % per DEVICE:")
    for d in by_dev.index:
        print(f"  {d:8s} {by_dev[d]:5.1f}%  (n={int(n_dev[d]):,})")
    # DEEPEN: device × modo → effetto PIATTAFORMA (Android>>iOS dentro i modi mobili) vs confound
    piv = gt.pivot_table(index="label", columns="device", values="absent", aggfunc="mean") * 100
    print("\nassente % per DEVICE × MODO (effetto piattaforma se Android>>iOS nei modi mobili):")
    print(piv.round(1).to_string())

    # 3) per ORA (giorno/notte)
    night = df[df.hour.isin([0,1,2,3,4,5,22,23])].absent.mean()*100
    day = df[~df.hour.isin([0,1,2,3,4,5,22,23])].absent.mean()*100
    print(f"\nassente % notte(22-6) {night:.1f}  vs giorno {day:.1f}")

    # 4) per INTENSITÀ MOTO (IMU, indipendente dal GPS)
    if MOTION in df.columns:
        m = df.dropna(subset=[MOTION])
        lo, hi = m[MOTION].quantile(.33), m[MOTION].quantile(.66)
        binned = pd.cut(m[MOTION], [-np.inf, lo, hi, np.inf], labels=["basso", "medio", "alto"])
        by_mot = 100 * m.groupby(binned, observed=True).absent.mean()
        print(f"\nassente % per intensità-moto IMU (terzili): {by_mot.round(1).to_dict()}")
        print(f"  motion medio: assente {m[m.absent][MOTION].mean():.3f} vs presente {m[~m.absent][MOTION].mean():.3f}")

    # 5) per-utente (range) — diffusione
    pu = df.groupby("userId").absent.mean()*100
    print(f"\nassente % per-utente: mediana {pu.median():.0f}  range [{pu.min():.0f}, {pu.max():.0f}]  (diffuso, vedi E4)")

    print("\n→ MNAR se varia sistematicamente per modo/device/ora/moto (non costante).")

    # ── figure (EN) ──
    plt.figure(figsize=(6, 4))
    plt.bar(by_mode.index, by_mode.values, color="tab:red")
    plt.axhline(base, ls="--", c="grey", label=f"overall {base:.0f}%")
    plt.ylabel("GPS-absent windows (%)"); plt.title("GPS absence by GT mode (MNAR)")
    plt.legend(); plt.grid(alpha=.3, axis="y"); savefig("e7_absent_by_mode")

    by_hour = 100 * df.groupby("hour").absent.mean()
    plt.figure(figsize=(7, 4))
    plt.plot(by_hour.index, by_hour.values, marker="o", ms=3)
    plt.axhline(base, ls="--", c="grey", label=f"overall {base:.0f}%")
    plt.xlabel("hour of day (Europe/Rome)"); plt.ylabel("GPS-absent windows (%)")
    plt.title("GPS absence by hour of day"); plt.legend(); plt.grid(alpha=.3)
    savefig("e7_absent_by_hour")

    # 3) device × modo — il finding chiave (effetto piattaforma Android) — con n per barra (C8)
    order = [m for m in ["Still", "Walk", "Bus", "Car", "Train", "Bike"] if m in piv.index]
    piv2 = piv.reindex(order)
    cnt2 = gt.pivot_table(index="label", columns="device", values="absent", aggfunc="size").reindex(order)
    ax = piv2.plot(kind="bar", figsize=(7.5, 4.2))
    for j, devcol in enumerate(piv2.columns):           # patches: colonna per colonna
        for i, mode in enumerate(piv2.index):
            r = ax.patches[j * len(piv2.index) + i]
            n = cnt2.loc[mode, devcol]
            if pd.notna(n):
                ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.2,
                        f"n={int(n):,}".replace(",", "."), ha="center", fontsize=6.5, color="dimgray", rotation=90)
    # label allineate al lessico Cap.3.2 (absent==0 vs "without usable GPS"<=0.5); niente title in-immagine (C8)
    ax.set_ylabel("Windows without usable GPS (%)"); ax.set_xlabel("Reference mode")
    ax.set_ylim(0, 122)
    plt.xticks(rotation=0); ax.grid(alpha=.3, axis="y"); ax.legend(title="device")
    savefig("e7_absent_device_mode")

    print(f"\nfigure → research/figures/e7_{{absent_by_mode,absent_by_hour,absent_device_mode}}.png|pdf")


if __name__ == "__main__":
    main()
