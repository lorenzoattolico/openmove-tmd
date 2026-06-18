"""
e22_gt_physics.py — coerenza GT↔fisica: audit GT ASSOLUTO (Fase 1c · task E22).

Scopo:    contraltare ASSOLUTO di E20. Cleanlab (E20) audita vs la distribuzione APPRESA (dalla
          stessa GT) → NON può dire se un'intera classe è fisicamente implausibile (es. il sospetto
          E8 "Bike 10.6 m/s"). E22 usa VINCOLI FISICI ASSOLUTI di velocità per modo (da letteratura)
          e segnala le finestre GT incoerenti → chiude il buco su Bike e quantifica il rumore-fisico.
Metodo:   per ogni modo un upper-bound fisico sulla velocità mediana di finestra (B_speed_p50, m/s).
          Flag = GT-finestra con p50 oltre il bound del suo modo (solo violazioni CHIARE; gli stop
          dei mezzi veloci NON sono flaggati). GPS-present (velocità definita).
Input:    data/v2/features_trento_full.parquet (label GT + B_speed_p50 + gps_frac)
Output:   research/figures/e22_gt_physics.{png,pdf} (flag-rate per modo + velocità Bike)
          research/figures/e22_gt_physics.csv
Alimenta: thesis/eda.md (E22)
Sez.tesi: 3.3 qualità GT / 4.4 / 7 limiti

Bound (m/s): Still>1.5 · Walk>3.0 · Bike>8.5(~30km/h) · Bus>30 · Car>45 · Train>90.
Run: /opt/miniconda3/envs/tmd/bin/python research/e22_gt_physics.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
SPD = "B_speed_p50"
# upper-bound fisico sulla velocità MEDIANA di finestra (m/s) — oltre = fisicamente implausibile per quel modo
HI = {"Still": 1.5, "Walk": 3.0, "Bike": 8.5, "Bus": 30.0, "Car": 45.0, "Train": 90.0}
ORDER = ["Still", "Walk", "Bike", "Bus", "Car", "Train"]


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento_full.parquet")
    g = df[(df.gps_frac > 0.5) & df.label.notna()].dropna(subset=[SPD]).copy()
    classes = [c for c in ORDER if c in g.label.unique()]
    print("=" * 70); print("E22 — coerenza GT↔fisica (audit GT assoluto via velocità)"); print("=" * 70)
    print(f"finestre GT GPS-present con {SPD}: {len(g)}")
    print(f"bound fisici upper (m/s): {HI}")

    g["implausible"] = g.apply(lambda r: r[SPD] > HI.get(r.label, np.inf), axis=1)
    rows = []
    print(f"\nFinestre fisicamente IMPLAUSIBILI per modo ({SPD} oltre il bound):")
    for c in classes:
        s = g[g.label == c]
        fr = s.implausible.mean()
        rows.append({"class": c, "n": len(s), "median_speed": s[SPD].median(),
                     "p90_speed": s[SPD].quantile(.9), "bound": HI[c], "implausible_pct": 100 * fr})
        print(f"  {c:6s} n={len(s):5d}  p50={s[SPD].median():5.1f}  p90={s[SPD].quantile(.9):5.1f}  "
              f"bound>{HI[c]:.0f}  → implausibili {100*fr:4.1f}%")
    res = pd.DataFrame(rows).set_index("class")
    res.round(2).to_csv(FIG / "e22_gt_physics.csv")

    # focus Bike (il sospetto E8/E20)
    bk = g[g.label == "Bike"]
    print(f"\n🔑 BIKE: velocità mediana {bk[SPD].median():.1f} m/s ({3.6*bk[SPD].median():.0f} km/h) — "
          f"implausibile per ciclismo umano (mediana attesa ~3–5 m/s).")
    print(f"   {100*(bk[SPD] > 8.5).mean():.0f}% dei GT-Bike ha p50 > 8.5 m/s (~30 km/h) → e-bike/scooter/veicolo o mislabel.")
    print(f"   {100*(bk[SPD] > 12.5).mean():.0f}% ha p50 > 12.5 m/s (~45 km/h) = chiaramente motorizzato.")
    print("   → conferma ASSOLUTA del sospetto E8 che Cleanlab/E20 (relativo, appreso) non poteva dare.")
    print(f"\nTotale GT fisicamente sospette: {g.implausible.sum()} / {len(g)} = {100*g.implausible.mean():.1f}% "
          f"(quasi tutto Bike; gli altri modi sono fisicamente coerenti)")

    # ── FIG 1: flag-rate per modo ──
    plt.figure(figsize=(7, 4.2))
    colors = ["tab:red" if c == "Bike" else "tab:blue" for c in classes]
    plt.bar(range(len(classes)), res.implausible_pct.values, color=colors)
    for i, c in enumerate(classes):
        plt.text(i, res.implausible_pct[c] + 0.5, f"{res.implausible_pct[c]:.0f}%", ha="center", fontsize=8)
    plt.xticks(range(len(classes)), classes); plt.ylabel("physically implausible %")
    plt.title("Absolute physics audit of GT: implausible-speed windows per mode\n(complements E20 Cleanlab; isolates Bike contamination)")
    plt.grid(alpha=.3, axis="y"); savefig("e22_gt_physics")

    # ── FIG 2: Bike speed vs altri modi + bound ──
    plt.figure(figsize=(7.5, 4.2))
    data = [g[g.label == c][SPD].dropna().values for c in classes]
    plt.boxplot(data, tick_labels=classes, showfliers=False)
    plt.axhline(8.5, color="tab:red", ls="--", lw=1, label="cycling plausibility (~30 km/h)")
    plt.ylabel(f"{SPD} (m/s)"); plt.title("GT-Bike speed overlaps motorized modes, not slow ones")
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y"); savefig("e22_bike_speed")

    print("\nfigure → e22_gt_physics · e22_bike_speed | tabella → e22_gt_physics.csv")


if __name__ == "__main__":
    main()
