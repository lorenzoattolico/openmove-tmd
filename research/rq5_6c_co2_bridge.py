"""
research/rq5_6c_co2_bridge.py — RQ5.6c: dal modal-split all'errore di stima CO2 (use-case).

Scopo:    chiudere il loop della motivazione (TEN-T 2027 / SUMI: monitoraggio CO2 da modal-split). La
          metrica che conta per la PA non è l'F1 né il TVD delle quote, ma l'**errore sulla CO2 aggregata**.
          CO2 = Σ_modo (km percorsi nel modo × fattore di emissione gCO2/pkm). Un errore di modal-split si
          propaga alla CO2 *pesato dai km e dai fattori di emissione* (Car↔Train pesano molto: 170 vs 40).
          Domanda: dato il nostro modello reale, quanto sbaglia la CO2 aggregata? E la quantification la cura?
Metodo:   eval rolling-OOF canonico (GPS-present, dove la distanza GPS è affidabile). Distanza per finestra =
          B_dist_total_m (dal parquet full). CO2_GT = Σ dist×EF[label]; CO2_pred = Σ dist×EF[pred]. Errore
          relativo. + quantification sui km (inversione confusione km-pesata, split onesto). + sensitivity sui EF.
Input:    data/v2/processed/eval_trento_20260612_202507.parquet · data/v2/features_trento_full.parquet
Output:   research/figures/rq5_6c_co2_bridge.{png,pdf} + riepilogo stdout.
Alimenta: thesis/results.md §RQ5 (CO2) + Cap.1 motivazione. Sez.tesi: 1.x / 6.6 / 7.

⚠ I fattori di emissione sono valori di RIFERIMENTO (gCO2e/passenger-km, ordine EEA/DEFRA), non misurati a
   Trento → riportati con sensitivity. La CO2 è un *indicatore d'uso*, non una misura certificata.
Run: python research/rq5_6c_co2_bridge.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data/v2/processed/eval_trento_20260612_202507.parquet"
FULL = ROOT / "data/v2/features_trento_full.parquet"
FIG = ROOT / "research/figures"
MODES = ["Still", "Walk", "Bus", "Car", "Train"]
# fattori di emissione gCO2e / passenger-km (riferimento; sensitivity sotto)
EF = {"Still": 0.0, "Walk": 0.0, "Bus": 100.0, "Car": 170.0, "Train": 40.0}
EF_LO = {"Still": 0, "Walk": 0, "Bus": 70, "Car": 120, "Train": 25}
EF_HI = {"Still": 0, "Walk": 0, "Bus": 120, "Car": 200, "Train": 60}


def co2(df, mode_col, ef):
    return float((df["km"] * df[mode_col].map(ef)).sum())


def main():
    ev = pd.read_parquet(EVAL)
    pc = "predicted_class_smooth" if "predicted_class_smooth" in ev else "predicted_class"
    ev = ev[(ev.gps_frac > 0.5) & ev.label.isin(MODES)].copy()
    full = pd.read_parquet(FULL)[["session_id", "ts_start", "B_dist_total_m"]]
    ev = ev.merge(full, on=["session_id", "ts_start"], how="left")
    ev["km"] = ev["B_dist_total_m"].fillna(0) / 1000.0
    ev = ev[ev.km >= 0]
    print("=" * 66); print("RQ5.6c — bridge modal-split → errore di stima CO2 (GPS-present)"); print("=" * 66)
    print(f"finestre = {len(ev)} | km totali = {ev.km.sum():.0f} | EF gCO2/pkm = {EF}\n")

    co2_gt = co2(ev, "label", EF)
    co2_pred = co2(ev, pc, EF)
    err = (co2_pred - co2_gt) / co2_gt * 100
    print(f"CO2 vera (GT)   = {co2_gt/1000:.1f} kgCO2")
    print(f"CO2 predetta    = {co2_pred/1000:.1f} kgCO2   → errore relativo NAIVE: {err:+.1f}%")

    # contributo per-modo (km e CO2)
    print(f"\n  {'modo':<7}{'km GT':>9}{'km pred':>9}{'EF':>6}{'CO2_GT%':>9}{'CO2_pr%':>9}")
    for m in MODES:
        kg = ev[ev.label == m].km.sum(); kp = ev[ev[pc] == m].km.sum()
        print(f"  {m:<7}{kg:>9.0f}{kp:>9.0f}{EF[m]:>6.0f}{100*kg*EF[m]/co2_gt:>9.1f}{100*kp*EF[m]/co2_pred:>9.1f}")

    # ── quantification sui km (confusione km-pesata, split onesto 10 seed) ──
    errs_naive, errs_corr = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(ev)); h = len(ev) // 2
        cal, tst = ev.iloc[idx[:h]], ev.iloc[idx[h:]]
        # M[i,j] = frazione dei km veri-i predetti-j (sul CAL)
        M = np.zeros((len(MODES), len(MODES)))
        for i, ci in enumerate(MODES):
            sub = cal[cal.label == ci]; tot = sub.km.sum()
            if tot > 0:
                for j, cj in enumerate(MODES):
                    M[i, j] = sub[sub[pc] == cj].km.sum() / tot
            else:
                M[i, i] = 1
        km_pred = np.array([tst[tst[pc] == cj].km.sum() for cj in MODES])
        km_corr, *_ = np.linalg.lstsq(M.T, km_pred, rcond=None)
        km_corr = np.clip(km_corr, 0, None)
        km_true = np.array([tst[tst.label == ci].km.sum() for ci in MODES])
        efv = np.array([EF[m] for m in MODES])
        c_gt = (km_true * efv).sum()
        errs_naive.append((km_pred @ efv - c_gt) / c_gt * 100)
        errs_corr.append((km_corr @ efv - c_gt) / c_gt * 100)
    print(f"\n  Errore CO2 (split onesto, 10 seed):")
    print(f"    NAIVE         : {np.mean(errs_naive):+.1f}% ± {np.std(errs_naive):.1f}")
    print(f"    quantification: {np.mean(errs_corr):+.1f}% ± {np.std(errs_corr):.1f}")

    # ── sensitivity sui fattori di emissione ──
    print(f"\n  Sensitivity errore-CO2-naive ai fattori di emissione:")
    for nm, ef in [("LOW (Car120/Bus70/Train25)", EF_LO), ("REF (Car170/Bus100/Train40)", EF),
                   ("HIGH (Car200/Bus120/Train60)", EF_HI)]:
        e = (co2(ev, pc, ef) - co2(ev, "label", ef)) / co2(ev, "label", ef) * 100
        print(f"    {nm:<32}: {e:+.1f}%")

    # ── figura ──
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    kmg = [ev[ev.label == m].km.sum() for m in MODES]; kmp = [ev[ev[pc] == m].km.sum() for m in MODES]
    x = np.arange(len(MODES)); w = 0.38
    ax[0].bar(x - w/2, kmg, w, label="GT", color="tab:green"); ax[0].bar(x + w/2, kmp, w, label="pred", color="tab:blue")
    ax[0].set_xticks(x); ax[0].set_xticklabels(MODES); ax[0].set_ylabel("km"); ax[0].set_title("Distance by mode (GT vs predicted)"); ax[0].legend()
    co2g = [ev[ev.label == m].km.sum()*EF[m]/1000 for m in MODES]; co2p = [ev[ev[pc] == m].km.sum()*EF[m]/1000 for m in MODES]
    ax[1].bar(x - w/2, co2g, w, label="GT", color="tab:green"); ax[1].bar(x + w/2, co2p, w, label="pred", color="tab:blue")
    ax[1].set_xticks(x); ax[1].set_xticklabels(MODES); ax[1].set_ylabel("kgCO2e"); ax[1].set_title("CO$_2$e by mode (GT vs predicted)"); ax[1].legend()
    # niente suptitle in-immagine (C8) e niente % nel titolo-pannello (l'errore headline -8.8% sta nel testo/caption)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"rq5_6c_co2_bridge.{ext}", dpi=150, bbox_inches="tight")
    print("\nfigura → rq5_6c_co2_bridge.{png,pdf}")


if __name__ == "__main__":
    main()
