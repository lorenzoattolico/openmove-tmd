"""
e4_sessions.py — EDA livello SESSIONE sul freeze (Fase 1c · task E4).

Scopo:    caratterizzare le sessioni R1 e il FENOMENO GUASTO-GPS (Cap.3.2):
          - copertura GPS per-sessione -> BIMODALE (presente vs assente), non graduale;
          - sessioni moving-GT-ma-GPS-assente, diffuse su molti utenti (ricalcola "20/46");
          - imu_only = recuperate dall'IMU-backbone (R1).
Input:    data/v2/features_trento_full.parquet  (finestre 120s aggregate per session_id; offline)
Output:   research/figures/e4_session_gpsfrac.{png,pdf}  (bimodalità copertura GPS per-sessione)
          research/figures/e4_user_strata.{png,pdf}      (conteggi finestre per-utente nei 3 strati GPS)
Alimenta: thesis/eda.md §2 (E4)
Sez.tesi: 3.2 Il fenomeno del guasto GPS (figura headline)

Run: /opt/miniconda3/envs/tmd/bin/python research/e4_sessions.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "v2" / "features_trento_full.parquet"
FIG = ROOT / "thesis" / "figures"
PRESENT = 0.5   # soglia gps_frac per "GPS presente" (dominio operativo)


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main() -> None:
    if not FEAT.exists():
        sys.exit(f"Manca {FEAT} — eseguito run_pipeline --all-features?")
    df = pd.read_parquet(FEAT, columns=["session_id", "userId", "sess_type",
                                        "gps_frac", "ts_start", "ts_end", "label"])
    df["moving"] = df.label.notna() & (df.label != "Still")

    g = df.groupby("session_id")
    sess = g.agg(userId=("userId", "first"), sess_type=("sess_type", "first"),
                 nwin=("gps_frac", "size"), mean_gf=("gps_frac", "mean"),
                 t0=("ts_start", "min"), t1=("ts_end", "max"),
                 has_moving=("moving", "any"),
                 has_gt=("label", lambda s: s.notna().any())).reset_index()
    sess["dur_min"] = (sess.t1 - sess.t0) / 60000.0
    sess["gps_present"] = sess.mean_gf > PRESENT

    n = len(sess)
    st = sess.sess_type.value_counts().to_dict()
    zero = (sess.mean_gf < 0.1).mean()
    full = (sess.mean_gf > 0.9).mean()
    mid = ((sess.mean_gf >= 0.1) & (sess.mean_gf <= 0.9)).mean()
    mov_abs = sess[sess.has_moving & ~sess.gps_present]
    imu_only_gf = sess.loc[sess.sess_type == "imu_only", "mean_gf"].mean()

    # ── riepilogo (→ thesis/eda.md) ─────────────────────────────────────────────
    print("=" * 64); print("E4 — EDA SESSIONI (freeze)"); print("=" * 64)
    print(f"sessioni: {n:,} | utenti: {sess.userId.nunique()} | sess_type: {st}")
    print(f"durata sessione (min)  mediana {sess.dur_min.median():.1f}  p90 {sess.dur_min.quantile(.9):.1f}")
    print(f"finestre/sessione  mediana {sess.nwin.median():.0f}")
    print(f"BIMODALITÀ copertura GPS per-sessione:  ~zero (<0.1) {100*zero:.0f}%  |  "
          f"~piena (>0.9) {100*full:.0f}%  |  intermedie (0.1–0.9) {100*mid:.0f}%")
    print(f"  (imu_only mean gps_frac = {imu_only_gf:.3f}  → atteso ~0: niente GPS, backbone IMU/R1)")
    print(f"MOVING-GT-ma-GPS-assente: {len(mov_abs)} sessioni su {int(mov_abs.userId.nunique())} utenti "
          f"(doc storico: 65 sess / 20–46 utenti)")
    print(f"R1: imu_only {st.get('imu_only',0)} sessioni recuperate dall'IMU-backbone "
          f"({100*st.get('imu_only',0)/n:.0f}% delle sessioni)")

    # ── figure (EN) ───────────────────────────────────────────────────────────
    # 1) bimodalità copertura GPS per-sessione (tracking vs imu_only)
    plt.figure(figsize=(6.5, 4))
    bins = np.linspace(0, 1.0, 26)
    tr = sess.loc[sess.sess_type == "tracking", "mean_gf"].clip(upper=1.0)
    io = sess.loc[sess.sess_type == "imu_only", "mean_gf"].clip(upper=1.0)
    plt.hist([io, tr], bins=bins, stacked=True, label=["imu_only (no GPS)", "tracking (GPS+IMU)"],
             color=["tab:red", "tab:blue"])
    plt.axvline(PRESENT, ls="--", c="grey", label="0.5 (present)")
    plt.xlabel("per-session mean GPS coverage (gps_frac)"); plt.ylabel("sessions")
    plt.title("Per-session GPS coverage is bimodal (present vs absent)")
    plt.legend(fontsize=8); plt.grid(alpha=.3)
    savefig("e4_session_gpsfrac")

    # 2) conteggi FINESTRE per-utente nei 3 strati GPS (assoluti) — massa no-GPS + deficit within-user/diffuso
    #    colori canonici (allineati e5): absent=red, sparse=orange, present=blue
    puw = (df.assign(stratum=np.select([df.gps_frac == 0, df.gps_frac <= PRESENT],
                                       ["absent", "sparse"], default="present"))
             .pivot_table(index="userId", columns="stratum", values="gps_frac",
                          aggfunc="size", fill_value=0))
    for c in ("present", "sparse", "absent"):
        if c not in puw.columns:
            puw[c] = 0
    puw = puw[["present", "sparse", "absent"]]
    n_mix = int(((puw["present"] > 0) & ((puw["absent"] + puw["sparse"]) > 0)).sum())
    n_any = int(((puw["absent"] + puw["sparse"]) > 0).sum())
    print(f"PER-UTENTE (finestre): {n_mix}/{len(puw)} utenti con MIX present+no-GPS | "
          f"{n_any}/{len(puw)} con ≥1 finestra no-GPS  → deficit within-user/per-finestra, diffuso")

    puw = puw.sort_values(["present", "sparse", "absent"], ascending=False)
    cols = [("present", "tab:blue", r"GPS-present ($>0.5$)"),
            ("sparse", "tab:orange", r"GPS-sparse ($0$–$0.5$]"),
            ("absent", "tab:red", r"GPS-absent ($=0$)")]
    x = np.arange(len(puw))
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bottom = np.zeros(len(puw))
    for c, color, lab in cols:
        ax.bar(x, puw[c].values, bottom=bottom, width=0.9, color=color, label=lab)
        bottom += puw[c].values
    # separatori verticali: i confini delle tre zone indotte dal sort present→sparse→absent
    p, s = puw["present"].values, puw["sparse"].values
    x_present_end = int(np.where(p > 0)[0].max())              # ultimo utente con present>0
    x_sparse_end = int(np.where((p == 0) & (s > 0))[0].max())  # ultimo no-present con sparse>0
    for xb in (x_present_end + 0.5, x_sparse_end + 0.5):
        ax.axvline(xb, color="k", ls="--", lw=1, alpha=.65)
    tr = ax.get_xaxis_transform()  # x in coord-dati, y in coord-assi
    ax.text((x_present_end + 0.5 + x_sparse_end + 0.5) / 2, 0.78, "no\nGPS-present",
            transform=tr, ha="center", va="top", fontsize=7.5, color="0.25")
    ax.text((x_sparse_end + 0.5 + len(puw)) / 2, 0.78, "absent-\nonly",
            transform=tr, ha="center", va="top", fontsize=7.5, color="0.25")
    ax.set_xlabel("user (sorted by GPS-present windows)"); ax.set_ylabel("windows")
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.grid(alpha=.3, axis="y")
    savefig("e4_user_strata")

    print(f"\nfigure → research/figures/e4_{{session_gpsfrac,user_strata}}.png|pdf")


if __name__ == "__main__":
    main()
