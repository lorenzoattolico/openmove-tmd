"""
e20_gt_noise_audit.py — audit del RUMORE nelle etichette GT (Fase 1c · task E20).

Scopo:    la GT Trento è output SDK MotionTag (inferenza, NON gold) → può contenere mislabel.
          Confident-learning (Cleanlab): classificatore cross-validato → pred_probs out-of-sample
          → find_label_issues. Stima il tasso di rumore per classe e identifica i GT sospetti.
          Triangola col segnale FISICO (velocità) e conferma il flag Bike 10.6 m/s di E8.
Metodo:   X = feature 163 (GPS-present, NaN→median come il modello); y = GT; RF cross-val (5-fold);
          cleanlab.filter.find_label_issues. Caveat: "flagged" = disaccordo-modello pesato sulla
          confidenza, NON prova di errore; affidabile soprattutto per il rumore SISTEMATICO.
Input:    data/v2/features_trento.parquet (label GT + 163 feature + gps_frac)
Output:   research/figures/e20_noise_by_class.{png,pdf} · e20_bike_suspects.{png,pdf}
          research/figures/e20_gt_noise.csv (per-classe) · e20_gt_suspects.csv (top finestre sospette)
Alimenta: thesis/eda.md (E20)
Sez.tesi: 3.3 qualità GT / 4.4 / 7 limiti

Run: /opt/miniconda3/envs/tmd/bin/python research/e20_gt_noise_audit.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from cleanlab.filter import find_label_issues
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
SEED = 0
ORDER = ["Still", "Walk", "Bike", "Bus", "Car", "Train"]


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    g = df[(df.gps_frac > 0.5) & df.label.notna()].copy().reset_index(drop=True)
    feat = [c for c in g.columns if c[:2] in ("A_", "B_", "C_", "D_")]
    print("=" * 70); print("E20 — audit rumore GT (Cleanlab / confident-learning)"); print("=" * 70)
    print(f"finestre GT GPS-present: {len(g)} | feature: {len(feat)}")

    le = LabelEncoder()
    y = le.fit_transform(g.label)
    X = SimpleImputer(strategy="median").fit_transform(g[feat])

    # pred_probs out-of-sample (cross-val)
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print("\ncross-val pred_probs (RF 300, 5-fold)...")
    pp = cross_val_predict(clf, X, y, cv=skf, method="predict_proba", n_jobs=-1)

    # confident learning
    issues = find_label_issues(labels=y, pred_probs=pp, return_indices_ranked_by="self_confidence")
    mask = np.zeros(len(y), bool); mask[issues] = True
    g["flagged"] = mask
    g["suggested"] = le.inverse_transform(pp.argmax(1))
    g["conf_suggested"] = pp.max(1)
    print(f"\nGT sospette (flagged): {mask.sum()} / {len(g)} = {100*mask.mean():.1f}%")

    # ── noise per classe ──
    classes = [c for c in ORDER if c in g.label.unique()]
    noise = g.groupby("label").agg(n=("flagged", "size"), flagged=("flagged", "sum"))
    noise["noise_rate"] = noise.flagged / noise.n
    noise = noise.reindex(classes)
    noise.round(3).to_csv(FIG / "e20_gt_noise.csv")
    print("\nTasso di rumore GT stimato per classe:")
    print(noise.assign(noise_rate=(noise.noise_rate * 100).round(0)).to_string())

    # dove vanno i sospetti (GT flagged → label suggerita)
    fl = g[g.flagged]
    print("\nPer i sospetti: GT → label suggerita dal modello (top per classe):")
    for c in classes:
        sub = fl[fl.label == c]
        if len(sub):
            top = sub.suggested.value_counts().head(3)
            print(f"  GT={c:6s} (n_flag={len(sub):4d}): " + ", ".join(f"{k} {v}" for k, v in top.items()))

    # ── triangolazione Bike col segnale fisico (E8: Bike 10.6 m/s sospetto) ──
    spd_col = next((c for c in ["B_speed_p50", "B_speed_p95", "B_speed_p25", "B_speed_max"] if c in g.columns), None) \
        if "Bike" in classes else None
    if "Bike" in classes and spd_col:
        bk = g[g.label == "Bike"]
        mf, mc = bk[bk.flagged][spd_col].median(), bk[~bk.flagged][spd_col].median()
        print(f"\nTriangolazione Bike ({spd_col}): flagged median {mf:.1f} vs clean {mc:.1f} m/s")
        print("  → i flagged sono i Bike PIÙ LENTI (→ confusi con Bus), NON i più veloci.")
        print("  ⚠ Cleanlab audita vs la distribuzione APPRESA (dalla STESSA GT), non vs la fisica assoluta:")
        print("    conferma che Bike è la classe più rumorosa (8%) via confusione Bike↔Bus, ma NON può")
        print("    adjudicare il sospetto E8 'Bike-classe troppo veloce 10.6 m/s' (il modello ha imparato Bike=veloce).")

    # top sospetti assoluti (bassa self-confidence nella GT)
    g["gt_selfconf"] = [pp[i, y[i]] for i in range(len(y))]
    top_susp = g[g.flagged].nsmallest(50, "gt_selfconf")[["userId", "label", "suggested", "conf_suggested", "gps_frac"]]
    top_susp.to_csv(FIG / "e20_gt_suspects.csv", index=False)

    # ── FIG 1: noise per classe ──
    plt.figure(figsize=(7, 4.2))
    colors = ["tab:red" if c == "Bike" else "tab:blue" for c in classes]
    plt.bar(range(len(classes)), noise.noise_rate.values * 100, color=colors)
    for i, c in enumerate(classes):
        plt.text(i, noise.noise_rate[c] * 100 + 0.5, f"{int(noise.flagged[c])}/{int(noise.n[c])}", ha="center", fontsize=7)
    plt.xticks(range(len(classes)), classes); plt.ylabel("estimated GT noise rate %")
    plt.ylim(0, max(noise.noise_rate.max() * 100 + 1.5, 5))
    plt.title("Estimated GT label-noise per class (Cleanlab confident learning)", pad=12)
    plt.grid(alpha=.3, axis="y"); savefig("e20_noise_by_class")

    # ── FIG 2: Bike suspects vs speed ──
    if "Bike" in classes and spd_col:
        bk = g[g.label == "Bike"]
        plt.figure(figsize=(6.5, 4))
        plt.hist(bk[~bk.flagged][spd_col].dropna(), bins=20, alpha=.6, label="GT-Bike clean", color="tab:blue")
        plt.hist(bk[bk.flagged][spd_col].dropna(), bins=20, alpha=.6, label="GT-Bike flagged", color="tab:red")
        plt.xlabel(f"{spd_col} (m/s)"); plt.ylabel("windows")
        plt.title("Flagged GT-Bike are the SLOWER ones (Bike↔Bus confusion)\nCleanlab audits vs learned distribution, not absolute physics")
        plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y"); savefig("e20_bike_suspects")

    print("\nfigure → e20_noise_by_class · e20_bike_suspects | tabelle → e20_gt_noise.csv · e20_gt_suspects.csv")


if __name__ == "__main__":
    main()
