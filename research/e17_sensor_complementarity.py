"""
e17_sensor_complementarity.py — complementarità GPS↔IMU (Fase 1c · task E17).

Scopo:    GPS (B/C/D) e IMU (A) portano informazione COMPLEMENTARE o RIDONDANTE? Giustifica
          la fusione multi-sensore e racconta la DEGRADAZIONE GRACEFUL: quando il GPS manca
          (E7 MNAR), resta solo l'IMU → quanto regge? Quale sensore serve a quale classe?
Metodo:   RF 200 cross-val (StratifiedGroupKFold per sessione, no-leakage E23) su GT GPS-present,
          3 viste: IMU-only (A) · GPS-only (B/C/D) · both. acc + macro-F1 + F1 per-classe.
          Complementarità = acc(both) − max(acc(A), acc(GPS)).
Input:    data/v2/features_trento.parquet (label GT + 163 feat + session_id + gps_frac)
Output:   research/figures/e17_sensor_sets.{png,pdf} (acc/F1 per set) · e17_perclass_sensor.{png,pdf}
          research/figures/e17_complementarity.csv
Alimenta: thesis/eda.md (E17)
Sez.tesi: 4.5 / 5 / 6.5 (degradazione GPS)

Lettura: se both > max(singoli) → complementari (fusione giustificata); l'IMU-only è il floor
         operativo quando il GPS fallisce. Per-classe: IMU↔gait (Walk), GPS↔velocità/infra (Car/Bus/Train).
Run: /opt/miniconda3/envs/tmd/bin/python research/e17_sensor_complementarity.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
SEED = 0
CLASSES = ["Still", "Walk", "Bus", "Car", "Train"]


def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()


def cv_pred(X, y, groups):
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(clf, X, y, cv=sgkf.split(X, y, groups), n_jobs=-1)


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    g = df[(df.gps_frac > 0.5) & df.label.notna() & df.label.isin(CLASSES)].copy().reset_index(drop=True)
    A = [c for c in g.columns if c[:2] == "A_"]
    GPS = [c for c in g.columns if c[:2] in ("B_", "C_", "D_")]
    le = LabelEncoder(); y = le.fit_transform(g.label)          # numpy ints (Arrow-safe)
    y_str = np.asarray(g.label)
    groups = g.session_id.to_numpy()
    print("=" * 70); print("E17 — complementarità GPS↔IMU"); print("=" * 70)
    print(f"finestre GT GPS-present (5 classi): {len(g)} | IMU(A) {len(A)} feat | GPS(B/C/D) {len(GPS)} feat")
    print("(session-grouped CV, no-leakage E23)")

    sets = {"IMU-only (A)": A, "GPS-only (B/C/D)": GPS, "both (A+B/C/D)": A + GPS}
    preds, rows, perclass = {}, [], {}
    for name, cols in sets.items():
        X = SimpleImputer(strategy="median").fit_transform(g[cols])
        p = le.inverse_transform(cv_pred(X, y, groups))
        preds[name] = p
        acc = accuracy_score(y_str, p); mf1 = f1_score(y_str, p, average="macro")
        rows.append({"set": name, "n_feat": len(cols), "accuracy": acc, "macro_f1": mf1})
        perclass[name] = f1_score(y_str, p, labels=CLASSES, average=None)
        print(f"  {name:18s} ({len(cols):3d} feat): acc {acc:.3f} | macro-F1 {mf1:.3f}")
    res = pd.DataFrame(rows).set_index("set")
    pc = pd.DataFrame(perclass, index=CLASSES)
    res.round(3).to_csv(FIG / "e17_complementarity.csv")

    compl = res.loc["both (A+B/C/D)", "accuracy"] - max(res.loc["IMU-only (A)", "accuracy"], res.loc["GPS-only (B/C/D)", "accuracy"])
    print(f"\n🔑 complementarità = acc(both) − max(singoli) = +{100*compl:.1f} pt "
          f"→ {'COMPLEMENTARI (fusione giustificata)' if compl > 0.005 else 'poca complementarità (quasi ridondanti)'}")
    print(f"🔑 floor operativo quando il GPS fallisce (IMU-only): acc {res.loc['IMU-only (A)','accuracy']:.3f}, macro-F1 {res.loc['IMU-only (A)','macro_f1']:.3f}")
    print("\nF1 per-classe per sensore (chi serve a chi):")
    print(pc.round(2).to_string())
    # delta per-classe IMU vs GPS
    pc["GPS−IMU"] = pc["GPS-only (B/C/D)"] - pc["IMU-only (A)"]
    print("\nper-classe GPS−IMU (>0 ⇒ il GPS domina; <0 ⇒ l'IMU domina):")
    print(pc["GPS−IMU"].round(2).to_string())

    # ── FIG 1: acc/F1 per set ──
    x = np.arange(len(res)); w = 0.38
    plt.figure(figsize=(7.5, 4.4))
    plt.bar(x - w/2, res.accuracy, w, label="accuracy", color="tab:blue")
    plt.bar(x + w/2, res.macro_f1, w, label="macro-F1", color="tab:gray")
    plt.xticks(x, res.index, fontsize=9); plt.ylim(0, 1.0); plt.ylabel("score")
    plt.title(f"Sensor complementarity: both > each (gain +{100*compl:.1f}pt)\nIMU-only = operating floor when GPS fails")
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y")
    for i, s in enumerate(res.index):
        plt.text(i, res.accuracy[s] + 0.02, f"{res.accuracy[s]:.2f}", ha="center", fontsize=8)
    savefig("e17_sensor_sets")

    # ── FIG 2: F1 per-classe IMU vs GPS ──
    x = np.arange(len(CLASSES)); w = 0.38
    plt.figure(figsize=(8, 4.4))
    plt.bar(x - w/2, pc["IMU-only (A)"].values, w, label="IMU-only (A)", color="tab:blue")
    plt.bar(x + w/2, pc["GPS-only (B/C/D)"].values, w, label="GPS-only (B/C/D)", color="tab:orange")
    plt.xticks(x, CLASSES); plt.ylim(0, 1.0); plt.ylabel("per-class F1")
    plt.title("Which sensor for which class (IMU↔gait, GPS↔speed/infrastructure)")
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y")
    savefig("e17_perclass_sensor")

    print("\nfigure → e17_sensor_sets · e17_perclass_sensor | tabella → e17_complementarity.csv")


if __name__ == "__main__":
    main()
