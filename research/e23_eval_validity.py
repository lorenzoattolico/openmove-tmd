"""
e23_eval_validity.py — validità del protocollo di valutazione (Fase 1c · task E23).

Scopo:    fissare COME riportare i numeri-modello. Dimostra empiricamente il LEAKAGE dello split
          casuale (finestre step 60s < lunghezza 120s → 50% overlap tra finestre adiacenti della
          stessa sessione → near-duplicate in train+test) e confronta i protocolli onesti:
          random (leaky) → session-grouped → user-grouped (LOUO) → temporal rolling-origin.
          Valida la scelta di `train.py` (rolling-origin media±std; temporal-singolo = fragile).
Metodo:   RF 200 in-domain su GT GPS-present (5 classi Still/Walk/Bus/Car/Train; Bike escluso, E20).
          Per ogni protocollo: accuracy + macro-F1 (media±std sui fold).
Input:    data/v2/features_trento.parquet (label GT + 163 feat + session_id + userId + ts_start + gps_frac)
Output:   research/figures/e23_eval_protocols.{png,pdf} · e23_eval_validity.csv
Alimenta: thesis/eda.md (E23)
Sez.tesi: 6.1 protocollo di valutazione / 6.2

Lettura: il gap random→session = leakage da overlap finestre; il calo →LOUO = dipendenza dall'utente
         (coerente con concentrazione/dropout E9). Il numero onesto = rolling-origin (media±std).
Run: /opt/miniconda3/envs/tmd/bin/python research/e23_eval_validity.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, TimeSeriesSplit
from sklearn.impute import SimpleImputer
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


def rf():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1)


def eval_folds(X, y, splits):
    """splits = lista di (train_idx, test_idx) → (acc list, macroF1 list)."""
    accs, f1s = [], []
    for tr, te in splits:
        m = rf().fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return np.array(accs), np.array(f1s)


def main():
    df = pd.read_parquet(ROOT / "data/v2/features_trento.parquet")
    g = df[(df.gps_frac > 0.5) & df.label.notna() & df.label.isin(CLASSES)].copy()
    g = g.sort_values("ts_start").reset_index(drop=True)
    feat = [c for c in g.columns if c[:2] in ("A_", "B_", "C_", "D_")]
    X = SimpleImputer(strategy="median").fit_transform(g[feat])
    y = g.label.values
    sess = g.session_id.values
    user = g.userId.values
    print("=" * 70); print("E23 — validità del protocollo di valutazione (in-domain)"); print("=" * 70)
    print(f"finestre GT GPS-present (5 classi): {len(g)} | sessioni {g.session_id.nunique()} | utenti {g.userId.nunique()}")
    print("(finestre: step 60s < lunghezza 120s → 50% overlap intra-sessione → lo split casuale perde)")

    results = {}

    # 1) random (LEAKY): finestre overlappate finiscono in train+test
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results["Random\n(leaky)"] = eval_folds(X, y, list(skf.split(X, y)))

    # 2) session-grouped: nessun overlap finestre / leakage intra-sessione
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    results["Session-\ngrouped"] = eval_folds(X, y, list(sgkf.split(X, y, groups=sess)))

    # 3) user-grouped (LOUO-style): generalizzazione a NUOVI utenti
    results["User-grouped\n(LOUO)"] = eval_folds(X, y, list(sgkf.split(X, y, groups=user)))

    # 4) temporal rolling-origin (expanding): train=passato, test=futuro (media±std sui fold)
    tss = TimeSeriesSplit(n_splits=5)
    results["Temporal\nrolling-origin"] = eval_folds(X, y, list(tss.split(X)))

    # ── report ──
    rows = []
    print(f"\n{'protocollo':22s} {'accuracy':>16s} {'macro-F1':>16s}")
    for k, (a, f) in results.items():
        kk = k.replace("\n", " ")
        print(f"{kk:22s}  {a.mean():.3f} ± {a.std():.3f}   {f.mean():.3f} ± {f.std():.3f}")
        rows.append({"protocol": kk, "acc_mean": a.mean(), "acc_std": a.std(),
                     "f1_mean": f.mean(), "f1_std": f.std()})
    pd.DataFrame(rows).to_csv(FIG / "e23_eval_validity.csv", index=False)

    leak = results["Random\n(leaky)"][0].mean() - results["Session-\ngrouped"][0].mean()
    louo_drop = results["Session-\ngrouped"][0].mean() - results["User-grouped\n(LOUO)"][0].mean()
    tmp = results["Temporal\nrolling-origin"]
    print(f"\n🔑 LEAKAGE (random − session-grouped) = +{100*leak:.1f} pt accuracy → lo split casuale GONFIA.")
    print(f"🔑 Calo →LOUO (session − user-grouped) = −{100*louo_drop:.1f} pt → dipendenza dall'UTENTE (E9: concentrazione/dropout).")
    print(f"🔑 Rolling-origin (onesto temporale): acc {tmp[0].mean():.3f} ± {tmp[0].std():.3f} — std>0 ⇒ un singolo taglio temporale è FRAGILE.")

    # ── figura ──
    labels = list(results.keys())
    accm = [results[k][0].mean() for k in labels]; accs = [results[k][0].std() for k in labels]
    f1m = [results[k][1].mean() for k in labels]; f1s = [results[k][1].std() for k in labels]
    x = np.arange(len(labels)); w = 0.38
    colors_a = ["tab:red" if "leaky" in k else "tab:blue" for k in labels]
    plt.figure(figsize=(8.5, 4.6))
    plt.bar(x - w/2, accm, w, yerr=accs, capsize=4, color=colors_a, label="accuracy")
    plt.bar(x + w/2, f1m, w, yerr=f1s, capsize=4, color="tab:gray", alpha=.8, label="macro-F1")
    plt.xticks(x, [k.replace("\n", " ") for k in labels], fontsize=8)
    plt.ylabel("score"); plt.ylim(0, 1.0)
    plt.title("Evaluation protocol matters: random split leaks (window overlap)\nhonest = session/user-grouped & temporal rolling-origin")
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y")
    plt.annotate(f"leakage +{100*leak:.1f}pt", xy=(0.5, max(accm[0], accm[1]) + 0.03),
                 ha="center", fontsize=8, color="tab:red")
    savefig("e23_eval_protocols")
    print("\nfigura → e23_eval_protocols | tabella → e23_eval_validity.csv")


if __name__ == "__main__":
    main()
