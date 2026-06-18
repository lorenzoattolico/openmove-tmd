"""
error_surfacing_freeze.py — RQ §6.6: il sistema audita il riferimento commerciale rumoroso.
RIGENERATO SUL FREEZE (14 giu, sostituisce la cifra pre-freeze archiviata).

Tesi (§6.6): ordinando le finestre per DISACCORDO-CONFIDENTE col GT (modello != GT con alta
max_prob), gli errori del GT emergono in cima — molto piu' che ordinando per incertezza.

Famiglia-errori-GT (la "Adige Car-as-Train" del §3.4): finestre con GT=Car ma fisicamente Treno
(rail_prop>0.35 AND speed>8 = le condizioni Train del labeler). Definita dalla FISICA, non da un
CSV senza chiave -> riproducibile sul corpus congelato.

Input: eval canonico rolling-OOF (data/v2/processed/eval_trento_20260612_202507.parquet, modello
trento_20260612_202512) + features_trento_full per rail_prop/speed (join su session_id+ts_start).
Output (operating domain GPS-present, 5 classi): recall della famiglia nel top-20% per
confident-disagreement vs per uncertainty, e base rate.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIVE = ["Still", "Walk", "Car", "Bus", "Train"]

e = pd.read_parquet(ROOT / "data/v2/processed/eval_trento_20260612_202507.parquet")
f = pd.read_parquet(ROOT / "data/v2/features_trento_full.parquet",
                    columns=["session_id", "ts_start", "C_osm_rail_prop", "B_speed_mean"])
d = e.merge(f, on=["session_id", "ts_start"], how="left")
d = d[(d["gps_frac"] > 0.5) & (d["label"].isin(FIVE))].dropna(
    subset=["C_osm_rail_prop", "B_speed_mean"]).copy()
N = len(d)

# famiglia errori-GT: Adige Car-as-Train (GT=Car ma rail+fast)
gt_err = (d["label"] == "Car") & (d["C_osm_rail_prop"] > 0.35) & (d["B_speed_mean"] > 8)
E = int(gt_err.sum())

disagree = d["predicted_class"] != d["label"]
d["s_conf"] = np.where(disagree, d["max_prob"], -1.0)   # confident-disagreement
d["s_unc"] = 1 - d["max_prob"]                            # uncertainty
k = int(0.20 * N)

print(f"operating domain (GPS-present, 5cl): {N} windows")
print(f"GT-error family (Car-as-Train, Adige): {E} windows, base rate {100*E/N:.1f}%\n")
for name, col in [("confident-disagreement", "s_conf"), ("uncertainty", "s_unc")]:
    top = d.nlargest(k, col)
    rec = gt_err[top.index].sum() / E
    print(f"  top-20% by {name:<22}: recovers {100*rec:.0f}% of the family")
