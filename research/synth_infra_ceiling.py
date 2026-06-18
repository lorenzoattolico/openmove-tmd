"""
synth_infra_ceiling.py — come si DECIDE (label-free) la soglia infrastrutturale del labeler
in un contesto nuovo, misurata su tracciati SINTETICI che seguono l'infrastruttura per
costruzione.

Obiettivo (NON ri-provare la robustezza-soglie, gia' fatta): mostrare la procedura
costruttiva per scegliere la soglia. Si generano viaggi on-route (bus su shape GTFS, treno su
rotaia OSM), si campionano alle FREQUENZE GPS REALI (da B_n_gps della GT MotionTag), si leggono
prop/align e si fa vedere che il valore giusto e' basso/conservativo, ricostruibile da mappe +
frequenza-GPS soli. I buffer sono quelli ESATTI della pipeline (gtfs_spatial + infra_align).

Validazione vs reali (GT): route_align Bus 0.74, bus_stops_prop 0.58, rail_prop Train 0.61,
rail_align 0.35. Soglie labeler: route_align 0.40, bus_stops_prop 0.20, rail_prop 0.35, rail_align 0.15.

Geometria treno: estratta da data/osm/nord-est-latest.osm.pbf (osmium, railway=rail), cache
in /tmp/rail_trento.geojsonseq (vedi commento in load_rail_lines).
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.labeling.infra_align import _route_trees

EARTH_R_M = 6_371_000.0
ROOT = Path(__file__).resolve().parents[1]
BBOX = {"lat": (45.35, 47.10), "lon": (10.50, 12.10)}

# buffer ESATTI pipeline (gtfs_spatial.PROP_THRESHOLDS + infra_align)
BUF = {"rail_prop": 30.0, "rail_align": 20.0, "bus_stops_prop": 50.0, "route_align": 50.0}
SPEED = {"bus": 7.0, "train": 20.0, "car": 12.0}          # m/s

# riferimenti reali (GT) [mediana on-route] e soglia labeler, per modo/feature
REAL = {("bus", "route_align"): (0.74, 0.40), ("bus", "bus_stops_prop"): (0.58, 0.20),
        ("train", "rail_align"): (0.35, 0.15), ("train", "rail_prop"): (0.61, 0.35)}
# fix per finestra 120s: reali (GT B_n_gps) Bus p25/50/75 = 71/116/120, Train 0/84/115.
# Sweep dal denso (~1Hz) al rado, per coprire il range reale incl. coda duty-cycling.
FIX_SWEEP = [120, 80, 40, 20, 10]
SIGMA_MAIN = 8.0
SIGMA_SWEEP = [0, 5, 10, 15]
WIN_S = 120.0
N_TRIPS = 120


# ─────────────────────────── geometria ───────────────────────────
def load_rail_lines(bbox):
    """Linee rotaia ORDINATE dal geojsonseq osmium (railway=rail). Rigenera la cache con:
       osmium tags-filter data/osm/nord-est-latest.osm.pbf w/railway=rail -o /tmp/rail_only.pbf --overwrite
       osmium export /tmp/rail_only.pbf --geometry-types=linestring --output-format=geojsonseq -o /tmp/rail_trento.geojsonseq --overwrite --no-progress"""
    path = Path("/tmp/rail_trento.geojsonseq")
    lat0, lat1 = bbox["lat"]; lon0, lon1 = bbox["lon"]
    lines = []
    for raw in open(path, encoding="utf-8"):
        raw = raw.strip().lstrip("\x1e")
        if not raw:
            continue
        try:
            geom = json.loads(raw)["geometry"]
        except Exception:
            continue
        parts = ([geom["coordinates"]] if geom["type"] == "LineString"
                 else geom["coordinates"] if geom["type"] == "MultiLineString" else [])
        for coords in parts:
            pl = np.array([(lat, lon) for lon, lat in coords
                           if lat0 <= lat <= lat1 and lon0 <= lon <= lon1], float)
            if len(pl) >= 2:
                lines.append(pl)
    return lines


# ─────────────────────────── tracce sintetiche ───────────────────────────
def _haversine_m(lat1, lon1, lat2, lon2):
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def subpath(poly, length_m, rng):
    """Sotto-percorso contiguo di ~length_m metri da un punto d'inizio casuale lungo poly."""
    seg = _haversine_m(poly[:-1, 0], poly[:-1, 1], poly[1:, 0], poly[1:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= length_m:
        return poly
    start = rng.uniform(0, total - length_m)
    lo = np.searchsorted(cum, start) - 1
    hi = np.searchsorted(cum, start + length_m) + 1
    return poly[max(lo, 0):hi + 1]


def sample_fixes(poly, n_fixes, sigma_m, rng):
    """n_fixes punti equispaziati lungo poly (per arco) + rumore GPS gaussiano (sigma m)."""
    seg = _haversine_m(poly[:-1, 0], poly[:-1, 1], poly[1:, 0], poly[1:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] == 0 or n_fixes < 1:
        return poly[:1]
    targets = np.linspace(0, cum[-1], int(n_fixes))
    out = np.empty((len(targets), 2))
    j = 0
    for i, t in enumerate(targets):
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        f = 0.0 if seg[j] == 0 else (t - cum[j]) / seg[j]
        out[i] = poly[j] + f * (poly[j + 1] - poly[j])
    if sigma_m > 0:
        out[:, 0] += rng.normal(0, sigma_m, len(out)) / 111_320.0
        out[:, 1] += rng.normal(0, sigma_m, len(out)) / (111_320.0 * np.cos(np.radians(out[:, 0])))
    return out


def random_offroute(bbox, length_m, rng):
    lat = rng.uniform(*bbox["lat"]); lon = rng.uniform(*bbox["lon"])
    ang = rng.uniform(0, 2 * np.pi)
    dlat = length_m * np.cos(ang) / 111_320.0
    dlon = length_m * np.sin(ang) / (111_320.0 * np.cos(np.radians(lat)))
    return np.array([[lat, lon], [lat + dlat, lon + dlon]])


# ─────────────────────────── metriche (buffer pipeline) ───────────────────────────
def _ndist(coords, bt):
    return bt.query(np.radians(coords), k=1)[0].flatten() * EARTH_R_M


def metrics(coords, idx, route_trees):
    m = {}
    drail = _ndist(coords, idx["osm_rail"]["balltree"])
    m["rail_prop"] = float((drail < BUF["rail_prop"]).mean())
    m["rail_align"] = float((drail < BUF["rail_align"]).mean())
    m["bus_stops_prop"] = float((_ndist(coords, idx["bus_stops"]["balltree"]) < BUF["bus_stops_prop"]).mean())
    rad = np.radians(coords)
    m["route_align"] = max(((t.query(rad, k=1)[0].flatten() < BUF["route_align"] / EARTH_R_M).mean()
                            for t in route_trees), default=0.0)
    return m


def gen_trip(mode, shapes, rails, n_fixes, sigma, rng):
    length = SPEED[mode] * WIN_S
    if mode == "bus":
        poly = subpath(shapes[rng.integers(len(shapes))], length, rng)
    elif mode == "train":
        poly = subpath(rails[rng.integers(len(rails))], length, rng)
    else:
        poly = random_offroute(BBOX, length, rng)
    return sample_fixes(poly, n_fixes, sigma, rng)


# ─────────────────────────── run ───────────────────────────
def main():
    rng = np.random.default_rng(42)
    shapes_path = ROOT / "data/gtfs/trento/urbano/shapes.txt"
    idx = pickle.load(open(ROOT / "data/processed/spatial_index_trento.pkl", "rb"))
    route_trees = _route_trees(shapes_path)
    sh = pd.read_csv(shapes_path)
    shapes = [g.sort_values("shape_pt_sequence")[["shape_pt_lat", "shape_pt_lon"]].values
              for _, g in sh.groupby("shape_id")]
    rails = load_rail_lines(BBOX)
    print(f"geometria: {len(shapes)} linee bus (GTFS) · {len(rails)} linee rotaia (OSM, bbox Trento)\n")

    def dist(mode, feat, n, sigma):
        vals = []
        for _ in range(N_TRIPS):
            tr = gen_trip(mode, shapes, rails, n, sigma, rng)
            if len(tr) >= 1:
                vals.append(metrics(tr, idx, route_trees)[feat])
        return np.array(vals)

    print(f"=== Sweep frequenza-GPS (sigma={SIGMA_MAIN}m, N={N_TRIPS} viaggi/cella) ===")
    print("on-route mediana [p25..p75]; reale e soglia a lato. Fix/120s: 120~1Hz -> 10 molto rado\n")
    for mode, feat in [("bus", "route_align"), ("bus", "bus_stops_prop"),
                       ("train", "rail_align"), ("train", "rail_prop")]:
        real, thr = REAL[(mode, feat)]
        print(f"{mode:>5} {feat:<15} (reale {real:.2f}, soglia {thr:.2f}):")
        for n in FIX_SWEEP:
            v = dist(mode, feat, n, SIGMA_MAIN)
            print(f"    {n:>3} fix  med {np.median(v):.3f}  [{np.percentile(v,25):.3f}..{np.percentile(v,75):.3f}]"
                  f"   frazione>soglia {np.mean(v > thr):.2f}")
        print()

    print("=== Baseline OFF-ROUTE (car) + cross-mode, sigma=8m, 120 fix ===")
    car = {f: dist("car", f, 120, SIGMA_MAIN) for f in ["route_align", "rail_prop"]}
    print(f"  car route_align med {np.median(car['route_align']):.3f} | rail_prop med {np.median(car['rail_prop']):.3f}")
    print(f"  bus  rail_prop  med {np.median(dist('bus','rail_prop',120,SIGMA_MAIN)):.3f}  (atteso ~0)")
    print(f"  train route_align med {np.median(dist('train','route_align',120,SIGMA_MAIN)):.3f}  (atteso ~0)")

    print("\n=== Sensibilita' al rumore GPS (N=120 fix) ===")
    for mode, feat in [("bus", "route_align"), ("train", "rail_prop")]:
        row = "  ".join(f"s{s}:{np.median(dist(mode,feat,120,s)):.3f}" for s in SIGMA_SWEEP)
        print(f"  {mode:>5} {feat:<12} {row}")


if __name__ == "__main__":
    main()
