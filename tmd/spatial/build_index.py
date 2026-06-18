"""Costruisce spatial_index_{city}.pkl da feed GTFS e PBF OSM.

bus_stops strategy: GTFS (rete ufficiale operatore) + OSM (highway=bus_stop),
merge con dedup a 30m. OSM estende la copertura oltre il bbox GTFS (es. Verona
per utenti che viaggiano fuori dalla rete TTE).

Prerequisiti:
  data/gtfs/trento/urbano/       — feed TTE urbano estratto
  data/gtfs/trento/extraurbano/  — feed TTE extraurbano estratto
  data/osm/nord-est-latest.osm.pbf (o qualsiasi PBF che copre l'area)
  osmium installato (brew install osmium-tool)

Output:
  data/processed/spatial_index_trento.pkl

Eseguire:
  python -m tmd.spatial.build_index [--city trento] [--no-osm]
"""

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from tmd.config import CityConfig

EARTH_R = 6_371_000.0

RAIL_TYPES = {2, 100, 101, 102, 106, 109}
BUS_TYPES  = {3, 700, 702, 704}

# tag per geometrie lineari (way)
OSM_WAY_FILTERS = {
    "osm_rail":     ["w/railway=rail"],
    "osm_motorway": ["w/highway=motorway", "w/highway=motorway_link"],
    "osm_cycleway": ["w/highway=cycleway"],
}

# Nodi stazione/fermata ferroviaria (include stazioni RFI non nel GTFS TTE).
OSM_NODE_FILTERS = {
    "rail_stations_osm": ["n/railway=station", "n/railway=halt", "n/railway=stop"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def in_bbox(lat, lon, bbox):
    return (bbox["lat"][0] <= lat <= bbox["lat"][1] and
            bbox["lon"][0] <= lon <= bbox["lon"][1])


def build_balltree(coords: np.ndarray) -> dict:
    if len(coords) == 0:
        return {}
    return {"balltree": BallTree(np.deg2rad(coords), metric="haversine"),
            "coords": coords, "n": len(coords)}


def subsample_grid(points: np.ndarray, cell_m: float = 10.0) -> np.ndarray:
    if len(points) == 0:
        return points
    lat_deg = cell_m / 111320.0
    lon_deg = cell_m / (111320.0 * np.cos(np.radians(np.mean(points[:, 0]))))
    grid_lat = np.round(points[:, 0] / lat_deg).astype(np.int64)
    grid_lon = np.round(points[:, 1] / lon_deg).astype(np.int64)
    keys = grid_lat * 10**7 + grid_lon
    _, idx = np.unique(keys, return_index=True)
    return points[np.sort(idx)]


# ── GTFS ──────────────────────────────────────────────────────────────────────

def load_gtfs_stops_by_mode(gtfs_dirs, bbox):
    all_stops  = []
    stop_types = {}

    for gtfs_dir in gtfs_dirs:
        if not gtfs_dir.exists():
            print(f"  [skip] {gtfs_dir.name} non trovato")
            continue
        try:
            stops = pd.read_csv(gtfs_dir / "stops.txt", dtype=str, low_memory=False)
            stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
            stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
            stops = stops.dropna(subset=["stop_lat", "stop_lon"])
            stops = stops[stops.apply(
                lambda r: in_bbox(r["stop_lat"], r["stop_lon"], bbox), axis=1)]
            all_stops.append(stops[["stop_id", "stop_lat", "stop_lon"]])

            if all((gtfs_dir / f).exists()
                   for f in ["stop_times.txt", "trips.txt", "routes.txt"]):
                st     = pd.read_csv(gtfs_dir / "stop_times.txt",
                                     usecols=["trip_id", "stop_id"],
                                     dtype=str, low_memory=False)
                trips  = pd.read_csv(gtfs_dir / "trips.txt",
                                     usecols=["trip_id", "route_id"],
                                     dtype=str, low_memory=False)
                routes = pd.read_csv(gtfs_dir / "routes.txt",
                                     usecols=["route_id", "route_type"],
                                     dtype=str, low_memory=False)
                routes["route_type"] = pd.to_numeric(
                    routes["route_type"], errors="coerce").fillna(3).astype(int)
                merged = (st.merge(trips, on="trip_id")
                            .merge(routes, on="route_id")
                          [["stop_id", "route_type"]].drop_duplicates())
                for sid, rt in zip(merged["stop_id"], merged["route_type"]):
                    stop_types.setdefault(sid, set()).add(int(rt))
        except Exception as e:
            print(f"  [warn] errore leggendo {gtfs_dir.name}: {e}")

    if not all_stops:
        print("  Nessun feed GTFS caricato")
        return {"bus_stops": np.empty((0, 2))}

    df     = pd.concat(all_stops, ignore_index=True).drop_duplicates("stop_id")
    coords = df[["stop_lat", "stop_lon"]].values

    bus_mask = np.array([
        bool(stop_types.get(sid, set()) & BUS_TYPES or not stop_types.get(sid))
        for sid in df["stop_id"]
    ])
    result = {"bus_stops": coords[bus_mask]}
    print(f"  {'bus_stops':<20}: {result['bus_stops'].shape[0]:>5,} fermate  "
          f"(rail da GTFS non usato — OSM più completo)")
    return result


# ── OSM PBF ───────────────────────────────────────────────────────────────────

def _osmium_filter(pbf_path: Path, tags: list, tmp_pbf: Path) -> bool:
    r = subprocess.run(
        ["osmium", "tags-filter", str(pbf_path)] + tags +
        ["--output", str(tmp_pbf), "--overwrite", "--no-progress"],
        capture_output=True, text=True)
    return r.returncode == 0 and tmp_pbf.exists()


def _osmium_export(tmp_pbf: Path, geometry_types: str,
                   bbox: dict) -> list:
    with tempfile.TemporaryDirectory() as tmp2:
        tmp_geo = Path(tmp2) / "out.geojsonseq"
        r = subprocess.run(
            ["osmium", "export", str(tmp_pbf),
             f"--geometry-types={geometry_types}",
             "--output-format=geojsonseq",
             "--output", str(tmp_geo),
             "--overwrite", "--no-progress"],
            capture_output=True, text=True)
        if r.returncode != 0 or not tmp_geo.exists():
            return []

        points = []
        lat_min, lat_max = bbox["lat"]
        lon_min, lon_max = bbox["lon"]
        with open(tmp_geo, encoding="utf-8") as f:
            for line in f:
                line = line.strip().lstrip("\x1e")
                if not line:
                    continue
                try:
                    geom = json.loads(line)["geometry"]
                    if geom["type"] == "Point":
                        lon, lat = geom["coordinates"]
                        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                            points.append([lat, lon])
                    elif geom["type"] == "LineString":
                        for lon, lat in geom["coordinates"]:
                            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                                points.append([lat, lon])
                    elif geom["type"] == "MultiLineString":
                        for part in geom["coordinates"]:
                            for lon, lat in part:
                                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                                    points.append([lat, lon])
                except Exception:
                    continue
    return points


def extract_osm_ways(pbf_path: Path, tags: list, bbox: dict) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pbf = Path(tmp) / "filtered.osm.pbf"
        if not _osmium_filter(pbf_path, tags, tmp_pbf):
            return np.empty((0, 2))
        points = _osmium_export(tmp_pbf, "linestring", bbox)
    if not points:
        return np.empty((0, 2))
    arr = np.unique(np.round(np.array(points), 6), axis=0)
    return subsample_grid(arr, cell_m=10.0)


def extract_osm_nodes(pbf_path: Path, tags: list, bbox: dict) -> np.ndarray:
    """Estrae nodi puntuali — per stazioni ferroviarie."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pbf = Path(tmp) / "filtered.osm.pbf"
        if not _osmium_filter(pbf_path, tags, tmp_pbf):
            return np.empty((0, 2))
        points = _osmium_export(tmp_pbf, "point", bbox)
    if not points:
        return np.empty((0, 2))
    return np.unique(np.round(np.array(points), 6), axis=0)


def extract_osm_bus_stops(pbf_path: Path, bbox: dict) -> np.ndarray:
    """
    Estrae fermate bus da OSM (highway=bus_stop + public_transport=stop_position).
    Filtra rumore (ferry, railway, subway). Dedup a 30m per rimuovere coppie
    platform/stop_position della stessa fermata.
    """
    tags = ["n/highway=bus_stop", "n/public_transport=stop_position"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pbf = Path(tmp) / "filtered.osm.pbf"
        if not _osmium_filter(pbf_path, tags, tmp_pbf):
            return np.empty((0, 2))
        with tempfile.TemporaryDirectory() as tmp2:
            tmp_geo = Path(tmp2) / "out.geojsonseq"
            r = subprocess.run(
                ["osmium", "export", str(tmp_pbf),
                 "--geometry-types=point",
                 "--output-format=geojsonseq",
                 "--output", str(tmp_geo),
                 "--overwrite", "--no-progress"],
                capture_output=True, text=True)
            if r.returncode != 0 or not tmp_geo.exists():
                return np.empty((0, 2))

            lat_min, lat_max = bbox["lat"]
            lon_min, lon_max = bbox["lon"]
            points = []
            with open(tmp_geo, encoding="utf-8") as f:
                for line in f:
                    line = line.strip().lstrip("\x1e")
                    if not line:
                        continue
                    try:
                        feat  = json.loads(line)
                        props = feat.get("properties", {})
                        # scarta rumore non-bus
                        if props.get("railway") or props.get("ferry"):
                            continue
                        if props.get("subway") == "yes":
                            continue
                        if (props.get("public_transport") == "stop_position"
                                and props.get("tram") == "yes"
                                and props.get("bus", "no") != "yes"):
                            continue
                        geom = feat["geometry"]
                        if geom["type"] == "Point":
                            lon, lat = geom["coordinates"][:2]
                            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                                points.append([lat, lon])
                    except Exception:
                        continue

    if not points:
        return np.empty((0, 2))

    arr = np.array(points)
    # dedup a 30m: rimuove coppie platform/stop_position della stessa fermata
    lat_mean = float(np.mean(arr[:, 0]))
    cell_m   = 30.0
    lat_deg  = cell_m / 111320.0
    lon_deg  = cell_m / (111320.0 * np.cos(np.radians(lat_mean)))
    gk = (np.round(arr[:, 0] / lat_deg).astype(np.int64) * 10**8
          + np.round(arr[:, 1] / lon_deg).astype(np.int64))
    _, uniq = np.unique(gk, return_index=True)
    return arr[np.sort(uniq)]


# ── Validazione ───────────────────────────────────────────────────────────────

def validate(idx: dict, cfg) -> bool:
    print("\n── Validazione ──")
    errors = []

    def check(ok, msg_ok, msg_fail):
        if ok:
            print(f"  ✓ {msg_ok}")
        else:
            print(f"  ✗ {msg_fail}")
            errors.append(msg_fail)

    def dist_m(layer, lat, lon):
        if layer not in idx or not idx[layer]:
            return float("nan")
        pt = np.deg2rad([[lat, lon]])
        return float(idx[layer]["balltree"].query(pt, k=1)[0][0, 0] * EARTH_R)

    check(idx.get("bus_stops", {}).get("n", 0) > 1000,
          f"bus_stops: {idx.get('bus_stops',{}).get('n',0):,} fermate (GTFS + OSM)",
          "bus_stops: troppo poche o assenti")

    check(idx.get("rail_stations", {}).get("n", 0) > 20,
          f"rail_stations: {idx.get('rail_stations',{}).get('n',0):,} stazioni",
          "rail_stations: troppo poche o assenti")

    check(idx.get("osm_rail", {}).get("n", 0) > 500,
          f"osm_rail: {idx.get('osm_rail',{}).get('n',0):,} punti",
          "osm_rail: troppo pochi o assenti")

    d = dist_m("bus_stops", 46.0726, 11.1197)
    check(d < 300, f"Trento FS bus: {d:.0f}m", f"Trento FS bus: {d:.0f}m (>300m)")

    d = dist_m("rail_stations", 46.0726, 11.1197)
    check(d < 300, f"Trento FS rail: {d:.0f}m", f"Trento FS rail: {d:.0f}m (>300m)")

    d = dist_m("rail_stations", 45.8908, 11.0339)
    check(d < 200, f"Rovereto FS rail: {d:.0f}m", f"Rovereto FS rail: {d:.0f}m (>200m)")

    if errors:
        print(f"\nFALLITO — {len(errors)} errori")
        return False
    print("\nOK — tutti i check superati")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--city",   default="trento")
    p.add_argument("--no-osm", action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    cfg     = CityConfig.from_yaml(
        PROJECT_ROOT / "tmd" / "configs" / "cities" / f"{args.city}.yaml")
    bbox    = {"lat": cfg.bounds["lat"], "lon": cfg.bounds["lon"]}
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = out_dir / f"spatial_index_{cfg.city}.pkl"

    idx: dict = {}

    # ── GTFS (solo bus_stops) ─────────────────────────────────────────────────
    print(f"\n── GTFS ({args.city}) ──")
    gtfs_dirs = sorted((PROJECT_ROOT / "data" / "gtfs" / args.city).glob("*/"))
    if not gtfs_dirs:
        print("  Nessun feed trovato in data/gtfs/{city}/*/")
    else:
        print(f"  Feed trovati: {[d.name for d in gtfs_dirs]}")
        gtfs_stops = load_gtfs_stops_by_mode(gtfs_dirs, bbox)
        for layer, coords in gtfs_stops.items():
            if len(coords) > 0:
                idx[layer] = build_balltree(coords)

    # ── OSM PBF ───────────────────────────────────────────────────────────────
    if not args.no_osm:
        pbf_candidates = sorted((PROJECT_ROOT / "data" / "osm").glob("*.osm.pbf")) \
                         if (PROJECT_ROOT / "data" / "osm").exists() else []
        pbf_path = None
        for p in pbf_candidates:
            if args.city in p.name or any(k in p.name.lower()
                                          for k in ["trentino", "nord-est", "nordest"]):
                pbf_path = p
                break
        if pbf_path is None and pbf_candidates:
            pbf_path = pbf_candidates[0]

        if pbf_path:
            try:
                subprocess.run(["osmium", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("  [skip] osmium non trovato")
                pbf_path = None

        if pbf_path:
            print(f"\n── OSM PBF: {pbf_path.name} ──")

            # bus_stops: merge GTFS + OSM con dedup 30m
            print(f"  Estrazione bus_stops OSM... ", end="", flush=True)
            osm_bus = extract_osm_bus_stops(pbf_path, bbox)
            print(f"{len(osm_bus):,} fermate (dopo dedup 30m)")
            if len(osm_bus) > 0:
                gtfs_bus = idx.get("bus_stops", {}).get("coords", np.empty((0, 2)))
                if len(gtfs_bus) > 0:
                    # aggiungi solo fermate OSM senza corrispondente GTFS entro 30m
                    bt_gtfs  = BallTree(np.radians(gtfs_bus), metric="haversine")
                    d, _     = bt_gtfs.query(np.radians(osm_bus), k=1)
                    new_only = osm_bus[d[:, 0] * EARTH_R > 30]
                    merged   = np.vstack([gtfs_bus, new_only])
                    print(f"  bus_stops OSM nuove (>30m da GTFS): {len(new_only):,}")
                else:
                    merged = osm_bus
                idx["bus_stops"] = build_balltree(merged)
                n_gtfs = len(gtfs_bus) if len(gtfs_bus) > 0 else 0
                print(f"  bus_stops (GTFS {n_gtfs:,} + OSM {len(merged)-n_gtfs:,}): "
                      f"{len(merged):,} totali")

            # way layers (geometrie lineari)
            for layer, tags in OSM_WAY_FILTERS.items():
                print(f"  Estrazione {layer}... ", end="", flush=True)
                pts = extract_osm_ways(pbf_path, tags, bbox)
                print(f"{len(pts):,} punti")
                if len(pts) > 0:
                    idx[layer] = build_balltree(pts)

            # node layers (stazioni ferroviarie da OSM — più completo di GTFS TTE)
            for layer_name, tags in OSM_NODE_FILTERS.items():
                print(f"  Estrazione {layer_name}... ", end="", flush=True)
                pts = extract_osm_nodes(pbf_path, tags, bbox)
                print(f"{len(pts):,} nodi")
                if len(pts) > 0:
                    existing = idx.get("rail_stations", {}).get("coords", np.empty((0, 2)))
                    merged = np.unique(
                        np.vstack([existing, pts]) if len(existing) else pts,
                        axis=0
                    )
                    idx["rail_stations"] = build_balltree(merged)
                    print(f"  rail_stations (GTFS + OSM): {len(merged):,} totali")
        else:
            print("\n── OSM PBF non disponibile ──")
    else:
        print("\n── OSM saltato (--no-osm) ──")

    # ── Layer mancanti ────────────────────────────────────────────────────────
    all_layers = ["bus_stops", "rail_stations", "subway_stations",
                  "osm_rail", "osm_subway", "osm_motorway", "osm_cycleway"]
    missing = [l for l in all_layers if l not in idx]
    if missing:
        print(f"\n  Layer assenti (NaN a inferenza): {missing}")

    print(f"\n── Layer costruiti ──")
    for name, layer in idx.items():
        print(f"  {name:<25}: {layer['n']:>6,} punti")

    ok = validate(idx, cfg)

    with open(out_pkl, "wb") as f:
        pickle.dump(idx, f)
    size_kb = out_pkl.stat().st_size / 1024
    print(f"\nSalvato: {out_pkl.name}  ({size_kb:,.0f} KB)")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
