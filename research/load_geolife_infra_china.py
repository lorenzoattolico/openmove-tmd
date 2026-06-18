"""
load_geolife_infra_china.py — copertura OSM COMPLETA di tutta la Cina per GeoLife (rail+bus).

Forense (_forensic_geolife): l'indice OSM era solo Pechino+Tianjin → 16% delle finestre (e **48% dei
Train**, fino a 2300 km) cadevano fuori → rail/bus prop spuriamente 0 → ceiling RF-su-GT e transfer
artificialmente bassi. Le traiettorie sono al 99.4% in mainland-Cina (la coda 0.6% = Nord America,
solo Car/Walk di 2 utenti → ESCLUSA via flag in_china).

Strategia (no Overpass, no rate-limit, copertura completa, riproducibile): estratto Geofabrik
china-latest.osm.pbf → filtro railway + fermate-bus con osmium (C++) → BallTree → ricalcolo le 3
feature-C IDENTICHE al pipeline (rail 30 m, bus 50 m, prop=(d<t).mean()) per tutte le finestre, e
marca in_china (bbox Greater China). Backup del parquet prima di scrivere.

Prereq: data/external/osm/china-latest.osm.pbf (download Geofabrik) + osmium CLI.
Run: /opt/miniconda3/envs/tmd/bin/python research/load_geolife_infra_china.py
Rigenera dopo: rq4_3_geolife.py, rq4_5_walk_universal.py (con filtro in_china).
"""
from __future__ import annotations
import sys, json, subprocess, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from load_geolife import read_user_points, read_labels, WIN_MS, MIN_PTS  # noqa: E402
from tmd.spatial.build_index import build_balltree, subsample_grid, EARTH_R  # noqa: E402

GEOLIFE = ROOT / "data/external/raw/geolife/Data"
OUT = ROOT / "data/processed/features_geolife.parquet"
OSM = ROOT / "data/external/osm"
PBF = OSM / "china-latest.osm.pbf"
RAIL_GJS, BUS_GJS = OSM / "china_rail.geojsonseq", OSM / "china_bus.geojsonseq"
RAIL_THRESH, BUS_THRESH = 30.0, 50.0           # m — == pipeline
CHINA = (17.0, 73.0, 54.0, 135.0)              # lat_min, lon_min, lat_max, lon_max (Greater China)


def osmium_extract():
    """Filtra railway (ways) e fermate-bus (nodi) dall'estratto Cina → geojsonseq. Cache su file."""
    if not PBF.exists():
        sys.exit(f"manca {PBF} — scarica prima: curl -L -o {PBF} https://download.geofabrik.de/asia/china-latest.osm.pbf")
    if not RAIL_GJS.exists():
        rp = OSM / "china_rail.osm.pbf"
        print("osmium: filtro railway...")
        subprocess.run(["osmium", "tags-filter", "-o", str(rp), "--overwrite", str(PBF),
                        "w/railway=rail,subway,light_rail"], check=True)
        subprocess.run(["osmium", "export", str(rp), "-f", "geojsonseq", "-o", str(RAIL_GJS), "--overwrite"], check=True)
        rp.unlink(missing_ok=True)
    if not BUS_GJS.exists():
        bp = OSM / "china_bus.osm.pbf"
        print("osmium: filtro fermate-bus...")
        subprocess.run(["osmium", "tags-filter", "-o", str(bp), "--overwrite", str(PBF),
                        "n/highway=bus_stop", "n/public_transport=stop_position"], check=True)
        subprocess.run(["osmium", "export", str(bp), "-f", "geojsonseq", "-o", str(BUS_GJS), "--overwrite"], check=True)
        bp.unlink(missing_ok=True)


def load_coords(path) -> np.ndarray:
    """geojsonseq → array (lat,lon). Point + (Multi)LineString. Coord GeoJSON = [lon,lat]."""
    pts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            g = (json.loads(line).get("geometry") or {})
            t, c = g.get("type"), g.get("coordinates")
            if t == "Point":
                pts.append((c[1], c[0]))
            elif t == "LineString":
                pts.extend((y, x) for x, y in c)
            elif t == "MultiLineString":
                for seg in c:
                    pts.extend((y, x) for x, y in seg)
    return np.array(pts, np.float64)


def main():
    osmium_extract()
    print("Carico geometrie OSM...")
    rail = load_coords(RAIL_GJS)
    rail = subsample_grid(np.unique(np.round(rail, 6), axis=0), cell_m=10.0)
    bus = np.unique(np.round(load_coords(BUS_GJS), 6), axis=0)
    print(f"  rail vertici (post-subsample 10m): {len(rail):,} | fermate bus: {len(bus):,}")
    bt_rail = build_balltree(rail)["balltree"]
    bt_bus = build_balltree(bus)["balltree"]

    print("Re-windowing GeoLife + ricalcolo feature-C...")
    rows = []
    for ud in sorted([d for d in GEOLIFE.iterdir() if d.is_dir() and (d / "labels.txt").exists()]):
        labs = read_labels(ud); pts = read_user_points(ud)
        if not labs or pts is None or len(pts) < MIN_PTS:
            continue
        seg = 0
        for (t0, t1, cls) in labs:
            lo = int(np.searchsorted(pts[:, 0], t0, "left")); hi = int(np.searchsorted(pts[:, 0], t1, "right"))
            if hi - lo < MIN_PTS:
                continue
            sub = pts[lo:hi]; seg += 1; w0 = sub[0, 0]
            while w0 + WIN_MS <= sub[-1, 0] + WIN_MS:
                wl = int(np.searchsorted(sub[:, 0], w0, "left")); wh = int(np.searchsorted(sub[:, 0], w0 + WIN_MS, "left")); w0 += WIN_MS
                if wh - wl < MIN_PTS:
                    continue
                ll = sub[wl:wh, 1:3]; rr = np.radians(ll)
                dr = bt_rail.query(rr, k=1)[0][:, 0] * EARTH_R
                db = bt_bus.query(rr, k=1)[0][:, 0] * EARTH_R
                la, ln = ll[:, 0].mean(), ll[:, 1].mean()
                rows.append({"session_id": f"geolife_{ud.name}_s{seg}", "ts_start": int(sub[wl, 0]),
                             "C_osm_rail_prop": float((dr < RAIL_THRESH).mean()),
                             "C_osm_rail_p10": float(np.percentile(dr, 10)),
                             "C_bus_stops_prop": float((db < BUS_THRESH).mean()),
                             "in_china": bool(CHINA[0] <= la <= CHINA[2] and CHINA[1] <= ln <= CHINA[3])})
    cdf = pd.DataFrame(rows)
    print(f"  {len(cdf):,} finestre | in_china {cdf.in_china.mean():.2%}")

    bak = OUT.with_suffix(".parquet.prechina.bak")
    geo = pd.read_parquet(OUT); n0 = len(geo)
    if not bak.exists():
        geo.to_parquet(bak, index=False); print(f"backup → {bak.name}")
    geo = geo.drop(columns=[c for c in ["C_osm_rail_prop", "C_osm_rail_p10", "C_bus_stops_prop", "in_china"] if c in geo.columns])
    geo = geo.merge(cdf, on=["session_id", "ts_start"], how="left")
    assert len(geo) == n0, f"merge ha cambiato righe {n0}->{len(geo)}"
    geo.to_parquet(OUT, index=False)
    print(f"\nMerge OK → {OUT.name} ({len(geo):,} righe)")
    inc = geo[geo.in_china]
    print(f"\nIN-CINA ({len(inc):,} finestre) — mediane rail_prop / bus_prop per classe:")
    print(inc.groupby("label")[["C_osm_rail_prop", "C_bus_stops_prop"]].median().round(3).to_string())
    print("\n% rail_prop==0 per classe IN-CINA (era: Train ~in-bbox 22% / fuori 100%):")
    print((inc.assign(r0=inc.C_osm_rail_prop == 0).groupby("label").r0.mean().round(3)).to_string())
    print("% bus_prop==0 per classe IN-CINA:")
    print((inc.assign(b0=inc.C_bus_stops_prop == 0).groupby("label").b0.mean().round(3)).to_string())


if __name__ == "__main__":
    main()
