# openmove-tmd

Transportation mode detection from smartphone GPS and IMU, **without manual labels**.
Master's thesis (Lorenzo Attolico, University of Trento) and the deployment code for OpenMove.

The system reads raw GPS and inertial signals, cuts them into 120-second windows, labels each window
with a physically grounded rule set (no hand annotation), trains a model on those weak labels, and
reports transportation modes — Still, Walk, Car, Bus, Train — down to the corrected modal split and a
CO2 estimate. How it works is documented in `docs/`.

## Repository map

| Path | What it is |
|------|------------|
| `tmd/` | the library: the whole validated pipeline |
| `tmd/config`, `tmd/configs` | per-city configuration (thresholds, bounds); `trento.yaml` + a template |
| `tmd/ingest` | pulls raw data from MongoDB (incremental, per-user cursors) |
| `tmd/datasets`, `tmd/sessions` | reads the local raw data; groups GPS+IMU points into sessions (trips) |
| `tmd/features` | the feature groups — A (IMU), B (GPS kinematics), C (infrastructure), D (GPS quality) |
| `tmd/spatial` | builds the GTFS/OSM spatial index used by group C |
| `tmd/labeling` | the universal physical labeler and the silver labels it produces |
| `tmd/models`, `tmd/training`, `tmd/inference` | the hierarchical model, its training, prediction + post-processing |
| `tmd/aggregate` | modal split -> quantification -> CO2 (the use-case output) |
| `tmd/cli` | the `tmd` command — a thin dispatcher over the steps |
| `research/` | the experiments behind the thesis (60 scripts: EDA `e1`–`e29`, RQs `rq1`–`rq6`, helpers) |
| `production/` | the deployment guide (`DEPLOY.md`, `DEPLOY.it.md`) |
| `models/` | the deployable Trento model, shipped so the repo is self-contained |
| `docs/` | usage, architecture, methodology, results |
| `tests/` | a small suite (aggregate, labeler, an inertness regression) |

## Requirements

- Python 3.11 or newer (a virtual environment, conda or venv, is recommended).
- Runtime dependencies (pandas, numpy, scikit-learn, pyarrow, scipy, pyyaml, pymongo) install
  automatically; the exact versions used for the thesis are pinned in `requirements.lock`.
- Data are **not** included (user mobility, kept private): the pipeline runs on a local snapshot, or
  reads from MongoDB via `tmd ingest`.

## Installation

    pip install .              # installs the `tmd` command and its dependencies, from this folder
    tmd --help

`pip install .` reads `pyproject.toml` (the project manifest) and installs *this* project. For the
research scripts (plotting, extra classifiers), add the optional set; for the exact pinned versions,
use the lock file:

    pip install ".[research]"
    pip install -r requirements.lock

## Quickstart (Trento)

The trained Trento model ships in `models/`, so you can classify without training:

    tmd predict   --model models/trento_20260612_202641.pkl --features data/v2/features_trento.parquet
    tmd aggregate

`predict` writes per-window modes; `aggregate` turns them into the modal split and CO2 estimate.

## Commands

`tmd <command>` — each is one operation, not a variant:

| Command | What it does |
|---------|--------------|
| `tmd ingest` | pull new raw data from MongoDB (incremental) |
| `tmd build-index` | build the GTFS/OSM spatial index for a city |
| `tmd process` | raw -> windows + features |
| `tmd build-model` | silver labeling + training (the frozen recipe) |
| `tmd predict` | features -> predicted modes |
| `tmd aggregate` | predictions -> modal split + CO2 |
| `tmd run` | process -> predict -> aggregate, chained |

Full arguments, examples, and where each step writes its output are in **`docs/usage.md`**.

## Reproducing the thesis

The `research/` scripts each regenerate one figure or number (see `research/README.md`). They read the
frozen snapshot and run from the repo root, e.g. `python research/e1_gps_raw.py`.

## Deploying

`production/DEPLOY.md` (and `DEPLOY.it.md`) describe the two deployment paths and a container skeleton.

## Documentation

- `docs/usage.md` — every command, its arguments, examples, and where output lands.
- `docs/architecture.md` — how the pipeline is built, stage by stage.
- `docs/methodology.md` — the label-free method, leakage control, and how it is evaluated.
- `docs/results.md` — the headline numbers.

## License

MIT — see `LICENSE`.
