# openmove-tmd

Transportation mode detection from smartphone GPS and IMU, **without manual labels**.
Master's thesis (Lorenzo Attolico, University of Trento) and the deployment code for OpenMove.

The system reads raw GPS and inertial signals, cuts them into 120-second windows, labels each window
with a physically grounded rule set (no hand annotation), trains a model on those weak labels, and
reports transportation modes — down to the corrected modal split and a CO2 estimate.

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
| `docs/` | architecture, methodology, results |
| `tests/` | a small suite (aggregate, labeler, an inertness regression) |

## Install

    pip install .
    tmd --help

The `tmd` command exposes the pipeline as operations, not variants:
`ingest`, `build-index`, `process`, `build-model`, `predict`, `aggregate`, `run`.

## Use

Data are not included (user mobility, kept private); the pipeline runs on a local snapshot.

- **Run on Trento now:** the trained model is in `models/` — `tmd predict` then `tmd aggregate`.
- **A new city:** `tmd build-index -> ingest -> build-model` re-runs the protocol on local public maps.

See `production/DEPLOY.md` to deploy, `research/README.md` to reproduce the thesis.

## Documentation

- `docs/architecture.md` — how the pipeline is built, stage by stage.
- `docs/methodology.md` — the label-free method, leakage control, and how it is evaluated.
- `docs/results.md` — the headline numbers and what the system achieves.
