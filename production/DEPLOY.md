# Deploying openmove-tmd

The production path: one pipeline, one model, no research variants. OpenMove operates it;
containerization and scheduling are OpenMove's choice. The thesis numbers are produced by the same
`tmd` package, so production reproduces them exactly.

## Install

    pip install .          # installs the `tmd` command
    tmd --help

Secrets live in the environment, never in the repo: `MONGO_URI` (only for `tmd ingest`).
Per-city configuration is `tmd/configs/cities/<city>.yaml`; copy `city.example.yaml` to add one.

## Two paths

**A — Run now (Trento).** The trained Trento model is shipped in `models/`. Classify recent data:

    tmd ingest  --city trento          # incremental pull (cursors); needs MONGO_URI
    tmd process --city trento          # raw -> features
    tmd predict --model models/trento_20260612_202641.pkl --features data/v2/features_trento.parquet
    tmd aggregate                      # modal split + CO2

or chained:

    tmd run --city trento

**B — A new city (or a refresh).** Re-run the protocol on local public maps, with no manual labels:

    tmd build-index --city <x>         # GTFS/OSM spatial index (once)
    tmd ingest --city <x>
    tmd build-model --city <x>         # universal labeler -> silver -> local model

The model is always local by construction; only the protocol transfers.

## The corrected aggregate

`tmd aggregate` reports the naive modal split. The deliverable is the *corrected* aggregate
(modal split -> CO2): a prevalence correction (quantification) de-biases it, but needs a small
labeled calibration sample (~400 moving windows). Below ~200 it would hurt, so the naive split is
emitted with a caveat instead.

## Containerization (OpenMove)

The package is a plain installable app; the container is a few lines:

    FROM python:3.11-slim
    COPY . /app
    RUN pip install /app
    # CMD ["tmd", "run", "--city", "trento"]   # cadence and scheduling: your orchestrator

Data are never in the image (user-mobility privacy): mount them, or pull with `tmd ingest`.
