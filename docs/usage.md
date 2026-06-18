# Usage

A complete reference: setup, every command, reproducing the thesis, deploying, and where output lands.

## Setup

1. Python 3.11 or newer, ideally in a virtual environment (conda or venv).
2. Install:

       pip install .                      # the package + the `tmd` command
       pip install ".[research]"          # also the plotting / extra deps used by research/
       pip install -r requirements.lock   # the exact pinned versions, for reproducibility

3. Data. Nothing is bundled (privacy). Either set `MONGO_URI` in the environment (used only by
   `tmd ingest`) to pull from MongoDB, or place a frozen snapshot under `data/`.

## The `tmd` command

`tmd --help` lists the commands. Each is one operation, not a tuning knob.

### tmd ingest
Pulls raw GPS, IMU, and labels from MongoDB into `data/raw`, incrementally (per-user cursors, dedup,
crash-safe state). Only new data is fetched on each run.

    tmd ingest --city trento

Requires `MONGO_URI` in the environment.

### tmd build-index
Builds the GTFS/OSM spatial index (rail lines, bus stops, motorway, cycleway) used by the
infrastructure features. Run once per city, or when the maps change.

    tmd build-index --city trento

Needs the GTFS feeds and an OSM extract on disk, and the `osmium` command-line tool.

### tmd process
Reads the raw data, builds sessions and 120-second windows, and extracts the A/B/C/D features.

    tmd process --city trento

Writes `data/v2/features_<city>.parquet`.

### tmd build-model
The frozen training recipe: applies the universal labeler to produce silver labels, then trains the
hierarchical random forest (rolling-origin evaluation, no specialists). One model, no variants.

    tmd build-model --city trento

Writes the model and its metadata to the registry under `data/v2/models/`.

### tmd predict
Runs a model on a features parquet and writes per-window predictions, with smoothing and
segment-coherence post-processing applied.

    tmd predict --model models/trento_20260612_202641.pkl --features data/v2/features_trento.parquet

Writes `data/v2/predictions.parquet` (override with `--out`).

### tmd aggregate
Turns predictions into the modal split and a CO2 estimate. Without a calibration set it reports the
naive split; the corrected split (prevalence correction / quantification) needs a small labeled sample.

    tmd aggregate --predictions data/v2/predictions.parquet

### tmd run
Chains `process -> predict (with the local model) -> aggregate`, for a recurring job.

    tmd run --city trento

## Reproducing the thesis

Each `research/*.py` script regenerates one figure or number, and is indexed in `research/README.md`.
They read the frozen snapshot and run from the repo root:

    python research/e1_gps_raw.py        # an EDA figure
    python research/rq4_4_leakfree.py    # an experiment

Figures are written to `research/figures/`.

## Deploying (OpenMove)

See `production/DEPLOY.md` for the full guide. In short: `pip install .`, then either run `tmd predict`
and `tmd aggregate` with the shipped model, or run `tmd build-model` to train a local model for a new
city. The container is a few lines (Dockerfile skeleton in the deploy guide); cadence and scheduling
are decided by the operator.

## Where things land

| Path | What |
|------|------|
| `data/raw` | raw GPS / IMU / labels from `ingest` |
| `data/v2/features_<city>.parquet` | extracted features (`process`) |
| `data/v2/models/` | trained models + metadata (`build-model`) |
| `data/v2/predictions.parquet` | per-window predictions (`predict`) |
| `research/figures/` | generated figures |

`data/` is never committed (privacy); these are local artifacts.
