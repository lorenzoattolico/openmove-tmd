# Architecture

`tmd` turns raw smartphone GPS and IMU into per-window transportation modes
(Still / Walk / Car / Bus / Train), without manual labels. This document is *how* the pipeline is
built; the *why* of the method is in `methodology.md`.

## The pipeline, stage by stage

    ingest      MongoDB -> data/raw                  incremental, per-user cursors, idempotent dedup
    sessions    points -> sessions (R1)              continuity over GPS and IMU together
    features    sessions -> 120s windows             four feature groups, ~230 columns
    labeling    windows -> silver labels             universal physical labeler, ABSTAIN on the ambiguous
    training    silver -> model                      hierarchical random forest, rolling-origin evaluation
    inference   model -> predictions                 smoothing + segment coherence
    aggregate   predictions -> modal split, CO2      with a quantification correction

Each stage is a package under `tmd/`; the CLIs in `tmd/cli/` are thin (parse arguments, call the
library, write output). Outputs land in `data/v2/`. The `tmd` command chains the stages.

## Feature groups

A window carries four families of features:

- **A — IMU.** Statistics, spectrum, jerk, autocorrelation of accelerometer and gyroscope (~130).
- **B — GPS kinematics.** Speed mean/max, stop fraction, path efficiency, bearing (~20).
- **C — infrastructure.** Proximity to rail, bus stops, motorway and cycleway, from GTFS and OSM (~15).
- **D — GPS quality.** Accuracy, gap fraction, a reliable-GPS flag (~5).

230 features are extracted; an unsupervised selection (redundancy, non-transferability, sparsity)
reduces them to the 163 the model sees.

## Sessions (R1)

A session is a stretch of continuous recording, cut only where **both** GPS and IMU are missing for
longer than a gap threshold. Building sessions on the union of the two sensors — rather than on GPS
alone — means a GPS dropout mid-trip no longer splits the trip: the IMU carries the continuity.

## The model

A two-level hierarchical classifier: the first level separates Still from Moving, the second assigns
the moving mode. The default base learner is a random forest. Trained models are saved in a registry
with their full configuration; the registry's loader resolves the model class by name, so a model
pickled under one package path still loads after the package is renamed.

## Post-processing

Raw per-window predictions are smoothed (majority vote over a short horizon) and passed through a
segment-coherence filter (minimum durations per mode, impossible transitions removed). The headline
numbers are the raw outputs; post-processing is reported only as a separate, labeled increment.

## Key choices (the short version — see `methodology.md`)

- The labeler's thresholds are physical constants, not fit to a dataset; it abstains rather than guess.
- Route-alignment features are withheld from training (leakage control); spatial proximity is kept.
- GPS availability is reported three ways (present / sparse / absent), never as one number.
- Deployment transfers the protocol, not the model: each context re-runs the pipeline on its own maps.
