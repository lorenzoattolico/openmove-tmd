# Architecture

`tmd` turns raw smartphone GPS and IMU into per-window transportation modes
(Still / Walk / Car / Bus / Train), without manual labels.

## Pipeline

    ingest      MongoDB -> data/raw            incremental, per-user cursors
    sessions    R1: continuity over GPS and IMU together
    features    120s windows -> groups A (IMU) / B (GPS) / C (infrastructure) / D (GPS quality)
    labeling    universal physical labeler -> silver labels (ABSTAIN on the ambiguous)
    training    hierarchical random forest on silver, rolling-origin evaluation
    inference   predict -> smoothing + segment coherence
    aggregate   modal split -> quantification -> CO2

All the logic is in the library (`tmd/<package>/`); the CLIs (`tmd/cli/`) are thin wrappers.
Outputs land in `data/v2/`.

## Packages

`config` (frozen CityConfig) · `datasets` · `ingest` (Mongo dump) · `sessions` · `features` ·
`spatial` (GTFS/OSM index) · `labeling` · `models` (HierarchicalTMD + registry) · `training` ·
`inference` · `evaluation` · `aggregate` · `cli`.

## Key choices

- **Universal labeler.** Its thresholds are physical constants (locomotion ceiling, sensor noise
  floor), not fit to a dataset; it abstains rather than guess, and the model generalizes from the
  certain cases.
- **Leakage control.** The two route-alignment features are dropped before training; spatial
  proximity (rail, bus stops, motorway) is kept, because it is universal physics, not memorized geography.
- **GPS availability is reported three ways** (present / sparse / absent), never as one inflated number.
- **Deployment transfers the protocol, not the model.** Each context re-runs labeler -> silver ->
  local training on its own public maps; the model is local by construction.
