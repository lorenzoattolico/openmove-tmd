# openmove-tmd

Transportation mode detection from smartphone GPS and IMU, without manual labels.
Master's thesis (Lorenzo Attolico, University of Trento) and the deployment code for OpenMove.

## Layout

- `tmd/` — the pipeline: feature extraction, the physical labeler, silver labels, the
  hierarchical model, inference and evaluation.
- `research/` — the experiments behind the thesis: exploratory analysis and the
  research-question matrix, each script reproducing one figure or number.
- `production/` — the deployment path: one configuration, one model, down to the corrected
  modal split and CO2 estimate.

## Install

    pip install .
    tmd --help

## Use

Data are not included (user mobility, kept private); the pipeline runs on a local snapshot.
To deploy on your own data see `production/DEPLOY.md`; to reproduce the thesis results see
`research/README.md`.
