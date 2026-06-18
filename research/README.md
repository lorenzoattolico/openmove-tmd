# research — reproducing the thesis

Each script reproduces one figure or number from the thesis. They read the frozen local snapshot
(`data/`, not included), are deterministic (fixed seeds), and run standalone from the repo root, e.g.

    python research/e1_gps_raw.py

Output figures go to `research/figures/`. The pipeline itself lives in `tmd/`; these scripts import
it and analyse its outputs.

## Exploratory analysis — `e1`–`e29`

Characterize the data and the features: GPS and IMU coverage, R1 sessions, windowing, the
230 -> 163 feature selection, GPS missingness (MNAR, device-side), ground-truth audits, the validity
of the evaluation protocol, and cross-dataset shift.

## Experiments (research questions) — `rq1`–`rq6`

The model matrix: silver vs supervised, feature groups, the GPS operating floor and its cure,
leave-one-user-out spread, calibration, cross-dataset transfer, modal split and the CO2 bridge,
the learning curve, and active learning.

## Helpers

`strat_eval` (3-way GPS-stratified macro-F1), `bootstrap_ci` (session-cluster intervals),
`device_map` (stable userId to device).

## External benchmarks

`protocol_on_shl` runs the whole protocol on SHL; `geolife_*` and `load_geolife*` cover the GeoLife
reach test. SHL and GeoLife features are consumed as frozen inputs — their extraction is documented,
not re-run here.
