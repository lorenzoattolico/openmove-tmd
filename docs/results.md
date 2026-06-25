# Results

Headline numbers for the Trento deployment and the cross-dataset tests. The full evaluation —
protocols, per-class breakdowns, confidence intervals — is in the thesis; this is a short orientation.

## In-domain (Trento, label-free)

The label-free model reaches a macro-F1 of **0.80** on the operating domain (GPS-present windows) and
**0.63** pooled over all windows, including the no-GPS regime. The gap to a supervised model trained on
the same reference is small on GPS-present, so silver labels nearly match supervised on the physical
modes (Still, Walk, Car).

Evaluation is rolling-origin, never random: random splits inflate accuracy by about 5 points through
overlapping windows. Every figure is stratified three ways by GPS availability rather than hidden
behind one number.

## The GPS operating floor

Macro-F1 by GPS availability is **0.80 / 0.57 / 0.18** on present / sparse / absent. The collapse at
zero-GPS is architectural, not a limit of the inertial sensor: an IMU-aware variant recovers the
no-GPS stratum to **0.46** (Walk 0.75, Still 0.92). The irreducible limit is motorized modes without GPS.

## Cross-dataset — the protocol transfers, not the model

Re-running the whole protocol on SHL (a second country, with true ground truth) gives silver **0.84**
vs supervised **0.92**: the label-free cost holds on a different corpus, and the universal labeler scores
**0.87–0.91** untuned. GeoLife (GPS-only, a third continent) is the reach test — the core physics
(Walk, Still, Car) transfers, while Bus and Train are bounded by map completeness and GPS coverage.

## Leakage control

Dropping the infrastructure feature group costs the transfer (**0.79 → 0.73**) but not the in-domain
score — evidence that the spatial features are universal physics, not memorized geography.

## Use case

The deliverable is the aggregate, not the per-window label: the modal split, de-biased by a standard
prevalence correction (quantification) from a small local calibration sample, maps to a CO2 estimate within the error margin that mobility surveys take as a reference.
