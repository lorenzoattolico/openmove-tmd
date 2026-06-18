# Methodology

How the system labels, learns, and is evaluated — and why those choices are defensible.

## Label-free: a universal physical labeler

There is no hand annotation. Each window is labeled by a cascade of physical rules:

    Still -> Walk -> Train -> Bus -> Car -> ABSTAIN

The thresholds are physical constants, not values fit to Trento: the human locomotion ceiling
(no pedestrian sustains a peak above ~5 m/s over two minutes), the MEMS noise floor of a still device,
the map-matching confidence to rail and bus stops. Because they are physical, the same labeler runs on
other cities and datasets without retuning — on a second country it agrees with the ground truth at
0.87–0.91 untuned.

When a window is ambiguous, the labeler **abstains** instead of guessing. High precision beats
coverage: the model learns from the certain cases and generalizes to the uncertain ones. These weak
labels are the **silver labels** the model trains on.

## Leakage control

Two infrastructure features measure how well a trajectory *follows* a specific bus route or rail line —
essentially the labeler's own verdict. They are dropped before training. The remaining spatial features
(proximity to rail, stops, motorway) are kept, because proximity is universal physics, not memorized
geography.

The test settles it: dropping the spatial group *costs the cross-dataset transfer* (0.79 -> 0.73) but
does **not** help the in-domain score. If the model were memorizing one city's geography, dropping
those features would hurt in-domain and help transfer — the opposite of what happens.

## Honest evaluation

- **Rolling-origin, never random.** Consecutive windows overlap by 50%, so a random split leaks
  near-duplicate windows between train and test and inflates accuracy by about 5 points. The model
  trains up to a time origin and is tested on the block that follows.
- **Three-way GPS stratification.** Every figure is split by GPS availability — present / sparse /
  absent — because a single pooled number hides the no-GPS floor (0.80 / 0.57 / 0.18).
- **Leave-one-user-out** for the "a new user installs the app tomorrow" question, reported with its
  per-user spread rather than just the mean.
- **Cross-dataset transfer** on SHL (a second country, with true ground truth) and GeoLife (a third
  continent, GPS-only) — to separate the limits of the method from the harshness of the deployment data.

## Deployment: the protocol transfers, not the model

In production the Trento model is not shipped to a new city. The **protocol** is re-run per context —
universal labeler -> silver labels -> local training -> post-processing — on that context's own public
maps and data, with no manual labels. The model is local by construction; what transfers is the method.
On SHL, re-running the whole protocol reaches a macro-F1 of 0.84, against 0.92 for a supervised model
trained on that dataset's real labels — a small, honest label-free cost on a second corpus.

## The use case

The deliverable is the aggregate, not the per-window label. A classifier over-counts some modes, so the
raw modal split carries a systematic bias; a standard prevalence correction (quantification,
Saerens/Forman) de-biases it from a small labeled calibration sample, and the corrected split maps to a
CO2 estimate within a survey-grade margin.
