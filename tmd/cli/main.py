"""tmd — CLI unificata della pipeline TMD.

Comandi (operazioni, non varianti):
  ingest       scarica i dati grezzi (Mongo -> data/raw)         [richiede MONGO_URI]
  build-index  costruisce l'indice spaziale GTFS/OSM             [una volta per citta']
  process      raw -> finestre + feature (parquet)
  build-model  silver labeling + training del modello (ricetta fissa)
  predict      feature -> modi predetti + post-processing
  aggregate    predizioni -> modal-split + CO2                   [il deliverable d'uso]
  run          process -> predict -> aggregate (job ricorrente)
  label train transfer   passi singoli della pipeline

Esempi:
  tmd build-model --city trento
  tmd run --city trento
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

# Building block -> modulo da eseguire (passthrough degli argomenti).
PASSTHROUGH = {
    "ingest":      "tmd.ingest.dump",
    "build-index": "tmd.spatial.build_index",
    "process":     "tmd.cli.run_pipeline",
    "label":       "tmd.cli.label_silver",
    "train":       "tmd.cli.train",
    "predict":     "tmd.cli.predict",
    "transfer":    "tmd.cli.eval_transfer",
    "aggregate":   "tmd.cli.aggregate_cmd",
}


def _run(module: str, argv: list[str]) -> None:
    sys.argv = [module.rsplit(".", 1)[-1]] + argv
    runpy.run_module(module, run_name="__main__")


def _opt(argv: list[str], name: str, default: str) -> str:
    return argv[argv.index(name) + 1] if name in argv else default


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]

    if cmd in PASSTHROUGH:
        _run(PASSTHROUGH[cmd], rest)

    elif cmd == "build-model":
        # Ricetta congelata: silver labeling + RF gerarchico, rolling-origin, locale.
        city = _opt(rest, "--city", "trento")
        _run("tmd.cli.label_silver", ["--city", city])
        _run("tmd.cli.train", [
            "--parquet", f"data/v2/features_{city}.parquet",
            "--city", city, "--groups", "A,B,C,D",
            "--source", "silver", "--eval-strategy", "rolling",
            "--clf", "rf", "--no-specialists",
        ])

    elif cmd == "run":
        # Job ricorrente: feature -> predizioni col modello locale -> aggregato.
        city = _opt(rest, "--city", "trento")
        _run("tmd.cli.run_pipeline", ["--city", city])
        latest = json.loads(Path(f"data/v2/models/latest_{city}.json").read_text())
        _run("tmd.cli.predict", ["--model", latest["path"],
                                 "--features", f"data/v2/features_{city}.parquet", "--city", city])
        _run("tmd.cli.aggregate_cmd", [])

    else:
        sys.exit(f"comando sconosciuto: {cmd!r} (usa 'tmd --help')")


if __name__ == "__main__":
    main()
