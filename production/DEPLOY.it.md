# Deployment di openmove-tmd

Il percorso di produzione: una pipeline, un modello, nessuna variante di ricerca. Lo opera OpenMove;
contenerizzazione e scheduling sono scelte vostre. I numeri della tesi sono prodotti dallo stesso
pacchetto `tmd`, quindi la produzione li riproduce esatti.

## Installazione

    pip install .          # installa il comando `tmd`
    tmd --help

I segreti stanno nell'ambiente, mai nel repo: `MONGO_URI` (solo per `tmd ingest`).
La config per-citta e in `tmd/configs/cities/<citta>.yaml`; per aggiungerne una copia `city.example.yaml`.

## Due percorsi

**A — Parti subito (Trento).** Il modello Trento allenato e incluso in `models/`. Classifica i dati recenti:

    tmd ingest  --city trento          # pull incrementale (cursori); richiede MONGO_URI
    tmd process --city trento          # raw -> feature
    tmd predict --model models/trento_20260612_202641.pkl --features data/v2/features_trento.parquet
    tmd aggregate                      # modal-split + CO2

oppure concatenato:

    tmd run --city trento

**B — Una citta nuova (o un refresh).** Ri-esegui il protocollo sulle mappe pubbliche locali, senza etichette manuali:

    tmd build-index --city <x>         # indice spaziale GTFS/OSM (una volta)
    tmd ingest --city <x>
    tmd build-model --city <x>         # labeler universale -> silver -> modello locale

Il modello e sempre locale per costruzione; cio che si trasferisce e il protocollo.

## L'aggregato corretto

`tmd aggregate` da il modal-split naive. Il deliverable e l'aggregato *corretto* (modal-split -> CO2):
la correzione di prevalenza (quantification) lo de-biasa, ma richiede un piccolo set di calibrazione
etichettato (~400 finestre moving). Sotto ~200 peggiorerebbe, quindi emette il naive con un caveat.

## Contenerizzazione (OpenMove)

Il pacchetto e un'app installabile; il container e poche righe:

    FROM python:3.11-slim
    COPY . /app
    RUN pip install /app
    # CMD ["tmd", "run", "--city", "trento"]   # cadenza e scheduling: il vostro orchestratore

I dati non stanno mai nell'immagine (privacy della mobilita utenti): montateli, o scaricateli con `tmd ingest`.
