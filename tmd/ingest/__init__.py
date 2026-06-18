"""tmd.ingest — dump Mongo → data/raw/ (Step 3). pymongo confinato qui.

Entry point: `python -m tmd.ingest.dump [--full] [--until DATA] [--dry-run] ...`
(L'__init__ non importa dump/mongo_source: così `import tmd.ingest` non tira pymongo.)
"""
