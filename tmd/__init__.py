"""
tmd — ricostruzione pulita del modulo TMD (OpenMove / UniTN).

Mondo nuovo e autosufficiente: NON importa da `tmd/` (vecchio, congelato).
Si costruisce un componente alla volta — protocollo e stato in REBUILD_PLAN.md.
Legge data/raw/ (condiviso), scrive in data/v2/ (output nuovi, isolati dal vecchio).
"""
