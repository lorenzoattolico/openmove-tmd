"""
scripts/eval_transfer.py — Valutazione cross-dataset (Trento → SHL o altro).

Carica uno o più modelli dal registry, li applica a un parquet target,
calcola F1 sulle classi comuni e stampa una tabella comparativa.

Esempi:
  # Confronta tutti i modelli trento_* salvati oggi su SHL validate
  python scripts/eval_transfer.py \\
      --models trento_20260528_124820 trento_20260527_175640 \\
      --target data/processed/features_shl_bootstrap.parquet \\
      --filter-split validate \\
      --classes Still Walk Car Bus Train

  # Usa l'ultimo modello trento e confronta raw vs smooth
  python scripts/eval_transfer.py \\
      --city trento \\
      --target data/processed/features_shl_bootstrap.parquet \\
      --filter-split validate \\
      --smooth
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # tmd/cli -> repo
sys.path.insert(0, str(PROJECT_ROOT))

from tmd.evaluation.transfer import evaluate_transfer, evaluate_rule_based
from tmd.models.registry import load_latest, list_models


def parse_args():
    p = argparse.ArgumentParser()

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--models",  nargs="+",
                     help="Nomi modelli dal registry (es. trento_20260528_124820) "
                          "o path .pkl")
    src.add_argument("--city",    help="Usa il modello latest per questa città")
    src.add_argument("--rule-based", dest="rule_based", action="store_true",
                     help="Valuta solo il classificatore rule-based (window_labeler)")

    p.add_argument("--target",       required=True,
                   help="Parquet con feature + label GT su cui valutare")
    p.add_argument("--filter-split", dest="filter_split", default=None,
                   help="Filtra a df['split'] == valore (es. 'validate' per SHL)")
    p.add_argument("--classes",      nargs="*", default=None,
                   help="Classi da includere nella valutazione "
                        "(default: intersezione automatica). "
                        "Es: --classes Still Walk Car Bus Train")
    p.add_argument("--smooth",       action="store_true",
                   help="Applica smoothing post-inferenza per sessione")
    p.add_argument("--with-rule-based", dest="with_rule_based", action="store_true",
                   help="Aggiunge il rule-based al confronto insieme ai modelli ML")
    p.add_argument("--city-cfg",     dest="city_cfg", default="trento",
                   help="Città da cui caricare la config per il rule-based (default: trento)")
    p.add_argument("--registry",     default="data/models")
    p.add_argument("--out-dir",      dest="out_dir", default="data/v2/processed",
                   help="Cartella dove salvare gli eval parquet (data/processed = era-1)")
    p.add_argument("--no-save",      dest="no_save", action="store_true",
                   help="Non salvare eval parquet")
    return p.parse_args()


def load_city_cfg(city: str):
    from tmd.config import CityConfig
    cfg_path = PROJECT_ROOT / "tmd" / "configs" / "cities" / f"{city}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config città non trovata: {cfg_path}")
    return CityConfig.from_yaml(cfg_path)


def resolve_models(args) -> list[tuple[str, Path]]:
    """Ritorna lista (tag, pkl_path) per ogni modello richiesto."""
    if getattr(args, "rule_based", False):
        return []

    registry_dir = PROJECT_ROOT / args.registry

    if args.city:
        _, info = load_latest(args.city, registry_dir)
        pkl = Path(info["path"])
        return [(pkl.stem, pkl)]

    results = []
    for m in args.models:
        p = Path(m)
        if p.exists():
            results.append((p.stem, p))
        elif (registry_dir / f"{m}.pkl").exists():
            results.append((m, registry_dir / f"{m}.pkl"))
        else:
            print(f"[WARN] Modello non trovato: {m} — skipped")
    return results


def print_table(all_metrics: list[dict]):
    # Unione ordinata di tutte le classi presenti in qualsiasi modello —
    # evita che la tabella mostri solo le classi del primo modello.
    seen_classes: dict[str, None] = {}
    for m in all_metrics:
        for c in m.get("eval_classes", []):
            seen_classes[c] = None
    classes = list(seen_classes)

    header = (f"{'Modello':<38} {'Source':<12} {'Groups':<10} "
              f"{'Trento F1':>9} {'SHL F1':>8} {'Abstain':>8}")
    for c in classes:
        header += f"  {c:>6}"
    print()
    print(header)
    print("─" * len(header))

    for m in all_metrics:
        trento = f"{m['trento_f1']:.4f}" if m["trento_f1"] else "—"
        abstain = f"{m.get('abstain_rate', 0.0):.1%}" if m.get("abstain_rate") else "—"
        row = (f"{m['model_tag']:<38} {m['source']:<12} {m['groups']:<10} "
               f"{trento:>9} {m['transfer_f1']:>8.4f} {abstain:>8}")
        for c in classes:
            f = m["f1_per_class"].get(c)
            row += f"  {f:>6.3f}" if f is not None else "     —"
        print(row)
    print()


def main():
    args = parse_args()

    import numpy as np

    df_target = pd.read_parquet(args.target)
    print(f"Target: {args.target}  ({len(df_target):,} finestre)")
    if args.filter_split and "split" in df_target.columns:
        n_split = (df_target["split"] == args.filter_split).sum()
        print(f"  split='{args.filter_split}': {n_split:,} finestre")

    # Inferisci win_s dal parquet — usato da evaluate_transfer quando smooth=True.
    # Hardcodare 120.0 dava smooth_window sbagliato su parquet 60s/30s.
    win_s = 120.0
    if "ts_start" in df_target.columns and "ts_end" in df_target.columns and len(df_target) > 0:
        win_s = float(np.median((df_target["ts_end"] - df_target["ts_start"]).values / 1000))
    print(f"  win_s inferito: {win_s:.0f}s")

    all_metrics = []
    eval_frames = []

    # ── Modelli ML ──────────────────────────────────────────────────────────────
    models = resolve_models(args)
    if not models and not getattr(args, "rule_based", False) \
            and not getattr(args, "with_rule_based", False):
        print("Nessun modello trovato.")
        sys.exit(1)

    for tag, pkl_path in models:
        print(f"\n── {tag} ──")
        try:
            metrics, eval_df = evaluate_transfer(
                model_path   = pkl_path,
                df_target    = df_target,
                classes      = args.classes,
                filter_split = args.filter_split,
                smooth       = args.smooth,
                win_s        = win_s,
            )
            print(f"  F1 transfer: {metrics['transfer_f1']:.4f}  "
                  f"(Trento: {metrics['trento_f1']})")
            for c, f in metrics["f1_per_class"].items():
                print(f"    {c:<10}: {f:.4f}")
            all_metrics.append(metrics)
            eval_frames.append(eval_df)
        except Exception as e:
            print(f"  [ERRORE] {e}")

    # ── Rule-based ──────────────────────────────────────────────────────────────
    if getattr(args, "rule_based", False) or getattr(args, "with_rule_based", False):
        print(f"\n── rule_based (window_labeler_universal, soglie {args.city_cfg}) ──")
        try:
            city_cfg = load_city_cfg(args.city_cfg)
            metrics, eval_df = evaluate_rule_based(
                df_target    = df_target,
                city_cfg     = city_cfg,
                classes      = args.classes,
                filter_split = args.filter_split,
                model_tag    = f"rule_based_{args.city_cfg}",
            )
            print(f"  F1 transfer: {metrics['transfer_f1']:.4f}  "
                  f"(abstain: {metrics['abstain_rate']:.1%})")
            for c, f in metrics["f1_per_class"].items():
                print(f"    {c:<10}: {f:.4f}")
            all_metrics.append(metrics)
            eval_frames.append(eval_df)
        except Exception as e:
            print(f"  [ERRORE] {e}")

    if not all_metrics:
        print("Nessun modello valutato con successo.")
        sys.exit(1)

    print_table(all_metrics)

    if not args.no_save and eval_frames:
        out_dir = PROJECT_ROOT / args.out_dir
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        target_stem = Path(args.target).stem

        for metrics, eval_df in zip(all_metrics, eval_frames):
            tag = metrics["model_tag"]
            out_path = out_dir / f"transfer_{tag}_on_{target_stem}_{ts}.parquet"
            eval_df.to_parquet(out_path, index=False)
            print(f"Salvato: {out_path.name}")

        summary_path = out_dir / f"transfer_summary_{target_stem}_{ts}.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Sommario: {summary_path.name}")


if __name__ == "__main__":
    main()
