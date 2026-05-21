#!/usr/bin/env python3
# deep_stylometry/experiments/retrieval_inspection_reaggregate.py
"""Post-hoc re-aggregation tool for retrieval inspection results.

Reads an existing ``pairs.jsonl`` and ``random_pairs.jsonl`` produced by
``retrieval_inspection.py`` and re-runs aggregation with the corrected ratio
and rank-bin logic, writing updated ``summary.json`` and ``report.txt``.

This lets you fix Issue 1 (zero-baseline inf-ratio reporting) entirely from
existing data. For Issue 2 (full-pool rank bins), old pairs.jsonl files have
``true_positive_rank=None`` for queries whose TP fell outside top-K; those
queries are reported under a ``>top_k`` bucket with a clear warning — the
21-50, 51-100, 101-500, >500 bins cannot be recovered without rerunning.

Run::

    python -m deep_stylometry.experiments.retrieval_inspection_reaggregate \\
        --input_dir ./analysis/retrieval_inspection_base4_ngram3 \\
        --output_dir ./analysis/retrieval_inspection_base4_ngram3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deep_stylometry.utils.retrieval_aggregation import (
    _aggregate_pair_records,
    _sanitize_for_json,
    _write_report,
)

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)
    return records


def reaggregate(
    input_dir: Path,
    output_dir: Path,
    top_k: Optional[int] = None,
    subset: str = "unknown",
    n_seeds: int = 0,
    n_queries: int = 0,
) -> Dict[str, Any]:
    """Re-aggregate retrieval inspection results from existing JSONL files.

    Args:
        input_dir: Directory containing ``pairs.jsonl`` and optionally
            ``random_pairs.jsonl`` and ``config_snapshot.yml``.
        output_dir: Directory to write updated ``summary.json`` and ``report.txt``.
        top_k: K used during the original retrieval run. If None, attempts to
            read from ``config_snapshot.yml``; falls back to 100.
        subset: Subset name for the report header.
        n_seeds: Number of seeds for the report header.
        n_queries: Queries per seed for the report header.

    Returns:
        The updated summary dict (also written to output_dir/summary.json).
    """
    pairs_path = input_dir / "pairs.jsonl"
    random_pairs_path = input_dir / "random_pairs.jsonl"
    snapshot_path = input_dir / "config_snapshot.yml"

    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs.jsonl not found in {input_dir}")

    # Load config_snapshot.yml to fill in missing args.
    snap: Dict[str, Any] = {}
    if snapshot_path.exists():
        with open(snapshot_path, encoding="utf-8") as fh:
            snap = yaml.safe_load(fh) or {}

    cli = snap.get("_cli_args", {})
    if top_k is None:
        top_k = int(cli.get("top_k", 100))
    if subset == "unknown":
        subset = cli.get("subset", "unknown")
    if n_seeds == 0:
        n_seeds = int(cli.get("n_seeds", 0))
    if n_queries == 0:
        n_queries = int(cli.get("n_queries_per_seed", 0))

    logger.info("Loading pairs from %s …", pairs_path)
    pair_records = _load_jsonl(pairs_path)
    logger.info("Loaded %d pair records.", len(pair_records))

    random_pairs: List[Dict[str, Any]] = []
    if random_pairs_path.exists():
        logger.info("Loading random pairs from %s …", random_pairs_path)
        random_pairs = _load_jsonl(random_pairs_path)
        logger.info("Loaded %d random pair records.", len(random_pairs))
    else:
        logger.warning(
            "random_pairs.jsonl not found in %s. "
            "Ratio blocks will show 'Insufficient data'. "
            "Re-run retrieval_inspection.py to generate this file.",
            input_dir,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _aggregate_pair_records(
        pair_records, random_pairs, top_k, stale_ranks=True
    )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize_for_json(summary), fh, indent=2, sort_keys=True, default=str)
    logger.info("Wrote %s", summary_path)

    report_path = output_dir / "report.txt"
    _write_report(report_path, summary, subset, n_seeds, n_queries)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Re-aggregate retrieval inspection results from existing pairs.jsonl "
            "and random_pairs.jsonl, producing corrected summary.json and report.txt."
        )
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing pairs.jsonl (and optionally random_pairs.jsonl, config_snapshot.yml).",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write updated summary.json and report.txt.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="K used in the original retrieval run (read from config_snapshot.yml if omitted).",
    )
    p.add_argument(
        "--subset",
        default="unknown",
        help="Subset name for the report header (read from config_snapshot.yml if omitted).",
    )
    p.add_argument(
        "--n_seeds",
        type=int,
        default=0,
        help="Number of seeds for the report header (read from config_snapshot.yml if omitted).",
    )
    p.add_argument(
        "--n_queries",
        type=int,
        default=0,
        help="Queries per seed for the report header (read from config_snapshot.yml if omitted).",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    reaggregate(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        subset=args.subset,
        n_seeds=args.n_seeds,
        n_queries=args.n_queries,
    )


if __name__ == "__main__":
    main()
