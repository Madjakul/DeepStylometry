# deep_stylometry/experiments/mechanistic/run_phase.py
"""Central orchestrator for the mechanistic interpretability study.

Usage:
    python -m deep_stylometry.experiments.mechanistic.run_phase \\
        --config configs/mechanistic.yml \\
        --phase 0 \\
        [--models mean li pli_ngram2 e5] \\
        [--checkpoints final] \\
        [--resume] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase DAG
# ---------------------------------------------------------------------------

# Maps phase -> list of prerequisite phases
_PHASE_DEPS: Dict[str, List[str]] = {
    "0":  [],
    "1a": ["0"],
    "1b": ["0", "1a"],
    "2":  ["0"],
    "3":  ["0", "1a"],
    "4":  ["0", "2"],
}


def _validate_dag(phases: List[str], resume: bool, root: Path) -> None:
    """Check that all prerequisites are satisfied or will be run."""
    scheduled = set(phases)
    for phase in phases:
        for dep in _PHASE_DEPS.get(phase, []):
            if dep not in scheduled:
                # Check if output already exists
                dep_output = _phase_output_exists(dep, root)
                if not dep_output:
                    if not resume:
                        logger.warning(
                            "Phase %s depends on phase %s, "
                            "which is not scheduled and has no cached output.",
                            phase, dep,
                        )


def _phase_output_exists(phase: str, root: Path) -> bool:
    """Quick check for phase-level cached output."""
    checks = {
        "0":  root / "phase0_probe_set" / "probe_set.json",
        "1a": root / "phase1_lisa" / "features.parquet",
        "1b": root / "phase1_probes",  # directory
        "2":  root / "phase2_patching",
        "3":  root / "phase3_dynamics" / "summary.parquet",
        "4":  root / "phase4_distractors",
    }
    p = checks.get(phase)
    if p is None:
        return False
    return p.exists()


# ---------------------------------------------------------------------------
# Results summary generator
# ---------------------------------------------------------------------------

def _generate_results_md(cfg, root: Path) -> None:
    lines = ["# Mechanistic Study Results\n"]

    # Phase 2 inflection layers
    lines.append("## Inflection Layers (Phase 2)\n")
    for model_id in cfg.models:
        from deep_stylometry.experiments.mechanistic.io_utils import phase2_path
        rec_path = phase2_path(root, model_id, "final", "recovery.npz")
        if not rec_path.exists():
            continue
        import numpy as np
        data = np.load(rec_path, allow_pickle=True)
        recovery = data["recovery"]
        tl = list(data["tier_labels"])
        a_idx = [i for i, t in enumerate(tl) if t == "A"]
        if a_idx:
            mean_rec = recovery[a_idx].mean(axis=0)
            inflection = next((i for i, r in enumerate(mean_rec) if r >= 50.0), int(mean_rec.argmax()))
            lines.append(f"- **{model_id}**: inflection layer = {inflection}  "
                         f"(max recovery = {mean_rec.max():.1f}%)\n")

    # Phase 1 top LISA features
    lines.append("\n## Top LISA Features by R² (Phase 1b, final checkpoint)\n")
    for model_id in cfg.models:
        from deep_stylometry.experiments.mechanistic.io_utils import phase1_probe_path
        probe_path = phase1_probe_path(root, model_id, "final", "probes.npz")
        if not probe_path.exists():
            continue
        import numpy as np
        data = np.load(probe_path, allow_pickle=True)
        r2 = data["r2_per_layer_per_feature"]
        feat_names = list(data["feature_names"])
        max_r2 = r2.max(axis=0)
        top5_idx = max_r2.argsort()[-5:][::-1]
        lines.append(f"**{model_id}**: " + ", ".join(
            f"{feat_names[i]} (R²={max_r2[i]:.3f})" for i in top5_idx
        ) + "\n")

    # Phase 4 failure rates
    lines.append("\n## Predictable Failure Rates (Phase 4)\n")
    for model_id in cfg.models:
        from deep_stylometry.experiments.mechanistic.io_utils import phase4_path
        rank_path = phase4_path(root, model_id, "rankings.parquet")
        if not rank_path.exists():
            continue
        import pandas as pd
        df = pd.read_parquet(rank_path)
        for tier in ["B", "C"]:
            t_df = df[df["tier"] == tier]
            if len(t_df) == 0:
                continue
            fail_rate = t_df["neg_higher_than_pos"].mean()
            lines.append(
                f"- **{model_id} Tier {tier}**: "
                f"{fail_rate:.1%} failure rate (n={len(t_df)})\n"
            )

    lines.append("\n## Observations\n")
    lines.append("*(Fill in after reviewing figures)*\n")

    out = root / "RESULTS.md"
    with open(out, "w") as f:
        f.writelines(lines)
    logger.info("Saved RESULTS.md to %s", out)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase_0(cfg, root, resume, dry_run) -> None:
    from deep_stylometry.experiments.mechanistic.probe_set import build_probe_set
    if dry_run:
        print("[DRY-RUN] Would run Phase 0: build probe set")
        return
    build_probe_set(cfg, resume=resume)


def run_phase_1a(cfg, root, resume, dry_run) -> None:
    from deep_stylometry.experiments.mechanistic.probe_set import build_probe_set
    from deep_stylometry.experiments.mechanistic.lisa_features import build_lisa_corpus
    if dry_run:
        print("[DRY-RUN] Would run Phase 1a: extract LISA features")
        return
    probe_set = build_probe_set(cfg, resume=True)
    build_lisa_corpus(cfg, probe_set, resume=resume)


def run_phase_1b(cfg, root, models, steps, resume, dry_run) -> None:
    """Extract activations + train probes for each (model, step)."""
    import torch
    from deep_stylometry.experiments.mechanistic.activation_extractor import (
        load_ds_model, load_e5_model, extract_and_cache,
    )
    from deep_stylometry.experiments.mechanistic.linear_probes import run_phase1b
    from deep_stylometry.experiments.mechanistic.residual_patching import _resolve_checkpoint

    if dry_run:
        for m in models:
            for s in steps:
                print(f"[DRY-RUN] Would extract activations + train probes: model={m} step={s}")
        return

    import pandas as pd
    feat_path = root / "phase1_lisa" / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError("LISA features not found. Run Phase 1a first.")

    df = pd.read_parquet(feat_path)
    train_rows = df[df["split"] == "train"]
    eval_rows = df[df["split"] == "eval"]
    train_texts = list(train_rows.get("text_preview", train_rows.iloc[:, 3]))
    eval_texts = list(eval_rows.get("text_preview", eval_rows.iloc[:, 3]))
    train_doc_ids = list(train_rows["doc_id"])
    eval_doc_ids = list(eval_rows["doc_id"])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_id in models:
        for step in steps:
            logger.info("Phase 1b: model=%s step=%s", model_id, step)
            model_entry = cfg.models.get(model_id)
            if model_entry is None:
                logger.warning("Unknown model_id %s", model_id)
                continue

            if model_id == "e5":
                model, tok = load_e5_model(device)
                is_e5 = True
                tok_for_extraction = tok
            else:
                ckpt = _resolve_checkpoint(model_entry.checkpoint_pattern, step)
                model, _ = load_ds_model(model_entry.config, ckpt, device)
                is_e5 = False
                tok_for_extraction = tokenizer

            model.eval()

            extract_and_cache(
                model_id, step, train_texts, train_doc_ids,
                model, tok_for_extraction, device, root,
                "probe_train", is_e5, resume,
            )
            extract_and_cache(
                model_id, step, eval_texts, eval_doc_ids,
                model, tok_for_extraction, device, root,
                "probe_eval", is_e5, resume,
            )

            run_phase1b(cfg, model_id, step, resume=resume)


def run_phase_2(cfg, root, models, steps, resume, dry_run) -> None:
    from deep_stylometry.experiments.mechanistic.probe_set import build_probe_set
    from deep_stylometry.experiments.mechanistic.residual_patching import run_phase2

    if dry_run:
        for m in models:
            for s in steps:
                print(f"[DRY-RUN] Would run Phase 2 patching: model={m} step={s}")
        return

    probe_set = build_probe_set(cfg, resume=True)
    for model_id in models:
        for step in steps:
            logger.info("Phase 2: model=%s step=%s", model_id, step)
            if model_id == "e5":
                logger.info("Skipping Phase 2 patching for E5 (not implemented).")
                continue
            run_phase2(cfg, model_id, step, probe_set, resume=resume)


def run_phase_3(cfg, root, models, steps, resume, dry_run) -> None:
    from deep_stylometry.experiments.mechanistic.training_dynamics import run_phase3
    if dry_run:
        print(f"[DRY-RUN] Would run Phase 3 dynamics: models={models}")
        return
    run_phase3(cfg, model_ids=models, steps=steps, resume=resume)


def run_phase_4(cfg, root, models, resume, dry_run) -> None:
    from deep_stylometry.experiments.mechanistic.probe_set import build_probe_set
    from deep_stylometry.experiments.mechanistic.distractor_analysis import run_phase4
    if dry_run:
        for m in models:
            print(f"[DRY-RUN] Would run Phase 4 distractors: model={m}")
        return
    probe_set = build_probe_set(cfg, resume=True)
    for model_id in models:
        run_phase4(cfg, model_id, probe_set, resume=resume)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mechanistic interpretability study orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/mechanistic.yml",
                        help="Path to mechanistic.yml config.")
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=["0", "1a", "1b", "2", "3", "4", "all"],
        default=["all"],
        help="Phase(s) to run.",
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model IDs to process. Default: all in config.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Checkpoint steps, e.g. final 5000 0. Default: from config.",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip phases with cached output.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned work and exit.")
    args = parser.parse_args()

    from deep_stylometry.experiments.mechanistic.config import MechanisticConfig
    from deep_stylometry.experiments.mechanistic.io_utils import output_root

    cfg = MechanisticConfig.from_yaml(args.config)
    root = output_root(cfg)
    root.mkdir(parents=True, exist_ok=True)

    # Resolve models
    if args.models is not None:
        models = args.models
    else:
        models = list(cfg.models.keys())

    # Resolve checkpoints
    if args.checkpoints is not None:
        steps = []
        for s in args.checkpoints:
            steps.append(s if s == "final" else int(s))
    else:
        steps = cfg.checkpoints.selected_steps

    # Expand 'all' to ordered phase list
    phases_raw = args.phase
    if "all" in phases_raw:
        phases = ["0", "1a", "1b", "2", "3", "4"]
    else:
        # Deduplicate while preserving order
        seen = set()
        phases = []
        for p in phases_raw:
            if p not in seen:
                phases.append(p)
                seen.add(p)

    _validate_dag(phases, args.resume, root)

    dry = args.dry_run

    if dry:
        print(f"Config:  {args.config}")
        print(f"Output:  {root}")
        print(f"Phases:  {phases}")
        print(f"Models:  {models}")
        print(f"Steps:   {steps}")
        print(f"Resume:  {args.resume}")
        print()

    for phase in phases:
        logger.info("=== Phase %s ===", phase)
        if phase == "0":
            run_phase_0(cfg, root, args.resume, dry)
        elif phase == "1a":
            run_phase_1a(cfg, root, args.resume, dry)
        elif phase == "1b":
            run_phase_1b(cfg, root, models, steps, args.resume, dry)
        elif phase == "2":
            run_phase_2(cfg, root, models, steps, args.resume, dry)
        elif phase == "3":
            run_phase_3(cfg, root, models, steps, args.resume, dry)
        elif phase == "4":
            run_phase_4(cfg, root, models, args.resume, dry)

    if not dry and "4" in phases:
        _generate_results_md(cfg, root)
        logger.info("Study complete. See %s/RESULTS.md", root)


if __name__ == "__main__":
    main()
