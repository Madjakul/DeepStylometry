# deep_stylometry/experiments/speed_benchmark.py
"""Inference-speed benchmark for all interaction methods.

Generates random embeddings of shape ``(N_queries, S, H)`` and
``(N_docs, S, H)`` and times the scoring step with each interaction module.

Usage
-----
python -m deep_stylometry.experiments.speed_benchmark \\
    --n_queries 256 \\
    --n_docs 1024 \\
    --seq_len 512 \\
    --hidden 768 \\
    --n_repeats 5 \\
    --batch_size 64 \\
    --device cuda
"""

import argparse
import time
from typing import List

import torch
import torch.nn.functional as F


def _time_scoring(
    pool,
    q_embs: torch.Tensor,
    k_embs: torch.Tensor,
    q_mask: torch.Tensor,
    k_mask: torch.Tensor,
    q_ids: torch.Tensor,
    k_ids: torch.Tensor,
    batch_size: int,
    n_repeats: int,
) -> float:
    """Return mean wall-clock time (seconds) over ``n_repeats`` runs."""
    from deep_stylometry.modules.patch_interaction import PatchInteraction

    n_q = q_embs.size(0)
    n_k = k_embs.size(0)
    times: List[float] = []

    for _ in range(n_repeats):
        if q_embs.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for q_start in range(0, n_q, batch_size):
            q_end = min(q_start + batch_size, n_q)
            for k_start in range(0, n_k, batch_size):
                k_end = min(k_start + batch_size, n_k)

                kwargs = dict(
                    query_embs=q_embs[q_start:q_end],
                    key_embs=k_embs[k_start:k_end],
                    q_mask=q_mask[q_start:q_end],
                    k_mask=k_mask[k_start:k_end],
                )
                if isinstance(pool, PatchInteraction):
                    kwargs["q_input_ids"] = q_ids[q_start:q_end]
                    kwargs["k_input_ids"] = k_ids[k_start:k_end]
                else:
                    kwargs["q_input_ids"] = q_ids[q_start:q_end]

                with torch.no_grad():
                    _ = pool(**kwargs)

        if q_embs.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return sum(times) / len(times)


def run_benchmark(
    n_queries: int,
    n_docs: int,
    seq_len: int,
    hidden: int,
    n_repeats: int,
    batch_size: int,
    device_str: str,
    patch_sizes: List[int],
) -> None:
    from deep_stylometry.utils.configs import BaseConfig
    from deep_stylometry.modules.mean_interaction import MeanInteraction
    from deep_stylometry.modules.late_interaction import LateInteraction
    from deep_stylometry.modules.patch_interaction import PatchInteraction

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Random data
    q_embs = torch.randn(n_queries, seq_len, hidden, device=device)
    k_embs = torch.randn(n_docs, seq_len, hidden, device=device)
    q_mask = torch.ones(n_queries, seq_len, dtype=torch.long, device=device)
    k_mask = torch.ones(n_docs, seq_len, dtype=torch.long, device=device)
    # Simple token IDs (whitespace/wholeword patching would need real IDs)
    q_ids = torch.randint(1, 50000, (n_queries, seq_len), device=device)
    k_ids = torch.randint(1, 50000, (n_docs, seq_len), device=device)

    def _make_pli_cfg(method: str, size: int = 3) -> BaseConfig:
        cfg = BaseConfig()
        cfg.model.pooling_method = "pli"
        cfg.model.patch_method = method
        cfg.model.patch_size = size
        cfg.model.patch_compression = "mean"
        cfg.model.lm_hidden_size = hidden
        cfg.model.skip_list = False
        cfg.train.precision = "32"
        return cfg

    base_cfg = BaseConfig()
    base_cfg.model.skip_list = False
    base_cfg.train.precision = "32"

    systems = [
        ("MeanInteraction", MeanInteraction()),
        ("LateInteraction (ColBERT)", LateInteraction(base_cfg)),
    ]
    for k in patch_sizes:
        cfg = _make_pli_cfg("ngram", k)
        systems.append((f"PatchInteraction-NGram-{k}", PatchInteraction(cfg)))

    header = f"\n{'System':<35} {'Mean time (s)':>15} {'Relative':>10}"
    print(header)
    print("-" * 65)

    baseline_t = None
    for name, pool in systems:
        pool = pool.to(device).eval()
        t = _time_scoring(pool, q_embs, k_embs, q_mask, k_mask,
                          q_ids, k_ids, batch_size, n_repeats)
        if baseline_t is None:
            baseline_t = t
        rel = t / baseline_t if baseline_t else 1.0
        print(f"{name:<35} {t:>15.3f} {rel:>9.2f}×")

    print(
        f"\nSetup: N_q={n_queries}, N_docs={n_docs}, S={seq_len}, "
        f"H={hidden}, repeats={n_repeats}, device={device}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_queries", type=int, default=256)
    parser.add_argument("--n_docs", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patch_sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    args = parser.parse_args()

    run_benchmark(
        n_queries=args.n_queries,
        n_docs=args.n_docs,
        seq_len=args.seq_len,
        hidden=args.hidden,
        n_repeats=args.n_repeats,
        batch_size=args.batch_size,
        device_str=args.device,
        patch_sizes=args.patch_sizes,
    )
