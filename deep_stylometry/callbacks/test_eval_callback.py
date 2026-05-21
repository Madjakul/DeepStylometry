# deep_stylometry/callbacks/test_eval_callback.py

import logging
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, List, Optional, Tuple

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ranx import Run
from tqdm import tqdm

from deep_stylometry.modules.late_interaction import LateInteraction
from deep_stylometry.modules.mean_interaction import MeanInteraction
from deep_stylometry.modules.patch_interaction import PatchInteraction
from deep_stylometry.utils.eval_utils import (build_qrels, evaluate_run,
                                              gather_targets)

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig

logger = logging.getLogger(__name__)


class _CenteredMeanPool(nn.Module):
    """Mean-pool + subtract trained running mean + L2-normalise.

    Mirrors the ``_pool_and_center`` path in ``DeepStylometry.training_step``
    but without the running-mean *update* — eval mode only.

    Parameters
    ----------
    mean_centerer : nn.Module
        The trained ``MeanCenterer`` instance from
        ``pl_module.mean_centerer``.  Its ``mu`` buffer carries the
        running mean accumulated during training.
    """

    def __init__(self, mean_centerer: nn.Module) -> None:
        super().__init__()
        self.centerer = mean_centerer

    def forward(
        self,
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        def pool_and_center(embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            m = mask.unsqueeze(-1).float()
            pooled = (embs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            return self.centerer.apply(pooled)

        q_vec = pool_and_center(query_embs, q_mask)
        k_vec = pool_and_center(key_embs, k_mask)
        return torch.matmul(q_vec, k_vec.T)


class TestEvalCallback(L.Callback):
    """Full-corpus retrieval evaluation callback for test time.

    Collects per-token embeddings from ``test_step`` outputs into an HDF5
    file (document side) and CPU tensors (query side), then scores all queries
    against all documents using mean pooling, full late interaction, and (when
    configured) patch-level late interaction. Reports MRR, nDCG, Recall, and
    accuracy at k in {5, 10, 20, 100}.

    Parameters
    ----------
    cfg : BaseConfig
        Global configuration.
    k : int, optional
        Number of top-k documents to retrieve (default: 100).
    q_chunk : int, optional
        Query batch size during full-corpus scoring (default: 64).
    k_chunk : int, optional
        Document batch size during full-corpus scoring (default: 64).
    max_seq_len : int, optional
        Maximum sequence length to store per embedding (default: 512).
    save_scores_dir : str, optional
        If given, saves ``{dense,li,pli}_topk_{scores,indices}.npy`` here.
    """

    def __init__(
        self,
        cfg: "BaseConfig",
        k: int = 100,
        q_chunk: int = 64,
        k_chunk: int = 64,
        max_seq_len: int = 512,
        save_scores_dir: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.K = k
        self.Q_CHUNK = q_chunk
        self.K_CHUNK = k_chunk
        self.max_seq_len = max_seq_len
        self.save_scores_dir = save_scores_dir
        self.tmp_dir = tempfile.mkdtemp()
        self.h5_path = os.path.join(self.tmp_dir, "corpus.h5")

    def _reset(self):
        self.q_embs: List[torch.Tensor] = []
        self.q_masks: List[torch.Tensor] = []
        self.q_ids: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.n_batches = 0
        self._triplet_correct: int = 0
        self._triplet_total: int = 0

    def on_test_epoch_start(self, trainer, pl_module):
        self._reset()
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.h5_file = h5py.File(self.h5_path, "w")
        self.h5_datasets = {}
        logger.info(f"TestEvalCallback: HDF5 → {self.h5_path}")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # --- Triplet accuracy (per-batch, no extra pass needed) ---
        # Mirrors validation_step logic: pad pos/neg to same seq len, score, compare.
        # Uses the model's own trained pool — the most informative triplet signal.
        pool = pl_module.contrastive_loss.pool
        q = outputs["q_embs"]
        qm = outputs["q_mask"]
        q_ids = outputs["q_input_ids"]

        pe, ne = outputs["pos_embs"], outputs["neg_embs"]
        pm, nm = outputs["pos_mask"], outputs["neg_mask"]
        p_ids, n_ids = outputs["pos_input_ids"], outputs["neg_input_ids"]

        # Pad pos/neg to same seq len (same logic as validation_step)
        max_seq = max(pe.size(1), ne.size(1))
        pe = F.pad(pe, (0, 0, 0, max_seq - pe.size(1)))
        ne = F.pad(ne, (0, 0, 0, max_seq - ne.size(1)))
        pm = F.pad(pm, (0, max_seq - pm.size(1)))
        nm = F.pad(nm, (0, max_seq - nm.size(1)))
        p_ids = F.pad(p_ids, (0, max_seq - p_ids.size(1)))
        n_ids = F.pad(n_ids, (0, max_seq - n_ids.size(1)))

        # Concatenate into key set: [pos; neg]
        k_embs = torch.cat([pe, ne], dim=0)
        k_mask = torch.cat([pm, nm], dim=0)
        k_ids = torch.cat([p_ids, n_ids], dim=0)

        # Build pool kwargs (same dispatch as InfoNCELoss)
        pool_kw = dict(query_embs=q, key_embs=k_embs, q_mask=qm, k_mask=k_mask)
        if isinstance(pool, (LateInteraction, PatchInteraction)):
            pool_kw["q_input_ids"] = q_ids
        if isinstance(pool, PatchInteraction):
            pool_kw["k_input_ids"] = k_ids

        all_scores = pool(**pool_kw)  # (B, 2B)
        bs = q.size(0)
        targets = torch.arange(bs, device=q.device)
        poss = all_scores[targets, targets]
        negs = all_scores[targets, targets + bs]
        self._triplet_correct += (poss > negs).sum().item()
        self._triplet_total += bs

        S = self.max_seq_len

        def pad(t, s):
            diff = s - t.size(1)
            if diff > 0:
                return F.pad(t, (0, 0, 0, diff) if t.dim() == 3 else (0, diff))
            return t[:, :s] if t.dim() == 2 else t[:, :s, :]

        # Queries → RAM
        self.q_embs.append(pad(outputs["q_embs"], S).half().cpu())
        self.q_masks.append(pad(outputs["q_mask"], S).cpu())
        self.q_ids.append(pad(outputs["q_input_ids"], S).cpu())

        # Documents → HDF5
        for prefix, embs_key, mask_key, ids_key in [
            ("pos", "pos_embs", "pos_mask", "pos_input_ids"),
            ("neg", "neg_embs", "neg_mask", "neg_input_ids"),
        ]:
            embs_np = pad(outputs[embs_key], S).half().cpu().numpy()
            masks_np = pad(outputs[mask_key], S).cpu().to(torch.int8).numpy()
            bs = embs_np.shape[0]

            for suffix, data in [("embs", embs_np), ("masks", masks_np)]:
                key = f"{prefix}_{suffix}"
                if key not in self.h5_datasets:
                    self.h5_datasets[key] = self.h5_file.create_dataset(
                        key,
                        shape=(0, *data.shape[1:]),
                        maxshape=(None, *data.shape[1:]),
                        dtype=data.dtype,
                        chunks=(min(64, bs), *data.shape[1:]),
                    )
                ds = self.h5_datasets[key]
                old = ds.shape[0]
                ds.resize(old + bs, axis=0)
                ds[old : old + bs] = data

            # Store input_ids if available in outputs
            if ids_key in outputs and outputs[ids_key] is not None:
                ids_np = pad(outputs[ids_key], S).cpu().to(torch.int32).numpy()
                key_ids = f"{prefix}_ids"
                if key_ids not in self.h5_datasets:
                    self.h5_datasets[key_ids] = self.h5_file.create_dataset(
                        key_ids,
                        shape=(0, *ids_np.shape[1:]),
                        maxshape=(None, *ids_np.shape[1:]),
                        dtype=ids_np.dtype,
                        chunks=(min(64, bs), *ids_np.shape[1:]),
                    )
                ds_ids = self.h5_datasets[key_ids]
                old = ds_ids.shape[0]
                ds_ids.resize(old + bs, axis=0)
                ds_ids[old : old + bs] = ids_np

        if outputs["target_indices"] is not None:
            self.targets.append(outputs["target_indices"].cpu())

        self.n_batches += 1
        if (batch_idx + 1) % 50 == 0:
            n = sum(t.size(0) for t in self.q_embs)
            logger.info(f"  [test] batch {batch_idx + 1}: {n} samples")

    def on_test_epoch_end(self, trainer, pl_module):
        if self.n_batches == 0:
            logger.warning("TestEvalCallback: no batches.")
            return

        self.h5_file.close()
        device = pl_module.device

        q_embs = torch.cat(self.q_embs, dim=0)  # (N, S, H) fp16 in RAM
        q_masks = torch.cat(self.q_masks, dim=0)  # (N, S)
        q_ids = torch.cat(self.q_ids, dim=0)  # (N, S)
        n_queries = q_embs.size(0)

        h5 = h5py.File(self.h5_path, "r")
        n_pos = h5["pos_embs"].shape[0]
        n_neg = h5["neg_embs"].shape[0]
        assert n_pos == n_neg, f"Corpus mismatch: {n_pos} pos != {n_neg} neg"
        n_corpus = n_pos + n_neg
        has_doc_ids = "pos_ids" in h5 and "neg_ids" in h5

        logger.info(f"  {n_queries} queries × {n_corpus} docs")

        # --- Triplet accuracy (accumulated across all test batches) ---
        if self._triplet_total > 0:
            triplet_acc = self._triplet_correct / self._triplet_total
            pl_module.log("test/triplet_accuracy", triplet_acc)
            logger.info(f"  test/triplet_accuracy: {triplet_acc:.4f}")

        targets = gather_targets(self.targets)
        hard_qrels, soft_qrels = build_qrels(n_queries, n_corpus, targets)

        # --- Score with MeanInteraction (or CenteredMeanPool if trained) ---
        if hasattr(pl_module, "mean_centerer") and pl_module.mean_centerer is not None:
            mean_pool = _CenteredMeanPool(pl_module.mean_centerer).to(device)
            logger.info("  Scoring with _CenteredMeanPool (mean_center=True)...")
        else:
            mean_pool = MeanInteraction().to(device)
            logger.info("  Scoring with MeanInteraction...")

        dense_run, dense_top_scores, dense_top_indices = self._score_full_corpus(
            pool=mean_pool,
            q_embs=q_embs,
            q_masks=q_masks,
            q_ids=q_ids,
            h5=h5,
            n_pos=n_pos,
            n_corpus=n_corpus,
            device=device,
            has_doc_ids=has_doc_ids,
        )
        for k in [5, 10, 20, 100]:
            metrics = evaluate_run(hard_qrels, soft_qrels, dense_run, k)
            pl_module.log_dict(
                {f"test/dense/{name}": v for name, v in metrics.items()},
                on_epoch=True,
            )
            logger.info(f"  test/dense @{k}: {metrics}")

        if self.save_scores_dir:
            os.makedirs(self.save_scores_dir, exist_ok=True)
            np.save(
                os.path.join(self.save_scores_dir, "dense_topk_scores.npy"),
                dense_top_scores.numpy(),
            )
            np.save(
                os.path.join(self.save_scores_dir, "dense_topk_indices.npy"),
                dense_top_indices.numpy(),
            )
            logger.info(f"  Saved dense score matrix → {self.save_scores_dir}")

        # --- Score with LateInteraction ---
        li = LateInteraction(self.cfg).to(device)
        logger.info("  Scoring with LateInteraction...")
        li_run, li_top_scores, li_top_indices = self._score_full_corpus(
            pool=li,
            q_embs=q_embs,
            q_masks=q_masks,
            q_ids=q_ids,
            h5=h5,
            n_pos=n_pos,
            n_corpus=n_corpus,
            device=device,
            has_doc_ids=has_doc_ids,
        )
        for k in [5, 10, 20, 100]:
            metrics = evaluate_run(hard_qrels, soft_qrels, li_run, k)
            pl_module.log_dict(
                {f"test/late_interaction/{name}": v for name, v in metrics.items()},
                on_epoch=True,
            )
            logger.info(f"  test/late_interaction @{k}: {metrics}")

        if self.save_scores_dir:
            np.save(
                os.path.join(self.save_scores_dir, "li_topk_scores.npy"),
                li_top_scores.numpy(),
            )
            np.save(
                os.path.join(self.save_scores_dir, "li_topk_indices.npy"),
                li_top_indices.numpy(),
            )
            logger.info(f"  Saved LI score matrix → {self.save_scores_dir}")

        # --- Score with PatchInteraction (when configured) ---
        if (
            self.cfg.model.pooling_method == "pli"
            or getattr(self.cfg.model, "patch_method", "none") != "none"
        ):
            trained_pool = getattr(pl_module.contrastive_loss, "pool", None)
            if isinstance(trained_pool, PatchInteraction):
                pli = trained_pool  # already on device; uses trained weights
                pli.eval()
            else:
                pli = PatchInteraction(self.cfg).to(device)

            logger.info(
                f"  Scoring with PatchInteraction ({self.cfg.model.patch_method})..."
            )
            pli_run, pli_top_scores, pli_top_indices = self._score_full_corpus(
                pool=pli,
                q_embs=q_embs,
                q_masks=q_masks,
                q_ids=q_ids,
                h5=h5,
                n_pos=n_pos,
                n_corpus=n_corpus,
                device=device,
                has_doc_ids=has_doc_ids,
            )
            for k in [5, 10, 20, 100]:
                metrics = evaluate_run(hard_qrels, soft_qrels, pli_run, k)
                pl_module.log_dict(
                    {
                        f"test/patch_interaction/{name}": v
                        for name, v in metrics.items()
                    },
                    on_epoch=True,
                )
                logger.info(f"  test/patch_interaction @{k}: {metrics}")

            if self.save_scores_dir:
                np.save(
                    os.path.join(self.save_scores_dir, "pli_topk_scores.npy"),
                    pli_top_scores.numpy(),
                )
                np.save(
                    os.path.join(self.save_scores_dir, "pli_topk_indices.npy"),
                    pli_top_indices.numpy(),
                )
                logger.info(f"  Saved PLI score matrix → {self.save_scores_dir}")

        h5.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        logger.info("TestEvalCallback: done.")

    def _score_full_corpus(
        self,
        pool,
        q_embs: torch.Tensor,
        q_masks: torch.Tensor,
        q_ids: torch.Tensor,
        h5: h5py.File,
        n_pos: int,
        n_corpus: int,
        device: torch.device,
        has_doc_ids: bool = False,
    ) -> Tuple[Run, torch.Tensor, torch.Tensor]:
        """Score all queries against all corpus documents.

        Returns
        -------
        Tuple[Run, torch.Tensor, torch.Tensor]
            Tuple of ``(Run, top_scores, top_indices)`` where the last two
            tensors have shape ``(n_queries, K)`` and can be saved to disk
            for post-hoc per-domain analysis.
        """
        n_queries = q_embs.size(0)
        top_scores = torch.full((n_queries, self.K), float("-inf"))
        top_indices = torch.zeros((n_queries, self.K), dtype=torch.long)

        for q_start in tqdm(range(0, n_queries, self.Q_CHUNK), desc="Scoring queries"):
            q_end = min(q_start + self.Q_CHUNK, n_queries)

            q_chunk = q_embs[q_start:q_end].float().to(device)
            q_mask_chunk = q_masks[q_start:q_end].to(device)
            q_id_chunk = q_ids[q_start:q_end].to(device)

            for k_start in tqdm(
                range(0, n_corpus, self.K_CHUNK), desc="Scoring docs", leave=False
            ):
                k_end = min(k_start + self.K_CHUNK, n_corpus)

                # Read doc chunk from HDF5 (may span pos/neg boundary)
                k_chunk, k_mask_chunk, k_id_chunk = self._read_docs(
                    h5, k_start, k_end, n_pos, device, has_doc_ids
                )

                # Build call kwargs
                call_kwargs: dict = dict(
                    query_embs=q_chunk,
                    key_embs=k_chunk,
                    q_mask=q_mask_chunk,
                    k_mask=k_mask_chunk,
                    q_input_ids=q_id_chunk,
                )
                if isinstance(pool, PatchInteraction) and k_id_chunk is not None:
                    call_kwargs["k_input_ids"] = k_id_chunk

                chunk_scores = pool(**call_kwargs).cpu()  # (q_chunk, k_chunk)

                # Merge with running top-K
                combined_scores = torch.cat(
                    [top_scores[q_start:q_end], chunk_scores], dim=1
                )
                combined_indices = torch.cat(
                    [
                        top_indices[q_start:q_end],
                        torch.arange(k_start, k_end)
                        .unsqueeze(0)
                        .expand(q_end - q_start, -1),
                    ],
                    dim=1,
                )
                best_scores, best_local = combined_scores.topk(self.K, dim=1)
                top_scores[q_start:q_end] = best_scores
                top_indices[q_start:q_end] = combined_indices.gather(1, best_local)

            logger.info(f"    queries {q_end}/{n_queries}")

        run = Run(
            {
                f"q{i}": {
                    f"d{int(top_indices[i, j])}": float(top_scores[i, j])
                    for j in range(self.K)
                    if top_scores[i, j] != float("-inf")
                }
                for i in range(n_queries)
            }
        )
        return run, top_scores, top_indices

    @staticmethod
    def _read_docs(h5, start, end, n_pos, device, has_doc_ids=False):
        parts_e, parts_m, parts_ids = [], [], []
        if start < n_pos:
            s = min(end, n_pos)
            parts_e.append(torch.from_numpy(h5["pos_embs"][start:s]))
            parts_m.append(torch.from_numpy(h5["pos_masks"][start:s]))
            if has_doc_ids:
                parts_ids.append(torch.from_numpy(h5["pos_ids"][start:s]).long())
        if end > n_pos:
            ns = max(start, n_pos) - n_pos
            ne = end - n_pos
            parts_e.append(torch.from_numpy(h5["neg_embs"][ns:ne]))
            parts_m.append(torch.from_numpy(h5["neg_masks"][ns:ne]))
            if has_doc_ids:
                parts_ids.append(torch.from_numpy(h5["neg_ids"][ns:ne]).long())

        k_embs = torch.cat(parts_e).float().to(device)
        k_masks = torch.cat(parts_m).to(device)
        k_ids = torch.cat(parts_ids).to(device) if parts_ids else None

        return k_embs, k_masks, k_ids
