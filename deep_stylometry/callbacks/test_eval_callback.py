# deep_stylometry/callbacks/test_eval_callback.py

import logging
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, List

import h5py
import lightning as L
import torch
import torch.nn.functional as F
from ranx import Run
from tqdm import tqdm

from deep_stylometry.modules.late_interaction import LateInteraction
from deep_stylometry.modules.mean_interaction import MeanInteraction
from deep_stylometry.utils.eval_utils import build_qrels, evaluate_run, gather_targets

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class TestEvalCallback(L.Callback):

    def __init__(
        self,
        cfg: "BaseConfig",
        k: int = 100,
        q_chunk: int = 256,
        k_chunk: int = 256,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.cfg = cfg
        self.K = k
        self.Q_CHUNK = q_chunk
        self.K_CHUNK = k_chunk
        self.max_seq_len = max_seq_len
        self.tmp_dir = tempfile.mkdtemp()
        self.h5_path = os.path.join(self.tmp_dir, "corpus.h5")

    def _reset(self):
        self.q_embs: List[torch.Tensor] = []
        self.q_masks: List[torch.Tensor] = []
        self.q_ids: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.n_batches = 0

    def on_test_epoch_start(self, trainer, pl_module):
        self._reset()
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.h5_file = h5py.File(self.h5_path, "w")
        self.h5_datasets = {}
        logging.info(f"TestEvalCallback: HDF5 → {self.h5_path}")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
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
        for prefix, embs_key, mask_key in [
            ("pos", "pos_embs", "pos_mask"),
            ("neg", "neg_embs", "neg_mask"),
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

        if outputs["target_indices"] is not None:
            self.targets.append(outputs["target_indices"].cpu())

        self.n_batches += 1
        if (batch_idx + 1) % 50 == 0:
            n = sum(t.size(0) for t in self.q_embs)
            logging.info(f"  [test] batch {batch_idx + 1}: {n} samples")

    def on_test_epoch_end(self, trainer, pl_module):
        if self.n_batches == 0:
            logging.warning("TestEvalCallback: no batches.")
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

        logging.info(f"  {n_queries} queries × {n_corpus} docs")

        targets = gather_targets(self.targets)
        hard_qrels, soft_qrels = build_qrels(n_queries, n_corpus, targets)

        # --- Score with MeanInteraction ---
        mean_pool = MeanInteraction().to(device)
        logging.info("  Scoring with MeanInteraction...")
        dense_run = self._score_full_corpus(
            pool=mean_pool,
            q_embs=q_embs,
            q_masks=q_masks,
            q_ids=q_ids,
            h5=h5,
            n_pos=n_pos,
            n_corpus=n_corpus,
            device=device,
        )
        for k in [5, 10, 20, 100]:
            metrics = evaluate_run(hard_qrels, soft_qrels, dense_run, k)
            pl_module.log_dict(
                {f"test/dense/{name}": v for name, v in metrics.items()},
                on_epoch=True,
            )
            logging.info(f"  test/dense @{k}: {metrics}")

        # --- Score with LateInteraction ---
        li = LateInteraction(self.cfg).to(device)
        logging.info("  Scoring with LateInteraction...")
        li_run = self._score_full_corpus(
            pool=li,
            q_embs=q_embs,
            q_masks=q_masks,
            q_ids=q_ids,
            h5=h5,
            n_pos=n_pos,
            n_corpus=n_corpus,
            device=device,
        )
        for k in [5, 10, 20, 100]:
            metrics = evaluate_run(hard_qrels, soft_qrels, li_run, k)
            pl_module.log_dict(
                {f"test/late_interaction/{name}": v for name, v in metrics.items()},
                on_epoch=True,
            )
            logging.info(f"  test/late_interaction @{k}: {metrics}")

        h5.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        logging.info("TestEvalCallback: done.")

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
    ) -> Run:
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
                k_chunk, k_mask_chunk = self._read_docs(
                    h5, k_start, k_end, n_pos, device
                )

                chunk_scores = pool(
                    query_embs=q_chunk,
                    key_embs=k_chunk,
                    q_mask=q_mask_chunk,
                    k_mask=k_mask_chunk,
                    q_input_ids=q_id_chunk,
                ).cpu()  # (q_chunk_size, k_chunk_size)

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

            logging.info(f"    queries {q_end}/{n_queries}")

        return Run(
            {
                f"q{i}": {
                    f"d{int(top_indices[i, j])}": float(top_scores[i, j])
                    for j in range(self.K)
                    if top_scores[i, j] != float("-inf")
                }
                for i in range(n_queries)
            }
        )

    @staticmethod
    def _read_docs(h5, start, end, n_pos, device):
        parts_e, parts_m = [], []
        if start < n_pos:
            s = min(end, n_pos)
            parts_e.append(torch.from_numpy(h5["pos_embs"][start:s]))
            parts_m.append(torch.from_numpy(h5["pos_masks"][start:s]))
        if end > n_pos:
            ns = max(start, n_pos) - n_pos
            ne = end - n_pos
            parts_e.append(torch.from_numpy(h5["neg_embs"][ns:ne]))
            parts_m.append(torch.from_numpy(h5["neg_masks"][ns:ne]))
        return (
            torch.cat(parts_e).float().to(device),
            torch.cat(parts_m).to(device),
        )
