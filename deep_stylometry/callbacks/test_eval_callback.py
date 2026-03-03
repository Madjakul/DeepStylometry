# deep_stylometry/callbacks/test_eval_callback.py

import os
import shutil
import tempfile
from collections import defaultdict

import lightning as L
import torch
import torch.nn.functional as F
from ranx import Run, evaluate

from deep_stylometry.utils.eval_utils import build_qrels, gather_targets


class TestEvalCallback(L.Callback):
    def __init__(self, k: int = 100, shortlist_k: int = 500, max_cache_size: int = 64):
        super().__init__()
        self.k = k
        self.shortlist_k = shortlist_k
        self.max_cache_size = max_cache_size
        self.tmp_dir = tempfile.mkdtemp()
        self._batch_cache = {}
        self._reset_state()

    def _reset_state(self):
        self.q_dense, self.p_dense, self.n_dense, self.targets = [], [], [], []
        self.n_batches = 0
        self._batch_cache.clear()

    def on_test_epoch_start(self, trainer, pl_module):
        self._reset_state()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        q_3d = pl_module.lm(batch["input_ids"], batch["attention_mask"]).half()
        p_3d = pl_module.lm(batch["pos_input_ids"], batch["pos_attention_mask"]).half()
        n_3d = pl_module.lm(batch["neg_input_ids"], batch["neg_attention_mask"]).half()

        # 1. DENSE STAGE: Pool to 1D and keep in RAM
        pool = lambda t, m: F.normalize((t * m.unsqueeze(-1)).sum(1), p=2, dim=-1)
        self.q_dense.append(pool(q_3d, batch["attention_mask"]).cpu())
        self.p_dense.append(pool(p_3d, batch["pos_attention_mask"]).cpu())
        self.n_dense.append(pool(n_3d, batch["neg_attention_mask"]).cpu())

        # 2. LATE INTERACTION STAGE: Save heavy 3D tensors directly to disk
        torch.save(
            {
                "q": q_3d.cpu(),
                "q_m": batch["attention_mask"].cpu(),
                "q_id": batch["input_ids"].cpu(),
                "p": p_3d.cpu(),
                "p_m": batch["pos_attention_mask"].cpu(),
                "n": n_3d.cpu(),
                "n_m": batch["neg_attention_mask"].cpu(),
            },
            os.path.join(self.tmp_dir, f"batch_{batch_idx}.pt"),
        )

        if "target_indices" in batch:
            self.targets.append(batch["target_indices"].cpu())
        self.n_batches += 1

    # Point 1: Safe manual cache instead of lru_cache
    def load_batch(self, file_idx: int):
        if file_idx not in self._batch_cache:
            # FIFO eviction (Python 3.7+ dicts preserve insertion order)
            if len(self._batch_cache) >= self.max_cache_size:
                self._batch_cache.pop(next(iter(self._batch_cache)))

            self._batch_cache[file_idx] = torch.load(
                os.path.join(self.tmp_dir, f"batch_{file_idx}.pt"), weights_only=True
            )
        return self._batch_cache[file_idx]

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        if self.n_batches == 0:
            return

        bs = pl_module.cfg.data.batch_size
        p_cat = torch.cat(self.p_dense, dim=0)
        n_cat = torch.cat(self.n_dense, dim=0)

        # Point 6: Ensure balanced corpus for the math to work
        assert len(p_cat) == len(
            n_cat
        ), f"Corpus mismatch! {len(p_cat)} pos != {len(n_cat)} neg"

        # --- DENSE RETRIEVAL (In-Memory) ---
        q_dense = torch.cat(self.q_dense, dim=0).to(pl_module.device)
        k_dense = torch.cat([p_cat, n_cat], dim=0).to(pl_module.device)
        n_corpus = k_dense.size(0)

        dense_scores = torch.matmul(q_dense, k_dense.T)
        dense_top_s, shortlists = dense_scores.topk(
            min(self.shortlist_k, n_corpus), dim=1
        )

        # Point 2: Log Dense Run (Covers Comparison 1 & 3 Baselines)
        dense_run_dict = {}
        for q_idx in range(q_dense.size(0)):
            c_ids = shortlists[q_idx].cpu().tolist()
            c_scores = dense_top_s[q_idx].cpu().tolist()
            dense_run_dict[f"q{q_idx}"] = {
                f"d{c_ids[j]}": float(c_scores[j])
                for j in range(min(self.k, len(c_ids)))
            }

        # --- LATE INTERACTION (Streamed from Disk) ---
        li = getattr(pl_module.contrastive_loss, "pool", None)
        li_run_dict = {}

        if li:
            for q_idx in range(q_dense.size(0)):
                # 1. Read Query from Disk
                q_data = self.load_batch(q_idx // bs)
                q_emb = q_data["q"][q_idx % bs].unsqueeze(0).to(pl_module.device)
                q_mask = q_data["q_m"][q_idx % bs].unsqueeze(0).to(pl_module.device)
                q_id = q_data["q_id"][q_idx % bs].unsqueeze(0).to(pl_module.device)

                cand_ids = shortlists[q_idx].cpu().tolist()

                # --- I/O OPTIMIZATION: Group fetches by file ---
                file_fetches = defaultdict(list)
                for list_idx, c_idx in enumerate(cand_ids):
                    is_neg = c_idx >= (n_corpus // 2)
                    offset = c_idx - (n_corpus // 2) if is_neg else c_idx
                    file_fetches[offset // bs].append(
                        {
                            "list_idx": list_idx,
                            "is_neg": is_neg,
                            "item_idx": offset % bs,
                        }
                    )

                cand_embs, cand_masks = [None] * len(cand_ids), [None] * len(cand_ids)

                # Batch load from disk safely using the new dict cache
                for file_idx, items in file_fetches.items():
                    c_data = self.load_batch(file_idx)
                    for item in items:
                        k_key = "n" if item["is_neg"] else "p"
                        cand_embs[item["list_idx"]] = c_data[k_key][item["item_idx"]]
                        cand_masks[item["list_idx"]] = c_data[f"{k_key}_m"][
                            item["item_idx"]
                        ]

                # Pad streamed candidates locally and Score
                m_len = max(m.size(0) for m in cand_masks)
                cand_embs = torch.stack(
                    [F.pad(e, (0, 0, 0, m_len - e.size(0))) for e in cand_embs]
                ).to(pl_module.device)
                cand_masks = torch.stack(
                    [F.pad(m, (0, m_len - m.size(0))) for m in cand_masks]
                ).to(pl_module.device)

                scores = li(
                    query_embs=q_emb,
                    key_embs=cand_embs,
                    q_mask=q_mask,
                    k_mask=cand_masks,
                    q_input_ids=q_id,
                ).squeeze(0)
                top_s, top_i = scores.topk(min(self.k, len(cand_ids)))
                li_run_dict[f"q{q_idx}"] = {
                    f"d{cand_ids[top_i[j]]}": float(top_s[j]) for j in range(len(top_i))
                }

        # --- LOGGING & CLEANUP ---
        hard_qrels, soft_qrels = build_qrels(
            q_dense.size(0), n_corpus, gather_targets(self.targets)
        )

        cutoffs = [5, 10, 20, 100]
        mrr_reqs = [f"mrr@{k}" for k in cutoffs]
        ndcg_reqs = [f"ndcg@{k}" for k in cutoffs]
        recall_reqs = [f"recall@{k}" for k in cutoffs]

        def compute_metrics(run_dict):
            run = Run(run_dict)
            # MRR uses hard qrels, nDCG and Recall use soft qrels
            mrr_res = evaluate(hard_qrels, run, metrics=mrr_reqs)
            ndcg_res = evaluate(soft_qrels, run, metrics=ndcg_reqs)
            recall_res = evaluate(soft_qrels, run, metrics=recall_reqs)

            # Merge dictionaries
            return {**mrr_res, **ndcg_res, **recall_res}

        # Log Dense First-Stage
        dense_metrics = compute_metrics(dense_run_dict)
        pl_module.log_dict(
            {f"test/dense_baseline/{k}": v for k, v in dense_metrics.items()},
            on_epoch=True,
        )

        # Log Late Interaction Second-Stage
        if li:
            li_metrics = compute_metrics(li_run_dict)
            pl_module.log_dict(
                {f"test/late_interaction_rerank/{k}": v for k, v in li_metrics.items()},
                on_epoch=True,
            )

        shutil.rmtree(self.tmp_dir)  # Nuke disk storage
        self._batch_cache.clear()  # Free RAM
