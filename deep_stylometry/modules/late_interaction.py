# deep_stylometry/modules/late_interaction.py

import logging
import string
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from tqdm import tqdm

from deep_stylometry.utils.helpers import get_tokenizer

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class LateInteraction(torch.nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        logging.info("Using Late Interaction pooling method")
        self.cfg = cfg

        if self.cfg.model.skip_list:
            tokenizer = get_tokenizer(cfg.model.base_checkpoint)
            punc_chars = set(string.punctuation) | {" "}
            punc_token_ids = set()
            for token_str, token_id in tqdm(
                tokenizer.get_vocab().items(), desc="Identifying punctuation tokens"
            ):
                # Decode the raw token to a string (handles Ġ, Ċ, byte-level prefixes, etc.)
                decoded = tokenizer.convert_tokens_to_string([token_str])
                if decoded and all(c in punc_chars for c in decoded):
                    punc_token_ids.add(token_id)
            self.register_buffer(
                "punc_token_ids",
                torch.tensor(list(punc_token_ids), dtype=torch.long),
                persistent=False,
            )
            logging.info(
                f"Initialized Late Interaction with {len(punc_token_ids)} punctuation"
                " tokens to skip."
            )

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "n_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "n_times_batch seq"],
        q_input_ids: Optional[Int[torch.Tensor, "batch seq"]] = None,
    ) -> Float[torch.Tensor, "batch n_times_batch"]:
        normalized_query_embs = F.normalize(query_embs, p=2, dim=-1)
        normalized_key_embs = F.normalize(key_embs, p=2, dim=-1)

        # Shape: (batch_q, batch_k, q_len, k_len)
        scores = torch.einsum(
            "ash, bth -> abst", normalized_query_embs, normalized_key_embs
        )

        min_finite = -q_mask.sum(dim=-1).view(-1, 1, 1, 1)  # (batch_q, 1, 1, 1)

        mask_inv = (
            (1.0 - k_mask.float()).unsqueeze(0).unsqueeze(2)
        )  # (1, batch_k, 1, k_len)
        scores = scores + (mask_inv * min_finite)

        # Shape: (batch_q, batch_k, q_len)
        scores = scores.max(dim=-1).values

        if self.cfg.model.skip_list and q_input_ids is not None:
            punc_mask = torch.isin(q_input_ids, self.punc_token_ids)  # (batch_q, q_len)
            keep_mask = ~punc_mask
            scores = scores * keep_mask.unsqueeze(1).float()

        scores = scores * q_mask.unsqueeze(1).float()
        scores = scores.sum(dim=-1)  # (batch_q, batch_k)

        return scores
