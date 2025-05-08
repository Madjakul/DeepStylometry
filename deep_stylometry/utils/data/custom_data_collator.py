from typing import Optional, Tuple

import torch
from transformers import DataCollatorForLanguageModeling


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=None,
        tf_experimental_compile=False,
        return_tensors="pt",
        generator=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            mlm=True,  # Ensure MLM is enabled
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
            tf_experimental_compile=tf_experimental_compile,
            return_tensors=return_tensors,
        )
        if generator is None:
            self.generator = torch.Generator()
        else:
            self.generator = generator

    def torch_mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        device = labels.device

        # 1) build prob matrix and zero out specials
        prob_mat = torch.full(labels.shape, self.mlm_probability, device=device)
        if special_tokens_mask is None:
            st_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(st_mask, dtype=torch.bool, device=device)
        else:
            special_tokens_mask = special_tokens_mask.bool().to(device)
        prob_mat.masked_fill_(special_tokens_mask, 0.0)

        # 2) sample the usual masked_indices
        masked_indices = torch.bernoulli(prob_mat, generator=self.generator).bool()

        # 3) force at least one non-special mask per row
        no_mask_row = ~masked_indices.any(dim=1)  # [batch]
        if no_mask_row.any():
            # random scores over every token
            rand_scores = torch.rand(labels.shape, device=device)
            # forbid specials by setting their scores very low
            rand_scores.masked_fill_(special_tokens_mask, -1.0)
            # for each row, pick the idx of the max score → guaranteed non-special
            forced_idx = rand_scores.argmax(dim=1)  # [batch]
            # now turn on that one position in each “empty” row
            masked_indices[no_mask_row, forced_idx[no_mask_row]] = True

        # 4) prepare labels (only compute loss on masked)
        labels[~masked_indices] = -100

        # 5) apply masking / random / keep logic
        #   mask token replacements
        mask_replace = (
            torch.bernoulli(
                torch.full(labels.shape, self.mask_replace_prob, device=device),
                generator=self.generator,
            ).bool()
            & masked_indices
        )
        inputs[mask_replace] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        #   random replacements
        random_replace = (
            torch.bernoulli(
                torch.full(labels.shape, self.random_replace_prob, device=device),
                generator=self.generator,
            ).bool()
            & masked_indices
            & ~mask_replace
        )
        random_words = torch.randint(
            low=0,
            high=len(self.tokenizer),
            size=labels.shape,
            generator=self.generator,
            device=device,
        )
        inputs[random_replace] = random_words[random_replace]

        return inputs, labels
