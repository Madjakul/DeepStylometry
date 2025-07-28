# deep_stylometry/utils/data/custom_data_collator.py

from typing import Optional

import torch
from transformers import DataCollatorForLanguageModeling


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """A custom data collator that ensures at least one token is masked per
    sequence, using vectorized operations. Inherits from
    DataCollatorForLanguageModeling and overrides torch_mask_tokens.

    If, after initial probabilistic masking, a sequence has no masked tokens,
    this collator will randomly select one non-special token from that sequence
    to mask. This selection is done in a vectorized way.

    Parameters
    ----------
    tokenizer: PreTrainedTokenizerBase
        The tokenizer used for encoding the text.
    mlm_probability: float
        The probability of masking a token. Default is 0.15.
    pad_to_multiple_of: int, optional
        If specified, the sequences will be padded to a multiple of this value.
    tf_experimental_compile: bool
        If True, the model will be compiled using TensorFlow's experimental compile.
    return_tensors: str
        The type of tensors to return. Default is "pt" (PyTorch).
    generator: torch.Generator, optional
        A random number generator to use for sampling. If None, a new generator will
        be created.

    Attributes
    ----------
    generator: torch.Generator
        The random number generator used for sampling.
    """

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
            mlm=True,
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
    ):
        """Prepare masked tokens inputs/labels for masked language modeling
        using vectorized operations to ensure at least one mask per sequence if
        possible. MLM strategy: 80% MASK, 10% random, 10% original.

        Parameters
        ----------
        inputs: torch.Tensor
            The input tensor containing the token IDs.
        special_tokens_mask: torch.Tensor, optional
            A mask indicating which tokens are special tokens. If None, the mask will be
            generated using the tokenizer's get_special_tokens_mask method.

        Returns
        -------
        tuple
            A tuple containing the masked inputs and the labels for the masked tokens.
            The labels are set to -100 for non-masked tokens, as per the standard
            PyTorch convention for ignoring certain tokens in loss computation.
        """
        labels = inputs.clone()
        device = labels.device

        # Build prob matrix and zero out specials
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

        # Sample the usual masked_indices
        masked_indices = torch.bernoulli(prob_mat, generator=self.generator).bool()

        # Force at least one non-special mask per row
        no_mask_row = ~masked_indices.any(dim=1)  # [batch]
        if no_mask_row.any():
            # Random scores over every token
            rand_scores = torch.rand(labels.shape, device=device)
            # Forbid specials by setting their scores very low
            rand_scores.masked_fill_(special_tokens_mask, -1.0)
            # For each row, pick the idx of the max score -> guaranteed non-special
            forced_idx = rand_scores.argmax(dim=1)  # [batch]
            # Now turn on that one position in each “empty” row
            masked_indices[no_mask_row, forced_idx[no_mask_row]] = True

        # Prepare labels (only compute loss on masked)
        labels[~masked_indices] = -100

        mask_replace = (
            torch.bernoulli(
                torch.full(labels.shape, self.mask_replace_prob, device=device),
                generator=self.generator,
            ).bool()
            & masked_indices
        )
        inputs[mask_replace] = self.tokenizer.convert_tokens_to_ids(  # type: ignore
            self.tokenizer.mask_token
        )

        #   Random replacements
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
