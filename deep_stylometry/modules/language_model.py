# deep_stylometry/modules/language_model.py

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM


class LanguageModel(nn.Module):
    """A wrapper class for loading a language model from Hugging Face's
    transformers library. This class can load either a decoder model (e.g.,
    GPT-2) or an encoder model (e.g., BERT) based on the specified model name.
    The class also provides a forward method that computes the loss and returns
    the last hidden states of the model.

    Parameters
    ----------
    model_name: str
        The name of the pretrained model to load from Hugging Face's transformers
        library.
    is_decoder_model: bool
        If True, load a decoder model (e.g., GPT-2). If False, load an encoder
        model (e.g., BERT). This parameter determines the type of model to load
        and affects the behavior of the forward method.

    Attributes
    ----------
    model: transformers.PreTrainedModel
        The loaded pretrained model from Hugging Face's transformers library.
    is_decoder: bool
        Indicates whether the loaded model is a decoder model (True) or an
        encoder model (False).
    hidden_size: int
        The hidden size of the model, which is the size of the hidden states
        produced by the model.
    vocab_size: int
        The vocabulary size of the model, which is the number of unique tokens
        that the model can handle.
    """

    def __init__(self, model_name: str, is_decoder_model: bool):
        super(LanguageModel, self).__init__()

        config = AutoConfig.from_pretrained(model_name)

        if is_decoder_model:
            logging.info(f"Loading pretrained decoder from {model_name}.")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
            self.is_decoder = True
        else:
            logging.info(f"Loading pretrained encoder from {model_name}.")
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
            self.is_decoder = False

        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Compute the loss and return the last hidden states of the model.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input tensor containing the token IDs.
        attention_mask: torch.Tensor
            A mask indicating which tokens should be attended to (1) and which
            should not (0).
        labels: torch.Tensor, optional
            The labels for the input data. If None and the model is a decoder,
            the input_ids will be used as labels.

        Returns
        -------
        loss: torch.Tensor
            The computed loss for the input data.
        last_hidden_states: torch.Tensor
            The last hidden states of the model, which are the output embeddings
            for the input data.
        """
        if self.is_decoder and labels is None:
            labels = input_ids.clone()
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = out.loss
        last_hidden_states = out.hidden_states[-1]
        return loss, last_hidden_states
