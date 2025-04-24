# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import Dict, Optional

import bitsandbytes as bnb
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import necessary Hugging Face classes
from transformers import (DataCollatorForLanguageModeling,
                          PreTrainedTokenizerBase)

from deep_stylometry.modules.clm_loss import CLMLoss
from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.language_model import LanguageModel
from deep_stylometry.optimizers import SOAP, SophiaG


class DeepStylometry(L.LightningModule):

    optim_map = {
        "adamw": torch.optim.AdamW,
        "adam8bit": bnb.optim.Adam8bit,
        "soap": SOAP,
        "sophia": SophiaG,
    }

    def __init__(
        self,
        optim_name: str,
        base_model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        seq_len: int,
        is_decoder_model: bool,
        lr: float = 2e-5,
        dropout: float = 0.1,
        weight_decay: float = 1e-2,
        # Renamed clm_weight to lm_weight
        lm_weight: float = 1.0,
        # Added mlm_probability
        mlm_probability: float = 0.15,
        do_late_interaction: bool = False,
        do_distance: bool = False,
        exp_decay: bool = False,
        alpha: float = 0.5,
        contrastive_weight: float = 1.0,
        contrastive_temp: float = 0.07,
        initial_gumbel_temp: float = 1.0,
        temp_annealing_rate: float = 0.999,
        min_gumbel_temp: float = 0.1,
        project_up: Optional[bool] = None,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters

        self.weight_decay = weight_decay
        self.lm = LanguageModel(base_model_name, is_decoder_model)
        # Load tokenizer associated with the base model
        self.tokenizer = tokenizer

        # Conditionally setup LM objective (Loss function or Data Collator)
        self.lm_objective = None
        if self.lm.is_decoder:
            logging.info("Model is decoder type. Using CLMLoss.")
            self.lm_objective = CLMLoss()
        else:
            logging.info(
                f"Model is encoder type. Using DataCollatorForLanguageModeling (MLM prob: {mlm_probability})."
            )
            # For MLM, we use the data collator to mask tokens on the fly
            self.lm_objective = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=mlm_probability,
                # return_tensors="pt" # Not needed if called with dict
            )

        self.lm_weight = lm_weight
        self.contrastive_weight = contrastive_weight
        self.gumbel_temp = initial_gumbel_temp
        self.temp_annealing_rate = temp_annealing_rate
        self.optim_name = optim_name
        self.batch_size = batch_size
        self.min_gumbel_temp = min_gumbel_temp
        self.dropout = dropout
        self.lr = lr

        # Projection layers setup
        if contrastive_weight > 0:
            self.contrastive_loss = InfoNCELoss(
                do_late_interaction=do_late_interaction,
                do_distance=do_distance,
                exp_decay=exp_decay,
                alpha=alpha,
                temperature=contrastive_temp,
                seq_len=seq_len,
            )
            hidden_size = self.lm.hidden_size
            if project_up is True:
                self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
                self.fc2 = nn.Linear(hidden_size * 4, hidden_size * 4)
            elif project_up is False:
                self.fc1 = nn.Linear(hidden_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
            else:  # Default projection if project_up is None but contrastive_weight > 0
                self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
                self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

    def _calculate_losses(self, batch: Dict[str, torch.Tensor]):
        # Extract original inputs

        lm_loss = torch.tensor(0.0)  # Initialize loss
        q_embs = None
        k_embs = None

        if self.lm.is_decoder:
            # --- CLM Path ---
            # Run main forward pass (gets hidden states and applies projection)
            q_embs, q_logits = self(batch["q_input_ids"], batch["q_attention_mask"])
            k_embs, _ = self(batch["k_input_ids"], batch["k_attention_mask"])

            # Calculate CLM loss using the CLMLoss instance
            # Assuming CLMLoss only needs q_logits like in the original code snippet
            if self.lm_weight > 0:
                lm_loss = self.lm_objective(
                    q_logits, batch["q_input_ids"], batch["q_attention_mask"]
                )

        else:
            # --- MLM Path ---
            # 1. Prepare masked inputs and labels using the data collator
            # The collator expects dicts or lists of dicts. Pass necessary fields.
            # We process 'q' inputs for the MLM loss calculation.
            # collator_input_q = {
            #     "input_ids": batch["q_input_ids"],
            #     "attention_mask": batch["q_attention_mask"],
            # }
            # collated_q = self.lm_objective(
            #     collator_input_q
            # )  # lm_objective is DataCollatorForLanguageModeling
            # q_input_ids_masked = collated_q["input_ids"]
            # q_labels_mlm = collated_q["labels"]
            #
            # # Also mask 'k' inputs for getting embeddings for contrastive loss
            # collator_input_k = {
            #     "input_ids": batch["k_input_ids"],
            #     "attention_mask": batch["k_attention_mask"],
            # }
            # collated_k = self.lm_objective(collator_input_k)
            # k_input_ids_masked = collated_k["input_ids"]
            q_input_ids = batch["q_input_ids"]
            q_attention_mask = batch["q_attention_mask"]
            collator_input_q = [
                {"input_ids": q_input_ids[i], "attention_mask": q_attention_mask[i]}
                for i in range(q_input_ids.size(0))
            ]
            collated_q = self.lm_objective(collator_input_q)
            q_input_ids_masked = collated_q["input_ids"].to(self.device)
            q_labels_mlm = collated_q["labels"].to(self.device)

            # Similarly for 'k' inputs
            k_input_ids = batch["k_input_ids"]
            k_attention_mask = batch["k_attention_mask"]
            collator_input_k = [
                {"input_ids": k_input_ids[i], "attention_mask": k_attention_mask[i]}
                for i in range(k_input_ids.size(0))
            ]
            collated_k = self.lm_objective(collator_input_k)
            k_input_ids_masked = collated_k["input_ids"].to(self.device)
            #
            # 2. Calculate MLM loss
            # Pass masked inputs and generated labels to the underlying HF model
            if self.lm_weight > 0:
                # output_hidden_states=False here, as we only need the loss
                outputs_q = self.lm.model(
                    input_ids=q_input_ids_masked,
                    attention_mask=batch["q_attention_mask"],  # Use original mask
                    labels=q_labels_mlm,
                    output_hidden_states=False,  # Only need loss here
                    return_dict=True,
                )
                lm_loss = outputs_q.loss

            # 3. Get Embeddings for Contrastive Loss
            # Run the main forward pass (which includes projection layers)
            # using the MASKED inputs.
            if self.contrastive_weight > 0:
                q_embs, _ = self(q_input_ids_masked, batch["q_attention_mask"])
                k_embs, _ = self(k_input_ids_masked, batch["k_attention_mask"])
            # --- End MLM Path ---

        # --- Contrastive Loss Calculation ---
        contrastive_loss = torch.tensor(0.0)  # Initialize
        if self.contrastive_weight > 0 and q_embs is not None and k_embs is not None:
            contrastive_loss = self.contrastive_loss(
                q_embs,
                k_embs,
                batch["author_label"],
                batch["q_attention_mask"],  # Use original attention masks
                batch["k_attention_mask"],
                gumbel_temp=self.gumbel_temp,
            )

        # --- Combine Losses ---
        total_loss = (self.lm_weight * lm_loss) + (
            self.contrastive_weight * contrastive_loss
        )

        # Log individual losses (use add_prefix=False if metric keys are already full)
        self.log(
            "lm_loss",
            lm_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        if self.contrastive_weight > 0:
            self.log(
                "contrastive_loss",
                contrastive_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                batch_size=self.batch_size,
            )

        return total_loss, lm_loss, contrastive_loss

    def configure_optimizers(self):
        # Filter out parameters that do not require gradients (if any)
        optim_params = [p for p in self.parameters() if p.requires_grad]
        logging.info(
            f"Configuring optimizer: {self.optim_name} with lr={self.lr}, weight_decay={self.weight_decay}"
        )
        logging.info(
            f"Number of trainable parameters: {sum(p.numel() for p in optim_params)}"
        )

        optimizer = self.optim_map[self.optim_name](
            optim_params, lr=self.lr, weight_decay=self.weight_decay
        )
        # Add scheduler here if needed
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [scheduler]
        return optimizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states, logits = self.lm(input_ids, attention_mask=attention_mask)

        if self.contrastive_weight > 0:
            embs = F.dropout(hidden_states, p=self.dropout, training=self.training)
            embs = F.gelu(self.fc1(embs))
            embs = F.dropout(embs, p=self.dropout, training=self.training)
            projected_embs = self.fc2(embs)
        else:
            projected_embs = hidden_states

        # Return projected embeddings (for contrastive) and raw logits (for CLM)
        return projected_embs, logits

    def training_step(self, batch, batch_idx: int):
        total_loss, lm_loss, contrastive_loss = self._calculate_losses(batch)

        # Log metrics
        self.log(
            "gumbel_temp",
            self.gumbel_temp,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        self.log(
            "train/total_loss",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/lm_loss",
            lm_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        if self.contrastive_weight > 0:
            self.log(
                "train/contrastive_loss",
                contrastive_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        # Anneal Gumbel temperature (if used by contrastive loss)
        # Check if self.contrastive_loss exists and has gumbel temp logic before annealing
        if hasattr(self, "contrastive_loss") and hasattr(
            self.contrastive_loss, "temperature"
        ):  # Simple check
            if self.gumbel_temp > self.min_gumbel_temp:
                self.gumbel_temp *= self.temp_annealing_rate

        return total_loss

    # Add validation_step and test_step if needed, mirroring _calculate_losses logic
    # Ensure to use appropriate logging keys (e.g., "val/total_loss")

    def validation_step(self, batch, batch_idx: int):
        total_loss, lm_loss, contrastive_loss = self._calculate_losses(batch)

        self.log(
            "val/total_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/lm_loss",
            lm_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        if self.contrastive_weight > 0:
            self.log(
                "val/contrastive_loss",
                contrastive_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        return total_loss

    def test_step(self, batch, batch_idx: int):
        total_loss, lm_loss, contrastive_loss = self._calculate_losses(batch)

        self.log(
            "test/total_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test/lm_loss",
            lm_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        if self.contrastive_weight > 0:
            self.log(
                "test/contrastive_loss",
                contrastive_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        return total_loss
