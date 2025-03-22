# deep_stylometry/modules/modeling_deep_stylometry.py


import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.clm_loss import CLMLoss
from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.language_model import LanguageModel


class DeepStylometry(L.LightningModule):

    optim_map = {
        # TODO: import the correct optim
        "adamw": torch.optim.AdamW,
        "adam8bit": bnb.optim.Adam8bit,
        "soap": SOAP,
        "sophia": SophiaG,
    }

    def __init__(
        self,
        optim_name: str,
        model_name: str,
        batch_size: int,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        clm_weight: float = 1.0,
        contrastive_temp: float = 0.07,
        initial_gumbel_temp: float = 0.5,
        temp_annealing_rate: float = 0.95,
    ):
        # TODO: correct initalizatio and adding linear layers
        super().__init__()
        self.save_hyperparameters()

        self.lm = LanguageModel(model_name)
        self.clm_loss = CLMLoss()
        self.contrastive_loss = InfoNCELoss(temperature=contrastive_temp)
        self.clm_weight = clm_weight
        self.gumbel_temp = initial_gumbel_temp
        self.temp_annealing_rate = temp_annealing_rate

    def configure_optimizers(self):
        # TODO: Add other optimizers (SOAP/SophiaG) as needed
        # Take into account optim_map
        if self.optim_name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        # Add other optimizers (SOAP/SophiaG) as needed
        return optimizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states, logits = self.lm(input_ids, attention_mask)
        return hidden_states, logits

    def _calculate_losses(self, batch):
        q_embs, q_logits = self(batch["q_input_ids"], batch["q_attention_mask"])
        k_embs, k_logits = self(batch["k_input_ids"], batch["k_attention_mask"])

        # CLM Loss with masks
        clm_loss = (
            self.clm_loss(q_logits, batch["q_input_ids"], batch["q_attention_mask"])
            + self.clm_loss(k_logits, batch["k_input_ids"], batch["k_attention_mask"])
        ) / 2

        # Contrastive Loss with masks
        contrastive_loss = self.contrastive_loss(
            q_embs,
            k_embs,
            batch["pair_labels"],
            batch["q_attention_mask"],
            batch["k_attention_mask"],
        )

        total_loss = self.clm_weight * clm_loss + contrastive_loss
        return total_loss, clm_loss, contrastive_loss

    def training_step(self, batch, batch_idx):
        # Anneal temperature each step
        self.gumbel_temp *= self.temp_annealing_rate
        total_loss, clm_loss, contrastive_loss = self._calculate_losses(batch)
        self.log("gumbel_temp", self.gumbel_temp, prog_bar=True)
        self.log_dict(
            {
                "train/total_loss": total_loss,
                "train/clm_loss": clm_loss,
                "train/contrastive_loss": contrastive_loss,
            },
            prog_bar=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, clm_loss, contrastive_loss = self._calculate_losses(batch)
        self.log_dict(
            {
                "val/total_loss": total_loss,
                "val/clm_loss": clm_loss,
                "val/contrastive_loss": contrastive_loss,
            },
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        total_loss, clm_loss, contrastive_loss = self._calculate_losses(batch)
        self.log_dict(
            {
                "test/total_loss": total_loss,
                "test/clm_loss": clm_loss,
                "test/contrastive_loss": contrastive_loss,
            }
        )
