# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from transformers import get_cosine_schedule_with_warmup

from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.language_model import LanguageModel
from deep_stylometry.optimizers import SOAP, SophiaG


class DeepStylometry(L.LightningModule):
    """DeepStylometry model for stylometry tasks. This model combines a
    language model with a contrastive loss function to learn representations of
    text data. The model can be trained using different optimizers and supports
    various hyperparameters for fine-tuning.

    The language model can be either a decoder model (e.g., GPT-2) or an
    encoder model (e.g., BERT) based on the specified model name. The
    contrastive loss is computed using InfoNCE, where the distance between
    query and key sentences is computed using cosine similarity over the average
    embeddings of the sentences or using a modified late interaction approach.
    The importance of the language model loss and contrastive loss can be
    controlled using the `lm_weight` and `contrastive_weight` hyperparameters.

    By default, the scheduler uses a cosine annealing schedule with warmup steps.

    Parameters
    ----------
    optim_name: str
        The name of the optimizer to use. Options are "adamw", "soap", or "sophia".
    base_model_name: str
        The name of the pretrained model to load from Hugging Face's transformers
        library.
    batch_size: int
        The batch size for training and evaluation.
    seq_len: int
        The maximum sequence length of the input sentences.
    is_decoder_model: bool
        If True, load a decoder model (e.g., GPT-2). If False, load an encoder
        model (e.g., BERT). This parameter determines the type of model to load
        and affects the behavior of the forward method.
    lr: float
        The learning rate for the optimizer.
    dropout: float
        The dropout rate for the model.
    weight_decay: float
        The weight decay for the optimizer.
    lm_weight: float
        The weight for the language model loss. Default is 1.0.
    contrastive_weight: float
        The weight for the contrastive loss. Default is 1.0.
    contrastive_temp: float
        The temperature parameter for the contrastive loss. Default is 7e-2.
    do_late_interaction: bool
        If True, use late interaction to compute the similarity scores.
    use_max: bool
        If True, use maximum cosine similarity for late interaction. If False, use Gumbel softmax.
    initial_gumbel_temp: float
        The initial temperature for the Gumbel softmax, if `use_max` is False. Default is 1.0.
        auto_anneal_gumbel: bool
        If True, automatically anneal the Gumbel temperature linearly alongside the optimizer
        steps during training. Default is True.
    gumbel_linear_delta: float
        The linear delta for the Gumbel temperature. If `auto_anneal_gumbel` is True, this
        parameter is ignored. Default is 1e-3.
    min_gumbel_temp: float
        The minimum Gumbel temperature. Default is 1e-6.
        do_distance: bool
        If True, use distance-based weighting for late interaction.
    exp_decay: bool
        If True, use exponential decay for the distance weights. Only if
        `do_distance` is True.
    alpha: float
        The alpha parameter for the exponential decay function. Only if
        `do_distance` is True.
    project_up: Optional[bool]
        If True, project the embeddings up to a higher dimension before
        computing the contrastive loss. If False, project down to the same
        dimension. If None, use the default projection (up to 4x the hidden
        size).

    Attributes
    ----------
    lm: LanguageModel
        The language model used for the task.
    is_decoder_model: bool
        Indicates whether the loaded model is a decoder model (True) or an
        encoder model (False).
    lm_weight: float
        The weight for the language model loss.
    contrastive_weight: float
        The weight for the contrastive loss.
    initial_gumbel_temp: float
        The initial temperature for the Gumbel softmax.
    gumbel_temp: float
        The current Gumbel temperature.
    gumbel_linear_delta: Optional[float]
        The linear delta for the Gumbel temperature.
    optim_name: str
        The name of the optimizer to use.
    batch_size: int
        The batch size for training and evaluation.
    min_gumbel_temp: float
        The minimum Gumbel temperature.
    auto_anneal_gumbel: bool
        If True, automatically anneal the Gumbel temperature linearly alongside
        the optimizer steps during training.
    dropout: float
        The dropout rate for the model.
    lr: float
        The learning rate for the optimizer.
    val_auroc: BinaryAUROC
        The AUROC metric for validation.
    val_f1: BinaryF1Score
        The F1 score metric for validation.
    val_precision: BinaryPrecision
        The precision metric for validation.
    val_recall: BinaryRecall
        The recall metric for validation.
    test_auroc: BinaryAUROC
        The AUROC metric for testing.
    test_f1: BinaryF1Score
        The F1 score metric for testing.
    test_precision: BinaryPrecision
        The precision metric for testing.
    test_recall: BinaryRecall
        The recall metric for testing.
    contrastive_loss: InfoNCELoss
        The contrastive loss function used for training.
    fc1: nn.Linear
        The first linear layer for projecting the embeddings.
    fc2: nn.Linear
        The second linear layer for projecting the embeddings.
    optim_map: Dict[str, Type[torch.optim.Optimizer]]
        A mapping of optimizer names to their corresponding PyTorch
        optimizer classes.
    """

    optim_map = {
        "adamw": torch.optim.AdamW,
        "soap": SOAP,
        "sophia": SophiaG,
    }

    def __init__(
        self,
        optim_name: str,
        base_model_name: str,
        batch_size: int,
        seq_len: int,
        is_decoder_model: bool,
        lr: float = 2e-5,
        dropout: float = 0.1,
        weight_decay: float = 1e-2,
        lm_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        contrastive_temp: float = 7e-2,
        do_late_interaction: bool = True,
        use_max: bool = False,
        initial_gumbel_temp: float = 1.0,
        auto_anneal_gumbel: bool = True,
        gumbel_linear_delta: float = 1e-3,
        min_gumbel_temp: float = 1e-6,
        do_distance: bool = True,
        exp_decay: bool = True,
        alpha: float = 1.0,
        project_up: Optional[bool] = None,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters
        self.weight_decay = weight_decay
        self.lm = LanguageModel(base_model_name, is_decoder_model)
        self.is_decoder_model = is_decoder_model
        self.lm_weight = lm_weight
        self.contrastive_weight = contrastive_weight
        self.initial_gumbel_temp = initial_gumbel_temp
        self.gumbel_temp = initial_gumbel_temp
        self.gumbel_linear_delta = gumbel_linear_delta
        self.optim_name = optim_name
        self.batch_size = batch_size
        self.min_gumbel_temp = min_gumbel_temp
        self.auto_anneal_gumbel = auto_anneal_gumbel
        self.dropout = dropout
        self.lr = lr
        # Validation metrics
        self.val_auroc = BinaryAUROC(thresholds=None)
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        # Test metrics
        self.test_auroc = BinaryAUROC(thresholds=None)
        self.test_f1 = BinaryF1Score()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        if contrastive_weight > 0:
            self.contrastive_loss = InfoNCELoss(
                do_late_interaction=do_late_interaction,
                do_distance=do_distance,
                exp_decay=exp_decay,
                alpha=alpha,
                temperature=contrastive_temp,
                seq_len=seq_len,
                use_max=use_max,
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

    def _compute_losses(self, batch: Dict[str, torch.Tensor]):
        lm_loss, _, q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels", None),
        )
        _, _, k_embs = self(
            input_ids=batch["k_input_ids"],
            attention_mask=batch["k_attention_mask"],
        )

        contrastive_loss = 0.0
        pos_query_scores = None
        pos_query_targets = None
        if self.contrastive_weight > 0:
            pos_query_scores, pos_query_targets, contrastive_loss = (
                self.contrastive_loss(
                    q_embs,
                    k_embs,
                    batch["author_label"],
                    batch["attention_mask"],  # Use original attention masks
                    batch["k_attention_mask"],
                    gumbel_temp=self.gumbel_temp,
                )
            )

        total_loss = (self.lm_weight * lm_loss) + (
            self.contrastive_weight * contrastive_loss
        )

        metrics = {
            "pos_query_scores": pos_query_scores,
            "pos_query_targets": pos_query_targets,
            "lm_loss": lm_loss * self.lm_weight,
            "contrastive_loss": contrastive_loss * self.contrastive_weight,
            "total_loss": total_loss,
        }

        return metrics

    def configure_optimizers(self):  # type: ignore[override]
        """Initializes the optimizer and learning rate scheduler. The default
        scheduler is a cosine annealing schedule with warmup steps. The
        optimizer is selected based on the `optim_name` parameter, which can be
        "adamw", "soap", or "sophia".

        Also initializes the Gumbel temperature if `auto_anneal_gumbel` is True.
        The Gumbel temperature is linearly annealed during training.

        Returns
        -------
        dict
            A dictionary containing the optimizer and learning rate scheduler.
            The optimizer is selected based on the `optim_name` parameter, and
            the scheduler is a cosine annealing schedule with warmup steps.
        """
        logging.info(
            f"Configuring optimizer: {self.optim_name} with lr={self.lr}, weight_decay={self.weight_decay}"
        )

        optimizer = self.optim_map[self.optim_name](
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Calculate steps dynamically
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.05 * total_steps))

        if self.auto_anneal_gumbel:
            total_temp_range = self.initial_gumbel_temp - self.min_gumbel_temp
            self.gumbel_linear_delta = total_temp_range / total_steps

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,  # 0.5 cosine cycle → single smooth decay
            last_epoch=-1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):  # type: ignore[override]
        """Ovverride the optimizer_step method to include custom logic for
        updating the Gumbel temperature. This method is called after each
        optimizer step during training.

        Parameters
        ----------
        epoch: int
            The current epoch number.
        batch_idx: int
            The index of the current batch.
        optimizer: torch.optim.Optimizer
            The optimizer used for training.
        optimizer_closure: Callable
            A closure that computes the loss and gradients for the optimizer.
        """
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        # Update Gumbel temperature after each optimizer step
        if self.auto_anneal_gumbel:
            new_temp = self.gumbel_temp - self.gumbel_linear_delta
            self.gumbel_temp = max(new_temp, self.min_gumbel_temp)
            self.log("gumbel_temp", self.gumbel_temp, prog_bar=True)

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
        labels: Optional[torch.Tensor]
            The labels for the input data. If None and the model is a decoder,
            the input_ids will be used as labels.

        Returns
        -------
        lm_loss: torch.Tensor
            The computed loss for the input data.
        last_hidden_states: torch.Tensor
            The last hidden states of the model, which are the output embeddings
            for the input data.
        projected_embs: torch.Tensor
            The projected embeddings of the model, which are the output
            embeddings for the input data after applying the linear layers.
        """
        lm_loss, last_hidden_states = self.lm(
            input_ids, attention_mask=attention_mask, labels=labels
        )

        if self.contrastive_weight > 0:
            embs = F.dropout(last_hidden_states, p=self.dropout, training=self.training)
            embs = F.gelu(self.fc1(embs))
            embs = F.dropout(embs, p=self.dropout, training=self.training)
            projected_embs = self.fc2(embs)
        else:
            projected_embs = last_hidden_states

        return lm_loss, last_hidden_states, projected_embs

    def training_step(self, batch, batch_idx: int):
        """Compute the total loss for the training step.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            The input batch containing the input IDs, attention masks, and
            labels for the training step.
        batch_idx: int
            The index of the current batch.

        Returns
        -------
        total_loss: torch.Tensor
            The total loss for the training step, which is a combination of the
            language model loss and the contrastive loss.
        """
        metrics = self._compute_losses(batch)

        # Log metrics
        self.log(
            "gumbel_temp",
            self.gumbel_temp,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        self.log_dict(
            {
                "train_total_loss": metrics["total_loss"],
                "train_lm_loss": metrics["lm_loss"],
                "train_contrastive_loss": metrics["contrastive_loss"],
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return metrics["total_loss"]

    def validation_step(self, batch, batch_idx: int):
        """Compute the total loss for the validation step, as well as the
        auroc, f1, precision, and recall metrics.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            The input batch containing the input IDs, attention masks, and
            labels for the validation step.
        batch_idx: int
            The index of the current batch.
        """
        metrics = self._compute_losses(batch)

        if self.contrastive_weight > 0 and metrics["pos_query_scores"] is not None:
            pos_query_scores = metrics["pos_query_scores"]
            pos_query_targets = metrics["pos_query_targets"]

            # Generate binary labels (1 for correct key, 0 otherwise)
            binary_labels = torch.zeros_like(pos_query_scores, dtype=torch.long)
            rows = torch.arange(
                pos_query_scores.size(0), device=pos_query_scores.device
            )
            binary_labels[rows, pos_query_targets] = 1

            # Flatten scores and labels
            flat_scores = pos_query_scores.flatten()
            flat_labels = binary_labels.flatten()

            # Update metrics
            self.val_auroc(flat_scores, flat_labels)
            self.val_f1(flat_scores, flat_labels)
            self.val_precision(flat_scores, flat_labels)
            self.val_recall(flat_scores, flat_labels)

        self.log_dict(
            {
                "val_total_loss": metrics["total_loss"],
                "val_lm_loss": metrics["lm_loss"],
                "val_contrastive_loss": metrics["contrastive_loss"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def on_validation_epoch_end(self):
        """Aggregate and log the validation metrics at the end of the
        validation epoch."""
        self.log_dict(
            {
                "val_auroc": self.val_auroc.compute(),
                "val_f1": self.val_f1.compute(),
                "val_precision": self.val_precision.compute(),
                "val_recall": self.val_recall.compute(),
            },
            prog_bar=True,
        )
        self.val_auroc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def test_step(self, batch, batch_idx: int):
        """Compute the total loss for the test step, as well as the auroc, f1,
        precision, and recall metrics.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            The input batch containing the input IDs, attention masks, and
            labels for the test step.
        batch_idx: int
            The index of the current batch.
        """
        metrics = self._compute_losses(batch)
        if self.contrastive_weight > 0 and metrics["pos_query_scores"] is not None:
            pos_query_scores = metrics["pos_query_scores"]
            pos_query_targets = metrics["pos_query_targets"]

            # Generate binary labels (1 for correct key, 0 otherwise)
            binary_labels = torch.zeros_like(pos_query_scores, dtype=torch.long)
            rows = torch.arange(
                pos_query_scores.size(0), device=pos_query_scores.device
            )
            binary_labels[rows, pos_query_targets] = 1

            # Flatten scores and labels
            flat_scores = pos_query_scores.flatten()
            flat_labels = binary_labels.flatten()

            # Update metrics
            self.test_auroc(flat_scores, flat_labels)
            self.test_f1(flat_scores, flat_labels)
            self.test_precision(flat_scores, flat_labels)
            self.test_recall(flat_scores, flat_labels)

        self.log_dict(
            {
                "test_total_loss": metrics["total_loss"],
                "test_lm_loss": metrics["lm_loss"],
                "test_contrastive_loss": metrics["contrastive_loss"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def on_test_epoch_end(self):
        """Aggregate and log the test metrics at the end of the test epoch."""
        self.log_dict(
            {
                "test_auroc": self.test_auroc.compute(),
                "test_f1": self.test_f1.compute(),
                "test_precision": self.test_precision.compute(),
                "test_recall": self.test_recall.compute(),
            },
            prog_bar=True,
        )
        self.test_auroc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
