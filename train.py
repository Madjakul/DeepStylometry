# train.py

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from deep_stylometry.modules.modeling_deep_stylometry import DeepStylometry
from deep_stylometry.utils.data.halvest_data import HALvestDataModule


class ConvergenceTestCallback(L.Callback):
    def __init__(self):
        self.losses = []
        self.steps = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.losses.append(float(outputs["loss"]))
        self.steps.append(trainer.global_step)


def run_convergence_test():
    # Initialize data module with small subset
    dm = HALvestDataModule(
        batch_size=4,
        num_proc=15,
        tokenizer_name="openai-community/gpt2",
        max_length=256,
    )

    # Patch setup to use only first 1000 examples
    # original_setup = dm.setup
    #
    # def patched_setup(stage: str):
    #     original_setup(stage)
    #     if stage == "fit":
    #         dm.train_dataset = dm.train_dataset.select(range(800))  # type: ignore

    # dm.setup = patched_setup

    dm.prepare_data()
    dm.setup("fit")

    # Initialize model with conservative hyperparams
    model = DeepStylometry(
        optim_name="adamw",
        base_model_name="openai-community/gpt2",
        batch_size=4,
        seq_len=256,
        lr=3e-4,
        clm_weight=1,
        contrastive_weight=0.5,
        do_late_interaction=True,
        do_distance=True,
    )

    # Setup training with progress tracking
    callback = ConvergenceTestCallback()
    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[callback],
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
        accelerator="gpu",
    )

    # Run training
    with tqdm(total=len(dm.train_dataloader())) as pbar:
        trainer.fit(model, dm)
        pbar.update(len(dm.train_dataloader()) - pbar.n)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(callback.steps, callback.losses)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Convergence on First 1,000 Examples")
    plt.grid(True)
    plt.savefig("convergence_plot.png")
    plt.close()

    # Save raw data
    pd.DataFrame({"step": callback.steps, "loss": callback.losses}).to_csv(
        "convergence_data.csv", index=False
    )


if __name__ == "__main__":
    run_convergence_test()
