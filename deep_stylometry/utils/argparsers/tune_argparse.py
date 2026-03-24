# deep_stylometry/utils/argparsers/tune_argparse.py

import argparse


class TuneArgparse:
    """Argument parser for PLI hyperparameter tuning via Optuna."""

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser(
            description="Optuna hyperparameter search for Patch-Level Late Interaction."
        )
        parser.add_argument(
            "--config_path",
            type=str,
            required=True,
            help="Path to a base config YAML (pooling_method will be overridden to pli).",
        )
        parser.add_argument(
            "--processed_ds_dir",
            type=str,
            required=True,
            help="Directory where the processed datasets are stored.",
        )
        parser.add_argument(
            "--logs_dir",
            type=str,
            required=True,
            help="Directory where Optuna logs will be saved.",
        )
        parser.add_argument(
            "--n_trials",
            type=int,
            default=30,
            help="Number of Optuna trials.",
        )
        parser.add_argument(
            "--max_steps",
            type=int,
            default=10000,
            help="Training steps per trial.",
        )
        parser.add_argument(
            "--output_path",
            type=str,
            default="configs/best_pli_params.yml",
            help="Path to write the best hyperparameters YAML.",
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=4,
            help="Number of dataloader worker processes.",
        )
        parser.add_argument(
            "--study_name",
            type=str,
            default="pli_search",
            help="Optuna study name.",
        )
        parser.add_argument(
            "--storage",
            type=str,
            default=None,
            help="Optuna storage URL (e.g. sqlite:///pli.db). None uses in-memory storage.",
        )
        args, _ = parser.parse_known_args()
        return args
