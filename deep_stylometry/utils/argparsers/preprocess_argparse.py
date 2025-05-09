# deep_stylometry/utils/argparsers/preprocess_argparse.py

import argparse


class PreprocessArgparse:
    """Argument parser for fine-tuning."""

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(
            description="Argument parser to flatten data from [StyleEmbedding dataset](https://huggingface.co/datasets/AnnaWegmann/StyleEmbeddingData)."
        )
        parser.add_argument(
            "--config_path",
            type=str,
            required=True,
            help="Path to the config file.",
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=None,
            help="Number of processes to use. Default is the number of CPUs.",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Path to the cache directory.",
        )
        args, _ = parser.parse_known_args()
        return args
