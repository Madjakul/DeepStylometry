# deep_strylometry/utils/argparsers/tune_argparse.py

import argparse


class TuneArgparse:
    """Argument parser for hyper-parameter tuning."""

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(
            description="Arguments used for hyper-parameter tuning."
        )
        parser.add_argument(
            "--config_path",
            type=str,
            required=True,
            help="Path to the config file.",
        )
        parser.add_argument(
            "--ray_storage_path",
            type=str,
            required=True,
            help="Directory where Ray will save the logs and experiments results.",
        )
        parser.add_argument(
            "--logs_dir",
            type=str,
            required=True,
            help="Directory where the logs will be saved.",
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=None,
            help="Number of processes to use. Default is the number of CPUs minus one.",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Path to the cache directory for HuggingFace.",
        )
        args, _ = parser.parse_known_args()
        return args
