# deep_strylometry/utils/argparsers/test_argparse.py

import argparse


class TestArgparse:
    """Argument parser for evaluation / retrieval testing."""

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(
            description="Arguments used to test a single subset on retrieval."
        )
        parser.add_argument(
            "--config_path",
            type=str,
            required=True,
            help="Path to the config file.",
        )
        parser.add_argument(
            "--processed_ds_dir",
            type=str,
            required=True,
            help="Directory where the processed datasets are stored.",
        )
        parser.add_argument(
            "--checkpoint_path",
            type=str,
            required=True,
            help="Path to the model checkpoint to load.",
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
        parser.add_argument(
            "--test_subset",
            type=str,
            default=None,
            help="Override cfg.data.test_subset (e.g. base-2, base-4, unrestricted, se).",
        )
        parser.add_argument(
            "--ds_name",
            type=str,
            default=None,
            help="Override cfg.data.ds_name (halvest or se).",
        )
        args, _ = parser.parse_known_args()
        return args
