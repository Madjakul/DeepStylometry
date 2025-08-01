# deep_stylometry/utils/argparsers/train_argparse.py

import argparse


class TrainArgparse:
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
            description="Arguments used to train/fine-tune a model."
        )
        parser.add_argument(
            "--config_path",
            type=str,
            required=True,
            help="Path to the config file.",
        )
        parser.add_argument(
            "--logs_dir",
            type=str,
            required=True,
            help="Directory where the logs are stored.",
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=None,
            help="Number of processes to use. Default is the number of CPUs.",
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
            help="Directory where the model's checkpoints are stored.",
        )
        parser.add_argument(
            "--checkpoint_path",
            type=str,
            default=None,
            help="""Path to a checkpoint file if it exists. This argument is only used 
                when testing an existing model.""",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Path to the cache directory for HuggingFace.",
        )
        args, _ = parser.parse_known_args()
        return args
