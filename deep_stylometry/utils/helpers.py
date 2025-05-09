# deep_strylometry/utils/helpers.py

import json
import logging
import os
import os.path as osp

import yaml
from transformers import AutoTokenizer

WIDTH = 88


def get_tokenizer(model_name: str, **kwargs):
    """Get a tokenizer from the model name.

    Parameters
    ----------
    model_name: str
        Name of the model.

    Returns
    -------
    tokenizer: transformers.PretrainedTokenizerBase
        Tokenizer for the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("Tokenizer has neither pad_token nor eos_token defined.")
    return tokenizer


def load_config_from_file(path: str):
    """Load a configuration file from a path.

    Parameters
    ----------
    path: str
        Path to the configuration file.

    Returns
    -------
    config: Dict[str, Any]
        Configuration dictionary.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    load_config_map = {
        "json": load_config_from_json,
        "yaml": load_config_from_yaml,
        "yml": load_config_from_yaml,
    }
    try:
        assert os.path.isfile(path)
    except AssertionError:
        raise FileNotFoundError(f"No file at {path}.")
    _, file_extension = osp.splitext(path)
    file_extension = file_extension[1:]
    if file_extension not in load_config_map:
        raise ValueError(f"File extension {file_extension} not supported.")
    logging.info(f"Loading configuration from {path}.")
    config = load_config_map[file_extension](path)
    return config


def load_config_from_json(path: str):
    """Load a JSON configuration file from a path.

    Parameters
    ----------
    path: str
        Path to the JSON configuration file.

    Returns
    -------
    config: Dict[str, Any]
        Configuration dictionary.
    """
    config = json.load(open(path))
    return config


def load_config_from_yaml(path: str):
    """Load a YAML configuration file from a path.

    Parameters
    ----------
    path: str
        Path to the YAML configuration file.

    Returns
    -------
    config: Dict[str, Any]
        Configuration dictionary.
    """
    config = yaml.load(open(path), Loader=yaml.FullLoader)
    return config
