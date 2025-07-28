# DeepStylometry

Embedding subtle stylometric features.

---

A deep learning architecture designed to investigate the best way of embedding
stylistic features from text using contrastive learning and interpretability.

## Requirements

- 3.9 <= Python <= 3.12
- A HuggingFace account in order to run the preprocessing script or get access to some pre-trained models.

### Logging into your HuggingFace account

```bash
huggingface-cli login
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

It is recommended to use the shell scripts provided in the [`scripts`](./scripts) folder to run the code.
Make sure to modify the parameters in the [`scripts`](./scripts)'s but also in the [`configs`](./configs)'s file you want to use to fit your need.
The scripts are designed to be run from any directory.

If you still want to run the Python scripts directly:

### Hyperparameter tuning

Set the hyperparameters you want in [`tune.yml`](./configs/tune.yml).

```
usage: tune.py [-h] --config_path CONFIG_PATH --ray_storage_path
               RAY_STORAGE_PATH --logs_dir LOGS_DIR [--num_proc NUM_PROC]
               [--cache_dir CACHE_DIR]

Arguments used for hyper-parameter tuning.

options:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the config file.
  --ray_storage_path RAY_STORAGE_PATH
                        Directory where Ray will save the logs and experiments
                        results.
  --logs_dir LOGS_DIR   Directory where the logs will be saved.
  --num_proc NUM_PROC   Number of processes to use. Default is the number of
                        CPUs minus one.
  --cache_dir CACHE_DIR
                        Path to the cache directory for HuggingFace.
```

### Training/Fine-tuning

Set the hyperparameters you want in [`train.yml`](./configs/train.yml).

```
usage: train.py [-h] --config_path CONFIG_PATH --logs_dir LOGS_DIR
                [--num_proc NUM_PROC] [--checkpoint_dir CHECKPOINT_DIR]
                [--checkpoint_path CHECKPOINT_PATH] [--cache_dir CACHE_DIR]

Arguments used to train/fine-tune a model.

options:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the config file.
  --logs_dir LOGS_DIR   Directory where the logs are stored.
  --num_proc NUM_PROC   Number of processes to use. Default is the number of
                        CPUs.
  --checkpoint_dir CHECKPOINT_DIR
                        Directory where the model's checkpoints are stored.
  --checkpoint_path CHECKPOINT_PATH
                        Path to a checkpoint file if it exists. This argument
                        is only used when testing an existing model.
  --cache_dir CACHE_DIR
                        Path to the cache directory for HuggingFace.
```

## Citation

To cite DeepStylometry:

```bib
TBD
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
