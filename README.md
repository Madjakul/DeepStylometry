# DeepStylometry

A codebase to investigate the best way to embed stylistic features from text.

---

## Requirements

- 3.9 <= Python <= 3.12
- A HuggingFace account in order to run the preprocessing script and get access to pre-trained models.
- x86_64 if you want to install flash-attention.

### Logging into your HuggingFace account

```

huggingface-cli login

```

## Installation

### x86 CPUs

If you have an x86_64 CPU, you can use the installation script to install torch and flash-attention cleanly. From the repository, run

```

./install.sh

```

Once torch and flash-attention are installed, you can install the remaining packages using the constraint file created by the installation script.

```

pip install --upgrade -r requirements.txt -c constraints.txt

```

### Other CPUs

You need to install torch manually before installing the remaining packages. You are not limited by flash-attention and can therefore use the torch version you want with the adequate CUDA version.

```

pip install torch --index-url https://download.pytorch.org/whl/cu<your cuda version>

````

#### Optional

You can create a constraint file to make sure the remaining packages do not mess with torch dependencies.

```sh
pip freeze | grep -E "^(torch==|nvidia-)" >constraints.txt
````

Finally, install the remaining packages using constraints if necessary.

```
pip install --upgrade -r requirements.txt -c constraints.txt
```

## Usage

It is recommended to use the bash scripts provided in the [`scripts`](./scripts) directory to run the code.
Make sure to modify the parameters in the [`scripts`](./scripts) as well as in the [`configs`](./configs) files.
The scripts are designed to be run from any directory.

If you still want to run the Python scripts directly:

### Training/Fine-tuning

Set the hyperparameters you want in [`train.yml`](./configs/train.yml).

```
usage: train.py [-h] --config_path CONFIG_PATH --processed_ds_dir
                PROCESSED_DS_DIR --logs_dir LOGS_DIR
                [--checkpoint_dir CHECKPOINT_DIR] [--num_proc NUM_PROC]
                [--cache_dir CACHE_DIR]

Arguments used to train/fine-tune a model.

options:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the config file.
  --processed_ds_dir PROCESSED_DS_DIR
                        Directory where the processed datasets are stored.
  --logs_dir LOGS_DIR   Directory where the logs are stored.
  --checkpoint_dir CHECKPOINT_DIR
                        Directory where the model checkpoints are stored.
  --num_proc NUM_PROC   Number of processes to use. Default is the number
                        of CPUs.
  --cache_dir CACHE_DIR
                        Path to the cache directory for HuggingFace.
```

[`train.py`](./train.py) only runs classification validation. It computes the alignment and uniformity loss as a proxy to monitor how well training is progressing. A checkpoint is saved after each validation.

### Testing

Change the configuration in [`test.yml`](./configs/test.yml).

```
usage: test.py [-h] --config_path CONFIG_PATH --processed_ds_dir
               PROCESSED_DS_DIR --checkpoint_path CHECKPOINT_PATH
               --logs_dir LOGS_DIR [--num_proc NUM_PROC]
               [--cache_dir CACHE_DIR]

Arguments used to test a single subset on retrieval.

options:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the config file.
  --processed_ds_dir PROCESSED_DS_DIR
                        Directory where the processed datasets are stored.
  --checkpoint_path CHECKPOINT_PATH
                        Path to the model checkpoint to load.
  --logs_dir LOGS_DIR   Directory where the logs will be saved.
  --num_proc NUM_PROC   Number of processes to use. Default is the number
                        of CPUs minus one.
  --cache_dir CACHE_DIR
                        Path to the cache directory for HuggingFace.
```

[`test.py`](./test.py) tests a checkpoint on retrieval only. Despite being trained with a single target, testing fetches all the relevant documents for a given query.

The test script returns the retrieval accuracy, or hit@1, and {nDCG, MRR, Recall}@{5, 10, 20, 100}.

## Results

Example results after 23k steps on a 4-sentence test set.

| Model               | Validation Accuracy | Recall@20     | Recall@100    | nDCG@20       | nDCG@100      |
| ------------------- | ------------------- | ------------- | ------------- | ------------- | ------------- |
| BM25                | NA                  | TBD           | TBD           | TBD           | TBD           |
| ModernBERT (single) | 87.37               | 12.06 / 14.7  | 29.38 / 32.8  | 6.34 / 8.1    | 10.08 / 12.14 |
| ModernBERT (multi)  | 95.22               | 28.53 / 48.12 | 44.65 / 67.16 | 19.67 / 36.21 | 23.26 / 40.53 |

The first metric (in %) benchmarks a model using a single dense vector. The second metric tracks the performance using multiple vectors.
Models trained with a single vector benefit from using multiple vectors during inference.

## Citation

To cite DeepStylometry:

```bib
TBD
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).