# DeepStylometry

Encoding subtle stylometric faetures.

---

## Requirements

- 3.9 <= Python <= 3.12
- A HuggingFace account in order to run the preprocessing script or get access to some pre-trained models.

### Logging in to you HuggingFace account

```bash
huggingface-cli login
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

It is recommended to use the shell scripts provided in the [`scripts`](./scripts) folder to run the code.
Make sure to modify the parameters in the [`scripts`](./scripts) but also in the [`configs`](./configs) you want to use to fit your need.
The scripts are designed to be run from any directory.

If you still want to run the Python scripts directly:

### Preprocessing

```
usage: preprocess.py [-h] --config_path CONFIG_PATH [--num_proc NUM_PROC] [--cache_dir CACHE_DIR]

Argument parser to flatten data from StyleEmbedding dataset.

options:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the config file.
  --num_proc NUM_PROC   Number of processes to use. Default is the number of CPUs.
  --cache_dir CACHE_DIR
                        Path to the cache directory.
```
