Train config with Sophia optimizer on Style Embedding

```yml
# train.yml

project_name: "deep-stylometry"
experiment_name: "train-roberta-sophia-512-se-li-gumbel-dist-autoexp-v1"
do_train: true
do_test: false

# --- datamodule ---
ds_name: "se" # or halvest
batch_size: 32
tokenizer_name: "FacebookAI/roberta-base"
max_length: 512
map_batch_size: 1000
load_from_cache_file: true
# config_name: null
mlm_collator: false # only if you use an encoder

# --- model ---
architecture: "deep-stylometry"
optim_name: "sophia"
base_model_name: "FacebookAI/roberta-base"
is_decoder_model: false
lr: 4.28e-5
dropout: 0.1
weight_decay: 0.015
lm_weight: 0.0
contrastive_weight: 1.0
contrastive_temp: 0.14
# --- late interaction
do_late_interaction: true
initial_gumbel_temp: 2.0 # null
auto_anneal_gumbel: true
# gumbel_linear_delta: 1e-3
min_gumbel_temp: 0.4
use_max: false
do_distance: true
exp_decay: true
# alpha: 0.5
# project_up: true

# --- wandb logger ---
use_wandb: true
log_model: true
watch: "all" # all, gradients, parameters or null

# --- trainer ---
device: "gpu"
num_devices: 3
strategy: "ddp_find_unused_parameters_true" # "ddp"
# --- early stopping
early_stopping: false
# early_stopping_metric: "val_total_loss"
# early_stopping_mode: "min"
# early_stopping_patience: 3
# --- checkpointing
checkpoint_metric: "val_auroc" # only if a checkpoint directory was provided
checkpoint_mode: "max"
save_top_k: 8
# --- other
max_epochs: 8 # max_steps: 7000
val_check_interval: 1.0
log_every_n_steps: 1
accumulate_grad_batches: 8
gradient_clip_val: 0.59
precision: "32" # 32-true or bf16-mixed
```
