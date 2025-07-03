Train config with no weightning

```yml
# train.yml

project_name: "deep-stylometry"
group_name: "train-deepstylometry-512-se-default"
do_train: true
do_test: false

# === datamodule ===
ds_name: "se" # or halvest
batch_size: 32
tokenizer_name: "FacebookAI/roberta-base"
max_length: 512
map_batch_size: 1000
load_from_cache_file: true
# config_name: null
mlm_collator: false # only if you use an encoder

# === model ===
architecture: "deep-stylometry"
base_model_name: "FacebookAI/roberta-base"
is_decoder_model: false
dropout: 0.1
lm_weight: 0.0
contrastive_weight: 1.0
contrastive_temp: 0.91
pooling_method: "li" # li or mean
# --- trainer
lr: 2.65e-5
betas: [0.8, 0.999]
eps: 1.0e-7
weight_decay: 0.02
num_cycles: 0.5
# --- late interaction
distance_weightning: "none" # none, exp or linear
initial_gumbel_temp: 1.0 # null
auto_anneal_gumbel: true
min_gumbel_temp: 0.5
use_max: false
alpha: 0.59

# === wandb logger ===
use_wandb: true
log_model: true
watch: "all" # all, gradients, parameters or null

# === trainer ===
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
save_top_k: 4
# --- other
max_epochs: 4 # max_steps: 7000
val_check_interval: null
check_val_every_n_epoch: null
log_every_n_steps: 1
test_every_n_epochs: 1
accumulate_grad_batches: 4
gradient_clip_val: null # null
precision: "32" # 32-true or bf16-mixed
```

Train config with exponential decay.

```yml
# train.yml

project_name: "deep-stylometry"
group_name: "train-deepstylometry-512-se-default"
do_train: true
do_test: false

# === datamodule ===
ds_name: "se" # or halvest
batch_size: 32
tokenizer_name: "FacebookAI/roberta-base"
max_length: 512
map_batch_size: 1000
load_from_cache_file: true
# config_name: null
mlm_collator: false # only if you use an encoder

# === model ===
architecture: "deep-stylometry"
base_model_name: "FacebookAI/roberta-base"
is_decoder_model: false
dropout: 0.1
lm_weight: 0.0
contrastive_weight: 1.0
contrastive_temp: 0.98
pooling_method: "li" # li or mean
# --- trainer
lr: 4.73e-5
betas: [0.7, 0.999]
eps: 1.0e-9
weight_decay: 0.09
num_cycles: 0.5
# --- late interaction
distance_weightning: "exp" # none, exp or linear
initial_gumbel_temp: 1.0 # null
auto_anneal_gumbel: true
min_gumbel_temp: 0.5
use_max: false
alpha: 0.22

# === wandb logger ===
use_wandb: true
log_model: true
watch: "all" # all, gradients, parameters or null

# === trainer ===
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
save_top_k: 4
# --- other
max_epochs: 4 # max_steps: 7000
val_check_interval: null
check_val_every_n_epoch: null
log_every_n_steps: 1
test_every_n_epochs: 1
accumulate_grad_batches: 4
gradient_clip_val: null # null
precision: "32" # 32-true or bf16-mixed
```

Train config linear decay

```yml
# train.yml

project_name: "deep-stylometry"
group_name: "train-deepstylometry-512-se-default"
do_train: true
do_test: false

# === datamodule ===
ds_name: "se" # or halvest
batch_size: 32
tokenizer_name: "FacebookAI/roberta-base"
max_length: 512
map_batch_size: 1000
load_from_cache_file: true
# config_name: null
mlm_collator: false # only if you use an encoder

# === model ===
architecture: "deep-stylometry"
base_model_name: "FacebookAI/roberta-base"
is_decoder_model: false
dropout: 0.1
lm_weight: 0.0
contrastive_weight: 1.0
contrastive_temp: 0.91
pooling_method: "li" # li or mean
# --- trainer
lr: 7.03e-5
betas: [0.8, 0.999]
eps: 1.0e-8
weight_decay: 0.066
num_cycles: 0.5
# --- late interaction
distance_weightning: "linear" # none, exp or linear
initial_gumbel_temp: 1.0 # null
auto_anneal_gumbel: true
min_gumbel_temp: 0.5
use_max: false
alpha: 0.98

# === wandb logger ===
use_wandb: true
log_model: true
watch: "all" # all, gradients, parameters or null

# === trainer ===
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
save_top_k: 4
# --- other
max_epochs: 4 # max_steps: 7000
val_check_interval: null
check_val_every_n_epoch: null
log_every_n_steps: 1
test_every_n_epochs: 1
accumulate_grad_batches: 4
gradient_clip_val: null # null
precision: "32" # 32-true or bf16-mixed
```
