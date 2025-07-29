## Configs

### Tune configs

Tune configuration for the triplet loss

```yml
# tune.yml

mode: "tune"
project_name: "deep-stylometry"
group_name: "tune-deepstylometry-512-se-triplet"

data:
  ds_name: "se" # or halvest
  batch_size: 32
  tokenizer_name: "FacebookAI/roberta-base"
  max_length: 512
  map_batch_size: 1000
  load_from_cache_file: true
  # config_name: null
  mlm_collator: false # only if you use an encoder

model:
  base_model_name: "FacebookAI/roberta-base"
  is_decoder_model: false
  add_linear_layers: true
  dropout:
    type: uniform
    min: 0.1
    max: 0.3
  pooling_method: "mean" # li or mean
  # --- late interaction
  distance_weightning: none
  alpha:
    type: loguniform
    min: 0.1
    max: 1.0
  use_softmax: true
  initial_gumbel_temp: 1.0
  auto_anneal_gumbel: true
  min_gumbel_temp: 0.5

tune:
  loss: "triplet"
  # tau:
  #   type: choice
  #   values: [0.05, 0.07, 0.1, 0.15, 0.2]
  # lambda_: 0.9 # Only if using hybrid loss
  margin: # Only if using triplet loss
    type: uniform
    min: 0.2
    max: 1.5
  lm_loss_weight: 0.0
  # --- optimizer ---
  lr:
    type: loguniform
    min: 7.0e-6
    max: 7.0e-5
  betas:
    type: choice
    values: [[0.9, 0.999], [0.8, 0.999], [0.7, 0.999]]
  eps:
    type: choice
    values: [1.0e-7, 1.0e-8, 1.0e-9]
  weight_decay:
    type: loguniform
    min: 0.01
    max: 0.2
  num_cycles: 0.5
  # --- trainer ---
  device: "gpu"
  num_devices_per_trial: 1
  num_cpus_per_trial: 10
  max_epochs: 4
  log_every_n_steps: 1
  accumulate_grad_batches: 1
  gradient_clip_val: null
  precision: "32" # 32-true or bf16-mixed
  # --- tuner ---
  metric: "val_auroc"
  mode: "max"
  num_samples: 30
  max_concurrent_trials: 3
  time_budget_s: 151200 # 42h (in seconds)
  max_t: 4
  grace_period: 2
  # --- wandb logger ---
  use_wandb: true
```

### Train configs

Train configs for triplet loss

```yml
# train.yml

mode: "train"
project_name: "deep-stylometry"
group_name: "train-deepstylometry-512-se-default"
do_train: true
do_test: false

data:
  ds_name: "se" # or halvest
  batch_size: 32
  tokenizer_name: "FacebookAI/roberta-base"
  max_length: 512
  map_batch_size: 1000
  load_from_cache_file: true
  config_name: null
  mlm_collator: false # only if you use an encoder

model:
  base_model_name: "FacebookAI/roberta-base"
  is_decoder_model: false
  add_linear_layers: true
  dropout: 0.3
  pooling_method: "mean" # li or mean
  # --- late interaction
  distance_weightning: "none" # none, exp or linear
  alpha: 0.28
  use_softmax: true
  initial_gumbel_temp: null # null
  auto_anneal_gumbel: true
  min_gumbel_temp: 0.5

train:
  loss: "triplet"
  tau: 0.5
  lambda_: 0.5 # only if using hybrid
  margin: 0.32 # Only if using triplet loss
  lm_loss_weight: 0.0
  # --- optimizer ---
  lr: 1.25e-5
  betas: [0.9, 0.999]
  eps: 1.0e-9
  weight_decay: 0.11
  num_cycles: 0.5
  # --- checkpointing
  checkpoint_metric: "val_auroc" # only if a checkpoint directory was provided
  checkpoint_mode: "max"
  save_top_k: 4
  # --- trainer ---
  device: "gpu"
  num_devices: 3
  strategy: "ddp_find_unused_parameters_true" # "ddp"
  process_group_backend: "gloo" # gloo, nccl, mpi
  max_epochs: 4 # max_steps: 7000
  val_check_interval: null
  check_val_every_n_epoch: null
  log_every_n_steps: 1
  accumulate_grad_batches: 1
  gradient_clip_val: null # null
  precision: "32" # 32-true or 16-mixed
  # --- wandb logger ---
  use_wandb: true
  log_model: true
  watch: "all" # all, gradients, parameters or null
```
