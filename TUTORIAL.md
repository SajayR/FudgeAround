# CIS Testbed Quickstart

This repo is intentionally tiny so you can swap architectures, datasets, and logging knobs without digging through an "ML ops" stack. Read this once and you’ll be able to drop in any new idea within minutes.

## Project Layout

- `train.py` – main CLI for training (multi-dataset aware).
- `evaluate.py` – eval-only CLI against saved checkpoints.
- `configs/` – YAML configs (`default.yaml` is the template) + the minimal loader.
- `data/datasets.py` – ImageFolder loader with normalization presets and DatasetSpec.
- `models/` – micro registry (`register`, `create`) and the Dinov2+LoRA builder.
- `utils/` – logging/optim/scheduler/metrics helpers.

## 1. Declaring Datasets

- Datasets are listed in your YAML under `datasets:`. Each entry can be:
  ```yaml
  datasets:
    - name: cifar-10
      val_split: test
      normalization: cifar10
    - name: flowers-102
      val_split: test
      batch_size: 64
  ```
- The loader expects a standard folder layout: `dataset_name/{train,val,test}/{class_name}/image.jpg`.
- Any keys you omit fall back to `data:` defaults (root, image size, batch size, augmentation, etc.).
- Add new datasets by simply adding a new entry or pass `--datasets <name>` on the CLI—no code changes needed as long as the folders exist under `/speedy/datasets`.

## 2. Dropping In A New Model

1. **Create a builder** in `models/your_model.py`:
   ```python
   from models import register
   import torch.nn as nn
   from data import DatasetInfo

   @register("my_resnet")
   def build_my_resnet(params: dict, dataset: DatasetInfo):
       num_classes = params.get("num_classes", dataset.num_classes)
       model = nn.Sequential( ... )  # build your backbone/head
       extras = {"note": "metadata for logs"}
       return model, extras
   ```
   - Use `@register("name")` so the registry knows the model.
   - The function receives `params` from `config['model']['params']` and the dataset metadata.
   - Return `(nn.Module, metadata_dict)`; no classes or fancy wrappers required.

2. **Hook it up in the config**:
   ```yaml
   model:
     name: my_resnet
     params:
       width: 128
       dropout: 0.2
  ```
  Any key in `params` is passed to your builder.

- Need a live example? See `models/dinov2_bs_relora.py` plus the matching
  `configs/dinov2_bs_relora.yaml` for a Dinov2 classifier wired for the
  BS-ReLoRA training loop.

## 3. Running Training

```bash
python train.py --config configs/default.yaml
```

### BS-ReLoRA cycles

With the default config the trainer runs BS-ReLoRA cycles instead of plain
epochs. Each cycle consists of a 32-step **probe** (freeze linear weights but
update Adam moments), SVD-based **subspace extraction** with soft deflation,
a 128-step **low-rank** phase that only optimises the LoRA cores, and a final
**merge** that folds the low-rank update into the base weights while damping
optimizer state.

Tweak the behaviour in `training.bs_relora`:

```yaml
training:
  bs_relora:
    enabled: true
    rank: 16
    oversample: 4
    probe_steps: 32
    low_rank_steps: 128
    cycles: auto
    trainable_modules: ["head"]
    target_modules: ["query", "key", "value", "dense"]
```

With `cycles: auto` (the default), the trainer estimates how many
probe+low-rank bundles fit into an epoch by looking at the dataloader
length, your gradient-accumulation factor, and the chosen probe/low-rank
step counts. The total number of cycles becomes
``epochs × ceil(optimizer_steps_per_epoch / (probe_steps + low_rank_steps))``.
Set an explicit integer if you need a fixed count instead.

`trainable_modules` is a small allowlist (module-name prefixes) that stays
trainable during the low-rank burst. By default only the classifier head is
updated while everything else—including the frozen base linear weights—is kept
static.

Set `enabled: false` to fall back to classic epoch-based training.

Common tweaks:

- Limit to specific datasets at runtime:
  ```bash
  python train.py --config configs/default.yaml --datasets cifar-10 flowers-102
  ```
- Override values on the fly:
  ```bash
  python train.py --config configs/default.yaml \
                  --override training.epochs=2 model.params.dropout=0.2 \
                  --no-wandb
  ```

Outputs for each dataset end up under `runs/<dataset>/<run_name>/`:

- `metrics.jsonl` – JSON per logging step (training + validation metrics).
- `summary.json` – final summary (best val accuracy, checkpoint path, etc.).
- `checkpoints/` – `epoch_XXX.pt` plus `best.pt` if validation metrics are logged.
- `train.log` – console + file logging.

## 4. Running Evaluation

Evaluate any checkpoint with the same config:

```bash
python evaluate.py \
    --config configs/default.yaml \
    --dataset cifar-10 \
    --checkpoint runs/cifar-10/<run>/checkpoints/best.pt
```

Use `--override` if you want to tweak evaluation-time settings (e.g., batch size or image size) – just keep them consistent with training.

## 5. Helpful Tricks

- **Dry run / sanity check loaders**: set epochs to zero via CLI override to confirm dataset discovery without training.
  ```bash
  python train.py --config configs/default.yaml --datasets cifar-10 --override training.epochs=0
  ```
- **Run naming**: every run is named `<dataset>_<model>_<suffix>`. The suffix comes from `experiment.name_suffix` (override it via CLI) and is omitted if blank. Example: `--override experiment.name_suffix=rank-16` yields `cifar-10_vit-base-lora_rank-16-<timestamp>` locally and `cifar-10_vit-base-lora_rank-16` in W&B. Setting `logging.run_name` still force-overrides the whole name when you need a one-off label.
- **Weights & Biases**: controlled by `logging.use_wandb`. CLI flag `--no-wandb` forces it off. Dataset names are also injected as W&B tags so you can filter quickly.
- **Precision**: `training.precision.dtype` supports `bf16` or `fp16`. Automatic fallback to fp32 on CPU.
- **Gradient accumulation**: `training.grad_accumulation` – the script handles scaling and logging per accumulated step.
- **Monitoring toggles**: flip `training.monitor.grad_norm`, `param_norm`, `memory` to inject extra metrics into logs.

## 6. Extend Metrics or Logging

- All logging is dictionary based. Inside `train_epoch` or `evaluate` in `train.py`, add new keys to the `metrics` dict before calling `logger.log`.
- For per-class stats, you can extend `utils/metrics.py` and log the outputs to `summary.json` at the end of evaluation.

## TL;DR Workflow

1. Drop your dataset under `/speedy/datasets/<name>` (train/test folders).
2. Edit YAML `datasets:` list or pass `--datasets` on the CLI.
3. Register any new model builder in `models/` and set `model.name` in the config.
4. `python train.py --config ...` to run training.
5. Inspect results under `runs/` (JSONL + checkpoints).
6. `python evaluate.py --config ... --checkpoint ...` for quick evals.

That’s it—no engines, no ceremony. Swap params, add models, iterate fast.
