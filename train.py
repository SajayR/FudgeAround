"""Simple multi-dataset training script with LoRA-ready Dinov2 backbone."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None

from configs import load_config
from data import DatasetSpec, create_dataloaders
from models import create as create_model
from utils.metrics import accuracy, top_k_accuracy
from utils.training_utils import (
    create_optimizer,
    create_scheduler,
    grad_norm,
    param_norm,
    save_checkpoint,
    set_seed,
    setup_logging,
)
import os

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

LOGGER = logging.getLogger("train")


def _slugify(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace("/", "-")
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^0-9A-Za-z_.-]", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def build_run_labels(
    experiment_cfg: Dict[str, any],
    model_cfg: Dict[str, any],
    logging_cfg: Dict[str, any],
    dataset_name: str,
) -> tuple[str, str]:
    custom = logging_cfg.get("run_name")
    if custom:
        custom_slug = _slugify(custom) or "run"
        return custom_slug, custom_slug

    dataset_slug = _slugify(dataset_name) or "dataset"
    model_slug = _slugify(model_cfg.get("name")) or "model"

    base_parts = [dataset_slug, model_slug]
    base_label = "_".join(base_parts)

    suffix = _slugify(
        experiment_cfg.get("name_suffix")
        or logging_cfg.get("run_name_suffix")
        or logging_cfg.get("run_suffix")
    )
    run_slug = base_label if not suffix else f"{base_label}_{suffix}"

    return base_label, run_slug


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train models on folder-based vision datasets"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path or name of the YAML config",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset names to train (filters config list)",
    )
    parser.add_argument("--override", nargs="*", help="Dotlist overrides key=value")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging even if config enables it",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def build_dataset_specs(
    config: Dict[str, any], selected: List[str] | None
) -> List[DatasetSpec]:
    data_defaults = config.get("data", {})
    entries = config.get("datasets", [])
    processed: List[Dict[str, any]] = []

    for entry in entries:
        if isinstance(entry, str):
            processed.append({"name": entry})
        else:
            processed.append(dict(entry))

    if selected:
        wanted = set(selected)
        filtered = [e for e in processed if e["name"] in wanted]
        missing = wanted - {e["name"] for e in filtered}
        for name in missing:
            filtered.append({"name": name})
        processed = filtered

    specs = []
    for entry in processed:
        spec = DatasetSpec(
            name=entry["name"],
            root=entry.get("root", data_defaults.get("root", "/speedy/datasets")),
            train_split=entry.get(
                "train_split", data_defaults.get("train_split", "train")
            ),
            val_split=entry.get("val_split", data_defaults.get("val_split")),
            image_size=int(
                entry.get("image_size", data_defaults.get("image_size", 224))
            ),
            batch_size=int(
                entry.get("batch_size", data_defaults.get("batch_size", 64))
            ),
            num_workers=int(
                entry.get("num_workers", data_defaults.get("num_workers", 8))
            ),
            pin_memory=bool(
                entry.get("pin_memory", data_defaults.get("pin_memory", True))
            ),
            persistent_workers=bool(
                entry.get(
                    "persistent_workers", data_defaults.get("persistent_workers", True)
                )
            ),
            augment=bool(entry.get("augment", data_defaults.get("augment", True))),
            normalization=entry.get(
                "normalization", data_defaults.get("normalization", "imagenet")
            ),
            drop_last=bool(
                entry.get("drop_last", data_defaults.get("drop_last", True))
            ),
        )
        specs.append(spec)

    if not specs:
        raise ValueError(
            "No datasets specified. Add entries under 'datasets' in the config or use --datasets."
        )
    return specs


class RunLogger:
    def __init__(self, run_dir: Path, wandb_cfg: Dict[str, any]):
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.metrics_file = self.metrics_path.open("a", encoding="utf-8")
        self.wandb_run = None
        self.enabled = bool(wandb_cfg.get("use_wandb", False)) and wandb is not None
        if self.enabled:
            self.wandb_run = self._init_wandb(wandb_cfg)

    def _init_wandb(self, cfg: Dict[str, any]):
        if wandb is None:
            return None
        try:
            return wandb.init(
                project=cfg.get("project"),
                entity=cfg.get("entity"),
                name=cfg.get("run_name"),
                group=cfg.get("group"),
                tags=cfg.get("tags"),
                notes=cfg.get("notes"),
                config=cfg.get("config"),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to init W&B: %s", exc)
            self.enabled = False
            return None

    def log(self, metrics: Dict[str, float], step: int | None = None) -> None:
        payload = {"time": time.time(), "step": step, "metrics": metrics}
        self.metrics_file.write(json.dumps(payload) + "\n")
        self.metrics_file.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def summary(self, metrics: Dict[str, float]) -> None:
        summary_path = self.run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        if self.wandb_run is not None:
            for key, value in metrics.items():
                self.wandb_run.summary[key] = value

    def close(self) -> None:
        if not self.metrics_file.closed:
            self.metrics_file.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def prepare_run_dirs(base_dir: Path, dataset_name: str, run_slug: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_slug = run_slug or _slugify(dataset_name) or "run"
    run_name = f"{safe_slug}-{timestamp}"
    run_dir = base_dir / dataset_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_device(device_cfg: str | None) -> torch.device:
    if device_cfg is None or device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    epochs: int,
    grad_accum: int,
    grad_clip: float | None,
    monitor_cfg: Dict[str, any],
    global_step: int,
    log_every: int,
    logger: RunLogger,
) -> Dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    total_loss = 0.0
    total_samples = 0
    running_top1 = 0.0
    running_top5 = 0.0
    epoch_start = time.time()

    progress = tqdm(loader, desc=f"Train {epoch + 1}/{epochs}")
    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=use_amp, dtype=amp_dtype, device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets) / grad_accum

        loss.backward()

        should_step = (step + 1) % grad_accum == 0
        if should_step:
            if monitor_cfg.get("grad_norm", True):
                gn = grad_norm(model.parameters())
            else:
                gn = None
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

            global_step += 1

            metrics = {
                "train/loss": float(loss.item() * grad_accum),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
            if monitor_cfg.get("grad_norm", True) and gn is not None:
                metrics["train/grad_norm"] = gn
            if monitor_cfg.get("param_norm", False):
                metrics["train/param_norm"] = param_norm(model.parameters())
            if monitor_cfg.get("memory", False) and torch.cuda.is_available():
                metrics["train/memory_gb"] = torch.cuda.memory_allocated() / 1e9
            if global_step % log_every == 0:
                logger.log(metrics, step=global_step)

        batch_loss = loss.item() * grad_accum * targets.size(0)
        total_loss += batch_loss
        total_samples += targets.size(0)
        running_top1 += accuracy(outputs.detach(), targets.detach()) * targets.size(0)
        running_top5 += top_k_accuracy(
            outputs.detach(), targets.detach(), 5
        ) * targets.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        avg_top1 = running_top1 / max(total_samples, 1)
        avg_top5 = running_top5 / max(total_samples, 1)
        progress.set_postfix(
            loss=f"{avg_loss:.4f}", top1=f"{avg_top1:.2f}", top5=f"{avg_top5:.2f}"
        )

    duration = time.time() - epoch_start
    return {
        "train/loss": total_loss / max(total_samples, 1),
        "train/top1": running_top1 / max(total_samples, 1),
        "train/top5": running_top5 / max(total_samples, 1),
        "train/samples": total_samples,
        "train/duration": duration,
        "train/throughput": total_samples / max(duration, 1e-9),
        "state/global_step": global_step,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    total_loss = 0.0
    total_samples = 0
    running_top1 = 0.0
    running_top5 = 0.0
    start = time.time()

    progress = tqdm(loader, desc="Eval", leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(enabled=use_amp, dtype=amp_dtype, device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)
        running_top1 += accuracy(outputs.detach(), targets.detach()) * targets.size(0)
        running_top5 += top_k_accuracy(
            outputs.detach(), targets.detach(), 5
        ) * targets.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        avg_top1 = running_top1 / max(total_samples, 1)
        avg_top5 = running_top5 / max(total_samples, 1)
        progress.set_postfix(
            loss=f"{avg_loss:.4f}", top1=f"{avg_top1:.2f}", top5=f"{avg_top5:.2f}"
        )

    duration = time.time() - start
    return {
        "val/loss": total_loss / max(total_samples, 1),
        "val/accuracy": running_top1 / max(total_samples, 1),
        "val/top5": running_top5 / max(total_samples, 1),
        "val/duration": duration,
        "val/throughput": total_samples / max(duration, 1e-9),
    }


def run_dataset(
    config: Dict[str, any], spec: DatasetSpec, device: torch.device
) -> Dict[str, float]:
    experiment_cfg = config.get("experiment", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    logging_cfg = config.get("logging", {})
    monitor_cfg = {"grad_norm": True, "param_norm": False, "memory": False}
    monitor_cfg.update(training_cfg.get("monitor", {}))

    output_root = Path(experiment_cfg.get("output_dir", "./runs"))
    base_label, run_slug = build_run_labels(
        experiment_cfg, model_cfg, logging_cfg, spec.name
    )
    run_dir = prepare_run_dirs(output_root, spec.name, run_slug)
    setup_logging(run_dir, level=getattr(logging, config.get("log_level", "INFO")))

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    wandb_cfg = deepcopy(logging_cfg)
    wandb_cfg.setdefault("config", config)
    wandb_cfg["run_name"] = run_slug
    existing_tags = list(wandb_cfg.get("tags", []) or [])
    additions = [spec.name]
    model_tag = model_cfg.get("name")
    if model_tag:
        additions.append(str(model_tag))
    if base_label and base_label not in additions:
        additions.append(base_label)
    # Preserve order while deduplicating
    seen = set()
    deduped_tags = []
    for tag in [*existing_tags, *additions]:
        if not tag:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        deduped_tags.append(tag)
    wandb_cfg["tags"] = deduped_tags

    run_logger = RunLogger(run_dir, wandb_cfg)
    summary: Dict[str, float] = {}

    try:
        train_loader, val_loader, dataset_info = create_dataloaders(spec)

        model_name = model_cfg.get("name", "dinov2_lora")
        model_params = deepcopy(model_cfg.get("params", {}))
        model_params.setdefault("num_classes", dataset_info.num_classes)
        model, model_meta = create_model(model_name, model_params, dataset_info)
        model.to(device)

        # Log every trainable tensor explicitly so it's clear what will update.
        trainable_tensors = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        total_trainable = sum(param.numel() for _, param in trainable_tensors)
        LOGGER.info(
            "Trainable tensors: %d (%.2f M parameters)",
            len(trainable_tensors),
            total_trainable / 1e6,
        )
        for name, param in trainable_tensors:
            LOGGER.info("  %s | shape=%s | params=%d", name, tuple(param.shape), param.numel())

        optimizer = create_optimizer(model, training_cfg.get("optimizer", {}))
        total_steps = len(train_loader) * max(int(training_cfg.get("epochs", 1)), 1)
        scheduler = create_scheduler(
            optimizer, training_cfg.get("scheduler", {}), total_steps
        )

        grad_accum = max(
            1,
            int(
                training_cfg.get("grad_accumulation", training_cfg.get("grad_accum", 1))
            ),
        )
        grad_clip = training_cfg.get("gradient_clip_norm")

        best_metric = None
        best_path = None
        global_step = 0

        epochs = int(training_cfg.get("epochs", 1))
        log_every = int(training_cfg.get("log_every", 10))
        eval_every = int(training_cfg.get("eval_every", 1))
        save_every = int(training_cfg.get("save_every", 1))

        for epoch in range(epochs):
            epoch_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                epochs,
                grad_accum,
                grad_clip,
                monitor_cfg,
                global_step,
                log_every,
                run_logger,
            )
            global_step = int(epoch_metrics["state/global_step"])

            if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
                val_metrics = evaluate(model, val_loader, device)
                run_logger.log(val_metrics, step=global_step)
            else:
                val_metrics = {}

            combined = {**epoch_metrics, **val_metrics, "epoch": epoch}
            run_logger.log(combined, step=global_step)

            metric_value = val_metrics.get("val/accuracy")
            checkpoints_dir = run_dir / "checkpoints"
            if metric_value is not None:
                is_best = best_metric is None or metric_value > best_metric
                if is_best:
                    best_metric = metric_value
                    best_path = checkpoints_dir / "best.pt"
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict()
                            if scheduler
                            else None,
                            "metrics": combined,
                            "dataset": dataset_info.__dict__,
                            "model": {
                                "name": model_name,
                                "params": model_params,
                                "meta": model_meta,
                            },
                            "config": config,
                        },
                        best_path,
                    )
            if (epoch + 1) % save_every == 0:
                ckpt_path = checkpoints_dir / f"epoch_{epoch + 1:03d}.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict()
                        if scheduler
                        else None,
                        "metrics": combined,
                        "dataset": dataset_info.__dict__,
                        "model": {
                            "name": model_name,
                            "params": model_params,
                            "meta": model_meta,
                        },
                        "config": config,
                    },
                    ckpt_path,
                )

        summary = {
            "best/val_accuracy": best_metric,
            "epochs": epochs,
            "global_step": global_step,
            "dataset/train_samples": dataset_info.train_samples,
            "dataset/val_samples": dataset_info.val_samples,
            "dataset/classes": dataset_info.num_classes,
        }
        if best_path is not None:
            summary["best_checkpoint"] = str(best_path)

        run_logger.summary(summary)
        return summary
    finally:
        run_logger.close()


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config, overrides=args.override)

    if args.no_wandb:
        config.setdefault("logging", {})["use_wandb"] = False

    config["log_level"] = args.log_level.upper()

    experiment_cfg = config.get("experiment", {})
    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    specs = build_dataset_specs(config, args.datasets)
    device = get_device(experiment_cfg.get("device"))
    LOGGER.info("Using device %s", device)

    results = {}
    for spec in specs:
        LOGGER.info("=== Dataset: %s ===", spec.name)
        results[spec.name] = run_dataset(config, spec, device)

    LOGGER.info("Training complete. Results: %s", results)


if __name__ == "__main__":
    main()
