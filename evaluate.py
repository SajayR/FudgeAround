"""Evaluate a trained checkpoint on the chosen dataset."""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.amp import autocast
from tqdm import tqdm

from configs import load_config
from data import DatasetSpec, create_dataloaders
from models import create as create_model
from utils.metrics import accuracy, top_k_accuracy
from utils.training_utils import set_seed

LOGGER = logging.getLogger("evaluate")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument("--config", required=True, help="Path or name of the config used during training")
    parser.add_argument("--dataset", required=True, help="Dataset name to evaluate")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--override", nargs="*", help="Config overrides key=value")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def build_spec(config: Dict[str, any], dataset_name: str) -> DatasetSpec:
    data_defaults = config.get("data", {})
    entries = config.get("datasets", [])
    target = None
    for entry in entries:
        if isinstance(entry, str) and entry == dataset_name:
            target = {"name": entry}
            break
        if isinstance(entry, dict) and entry.get("name") == dataset_name:
            target = dict(entry)
            break
    if target is None:
        target = {"name": dataset_name}

    return DatasetSpec(
        name=dataset_name,
        root=target.get("root", data_defaults.get("root", "/speedy/datasets")),
        train_split=target.get("train_split", data_defaults.get("train_split", "train")),
        val_split=target.get("val_split", data_defaults.get("val_split")),
        image_size=int(target.get("image_size", data_defaults.get("image_size", 224))),
        batch_size=int(target.get("batch_size", data_defaults.get("batch_size", 64))),
        num_workers=int(target.get("num_workers", data_defaults.get("num_workers", 8))),
        pin_memory=bool(target.get("pin_memory", data_defaults.get("pin_memory", True))),
        persistent_workers=bool(target.get("persistent_workers", data_defaults.get("persistent_workers", True))),
        augment=bool(target.get("augment", data_defaults.get("augment", True))),
        normalization=target.get("normalization", data_defaults.get("normalization", "imagenet")),
        drop_last=bool(target.get("drop_last", data_defaults.get("drop_last", True))),
    )


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def evaluate(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    total_loss = 0.0
    total_samples = 0
    running_top1 = 0.0
    running_top5 = 0.0
    progress = tqdm(loader, desc="Eval", leave=False)

    with torch.no_grad():
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(enabled=use_amp, dtype=amp_dtype, device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            running_top1 += accuracy(outputs.detach(), targets.detach()) * targets.size(0)
            running_top5 += top_k_accuracy(outputs.detach(), targets.detach(), 5) * targets.size(0)

            avg_loss = total_loss / max(total_samples, 1)
            avg_top1 = running_top1 / max(total_samples, 1)
            avg_top5 = running_top5 / max(total_samples, 1)
            progress.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_top1:.2f}", top5=f"{avg_top5:.2f}")

    return {
        "val/loss": total_loss / max(total_samples, 1),
        "val/accuracy": running_top1 / max(total_samples, 1),
        "val/top5": running_top5 / max(total_samples, 1),
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    config = load_config(args.config, overrides=args.override)
    experiment_cfg = config.get("experiment", {})
    set_seed(int(experiment_cfg.get("seed", 42)))

    spec = build_spec(config, args.dataset)
    _, val_loader, dataset_info = create_dataloaders(spec)

    device = get_device(args.device if args.device else experiment_cfg.get("device", "auto"))
    LOGGER.info("Using device %s", device)

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "dinov2_lora")
    model_params = deepcopy(model_cfg.get("params", {}))
    model_params.setdefault("num_classes", dataset_info.num_classes)
    model, _ = create_model(model_name, model_params, dataset_info)
    model.to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    LOGGER.info("Loaded checkpoint from %s (epoch=%s)", checkpoint_path, state.get("epoch"))

    metrics = evaluate(model, val_loader, device)
    LOGGER.info("Metrics: %s", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
