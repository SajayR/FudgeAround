"""Lightweight training helpers."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR

LOGGER = logging.getLogger(__name__)


def setup_logging(output_dir: Path, level: int = logging.INFO) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler()]
    log_file = output_dir / "train.log"
    handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    LOGGER.info("Seed set to %d", seed)


def create_optimizer(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    param_groups: Optional[List[Dict[str, Any]]] = None,
) -> torch.optim.Optimizer:
    prepared_groups: List[Dict[str, Any]] = []
    if param_groups:
        for group in param_groups:
            group_copy = dict(group)
            group_params = [p for p in group_copy.get("params", []) if p.requires_grad]
            if not group_params:
                continue
            group_copy["params"] = group_params
            prepared_groups.append(group_copy)
    if prepared_groups:
        params = prepared_groups
        num_trainable = sum(p.numel() for group in prepared_groups for p in group["params"])
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        num_trainable = sum(p.numel() for p in params)
    name = cfg.get("type", "adamw").lower()
    lr = float(cfg.get("lr", cfg.get("learning_rate", 3e-4)))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    if name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        eps = float(cfg.get("eps", 1e-8))
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif name == "adam":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        eps = float(cfg.get("eps", 1e-8))
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        optimizer = SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    LOGGER.info(
        "Optimizer: %s lr=%.2e weight_decay=%.2e params=%d",
        name,
        lr,
        weight_decay,
        num_trainable,
    )
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], total_steps: int) -> Optional[LambdaLR]:
    name = (cfg or {}).get("type", "cosine").lower()
    if name in {"none", "constant"} or total_steps <= 0:
        return None

    warmup_fraction = float(cfg.get("warmup", cfg.get("warmup_epochs", 0.0)))
    if warmup_fraction < 1.0:
        warmup_steps = int(total_steps * warmup_fraction)
    else:
        warmup_steps = int(warmup_fraction)

    min_lr_mult = float(cfg.get("min_lr_mult", cfg.get("min_lr_ratio", 0.01)))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step / float(warmup_steps), 1e-6)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if name == "linear":
            return (1.0 - progress) * (1.0 - min_lr_mult) + min_lr_mult
        return min_lr_mult + 0.5 * (1.0 - min_lr_mult) * (1.0 + math.cos(progress * math.pi))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    LOGGER.info("Scheduler: %s warmup=%d steps total=%d", name, warmup_steps, total_steps)
    return scheduler


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    LOGGER.info("Saved checkpoint to %s", path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    LOGGER.info("Loaded checkpoint from %s", path)
    return checkpoint


def grad_norm(parameters, norm_type: float = 2.0) -> float:
    norms = []
    for p in parameters:
        if p.grad is not None:
            norms.append(p.grad.detach().data.norm(norm_type))
    if not norms:
        return 0.0
    total = torch.norm(torch.stack(norms), norm_type)
    return float(total.item())


def param_norm(parameters, norm_type: float = 2.0) -> float:
    values = [p.detach().data.norm(norm_type) for p in parameters if p.requires_grad]
    if not values:
        return 0.0
    total = torch.norm(torch.stack(values), norm_type)
    return float(total.item())


__all__ = [
    "setup_logging",
    "set_seed",
    "create_optimizer",
    "create_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "grad_norm",
    "param_norm",
]
