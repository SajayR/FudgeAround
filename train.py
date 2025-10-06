"""Simple multi-dataset training script with LoRA-ready Dinov2 backbone."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
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
from utils.bs_relora import BSReLoRAConfig, BSReLoRAController
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
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    epochs: int,
    grad_accum: int,
    precision_cfg: Dict[str, any],
    monitor_cfg: Dict[str, any],
    global_step: int,
    log_every: int,
    logger: RunLogger,
) -> Dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    use_amp = precision_cfg.get("enabled", True) and device.type == "cuda"
    dtype = precision_cfg.get("dtype", "bf16").lower()
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

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

        scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()

        should_step = (step + 1) % grad_accum == 0
        if should_step:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if monitor_cfg.get("grad_norm", True):
                gn = grad_norm(model.parameters())
            else:
                gn = None
            clip_val = precision_cfg.get("grad_clip", None)
            if clip_val:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

            global_step += 1

            metrics = {
                "train/loss_step": float(loss.item() * grad_accum),
                "train/lr": optimizer.param_groups[0]["lr"],
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
def evaluate(
    model: nn.Module, loader, device: torch.device, precision_cfg: Dict[str, any]
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    use_amp = precision_cfg.get("enabled", True) and device.type == "cuda"
    dtype = precision_cfg.get("dtype", "bf16").lower()
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

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


def _infinite_loader(loader):
    """Yield batches from ``loader`` indefinitely without caching the dataset."""

    while True:
        for batch in loader:
            yield batch


def _select_bs_relora_layers(model: nn.Module, keywords: Iterable[str]) -> List[tuple[str, nn.Linear]]:
    lowered = [kw.lower() for kw in (keywords or []) if kw]
    matches: List[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not lowered or any(key in name.lower() for key in lowered):
                matches.append((name, module))
    return matches


def _create_low_rank_optimizer(params: List[nn.Parameter], cfg: Dict[str, any]):
    parameters = [p for p in params if p.requires_grad]
    if not parameters:
        return None
    name = (cfg or {}).get("type", "adamw").lower()
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    eps = float(cfg.get("eps", 1e-8))

    if name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported low-rank optimizer: {name}")
    return optimizer


def _run_bs_relora_phase(
    phase_name: str,
    num_steps: int,
    model: nn.Module,
    controller: BSReLoRAController,
    batch_iter,
    device: torch.device,
    base_optimizer: torch.optim.Optimizer,
    low_rank_optimizer: torch.optim.Optimizer | None,
    scheduler,
    scaler: GradScaler,
    precision_cfg: Dict[str, any],
    monitor_cfg: Dict[str, any],
    grad_accum: int,
    log_every: int,
    run_logger: RunLogger,
    global_step: int,
    capture_probe: bool,
) -> tuple[Dict[str, float], int]:
    """Run one BS-ReLoRA phase (probe or low-rank)."""

    model.train()
    criterion = nn.CrossEntropyLoss()

    use_amp = precision_cfg.get("enabled", True) and device.type == "cuda"
    dtype = precision_cfg.get("dtype", "bf16").lower()
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    total_loss = 0.0
    total_samples = 0
    running_top1 = 0.0
    running_top5 = 0.0
    start = time.time()

    grad_clip = precision_cfg.get("grad_clip", None)

    total_micro_steps = num_steps * max(grad_accum, 1)
    progress = tqdm(range(total_micro_steps), desc=f"{phase_name.capitalize()} phase", leave=False)

    for micro_idx in progress:
        images, targets = next(batch_iter)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=use_amp, dtype=amp_dtype, device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets) / grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = ((micro_idx + 1) % grad_accum == 0)
        samples = targets.size(0)
        batch_loss = loss.item() * grad_accum * samples
        total_loss += batch_loss
        total_samples += samples
        running_top1 += accuracy(outputs.detach(), targets.detach()) * samples
        running_top5 += top_k_accuracy(outputs.detach(), targets.detach(), 5) * samples

        avg_loss = total_loss / max(total_samples, 1)
        avg_top1 = running_top1 / max(total_samples, 1)
        avg_top5 = running_top5 / max(total_samples, 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_top1:.2f}", top5=f"{avg_top5:.2f}")

        if not should_step:
            continue

        if capture_probe:
            controller.prepare_step()

        if scaler.is_enabled():
            scaler.unscale_(base_optimizer)
            if low_rank_optimizer is not None:
                scaler.unscale_(low_rank_optimizer)

        grad_norm_value = None
        if monitor_cfg.get("grad_norm", True):
            grad_norm_value = grad_norm(model.parameters())

        adapter_grad_norm = None
        adapter_param_norm = None
        if not capture_probe:
            adapter_grad_norm = controller.adapter_grad_norm()
            adapter_param_norm = controller.adapter_param_norm()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if scaler.is_enabled():
            if capture_probe:
                scaler.step(base_optimizer)
            if low_rank_optimizer is not None:
                scaler.step(low_rank_optimizer)
            scaler.update()
        else:
            if capture_probe:
                base_optimizer.step()
            if low_rank_optimizer is not None:
                low_rank_optimizer.step()

        if capture_probe:
            controller.finish_step(capture=True)

        base_optimizer.zero_grad(set_to_none=True)
        if low_rank_optimizer is not None:
            low_rank_optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and capture_probe:
            scheduler.step()

        global_step += 1

        metrics = {
            f"{phase_name}/loss_step": float(batch_loss / max(samples, 1)),
        }
        if capture_probe:
            metrics["train/lr"] = base_optimizer.param_groups[0]["lr"]
        elif low_rank_optimizer is not None:
            metrics["lowrank/lr"] = low_rank_optimizer.param_groups[0]["lr"]
        if grad_norm_value is not None:
            metrics[f"{phase_name}/grad_norm"] = grad_norm_value
        if monitor_cfg.get("param_norm", False):
            metrics[f"{phase_name}/param_norm"] = param_norm(model.parameters())
        if adapter_param_norm is not None:
            metrics["lowrank/adapter_param_norm"] = adapter_param_norm
        if adapter_grad_norm is not None:
            metrics["lowrank/adapter_grad_norm"] = adapter_grad_norm
        if monitor_cfg.get("memory", False) and torch.cuda.is_available():
            metrics[f"{phase_name}/memory_gb"] = torch.cuda.memory_allocated() / 1e9
        if log_every and global_step % log_every == 0:
            run_logger.log(metrics, step=global_step)

    duration = time.time() - start
    phase_metrics = {
        f"{phase_name}/loss": total_loss / max(total_samples, 1),
        f"{phase_name}/top1": running_top1 / max(total_samples, 1),
        f"{phase_name}/top5": running_top5 / max(total_samples, 1),
        f"{phase_name}/samples": total_samples,
        f"{phase_name}/duration": duration,
        f"{phase_name}/throughput": total_samples / max(duration, 1e-9),
        f"{phase_name}/optimizer_steps": num_steps,
    }
    return phase_metrics, global_step


def _train_with_bs_relora(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    precision_cfg: Dict[str, any],
    monitor_cfg: Dict[str, any],
    training_cfg: Dict[str, any],
    bs_cfg: Dict[str, any],
    model_name: str,
    model_params: Dict[str, any],
    model_meta: Dict[str, any],
    dataset_info,
    run_logger: RunLogger,
    run_dir: Path,
    config: Dict[str, any],
) -> Dict[str, float]:
    LOGGER.info("Running BS-ReLoRA training")
    grad_accum = max(
        1,
        int(
            training_cfg.get("grad_accumulation", training_cfg.get("grad_accum", 1))
        ),
    )
    log_every = int(training_cfg.get("log_every", 10))
    cycles = max(1, int(bs_cfg.get("cycles", 1)))
    probe_steps = max(1, int(bs_cfg.get("probe_steps", 32)))
    low_rank_steps = max(1, int(bs_cfg.get("low_rank_steps", 128)))
    eval_every = int(bs_cfg.get("eval_every", training_cfg.get("eval_every", 1)))
    save_every = int(bs_cfg.get("save_every", training_cfg.get("save_every", 1)))

    lora_defaults = model_params.get("lora", {}) if isinstance(model_params.get("lora", {}), dict) else {}
    target_modules = bs_cfg.get("target_modules") or lora_defaults.get("target_modules") or []
    layer_selector = _select_bs_relora_layers(model, target_modules)
    if not layer_selector:
        raise RuntimeError(
            "BS-ReLoRA is enabled but no target nn.Linear modules were located. "
            "Provide `training.bs_relora.target_modules` or ensure the model exposes linear layers."
        )

    rank_default = lora_defaults.get("r") or lora_defaults.get("rank") or 16
    rank = int(bs_cfg.get("rank", bs_cfg.get("r", rank_default)))
    oversample = int(bs_cfg.get("oversample", bs_cfg.get("p", 4)))
    power_iterations = int(bs_cfg.get("power_iterations", 1))
    memory_cap = int(
        bs_cfg.get(
            "memory_rank_cap",
            bs_cfg.get("memory_max_rank", rank * int(bs_cfg.get("memory_cycles", 4))),
        )
    )
    lambda_u = float(bs_cfg.get("lambda_u", 0.5))
    lambda_v = float(bs_cfg.get("lambda_v", 0.5))
    gamma = float(bs_cfg.get("gamma", 0.5))
    alpha_m_limits = tuple(bs_cfg.get("alpha_m_limits", (0.05, 0.5)))
    alpha_v_limits = tuple(bs_cfg.get("alpha_v_limits", (0.5, 0.9)))
    if len(alpha_m_limits) != 2 or len(alpha_v_limits) != 2:
        raise ValueError("alpha*_limits must be sequences of length 2")
    rho_eps = float(bs_cfg.get("rho_eps", 1e-12))

    relora_config = BSReLoRAConfig(
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        probe_steps=probe_steps,
        low_rank_steps=low_rank_steps,
        memory_rank_cap=memory_cap,
        lambda_u=lambda_u,
        lambda_v=lambda_v,
        gamma=gamma,
        alpha_m_limits=(float(alpha_m_limits[0]), float(alpha_m_limits[1])),
        alpha_v_limits=(float(alpha_v_limits[0]), float(alpha_v_limits[1])),
        rho_eps=rho_eps,
    )

    controller = BSReLoRAController(model, layer_selector, relora_config)
    LOGGER.info(
        "BS-ReLoRA targeting %d linear layers: %s",
        len(controller.layer_names()),
        controller.layer_names(),
    )

    batch_iter = _infinite_loader(train_loader)
    low_rank_opt_cfg = bs_cfg.get("low_rank_optimizer", {})

    best_metric = None
    best_path = None
    global_step = 0

    checkpoints_dir = run_dir / "checkpoints"

    for cycle in range(cycles):
        LOGGER.info("Cycle %d/%d - probe phase", cycle + 1, cycles)
        controller.start_probe()
        probe_metrics, global_step = _run_bs_relora_phase(
            phase_name="probe",
            num_steps=probe_steps,
            model=model,
            controller=controller,
            batch_iter=batch_iter,
            device=device,
            base_optimizer=optimizer,
            low_rank_optimizer=None,
            scheduler=scheduler,
            scaler=scaler,
            precision_cfg=precision_cfg,
            monitor_cfg=monitor_cfg,
            grad_accum=grad_accum,
            log_every=log_every,
            run_logger=run_logger,
            global_step=global_step,
            capture_probe=True,
        )
        controller.finish_probe()
        controller.extract_subspaces()
        controller.activate_low_rank()
        controller.set_base_trainable(False)

        low_rank_optimizer = _create_low_rank_optimizer(controller.adapter_parameters(), low_rank_opt_cfg)

        LOGGER.info("Cycle %d/%d - low-rank phase", cycle + 1, cycles)
        low_metrics, global_step = _run_bs_relora_phase(
            phase_name="lowrank",
            num_steps=low_rank_steps,
            model=model,
            controller=controller,
            batch_iter=batch_iter,
            device=device,
            base_optimizer=optimizer,
            low_rank_optimizer=low_rank_optimizer,
            scheduler=scheduler,
            scaler=scaler,
            precision_cfg=precision_cfg,
            monitor_cfg=monitor_cfg,
            grad_accum=grad_accum,
            log_every=log_every,
            run_logger=run_logger,
            global_step=global_step,
            capture_probe=False,
        )

        merge_metrics = controller.merge_and_damp(optimizer)
        controller.restore_original_requires_grad()
        controller.deactivate_low_rank()

        cycle_metrics = {**probe_metrics, **low_metrics, **merge_metrics, "cycle": cycle}
        run_logger.log(cycle_metrics, step=global_step)

        should_eval = (cycle + 1) % max(eval_every, 1) == 0 or (cycle + 1) == cycles
        val_metrics: Dict[str, float] = {}
        if should_eval:
            LOGGER.info("Evaluating after cycle %d", cycle + 1)
            val_metrics = evaluate(model, val_loader, device, precision_cfg)
            run_logger.log(val_metrics, step=global_step)
            metric_value = val_metrics.get("val/accuracy")
            if metric_value is not None:
                is_best = best_metric is None or metric_value > best_metric
                if is_best:
                    best_metric = metric_value
                    best_path = checkpoints_dir / "best.pt"
                    save_checkpoint(
                        {
                            "cycle": cycle,
                            "global_step": global_step,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict() if scheduler else None,
                            "metrics": {**cycle_metrics, **val_metrics},
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

        if (cycle + 1) % max(save_every, 1) == 0:
            ckpt_path = checkpoints_dir / f"cycle_{cycle + 1:03d}.pt"
            save_checkpoint(
                {
                    "cycle": cycle,
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "metrics": {**cycle_metrics, **val_metrics},
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
        "cycles": cycles,
        "probe_steps": probe_steps,
        "low_rank_steps": low_rank_steps,
        "global_step": global_step,
        "dataset/train_samples": dataset_info.train_samples,
        "dataset/val_samples": dataset_info.val_samples,
        "dataset/classes": dataset_info.num_classes,
    }
    if best_path is not None:
        summary["best_checkpoint"] = str(best_path)
    return summary


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

        optimizer = create_optimizer(model, training_cfg.get("optimizer", {}))
        bs_cfg = training_cfg.get("bs_relora", {}) or {}
        bs_enabled = bool(bs_cfg.get("enabled", False))
        if bs_enabled:
            probe_steps_conf = max(1, int(bs_cfg.get("probe_steps", 32)))
            low_rank_steps_conf = max(1, int(bs_cfg.get("low_rank_steps", 128)))
            cycles_conf = max(1, int(bs_cfg.get("cycles", 1)))
            total_steps = max(1, cycles_conf * (probe_steps_conf + low_rank_steps_conf))
        else:
            total_steps = len(train_loader) * max(int(training_cfg.get("epochs", 1)), 1)

        scheduler = create_scheduler(
            optimizer, training_cfg.get("scheduler", {}), total_steps
        )

        precision_cfg = training_cfg.get(
            "precision", {"enabled": True, "dtype": "bf16", "grad_scaler": False}
        )
        precision_cfg.setdefault("grad_clip", training_cfg.get("gradient_clip_norm"))
        grad_accum = max(
            1,
            int(
                training_cfg.get("grad_accumulation", training_cfg.get("grad_accum", 1))
            ),
        )
        scaler = GradScaler(
            enabled=precision_cfg.get("enabled", True)
            and device.type == "cuda"
            and precision_cfg.get("dtype", "bf16").lower() == "fp16"
            and precision_cfg.get("grad_scaler", True)
        )

        if bs_enabled:
            summary = _train_with_bs_relora(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                precision_cfg=precision_cfg,
                monitor_cfg=monitor_cfg,
                training_cfg=training_cfg,
                bs_cfg=bs_cfg,
                model_name=model_name,
                model_params=model_params,
                model_meta=model_meta,
                dataset_info=dataset_info,
                run_logger=run_logger,
                run_dir=run_dir,
                config=config,
            )
            run_logger.summary(summary)
            return summary

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
                scaler,
                device,
                epoch,
                epochs,
                grad_accum,
                precision_cfg,
                monitor_cfg,
                global_step,
                log_every,
                run_logger,
            )
            global_step = int(epoch_metrics["state/global_step"])

            if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
                val_metrics = evaluate(model, val_loader, device, precision_cfg)
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
