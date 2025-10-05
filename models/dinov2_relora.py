"""Dinov2 backbone wrapped with ReLoRA adapters."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

import torch.nn as nn
from transformers import Dinov2Model

from .relora import ReLoRAConfig, ReLoRAController, replace_linears_with_relora

LOGGER = logging.getLogger(__name__)


class DinoV2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, pixel_values):
        features = self.backbone(pixel_values=pixel_values)
        cls_token = features.last_hidden_state[:, 0]
        return self.head(cls_token)


def _collect_named_parameters(module: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    params: List[Tuple[str, nn.Parameter]] = []
    for name, param in module.named_parameters():
        params.append((name, param))
    return params


def _split_relora_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    base_params: List[nn.Parameter] = []
    adapter_params: List[nn.Parameter] = []
    for _, param in _collect_named_parameters(model):
        if not param.requires_grad:
            continue
        if getattr(param, "_is_relora_adapter", False):
            adapter_params.append(param)
        else:
            base_params.append(param)
    return base_params, adapter_params


def _build_relora_config(params: Dict[str, Any]) -> Tuple[ReLoRAConfig, Iterable[str]]:
    relora_params = params.get("relora", {}) or {}
    target_modules = relora_params.get(
        "target_modules",
        ["query", "key", "value", "dense", "fc1", "fc2"],
    )
    if not isinstance(target_modules, (list, tuple)):
        target_modules = list(target_modules)

    rank = int(relora_params.get("r", relora_params.get("rank", 16)))
    alpha = float(relora_params.get("alpha", relora_params.get("lora_alpha", 32.0)))
    dropout = float(relora_params.get("dropout", relora_params.get("lora_dropout", 0.0)))
    merge_interval = int(
        relora_params.get(
            "merge_interval",
            relora_params.get("merge_steps", relora_params.get("q", 200))),
        )
    warm_start_steps = int(relora_params.get("warm_start_steps", relora_params.get("warm_start", 0)))
    adapter_warmup_steps = int(
        relora_params.get(
            "adapter_warmup_steps",
            relora_params.get("warmup_steps", relora_params.get("adapter_warmup", 0))),
        )
    freeze_base = bool(relora_params.get("freeze_base", True))
    prune_b_state = bool(
        relora_params.get(
            "prune_b_state",
            relora_params.get("prune_moment_b", relora_params.get("prune_b", True)),
        )
    )

    config = ReLoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        merge_interval=merge_interval,
        warm_start_steps=warm_start_steps,
        adapter_warmup_steps=adapter_warmup_steps,
        freeze_base=freeze_base,
        prune_b_state=prune_b_state,
    )
    return config, target_modules


def build_model(model_name: str, num_classes: int, params: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    dropout = float(params.get("dropout", 0.0))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", False))

    LOGGER.info("Loading Dinov2 backbone '%s'", model_name)
    backbone = Dinov2Model.from_pretrained(model_name)

    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing")

    relora_config, target_modules = _build_relora_config(params)

    controller: ReLoRAController | None = None
    replaced_modules: List[str] = []
    if relora_config.rank > 0:
        controller = replace_linears_with_relora(backbone, target_modules, relora_config)
        replaced_modules = controller.module_names or []
        if not replaced_modules:
            LOGGER.warning("ReLoRA found no target linear modules to wrap")
    else:
        LOGGER.info("ReLoRA disabled (rank=0)")

    model = DinoV2Classifier(backbone, backbone.config.hidden_size, num_classes, dropout)
    if controller is not None:
        model.relora_controller = controller  # type: ignore[attr-defined]

    base_params, adapter_params = _split_relora_params(model)
    optimizer_groups: List[Dict[str, Any]] = []
    if base_params:
        optimizer_groups.append({"params": base_params})
    if adapter_params:
        adapter_group: Dict[str, Any] = {
            "params": adapter_params,
            "relora_adapter_group": True,
        }
        relora_optimizer_cfg = params.get("relora_optimizer", {}) or {}
        if "lr" in relora_optimizer_cfg:
            adapter_group["lr"] = float(relora_optimizer_cfg["lr"])
        if "weight_decay" in relora_optimizer_cfg:
            adapter_group["weight_decay"] = float(relora_optimizer_cfg["weight_decay"])
        optimizer_groups.append(adapter_group)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    extras: Dict[str, Any] = {
        "backbone": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params else 0.0,
        "relora": {
            "rank": relora_config.rank,
            "alpha": relora_config.alpha,
            "dropout": relora_config.dropout,
            "merge_interval": relora_config.merge_interval,
            "warm_start_steps": relora_config.warm_start_steps,
            "adapter_warmup_steps": relora_config.adapter_warmup_steps,
            "freeze_base": relora_config.freeze_base,
            "prune_b_state": relora_config.prune_b_state,
            "target_modules": list(target_modules),
            "replaced_modules": replaced_modules,
        },
    }
    if controller is not None:
        extras["relora"]["wrapped_layers"] = len(controller.modules)
        extras["optimizer_param_groups"] = optimizer_groups
    else:
        extras["optimizer_param_groups"] = optimizer_groups

    return model, extras


__all__ = ["build_model"]
