"""Dinov2 builder with TwinLoRA adapters on MLP blocks."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from transformers import Dinov2Model

from .twinlora import TwinLoRAConfig, build_twinlora_wrappers

LOGGER = logging.getLogger(__name__)


class DinoV2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=pixel_values)
        cls_token = features.last_hidden_state[:, 0]
        return self.head(cls_token)


def _freeze_module(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def _build_twinlora_config(params: Dict[str, Any]) -> TwinLoRAConfig:
    cfg = TwinLoRAConfig(
        rank=int(params.get("rank", params.get("r", 16))),
        shared_rank=int(params.get("shared_rank", params.get("s", 8))),
        alpha=float(params.get("alpha", params.get("lora_alpha", 16.0))),
        twin_alpha=float(params.get("twin_alpha", params.get("shared_alpha", 16.0))),
        dropout=float(params.get("dropout", 0.0)),
        twin_dropout=float(params.get("twin_dropout", params.get("shared_dropout", 0.0))),
        init=str(params.get("init", "kaiming")),
    )
    cfg.validate()
    return cfg


def _resolve_layers(total_layers: int, selection: Optional[Iterable[int]]) -> List[int]:
    if selection is None:
        return list(range(total_layers))
    resolved = []
    for item in selection:
        if not 0 <= item < total_layers:
            raise IndexError(f"TwinLoRA layer index {item} out of range (total_layers={total_layers})")
        resolved.append(int(item))
    return sorted(set(resolved))


def _apply_twinlora(backbone: Dinov2Model, cfg: TwinLoRAConfig, layers: Optional[Iterable[int]] = None) -> int:
    encoder_layers = backbone.encoder.layer
    target_indices = _resolve_layers(len(encoder_layers), layers)
    applied = 0

    for idx, layer in enumerate(encoder_layers):
        if idx not in target_indices:
            continue
        mlp = layer.mlp
        down_linear = mlp.fc1
        up_linear = mlp.fc2
        down_wrapper, up_wrapper, adapter = build_twinlora_wrappers(
            down_linear,
            up_linear,
            cfg,
            dtype=down_linear.weight.dtype,
            device=down_linear.weight.device,
        )
        mlp.fc1 = down_wrapper
        mlp.fc2 = up_wrapper
        mlp.twinlora_adapter = adapter  # register as submodule
        applied += 1

    return applied


def build_model(model_name: str, num_classes: int, params: Dict[str, Any]) -> tuple[nn.Module, Dict[str, Any]]:
    dropout = float(params.get("dropout", 0.1))
    freeze_backbone = bool(params.get("freeze_backbone", False))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", False))

    LOGGER.info("Loading Dinov2 backbone '%s'", model_name)
    backbone = Dinov2Model.from_pretrained(model_name)

    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing")

    if freeze_backbone:
        _freeze_module(backbone, False)

    twin_params = params.get("twinlora", {}) or {}
    twin_enabled = bool(twin_params.get("enabled", True))
    twin_config = None
    twin_layers = None
    applied_layers = 0

    if twin_enabled:
        twin_config = _build_twinlora_config(twin_params)
        twin_layers = twin_params.get("layers")
        applied_layers = _apply_twinlora(backbone, twin_config, twin_layers)
        LOGGER.info("Applied TwinLoRA to %d transformer blocks", applied_layers)

    model = DinoV2Classifier(backbone, backbone.config.hidden_size, num_classes, dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    extras = {
        "backbone": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params else 0.0,
        "twinlora": {
            "enabled": twin_enabled,
            "config": asdict(twin_config) if twin_config else None,
            "layers": _resolve_layers(len(backbone.encoder.layer), twin_layers) if twin_enabled else [],
            "applied_layers": applied_layers,
        },
    }

    return model, extras


__all__ = ["build_model"]
