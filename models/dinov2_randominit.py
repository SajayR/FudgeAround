"""Dinov2 backbone with random initialization."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from transformers import Dinov2Config, Dinov2Model

LOGGER = logging.getLogger(__name__)


class DinoV2Classifier(nn.Module):
    """Wrap Dinov2 backbone with a simple linear head."""

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


def build_model(model_name: str, num_classes: int, params: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    dropout = float(params.get("dropout", 0.1))
    freeze_backbone = bool(params.get("freeze_backbone", False))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", False))

    LOGGER.info("Loading Dinov2 config '%s' with random initialization", model_name)
    config = Dinov2Config.from_pretrained(model_name)
    backbone = Dinov2Model(config)

    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing")

    if freeze_backbone:
        _freeze_module(backbone, False)

    model = DinoV2Classifier(backbone, config.hidden_size, num_classes, dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    extras = {
        "backbone": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params else 0.0,
        "random_init": True,
    }

    return model, extras


__all__ = ["build_model"]
