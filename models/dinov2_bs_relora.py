"""Dinov2 classifier tailored for BS-ReLoRA training."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch.nn as nn
from transformers import Dinov2Model

from .dinov2_lora import DinoV2Classifier, _freeze_module

LOGGER = logging.getLogger(__name__)


def build_model(
    model_name: str,
    num_classes: int,
    params: Dict[str, Any],
) -> tuple[nn.Module, Dict[str, Any]]:
    """Construct a Dinov2 classifier without PEFT LoRA adapters.

    The BS-ReLoRA controller will attach its own low-rank adapters at runtime,
    so we keep the backbone untouched aside from optional freezing or gradient
    checkpoint toggles.
    """

    dropout = float(params.get("dropout", 0.1))
    freeze_backbone = bool(params.get("freeze_backbone", False))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", False))

    LOGGER.info("Loading Dinov2 backbone '%s' for BS-ReLoRA", model_name)
    backbone = Dinov2Model.from_pretrained(model_name)

    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing")

    if freeze_backbone:
        _freeze_module(backbone, False)

    model = DinoV2Classifier(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        num_classes=num_classes,
        dropout=dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    extras: Dict[str, Any] = {
        "backbone": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params else 0.0,
        "bs_relora_ready": True,
    }

    return model, extras


__all__ = ["build_model"]

