"""Lightweight model registry."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from data import DatasetInfo
from .commutator_lora import CommutatorLoRA
from .bracket_adapter import BracketAdapter

_MODEL_BUILDERS: Dict[str, callable] = {}


def register(name: str):
    def decorator(fn):
        key = name.lower()
        if key in _MODEL_BUILDERS:
            raise KeyError(f"Model '{name}' already registered")
        _MODEL_BUILDERS[key] = fn
        return fn

    return decorator


def create(name: str, params: Dict[str, Any], dataset: DatasetInfo) -> Tuple[nn.Module, Dict[str, Any]]:
    key = name.lower()
    if key not in _MODEL_BUILDERS:
        raise KeyError(f"Unknown model '{name}'. Registered models: {list(_MODEL_BUILDERS.keys())}")
    builder = _MODEL_BUILDERS[key]
    return builder(params, dataset)


@register("dinov2_lora")
def _build_dinov2(params: Dict[str, Any], dataset: DatasetInfo):
    from .dinov2_lora import build_model

    model_name = params.get("model_name", "facebook/dinov2-small")
    model, extras = build_model(model_name, dataset.num_classes, params)
    return model, extras


@register("dinov2_commutator")
def _build_dinov2_commutator(params: Dict[str, Any], dataset: DatasetInfo):
    from .dinov2_commutator import build_model

    model_name = params.get("model_name", "facebook/dinov2-small")
    model, extras = build_model(model_name, dataset.num_classes, params)
    return model, extras


@register("dinov2_relora")
def _build_dinov2_relora(params: Dict[str, Any], dataset: DatasetInfo):
    from .dinov2_relora import build_model

    model_name = params.get("model_name", "facebook/dinov2-small")
    model, extras = build_model(model_name, dataset.num_classes, params)
    return model, extras


@register("dinov2_bisided")
def _build_dinov2_bisided(params: Dict[str, Any], dataset: DatasetInfo):
    from .dinov2_bracket import build_model

    model_name = params.get("model_name", "facebook/dinov2-small")
    model, extras = build_model(model_name, dataset.num_classes, params)
    return model, extras


@register("vit_base_lora")
def _build_vit_base(params: Dict[str, Any], dataset: DatasetInfo):
    from .vit_base_lora import build_model

    model_name = params.get("model_name", "google/vit-base-patch16-224-in21k")
    model, extras = build_model(model_name, dataset.num_classes, params)
    return model, extras


@register("vit_base_commutator")
def _build_vit_base_commutator(params: Dict[str, Any], dataset: DatasetInfo):
    from .vit_base_commutator import build_model

    model_name = params.get("model_name", "google/vit-base-patch16-224-in21k")
    model, extras = build_model(model_name, dataset.num_classes, params)
    return model, extras


__all__ = ["create", "register", "CommutatorLoRA", "BracketAdapter"]
