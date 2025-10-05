"""ReLoRA linear module and training controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
from torch import Tensor
import torch.nn as nn


def _kaiming_init(matrix: Tensor) -> None:
    """Apply Kaiming uniform init in-place, matching PyTorch defaults."""

    if matrix.numel() == 0:
        return
    nn.init.kaiming_uniform_(matrix, a=math.sqrt(5))


class ReLoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` with low-rank adapters for ReLoRA training.

    Bias parameters remain trainable to mirror common LoRA practice; only the
    base weight is frozen once adapters activate.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float = 32.0,
        scaling: Optional[float] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be an nn.Linear instance")
        if rank < 0:
            raise ValueError("rank must be non-negative")

        self.base_linear = base_linear
        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.rank = int(rank)

        if scaling is not None and alpha is not None:
            # scaling overrides alpha; other code may still want alpha for logging
            self.scaling = float(scaling)
        else:
            self.scaling = float(alpha) / float(self.rank) if self.rank > 0 else 0.0
        self.alpha = float(alpha)

        self.adapter_dropout = nn.Dropout(dropout) if dropout > 0 else None

        if self.rank > 0:
            self.A = nn.Parameter(torch.empty(self.rank, self.in_features))
            self.B = nn.Parameter(torch.empty(self.out_features, self.rank))
        else:
            # Register zero-sized parameters for easier handling downstream
            self.register_parameter("A", nn.Parameter(torch.empty(0, 0)))
            self.register_parameter("B", nn.Parameter(torch.empty(0, 0)))

        # Initially allow base weight to train during warm start.
        self.base_linear.weight.requires_grad_(True)
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad_(True)

        self.adapter_active: bool = False
        self._reset_adapter_parameters()

        # Flag adapter params for grouping
        if self.rank > 0:
            self.A._is_relora_adapter = True  # type: ignore[attr-defined]
            self.B._is_relora_adapter = True  # type: ignore[attr-defined]

    def _reset_adapter_parameters(self) -> None:
        if self.rank == 0:
            return
        _kaiming_init(self.A)
        nn.init.zeros_(self.B)

    def reset_adapter_parameters(self) -> None:
        self._reset_adapter_parameters()

    def activate_adapter(self, freeze_base: bool = True, reinit: bool = True) -> None:
        if self.rank == 0:
            return
        if freeze_base:
            self.base_linear.weight.requires_grad_(False)
        if reinit:
            self._reset_adapter_parameters()
        self.adapter_active = True
        self.A.requires_grad_(True)
        self.B.requires_grad_(True)

    def deactivate_adapter(self) -> None:
        self.adapter_active = False
        self.base_linear.weight.requires_grad_(True)
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad_(True)

    def adapter_parameters(self) -> List[nn.Parameter]:
        if self.rank == 0:
            return []
        return [self.A, self.B]

    def materialize_delta_weight(self) -> Tensor:
        if self.rank == 0:
            return torch.zeros_like(self.base_linear.weight)
        return self.scaling * self.B @ self.A

    @torch.no_grad()
    def merge_into_base(self) -> None:
        if self.rank == 0:
            return
        assert self.base_linear.weight.shape == (
            self.B.size(0),
            self.A.size(1),
        ), (
            "Shape mismatch: base weight",
            self.base_linear.weight.shape,
            "delta",
            (self.B.size(0), self.A.size(1)),
        )
        delta = (
            self.B.float()
            @ self.A.float()
        ) * float(self.scaling)
        delta = delta.to(self.base_linear.weight.dtype)
        self.base_linear.weight.add_(delta)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.base_linear.weight
        bias = self.base_linear.bias
        compute_dtype = torch.promote_types(x.dtype, weight.dtype)
        x_compute = x.to(compute_dtype)
        weight_compute = weight.to(compute_dtype)
        out = nn.functional.linear(x_compute, weight_compute, bias=bias)
        if not self.adapter_active or self.rank == 0:
            return out.to(x.dtype)

        A = self.A.to(compute_dtype)
        B = self.B.to(compute_dtype)
        down = x_compute.matmul(A.t())
        if self.adapter_dropout is not None:
            down = self.adapter_dropout(down)
        up = down.matmul(B.t())
        out = out + self.scaling * up
        return out.to(x.dtype)


@dataclass
class ReLoRAConfig:
    rank: int
    alpha: float
    dropout: float
    merge_interval: int
    warm_start_steps: int
    adapter_warmup_steps: int
    freeze_base: bool = True
    prune_b_state: bool = True


class ReLoRAController:
    """Manage activation, merging, and warmup for a collection of ReLoRALinear modules."""

    def __init__(
        self,
        modules: Sequence[ReLoRALinear],
        config: ReLoRAConfig,
        module_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.modules = list(modules)
        self.config = config
        self.module_names = list(module_names) if module_names is not None else None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.activated: bool = False
        self._warmup_active: bool = False
        self._warmup_step: int = 0
        self._warmup_targets: List[float] = []
        self._adapter_group_indices: List[int] = []

    def adapter_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for module in self.modules:
            params.extend(module.adapter_parameters())
        return params

    def attach_optimizer(
        self, optimizer: torch.optim.Optimizer, scheduler=None
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._adapter_group_indices = []
        if optimizer is None:
            return
        for idx, group in enumerate(optimizer.param_groups):
            if group.get("relora_adapter_group"):
                self._adapter_group_indices.append(idx)

    def on_batch_start(self, completed_steps: int) -> None:
        if self.activated:
            return
        if completed_steps >= self.config.warm_start_steps:
            for module in self.modules:
                module.activate_adapter(
                    freeze_base=self.config.freeze_base, reinit=True
                )
            self.activated = True
            self.start_adapter_warmup()

    def start_adapter_warmup(self) -> None:
        if not self.optimizer:
            return
        steps = max(int(self.config.adapter_warmup_steps), 0)
        if steps == 0:
            self._warmup_active = False
            return
        if not self._adapter_group_indices:
            # Fallback: detect groups containing adapter params if flag missing.
            adapter_param_ids = {id(p) for p in self.adapter_parameters()}
            self._adapter_group_indices = []
            for idx, group in enumerate(self.optimizer.param_groups):
                if any(id(p) in adapter_param_ids for p in group["params"]):
                    self._adapter_group_indices.append(idx)
        if not self._adapter_group_indices:
            return
        self._warmup_active = True
        self._warmup_step = 0
        self._warmup_targets = []
        for idx in self._adapter_group_indices:
            group = self.optimizer.param_groups[idx]
            target_lr = float(group.get("relora_base_lr", group["lr"]))
            group["relora_base_lr"] = target_lr
            group["lr"] = 0.0
            self._warmup_targets.append(target_lr)

    def _advance_warmup(self) -> None:
        if not self._warmup_active or not self.optimizer:
            return
        steps = max(int(self.config.adapter_warmup_steps), 1)
        self._warmup_step += 1
        progress = min(self._warmup_step / steps, 1.0)
        for target, idx in zip(self._warmup_targets, self._adapter_group_indices):
            self.optimizer.param_groups[idx]["lr"] = target * progress
        if progress >= 1.0:
            self._warmup_active = False

    def _prune_optimizer_states(self, module: ReLoRALinear) -> None:
        if not self.optimizer:
            return
        if module.rank == 0:
            return
        params = [(module.A, False), (module.B, True)]
        for param, is_b in params:
            if is_b and not self.config.prune_b_state:
                continue
            state = self.optimizer.state.get(param)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key].zero_()

    def _merge_and_reset(self) -> None:
        if not self.modules:
            return
        with torch.no_grad():
            for module in self.modules:
                if module.rank == 0:
                    continue
                module.merge_into_base()
                module.reset_adapter_parameters()
                self._prune_optimizer_states(module)
        self.start_adapter_warmup()

    def on_step_end(self, completed_steps: int) -> None:
        if not self.activated:
            return
        interval = max(int(self.config.merge_interval), 0)
        if interval > 0 and completed_steps > 0 and completed_steps % interval == 0:
            self._merge_and_reset()
        self._advance_warmup()


def iter_named_linear_modules(
    module: nn.Module, target_suffixes: Iterable[str]
) -> List[tuple[str, nn.Linear]]:
    suffixes = list(target_suffixes)
    matches: List[tuple[str, nn.Linear]] = []
    for name, submodule in module.named_modules():
        if not isinstance(submodule, nn.Linear):
            continue
        if suffixes and not any(name.endswith(sfx) for sfx in suffixes):
            continue
        matches.append((name, submodule))
    return matches


def replace_linears_with_relora(
    root: nn.Module,
    target_suffixes: Iterable[str],
    cfg: ReLoRAConfig,
) -> ReLoRAController:
    matches = iter_named_linear_modules(root, target_suffixes)
    relora_modules: List[ReLoRALinear] = []

    for name, linear in matches:
        adapter = ReLoRALinear(
            base_linear=linear,
            rank=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
        )
        relora_modules.append(adapter)

        # Replace module in place while keeping references consistent
        parent = root
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
        attr = parts[-1]
        if attr.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(attr)] = adapter
        else:
            setattr(parent, attr, adapter)

    controller = ReLoRAController(relora_modules, cfg, [name for name, _ in matches])
    return controller


__all__ = [
    "ReLoRALinear",
    "ReLoRAConfig",
    "ReLoRAController",
    "replace_linears_with_relora",
]
