"""TwinLoRA adapter implementation for coupled FFN pairs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TwinLoRAConfig:
    """Configuration for the TwinLoRA adapter.

    Attributes:
        rank: LoRA rank ``r`` – controls the per-layer low-rank sheets.
        shared_rank: Shared interface rank ``s`` for the coupled path.
        alpha: Scaling factor applied to the LoRA updates (equivalent to ``lora_alpha``).
        twin_alpha: Scaling factor applied to the shared interface updates.
        dropout: Dropout probability applied to the inputs of the LoRA path.
        twin_dropout: Dropout probability applied to the inputs of the shared path.
        init: Weight init strategy for the low-rank A matrices (``"kaiming"`` or ``"zeros"``).
    """

    rank: int = 16
    shared_rank: int = 8
    alpha: float = 16.0
    twin_alpha: float = 16.0
    dropout: float = 0.0
    twin_dropout: float = 0.0
    init: str = "kaiming"

    def validate(self) -> None:
        if self.rank < 0:
            raise ValueError("TwinLoRA rank must be >= 0")
        if self.shared_rank < 0:
            raise ValueError("TwinLoRA shared_rank must be >= 0")
        if self.rank == 0 and self.shared_rank == 0:
            raise ValueError("At least one of rank or shared_rank must be > 0")
        if self.alpha < 0 or self.twin_alpha < 0:
            raise ValueError("Scaling factors alpha and twin_alpha must be non-negative")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if not (0.0 <= self.twin_dropout < 1.0):
            raise ValueError("twin_dropout must be in [0, 1)")
        if self.init not in {"kaiming", "zeros"}:
            raise ValueError("init must be either 'kaiming' or 'zeros'")


class TwinLoRAAdapter(nn.Module):
    """Container for the shared parameters of a TwinLoRA FFN pair."""

    def __init__(
        self,
        down: nn.Linear,
        up: nn.Linear,
        config: TwinLoRAConfig,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        config.validate()

        self.rank = int(config.rank)
        self.shared_rank = int(config.shared_rank)
        self.alpha = float(config.alpha)
        self.twin_alpha = float(config.twin_alpha)
        self.register_buffer("scaling", torch.tensor(self.alpha / self.rank if self.rank > 0 else 0.0))
        self.register_buffer(
            "twin_scaling", torch.tensor(self.twin_alpha / self.shared_rank if self.shared_rank > 0 else 0.0)
        )

        self.down_in = down.in_features
        self.down_out = down.out_features
        self.up_in = up.in_features
        self.up_out = up.out_features

        if self.up_in != self.down_out:
            raise ValueError(
                "TwinLoRA expects the up-projection input dim to equal the down-projection output dim"
            )
        if self.up_out != self.down_in:
            raise ValueError(
                "TwinLoRA expects the up-projection output dim to equal the down-projection input dim"
            )

        factory_kwargs = {"device": device, "dtype": dtype}

        if self.rank > 0:
            self.A1 = nn.Parameter(torch.zeros(self.rank, self.down_in, **factory_kwargs))
            self.B1 = nn.Parameter(torch.zeros(self.down_out, self.rank, **factory_kwargs))
            self.A2 = nn.Parameter(torch.zeros(self.rank, self.up_in, **factory_kwargs))
            self.B2 = nn.Parameter(torch.zeros(self.up_out, self.rank, **factory_kwargs))
        else:
            self.register_parameter("A1", None)
            self.register_parameter("B1", None)
            self.register_parameter("A2", None)
            self.register_parameter("B2", None)

        if self.shared_rank > 0:
            self.U = nn.Parameter(torch.zeros(self.down_out, self.shared_rank, **factory_kwargs))
            self.R = nn.Parameter(torch.zeros(self.shared_rank, self.down_in, **factory_kwargs))
            self.P = nn.Parameter(torch.zeros(self.up_out, self.shared_rank, **factory_kwargs))
        else:
            self.register_parameter("U", None)
            self.register_parameter("R", None)
            self.register_parameter("P", None)

        self.lora_dropout: nn.Module = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        self.shared_dropout: nn.Module = (
            nn.Dropout(config.twin_dropout) if config.twin_dropout > 0.0 else nn.Identity()
        )

        self._init_weights(method=config.init)

    def _init_weights(self, method: str = "kaiming") -> None:
        if self.rank > 0:
            if method == "kaiming":
                nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.A2, a=math.sqrt(5))
            elif method == "zeros":
                nn.init.zeros_(self.A1)
                nn.init.zeros_(self.A2)
            nn.init.zeros_(self.B1)
            nn.init.zeros_(self.B2)

        if self.shared_rank > 0:
            if method == "kaiming":
                nn.init.kaiming_uniform_(self.R, a=math.sqrt(5))
            elif method == "zeros":
                nn.init.zeros_(self.R)
            nn.init.zeros_(self.U)
            nn.init.zeros_(self.P)

    @property
    def device(self) -> torch.device:
        params = [p for p in self.parameters() if p is not None]
        if not params:
            raise RuntimeError("TwinLoRAAdapter has no parameters")
        return params[0].device

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, shared_rank={self.shared_rank}, "
            f"alpha={self.alpha}, twin_alpha={self.twin_alpha}, "
            f"down=({self.down_out}, {self.down_in}), up=({self.up_out}, {self.up_in})"
        )


class TwinLoRADownLinear(nn.Module):
    """Down projection wrapper that injects TwinLoRA updates."""

    def __init__(self, base: nn.Linear, adapter: TwinLoRAAdapter) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        adapter = self.adapter

        if adapter.rank > 0:
            projected = F.linear(adapter.lora_dropout(x), adapter.A1)
            result = result + adapter.scaling * F.linear(projected, adapter.B1)

        if adapter.shared_rank > 0:
            shared_proj = F.linear(adapter.shared_dropout(x), adapter.R)
            result = result + adapter.twin_scaling * F.linear(shared_proj, adapter.U)

        return result


class TwinLoRAUpLinear(nn.Module):
    """Up projection wrapper that injects TwinLoRA updates."""

    def __init__(self, base: nn.Linear, adapter: TwinLoRAAdapter) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        result = self.base(h)
        adapter = self.adapter

        if adapter.rank > 0:
            projected = F.linear(adapter.lora_dropout(h), adapter.A2)
            result = result + adapter.scaling * F.linear(projected, adapter.B2)

        if adapter.shared_rank > 0:
            interface = F.linear(adapter.shared_dropout(h), adapter.U.transpose(0, 1))
            result = result + adapter.twin_scaling * F.linear(interface, adapter.P)

        return result


def build_twinlora_wrappers(
    down: nn.Linear,
    up: nn.Linear,
    config: TwinLoRAConfig,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[TwinLoRADownLinear, TwinLoRAUpLinear, TwinLoRAAdapter]:
    """Create TwinLoRA wrappers for a down/up projection pair."""

    adapter = TwinLoRAAdapter(down, up, config, dtype=dtype, device=device)
    down_wrapper = TwinLoRADownLinear(down, adapter)
    up_wrapper = TwinLoRAUpLinear(up, adapter)
    return down_wrapper, up_wrapper, adapter


__all__ = [
    "TwinLoRAConfig",
    "TwinLoRAAdapter",
    "TwinLoRADownLinear",
    "TwinLoRAUpLinear",
    "build_twinlora_wrappers",
]
