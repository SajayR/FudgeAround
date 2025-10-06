"""BS-ReLoRA implementation helpers.

This module implements the building blocks for the BS-ReLoRA training
algorithm as described in the specification provided with the task.
The code is intentionally framework agnostic – the controller operates on
``nn.Linear`` modules and interacts with existing optimizers via a small set of
hooks, so the higher level training loop can remain relatively light-weight.

Key abstractions:

``randomized_svd``
    Fast top-k SVD used to recover the dominant subspace of the averaged Adam
    step matrix gathered during probe mode.

``soft_deflate``
    Implements the soft projection operator that discourages reusing the spans
    captured in previous BS-ReLoRA cycles.

``BSReLoRALayerState``
    Tracks per-layer buffers (probe accumulators, memory subspaces) and owns
    the lightweight adapter that injects low-rank updates during the low-rank
    phase.

``BSReLoRAController``
    Orchestrates the lifecycle across all target layers (probe → SVD →
    deflation → low-rank optimisation → merge) and exposes hooks that the
    training loop can call around optimizer steps.

The module purposefully avoids any project-specific imports so it can be
unit-tested in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


Tensor = torch.Tensor


def _ensure_2d(tensor: Tensor, name: str) -> Tensor:
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {tuple(tensor.shape)}")
    return tensor


def randomized_svd(
    matrix: Tensor,
    rank: int,
    oversample: int = 4,
    n_power_iter: int = 1,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute a truncated randomized SVD.

    The implementation follows the algorithm described in Halko et al. (2011).
    Only the top ``rank`` singular vectors are returned (post truncation).

    Args:
        matrix: ``(m, n)`` tensor.
        rank: Target rank ``r``.
        oversample: Oversampling factor ``p`` to stabilise the approximation.
        n_power_iter: Number of power iterations (>= 0).
        eps: Numerical jitter to keep QR/SVD stable on near-zero matrices.

    Returns:
        (U, S, Vh) with shapes ``(m, r)``, ``(r,)``, ``(r, n)`` respectively.
    """

    matrix = _ensure_2d(matrix, "matrix")
    if rank <= 0:
        raise ValueError("rank must be positive")

    m, n = matrix.shape
    device = matrix.device
    dtype = matrix.dtype

    r_star = min(rank + oversample, min(m, n))

    if matrix.norm().item() < eps:
        # Degenerate case – return orthonormal bases spanning random subspaces.
        q_left = torch.linalg.qr(torch.randn(m, r_star, device=device, dtype=dtype))[0][:, : rank]
        q_right = torch.linalg.qr(torch.randn(n, r_star, device=device, dtype=dtype))[0][:, : rank]
        sing = torch.zeros(rank, device=device, dtype=dtype)
        return q_left, sing, q_right.T

    # Step 1: sample a Gaussian test matrix.
    omega = torch.randn(n, r_star, device=device, dtype=dtype)

    # Step 2: form sample matrix Y = A * Omega.
    y = matrix @ omega

    # Power iterations improve spectral decay capture.
    if n_power_iter > 0:
        for _ in range(n_power_iter):
            y = matrix @ (matrix.transpose(0, 1) @ y)

    # Step 3: Orthonormalise the sample matrix (thin QR).
    q, _ = torch.linalg.qr(y, mode="reduced")  # (m, r_star)

    # Step 4: Project A into the sampled column space.
    b = q.transpose(0, 1) @ matrix  # (r_star, n)

    # Step 5: SVD on the reduced matrix.
    u_tilde, sing, v_h = torch.linalg.svd(b, full_matrices=False)

    u = q @ u_tilde  # (m, r_star)

    # Truncate to the requested rank.
    u = u[:, :rank]
    sing = sing[:rank]
    v_h = v_h[:rank, :]
    return u, sing, v_h


def soft_deflate(
    subspace: Tensor,
    memory: Optional[Tensor],
    strength: float,
) -> Tensor:
    r"""Apply soft deflation against the memory subspace.

    Args:
        subspace: ``(d, r)`` matrix with orthonormal columns.
        memory: ``(d, k)`` matrix (or ``None``) representing stored bases.
        strength: Scalar \lambda \in [0, 1]. ``0`` keeps the subspace as-is,
            ``1`` projects out the memory component completely.

    Returns:
        ``(d, r)`` matrix with re-orthonormalised columns after soft deflation.
    """

    subspace = _ensure_2d(subspace, "subspace")
    if memory is None or memory.numel() == 0:
        # Nothing to deflate against – perform a QR to guarantee orthonormality.
        q, _ = torch.linalg.qr(subspace, mode="reduced")
        return q

    memory = _ensure_2d(memory, "memory")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be within [0, 1]")

    if strength == 0.0:
        q, _ = torch.linalg.qr(subspace, mode="reduced")
        return q

    # Project onto the memory span and subtract a scaled component.
    projection = memory @ (memory.transpose(0, 1) @ subspace)
    adjusted = subspace - strength * projection
    q, _ = torch.linalg.qr(adjusted, mode="reduced")
    return q


class _Adapter(nn.Module):
    """Lightweight container for trainable BS-ReLoRA cores."""

    def __init__(self) -> None:
        super().__init__()
        # 0x0 parameters keep the module registered without polluting optimisers
        # until ``activate`` is called.
        self.core_L = nn.Parameter(torch.zeros(0, 0), requires_grad=False)
        self.core_R = nn.Parameter(torch.zeros(0, 0), requires_grad=False)
        self.active: bool = False

    def activate(self, rank: int, device: torch.device, dtype: torch.dtype) -> None:
        # Initialise to a non-degenerate state so gradients propagate on the
        # very first low-rank step. Identity for ``L`` preserves the neutral
        # merge while allowing ``R`` to receive signal immediately.
        eye = torch.eye(rank, device=device, dtype=dtype)
        self.core_L = nn.Parameter(eye.clone())
        self.core_R = nn.Parameter(torch.zeros(rank, rank, device=device, dtype=dtype))
        self.active = True

    def deactivate(self, device: torch.device, dtype: torch.dtype) -> None:
        self.core_L = nn.Parameter(torch.zeros(0, 0, device=device, dtype=dtype), requires_grad=False)
        self.core_R = nn.Parameter(torch.zeros(0, 0, device=device, dtype=dtype), requires_grad=False)
        self.active = False

    def parameters_list(self) -> List[nn.Parameter]:
        if not self.active:
            return []
        return [self.core_L, self.core_R]


@dataclass
class BSReLoRALayerState:
    """State container for a single BS-ReLoRA managed ``nn.Linear`` layer."""

    module: nn.Linear
    name: str
    device: torch.device
    dtype: torch.dtype
    memory_rank_cap: int

    def __post_init__(self) -> None:
        weight = self.module.weight
        self.out_features, self.in_features = weight.shape
        self.param: nn.Parameter = self.module.weight
        self.snapshot: Optional[Tensor] = None
        self.probe_accum = torch.zeros_like(weight, dtype=torch.float32, device=self.device)
        self.probe_steps: int = 0
        self.avg_step: Optional[Tensor] = None
        self.memory_u = torch.zeros(self.out_features, 0, device=self.device, dtype=torch.float32)
        self.memory_v = torch.zeros(self.in_features, 0, device=self.device, dtype=torch.float32)
        self.tilde_u: Optional[Tensor] = None
        self.tilde_v: Optional[Tensor] = None
        self.adapter = _Adapter()
        # Register the adapter as a submodule for visibility / parameter discovery.
        self.module.add_module("_bs_relora_adapter", self.adapter)
        self.hook_handle = self.module.register_forward_hook(self._forward_hook, with_kwargs=False)
        self._original_requires_grad = bool(self.module.weight.requires_grad)

    # ------------------------------------------------------------------
    # Probe helpers
    # ------------------------------------------------------------------
    def reset_probe(self) -> None:
        self.probe_accum.zero_()
        self.probe_steps = 0
        self.avg_step = None

    def store_snapshot(self) -> None:
        self.snapshot = self.module.weight.detach().clone()

    def restore_and_accumulate(
        self,
        capture: bool,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        if self.snapshot is None:
            raise RuntimeError("store_snapshot must be called before restore_and_accumulate")
        weight = self.module.weight
        if capture:
            if optimizer is None:
                raise ValueError("optimizer is required to capture Adam steps")
            step = self._adam_preconditioned_step(optimizer)
            self.probe_accum.add_(step)
            self.probe_steps += 1
        weight.data.copy_(self.snapshot)
        self.snapshot = None

    def finalise_probe(self) -> None:
        if self.probe_steps == 0:
            self.avg_step = torch.zeros_like(self.probe_accum)
        else:
            self.avg_step = self.probe_accum / float(self.probe_steps)

    # ------------------------------------------------------------------
    # Subspace + memory helpers
    # ------------------------------------------------------------------
    def compute_subspace(
        self,
        rank: int,
        oversample: int,
        power_iter: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.avg_step is None:
            raise RuntimeError("finalise_probe must run before compute_subspace")
        return randomized_svd(self.avg_step, rank=rank, oversample=oversample, n_power_iter=power_iter)

    def apply_soft_deflation(self, u: Tensor, v: Tensor, lambda_u: float, lambda_v: float) -> Tuple[Tensor, Tensor]:
        u_tilde = soft_deflate(u, self.memory_u if self.memory_u.numel() else None, lambda_u)
        v_tilde = soft_deflate(v, self.memory_v if self.memory_v.numel() else None, lambda_v)
        self.tilde_u = u_tilde
        self.tilde_v = v_tilde
        return u_tilde, v_tilde

    def update_memory(self, u_new: Tensor, v_new: Tensor) -> None:
        def _update(mem: Tensor, new: Tensor, cap: int) -> Tensor:
            if new.numel() == 0:
                return mem
            combined = torch.cat([mem, new], dim=1) if mem.numel() else new
            q, _ = torch.linalg.qr(combined, mode="reduced")
            if cap and q.shape[1] > cap:
                q = q[:, -cap:]
            return q

        self.memory_u = _update(self.memory_u, u_new, self.memory_rank_cap)
        self.memory_v = _update(self.memory_v, v_new, self.memory_rank_cap)

    # ------------------------------------------------------------------
    # Low-rank adapter helpers
    # ------------------------------------------------------------------
    def activate_adapter(self, rank: int) -> None:
        if self.tilde_u is None or self.tilde_v is None:
            raise RuntimeError("Soft deflation must run before adapter activation")
        self.adapter.activate(rank, device=self.device, dtype=torch.float32)

    def deactivate_adapter(self) -> None:
        self.adapter.deactivate(device=self.device, dtype=torch.float32)
        self.tilde_u = None
        self.tilde_v = None

    def set_base_trainable(self, trainable: bool) -> None:
        self.module.weight.requires_grad_(trainable)

    def adapter_parameters(self) -> List[nn.Parameter]:
        return self.adapter.parameters_list()

    def compute_delta_weight(self) -> Tensor:
        if self.adapter.active is False or self.tilde_u is None or self.tilde_v is None:
            return torch.zeros_like(self.module.weight, dtype=torch.float32)
        core = self.adapter.core_R @ self.adapter.core_L  # (r, r)
        delta = self.tilde_u @ (core @ self.tilde_v.transpose(0, 1))
        return delta

    def adapter_param_norm(self, norm_type: float = 2.0) -> float:
        params = self.adapter.parameters_list()
        if not params:
            return 0.0
        norms = [p.detach().data.norm(norm_type) for p in params]
        total = torch.norm(torch.stack(norms), norm_type)
        return float(total.item())

    def adapter_grad_norm(self, norm_type: float = 2.0) -> float:
        params = self.adapter.parameters_list()
        grads = [p.grad.detach().data.norm(norm_type) for p in params if p.grad is not None]
        if not grads:
            return 0.0
        total = torch.norm(torch.stack(grads), norm_type)
        return float(total.item())

    def merge_into_weight(self) -> Tensor:
        delta = self.compute_delta_weight()
        self.module.weight.data.add_(delta.to(self.module.weight.dtype))
        return delta

    # ------------------------------------------------------------------
    # Optimiser state adjustments
    # ------------------------------------------------------------------
    def damp_optimizer_state(
        self,
        optimizer_state: Dict[str, Tensor],
        delta: Tensor,
        rho_eps: float,
        alpha_m_limits: Tuple[float, float],
        alpha_v_limits: Tuple[float, float],
        gamma: float,
    ) -> Dict[str, Tensor]:
        weight = self.module.weight.detach()
        rho = float(delta.norm().item() / (weight.norm().item() + rho_eps))
        alpha_m = _interpolate_alpha(rho, alpha_m_limits)
        alpha_v = _interpolate_alpha(rho, alpha_v_limits)

        exp_avg = optimizer_state.get("exp_avg")
        exp_avg_sq = optimizer_state.get("exp_avg_sq")
        if exp_avg is not None:
            exp_avg.mul_(alpha_m)
            if self.tilde_u is not None and self.tilde_v is not None and gamma > 0.0:
                proj = self._project_onto_span(exp_avg)
                exp_avg.sub_(gamma * proj)
        if exp_avg_sq is not None:
            v_min = exp_avg_sq.new_full((), 1e-12)
            exp_avg_sq.mul_(alpha_v).add_(v_min * (1.0 - alpha_v))
        return optimizer_state

    def _project_onto_span(self, matrix: Tensor) -> Tensor:
        # Implements U (U^T matrix V) V^T without forming dense projectors.
        if self.tilde_u is None or self.tilde_v is None:
            return torch.zeros_like(matrix)
        temp = torch.matmul(matrix, self.tilde_v)
        temp = torch.matmul(self.tilde_u.transpose(0, 1), temp)
        temp = torch.matmul(self.tilde_u, temp)
        proj = torch.matmul(temp, self.tilde_v.transpose(0, 1))
        return proj

    # ------------------------------------------------------------------
    # Forward hook
    # ------------------------------------------------------------------
    def _forward_hook(self, module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> Tensor:
        if not self.adapter.active or self.tilde_u is None or self.tilde_v is None:
            return output
        x = inputs[0]
        original_dtype = output.dtype
        x_proj = x.to(torch.float32)
        core = self.adapter.core_R @ self.adapter.core_L  # (r, r)
        delta = torch.matmul(x_proj, self.tilde_v @ core.transpose(0, 1))
        delta = torch.matmul(delta, self.tilde_u.transpose(0, 1))
        return output + delta.to(original_dtype)

    def _adam_preconditioned_step(self, optimizer: torch.optim.Optimizer) -> Tensor:
        state = optimizer.state.get(self.param)
        if not state:
            return torch.zeros_like(self.probe_accum)

        group = None
        for param_group in optimizer.param_groups:
            for candidate in param_group["params"]:
                if candidate is self.param:
                    group = param_group
                    break
            if group is not None:
                break
        if group is None:
            raise RuntimeError("Parameter not found in optimizer param groups")

        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        step_t = state.get("step", 0)
        if exp_avg is None or exp_avg_sq is None or exp_avg.numel() == 0:
            return torch.zeros_like(self.probe_accum)

        if isinstance(step_t, torch.Tensor):
            step_val = int(step_t.item())
        else:
            step_val = int(step_t)
        if step_val <= 0:
            return torch.zeros_like(self.probe_accum)

        beta1, beta2 = group.get("betas", (0.9, 0.999))
        bias_correction1 = 1.0 - float(beta1) ** step_val
        bias_correction2 = 1.0 - float(beta2) ** step_val
        if bias_correction1 == 0.0 or bias_correction2 == 0.0:
            return torch.zeros_like(self.probe_accum)

        m_hat = exp_avg / bias_correction1
        v_hat = exp_avg_sq / bias_correction2
        denom = v_hat.sqrt().add_(group.get("eps", 1e-8))
        step = group.get("lr", 1.0) * (m_hat / denom)
        return step.to(torch.float32)


def _interpolate_alpha(rho: float, limits: Tuple[float, float]) -> float:
    lo, hi = limits
    if not (0.0 < lo <= hi <= 1.0):
        raise ValueError("alpha limits must satisfy 0 < lo <= hi <= 1")
    # Map rho logarithmically between lo and hi. Typical rho ~ 1e-3.
    rho = max(rho, 1e-8)
    t = (math.log10(rho) + 6.0) / 6.0
    t = min(max(t, 0.0), 1.0)
    value = hi - (hi - lo) * t
    return float(value)


@dataclass
class BSReLoRAConfig:
    rank: int
    oversample: int
    power_iterations: int
    probe_steps: int
    low_rank_steps: int
    memory_rank_cap: int
    lambda_u: float
    lambda_v: float
    gamma: float
    alpha_m_limits: Tuple[float, float]
    alpha_v_limits: Tuple[float, float]
    rho_eps: float


class BSReLoRAController:
    """Coordinates BS-ReLoRA across a set of target ``nn.Linear`` layers."""

    def __init__(
        self,
        model: nn.Module,
        layer_selector: Iterable[Tuple[str, nn.Linear]],
        config: BSReLoRAConfig,
    ) -> None:
        self.config = config
        self.layers: List[BSReLoRALayerState] = []
        for name, module in layer_selector:
            layer_state = BSReLoRALayerState(
                module=module,
                name=name,
                device=module.weight.device,
                dtype=module.weight.dtype,
                memory_rank_cap=config.memory_rank_cap,
            )
            self.layers.append(layer_state)

    # ------------------------------------------------------------------
    # Cycle orchestration helpers
    # ------------------------------------------------------------------
    def start_probe(self) -> None:
        for layer in self.layers:
            layer.reset_probe()

    def prepare_step(self) -> None:
        for layer in self.layers:
            layer.store_snapshot()

    def finish_step(
        self,
        capture: bool,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        for layer in self.layers:
            layer.restore_and_accumulate(capture=capture, optimizer=optimizer)

    def finish_probe(self) -> None:
        for layer in self.layers:
            layer.finalise_probe()

    def extract_subspaces(self) -> None:
        for layer in self.layers:
            u, _, v_h = layer.compute_subspace(
                rank=self.config.rank,
                oversample=self.config.oversample,
                power_iter=self.config.power_iterations,
            )
            v = v_h.transpose(0, 1)
            layer.apply_soft_deflation(u, v, self.config.lambda_u, self.config.lambda_v)
            layer.update_memory(layer.tilde_u, layer.tilde_v)

    def activate_low_rank(self) -> None:
        for layer in self.layers:
            layer.activate_adapter(self.config.rank)

    def deactivate_low_rank(self) -> None:
        for layer in self.layers:
            layer.deactivate_adapter()

    def adapter_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for layer in self.layers:
            params.extend(layer.adapter_parameters())
        return params

    def adapter_param_norm(self, norm_type: float = 2.0) -> float:
        params = self.adapter_parameters()
        if not params:
            return 0.0
        norms = [p.detach().data.norm(norm_type) for p in params]
        total = torch.norm(torch.stack(norms), norm_type)
        return float(total.item())

    def adapter_grad_norm(self, norm_type: float = 2.0) -> float:
        params = self.adapter_parameters()
        grads = [p.grad.detach().data.norm(norm_type) for p in params if p.grad is not None]
        if not grads:
            return 0.0
        total = torch.norm(torch.stack(grads), norm_type)
        return float(total.item())

    def merge_and_damp(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for layer in self.layers:
            delta = layer.merge_into_weight()
            state = optimizer.state.get(layer.module.weight)
            if isinstance(state, dict):
                layer.damp_optimizer_state(
                    optimizer_state=state,
                    delta=delta,
                    rho_eps=self.config.rho_eps,
                    alpha_m_limits=self.config.alpha_m_limits,
                    alpha_v_limits=self.config.alpha_v_limits,
                    gamma=self.config.gamma,
                )
            metrics[f"bs_relora/{layer.name}_rho"] = float(
                delta.norm().item() / (layer.module.weight.norm().item() + self.config.rho_eps)
            )
        return metrics

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def layer_names(self) -> Sequence[str]:
        return [layer.name for layer in self.layers]

    def set_base_trainable(self, trainable: bool) -> None:
        for layer in self.layers:
            layer.set_base_trainable(trainable)

    def restore_original_requires_grad(self) -> None:
        for layer in self.layers:
            layer.set_base_trainable(layer._original_requires_grad)


__all__ = [
    "BSReLoRAConfig",
    "BSReLoRAController",
    "BSReLoRALayerState",
    "randomized_svd",
    "soft_deflate",
]
