import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.bs_relora import (  # noqa: E402
    BSReLoRAConfig,
    BSReLoRAController,
    randomized_svd,
    soft_deflate,
)


def test_randomized_svd_matches_top_singular_vectors():
    torch.manual_seed(0)
    matrix = torch.randn(20, 10)
    rank = 3
    u, s, v_h = randomized_svd(matrix, rank=rank, oversample=6, n_power_iter=2)

    assert u.shape == (20, rank)
    assert s.shape == (rank,)
    assert v_h.shape == (rank, 10)

    recon = u @ torch.diag(s) @ v_h
    full_u, full_s, full_vh = torch.linalg.svd(matrix, full_matrices=False)
    optimal = full_u[:, :rank] @ torch.diag(full_s[:rank]) @ full_vh[:rank]

    error = (recon - optimal).norm() / optimal.norm()
    assert error.item() < 5e-2


def test_soft_deflate_removes_memory_component():
    memory = torch.tensor([[1.0], [0.0], [0.0]])
    subspace = torch.tensor([[1.0], [1.0], [0.0]]) / math.sqrt(2.0)
    result = soft_deflate(subspace, memory, strength=1.0)

    # Resulting vector should be orthogonal to memory and unit norm.
    assert torch.allclose(result.transpose(0, 1) @ memory, torch.zeros(1, 1), atol=1e-5)
    assert torch.allclose(result.norm(), torch.tensor(1.0), atol=1e-5)


def test_layer_forward_and_merge_updates_weight():
    linear = nn.Linear(3, 3, bias=False)
    linear.weight.data.copy_(torch.eye(3))
    layer = list(BSReLoRAController(
        model=linear,
        layer_selector=[("linear", linear)],
        config=BSReLoRAConfig(
            rank=1,
            oversample=0,
            power_iterations=0,
            probe_steps=1,
            low_rank_steps=1,
            memory_rank_cap=4,
            lambda_u=0.0,
            lambda_v=0.0,
            gamma=0.0,
            alpha_m_limits=(0.1, 0.5),
            alpha_v_limits=(0.5, 0.9),
            rho_eps=1e-12,
        ),
    ).layers)[0]

    u = torch.randn(3, 1)
    u = u / u.norm()
    v = torch.randn(3, 1)
    v = v / v.norm()
    layer.tilde_u = u
    layer.tilde_v = v
    layer.activate_adapter(rank=1)
    layer.adapter.core_L.data.fill_(2.0)
    layer.adapter.core_R.data.fill_(3.0)

    x = torch.randn(2, 3)
    base_out = linear(x)
    with torch.enable_grad():
        delta_out = layer._forward_hook(linear, (x,), base_out)

    expected_delta = x @ (v * 6.0) @ u.transpose(0, 1)
    assert torch.allclose(delta_out - base_out, expected_delta, atol=1e-6)

    delta_weight = layer.merge_into_weight()
    expected_weight_delta = u @ (torch.tensor([[6.0]]) @ v.transpose(0, 1))
    assert torch.allclose(delta_weight, expected_weight_delta, atol=1e-6)
    assert torch.allclose(linear.weight, torch.eye(3) + expected_weight_delta, atol=1e-6)


def test_probe_accumulates_step_and_restores_weight():
    linear = nn.Linear(2, 2, bias=False)
    torch.nn.init.eye_(linear.weight)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-2)

    config = BSReLoRAConfig(
        rank=1,
        oversample=0,
        power_iterations=0,
        probe_steps=1,
        low_rank_steps=1,
        memory_rank_cap=2,
        lambda_u=0.0,
        lambda_v=0.0,
        gamma=0.0,
        alpha_m_limits=(0.1, 0.5),
        alpha_v_limits=(0.5, 0.9),
        rho_eps=1e-12,
    )

    controller = BSReLoRAController(linear, [("proj", linear)], config)
    layer = controller.layers[0]

    controller.start_probe()
    x = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([0])

    logits = linear(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    controller.prepare_step()
    optimizer.step()
    controller.finish_step(capture=True)
    optimizer.zero_grad()

    controller.finish_probe()

    # Weight should be restored to identity.
    assert torch.allclose(linear.weight, torch.eye(2), atol=1e-6)
    assert layer.probe_steps == 1
    assert torch.allclose(layer.avg_step, layer.probe_accum)
