import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]

RELORA_PATH = ROOT_DIR / "models" / "relora.py"
spec = importlib.util.spec_from_file_location("relora_module", RELORA_PATH)
relora_module = importlib.util.module_from_spec(spec)
import sys

sys.modules[spec.name] = relora_module
assert spec.loader is not None
spec.loader.exec_module(relora_module)

ReLoRALinear = relora_module.ReLoRALinear
ReLoRAConfig = relora_module.ReLoRAConfig
ReLoRAController = relora_module.ReLoRAController


def _build_linear(d_in: int, d_out: int, rank: int) -> ReLoRALinear:
    base = nn.Linear(d_in, d_out, bias=True)
    return ReLoRALinear(base_linear=base, rank=rank, alpha=32.0, dropout=0.0)


def test_relora_forward_matches_merge() -> None:
    torch.manual_seed(0)
    adapter = _build_linear(8, 6, rank=4)
    adapter.activate_adapter()
    x = torch.randn(3, 8)

    with torch.no_grad():
        delta = adapter.materialize_delta_weight()
        weight_eff = adapter.base_linear.weight + delta
        expected = F.linear(x, weight_eff, bias=adapter.base_linear.bias)

    out = adapter(x)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4)


def test_relora_controller_merge_and_prune_states() -> None:
    torch.manual_seed(1)
    adapter = _build_linear(4, 5, rank=2)
    config = ReLoRAConfig(
        rank=2,
        alpha=16.0,
        dropout=0.0,
        merge_interval=2,
        warm_start_steps=0,
        adapter_warmup_steps=4,
        freeze_base=True,
        prune_b_state=True,
    )
    controller = ReLoRAController([adapter], config, module_names=["linear"])

    base_group = {"params": [adapter.base_linear.weight, adapter.base_linear.bias], "lr": 1e-3}
    adapter_group = {
        "params": [adapter.A, adapter.B],
        "lr": 1e-3,
        "relora_adapter_group": True,
        "weight_decay": 0.0,
    }
    optimizer = torch.optim.Adam([base_group, adapter_group])
    controller.attach_optimizer(optimizer)

    global_step = 0
    for _ in range(2):
        controller.on_batch_start(global_step)
        x = torch.randn(7, 4)
        loss = adapter(x).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if global_step == 2:
            weight_before = adapter.base_linear.weight.detach().clone()
            delta = adapter.materialize_delta_weight().detach().clone()
        controller.on_step_end(global_step)

    weight_after = adapter.base_linear.weight.detach()
    assert torch.allclose(weight_after, weight_before + delta, atol=1e-6)

    state_A = optimizer.state[adapter.A]
    assert torch.count_nonzero(state_A["exp_avg"]) == 0
    assert torch.count_nonzero(state_A["exp_avg_sq"]) == 0

    state_B = optimizer.state[adapter.B]
    assert torch.count_nonzero(state_B["exp_avg"]) == 0
    assert torch.count_nonzero(state_B["exp_avg_sq"]) == 0

    adapter_lr = optimizer.param_groups[1]["lr"]
    target_lr = optimizer.param_groups[1]["relora_base_lr"]
    # After merge the warmup should have progressed one step (1/4 of target)
    assert math.isclose(
        adapter_lr,
        target_lr * 1.0 / config.adapter_warmup_steps,
        rel_tol=1e-6,
        abs_tol=1e-9,
    )


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__])
