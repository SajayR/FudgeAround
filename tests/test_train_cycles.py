import pytest
import torch.nn as nn

from train import _freeze_model_except, _resolve_cycle_counts, _restore_requires_grad_state


def test_resolve_cycle_counts_manual():
    cycles, info = _resolve_cycle_counts(
        steps_per_epoch=100,
        grad_accum=2,
        epochs=3,
        probe_steps=10,
        low_rank_steps=20,
        cycles_cfg=5,
    )

    assert cycles == 5
    assert info["auto"] is False
    assert info["steps_per_cycle"] == 30
    assert info["cycles_per_epoch"] is None
    assert info["optimizer_steps_per_epoch"] is None


def test_resolve_cycle_counts_auto():
    cycles, info = _resolve_cycle_counts(
        steps_per_epoch=96,
        grad_accum=2,
        epochs=3,
        probe_steps=8,
        low_rank_steps=16,
        cycles_cfg="auto",
    )

    # optimizer steps per epoch = ceil(96/2) = 48; steps per cycle = 24 -> 2 cycles per epoch
    assert cycles == 6
    assert info["auto"] is True
    assert info["cycles_per_epoch"] == 2
    assert info["optimizer_steps_per_epoch"] == 48
    assert info["steps_per_cycle"] == 24


@pytest.mark.parametrize("cycles_cfg", [None, "auto"])
def test_resolve_cycle_counts_auto_requires_length(cycles_cfg):
    with pytest.raises(ValueError):
        _resolve_cycle_counts(
            steps_per_epoch=None,
            grad_accum=1,
            epochs=1,
            probe_steps=32,
            low_rank_steps=64,
            cycles_cfg=cycles_cfg,
        )


def test_freeze_model_except_keeps_allowed_prefixes():
    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.head = nn.Sequential(nn.Linear(4, 2))

    model = Toy()
    state = _freeze_model_except(model, ["head"])

    assert not model.backbone.weight.requires_grad
    assert not model.backbone.bias.requires_grad
    assert model.head[0].weight.requires_grad
    assert model.head[0].bias.requires_grad

    # Restore original flags (all True) and ensure they match the saved state.
    _restore_requires_grad_state(model, state)
    assert model.backbone.weight.requires_grad
    assert model.backbone.bias.requires_grad
    assert model.head[0].weight.requires_grad
    assert model.head[0].bias.requires_grad
