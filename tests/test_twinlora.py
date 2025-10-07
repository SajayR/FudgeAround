from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.twinlora import (
    TwinLoRAAdapter,
    TwinLoRAConfig,
    TwinLoRADownLinear,
    TwinLoRAUpLinear,
    build_twinlora_wrappers,
)
from models import dinov2_twinlora
from models.dinov2_twinlora import build_model as build_dinov2_twin
from transformers import Dinov2Config, Dinov2Model


def _manual_down(adapter: TwinLoRAAdapter, x: torch.Tensor) -> torch.Tensor:
    term = torch.zeros(x.size(0), adapter.down_out, device=x.device, dtype=x.dtype)
    if adapter.rank > 0:
        term = term + (adapter.B1 @ (adapter.A1 @ x.T)).T * adapter.scaling
    if adapter.shared_rank > 0:
        term = term + (adapter.U @ (adapter.R @ x.T)).T * adapter.twin_scaling
    return term


def _manual_up(adapter: TwinLoRAAdapter, h: torch.Tensor) -> torch.Tensor:
    term = torch.zeros(h.size(0), adapter.up_out, device=h.device, dtype=h.dtype)
    if adapter.rank > 0:
        term = term + (adapter.B2 @ (adapter.A2 @ h.T)).T * adapter.scaling
    if adapter.shared_rank > 0:
        term = term + (adapter.P @ (adapter.U.transpose(0, 1) @ h.T)).T * adapter.twin_scaling
    return term


def test_twinlora_wrappers_match_manual_formula():
    torch.manual_seed(0)
    batch, d, m = 3, 5, 7
    down = nn.Linear(d, m, bias=False)
    up = nn.Linear(m, d, bias=False)
    nn.init.zeros_(down.weight)
    nn.init.zeros_(up.weight)

    cfg = TwinLoRAConfig(rank=3, shared_rank=2, alpha=2.0, twin_alpha=1.5, dropout=0.0, twin_dropout=0.0, init="zeros")
    down_wrapper, up_wrapper, adapter = build_twinlora_wrappers(down, up, cfg)

    with torch.no_grad():
        adapter.A1.uniform_(-0.5, 0.5)
        adapter.B1.uniform_(-0.5, 0.5)
        adapter.A2.uniform_(-0.5, 0.5)
        adapter.B2.uniform_(-0.5, 0.5)
        adapter.R.uniform_(-0.5, 0.5)
        adapter.U.uniform_(-0.5, 0.5)
        adapter.P.uniform_(-0.5, 0.5)

    x = torch.randn(batch, d)
    h = torch.randn(batch, m)

    expected_down = _manual_down(adapter, x)
    expected_up = _manual_up(adapter, h)

    torch.testing.assert_close(down_wrapper(x), expected_down, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(up_wrapper(h), expected_up, rtol=1e-5, atol=1e-6)


def test_apply_twinlora_replaces_mlp_modules():
    config = Dinov2Config(
        image_size=32,
        patch_size=8,
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
    )
    backbone = Dinov2Model(config)
    twin_cfg = TwinLoRAConfig(rank=2, shared_rank=2, alpha=8.0, twin_alpha=4.0)
    applied = dinov2_twinlora._apply_twinlora(backbone, twin_cfg)

    assert applied == len(backbone.encoder.layer)

    for layer in backbone.encoder.layer:
        assert isinstance(layer.mlp.fc1, TwinLoRADownLinear)
        assert isinstance(layer.mlp.fc2, TwinLoRAUpLinear)
        assert isinstance(layer.mlp.twinlora_adapter, TwinLoRAAdapter)
        assert not layer.mlp.fc1.base.weight.requires_grad
        assert not layer.mlp.fc2.base.weight.requires_grad


def test_build_model_uses_twinlora_config(monkeypatch):
    config = Dinov2Config(
        image_size=32,
        patch_size=8,
        num_hidden_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
    )
    backbone = Dinov2Model(config)

    def fake_from_pretrained(cls, name):  # noqa: ARG001
        # Return a fresh copy so state mutations don't persist between calls
        return Dinov2Model(config)

    monkeypatch.setattr(dinov2_twinlora.Dinov2Model, "from_pretrained", classmethod(fake_from_pretrained))

    params = {
        "model_name": "facebook/dinov2-small",
        "dropout": 0.0,
        "twinlora": {
            "enabled": True,
            "rank": 2,
            "shared_rank": 1,
            "alpha": 4.0,
            "twin_alpha": 2.0,
            "layers": [0],
        },
    }

    model, extras = build_dinov2_twin("facebook/dinov2-small", num_classes=10, params=params)

    assert isinstance(model, dinov2_twinlora.DinoV2Classifier)
    assert extras["twinlora"]["applied_layers"] == 1
    assert extras["twinlora"]["layers"] == [0]

    adapter = model.backbone.encoder.layer[0].mlp.twinlora_adapter
    assert adapter.rank == 2
    assert adapter.shared_rank == 1
    assert adapter.A1.requires_grad
    assert adapter.U.requires_grad
    assert not model.backbone.encoder.layer[0].mlp.fc1.base.weight.requires_grad
