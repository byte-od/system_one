"""Testes unitários para módulos de alto nível ScarLinear e ScarGatedLayer."""

import torch
import torch.nn as nn
import pytest

from scar_cache.modules import ScarGatedLayer, ScarLinear


def test_scar_linear_forward_backward():
    """Garante compatibilidade total com autograd padrão do PyTorch."""
    layer = ScarLinear(in_features=16, out_features=4, mode="hybrid")
    x = torch.randn(8, 16, requires_grad=True)

    out = layer(x)
    assert out.shape == (8, 4)

    loss = out.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert x.grad is not None


def test_scar_linear_record_from_loss():
    """Testa a gravação autônoma de cicatriz a partir do loss autograd."""
    layer = ScarLinear(in_features=16, out_features=4, mode="hybrid")
    x = torch.randn(8, 16, requires_grad=True)

    out = layer(x)
    loss = out.sum()

    # Registra cicatriz sem backprop nos pesos lentos
    layer.record_from_loss(loss, x, surprise=1.5)
    assert layer.scars_count == 1
    assert torch.norm(layer.cache.W_scar).item() > 0


def test_scar_linear_projection_dimension():
    """Testa quando in_features != slot_dim (usando projeção interna)."""
    layer = ScarLinear(in_features=64, out_features=10, slot_dim=16, mode="hybrid")
    assert layer.needs_projection
    assert layer.cache.slot_dim == 16

    x = torch.randn(4, 64)
    out = layer(x)
    assert out.shape == (4, 10)

    # Grava cicatriz no espaço in_features (que é projetada internamente)
    grad_x = torch.randn(4, 64)
    layer.record_scar(x, grad_x, surprise=2.0)
    assert layer.scars_count == 1


def test_scar_gated_layer_wrapper():
    """Testa o wrapper ScarGatedLayer sobre um MLP sequencial padrão."""
    base_mlp = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    )
    gated_mlp = ScarGatedLayer(base_mlp, slot_dim=16, mode="hybrid")

    x = torch.randn(5, 32)
    out, affinity = gated_mlp(x, return_affinity=True)
    assert out.shape == (5, 16)
    assert affinity is None  # Nenhuma cicatriz ainda

    # Registra uma cicatriz na saída
    grad_out = torch.randn(5, 16)
    gated_mlp.record_scar(out, grad_out, surprise=1.0)
    assert gated_mlp.scars_count == 1

    out2, affinity2 = gated_mlp(x, return_affinity=True)
    assert affinity2 is not None
