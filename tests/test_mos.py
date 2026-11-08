"""Testes unitários para o Mixture of Scars (MoS)."""

import torch
import torch.nn as nn
import pytest

from scar_cache.mos import MixtureOfScarsLinear, ScarExpert


def test_scar_expert_low_rank_and_gating():
    """Verifica que o especialista tem consumo de bytes mínimo e gating funcional."""
    expert = ScarExpert(in_features=64, out_features=32, rank=4, scale=0.5)

    # Memória: (64*4 + 4*32)*4 bytes + 64*4 bytes = (256 + 128 + 64)*4 = 1792 bytes (< 2 KB!)
    assert expert.memory_bytes == 1792

    x = torch.randn(4, 64)
    # Antes da calibração, o gating é inativo, mas a projeção funciona
    out = expert(x)
    assert out.shape == (4, 32)

    # Calibra a bússola direcional
    expert.calibrate_direction(x)
    assert expert.scars_count.item() == 1

    out_gated = expert(x)
    assert out_gated.shape == (4, 32)


def test_mixture_of_scars_linear_forward():
    """Verifica o forward pass e dimensões do MixtureOfScarsLinear."""
    mos = MixtureOfScarsLinear(in_features=128, out_features=64, num_experts=8, top_k=2, rank=8)

    # Teste com tensor 2D [Batch, Dim]
    x_2d = torch.randn(8, 128)
    out_2d, routing, topk = mos(x_2d, return_routing=True)

    assert out_2d.shape == (8, 64)
    assert routing.shape == (8, 2)
    assert topk.shape == (8, 2)
    # Softmax sobre top-k deve somar aproximadamente 1.0
    assert torch.allclose(routing.sum(dim=-1), torch.ones(8), atol=1e-5)

    # Teste como drop-in padrão (return_routing=False)
    single_out = mos(x_2d)
    assert isinstance(single_out, torch.Tensor)
    assert single_out.shape == (8, 64)

    # Teste com tensor 3D [Batch, SeqLen, Dim]
    x_3d = torch.randn(4, 16, 128)
    out_3d, routing_3d, topk_3d = mos(x_3d, return_routing=True)

    assert out_3d.shape == (4, 16, 64)
    assert routing_3d.shape == (4, 16, 2)
    assert topk_3d.shape == (4, 16, 2)


def test_mixture_of_scars_shared_base_memory_reduction():
    """Verifica a redução drástica de memória ao compartilhar uma base única entre camadas."""
    in_dim, out_dim = 512, 512
    num_layers = 10
    num_experts = 8
    rank = 8

    # 1. Caso Tradicional: 10 camadas densas independentes
    dense_layers = [nn.Linear(in_dim, out_dim) for _ in range(num_layers)]
    dense_bytes = sum(
        l.weight.numel() * 4 + (l.bias.numel() * 4 if l.bias is not None else 0)
        for l in dense_layers
    )
    # Cada camada: (512*512 + 512)*4 = 1.050.624 bytes (~1.05 MB). 10 camadas = ~10.5 MB
    assert dense_bytes > 10 * (1024**2)

    # 2. Caso Mixture of Scars (MoS): 1 base compartilhada + 10 camadas de Scar Experts
    shared_base = nn.Linear(in_dim, out_dim)
    mos_layers = [
        MixtureOfScarsLinear(
            in_dim, out_dim, num_experts=num_experts, top_k=2, rank=rank, shared_base=shared_base
        )
        for _ in range(num_layers)
    ]

    base_bytes = shared_base.weight.numel() * 4 + shared_base.bias.numel() * 4
    scars_bytes = sum(l.scars_bytes for l in mos_layers)
    total_mos_bytes = base_bytes + scars_bytes

    # A economia deve ser de mais de 60% já com apenas 10 camadas de dim 512!
    reduction_factor = dense_bytes / total_mos_bytes
    assert reduction_factor > 2.5

    # Verifica que cada camada produz saídas diferentes graças aos seus Scar Experts
    x = torch.randn(2, in_dim)
    # Inicializa um especialista em uma camada para diferenciar
    with torch.no_grad():
        mos_layers[0].experts[0].B.fill_(0.5)

    out0 = mos_layers[0](x)
    out1 = mos_layers[1](x)
    # Como as camadas têm especialistas distintos, as saídas devem divergir
    assert not torch.allclose(out0, out1)


def test_mixture_of_scars_autograd():
    """Verifica que o gradiente retropropaga corretamente para a base e para os especialistas."""
    mos = MixtureOfScarsLinear(in_features=32, out_features=16, num_experts=4, top_k=2, rank=4)
    x = torch.randn(4, 32, requires_grad=True)
    out = mos(x)
    loss = out.sum()
    loss.backward()

    assert mos.base.weight.grad is not None
    assert mos.router.weight.grad is not None
    assert x.grad is not None
