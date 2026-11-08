"""Testes Unitários do Kernel Fused MoS (Fase 3)."""

import pytest
import torch
import torch.nn as nn
from scar_cache.real_70b import MoSResidualLinear
from scar_cache.triton_mos_gemm import (
    pytorch_fused_mos_linear,
    fused_mos_linear,
    HAS_TRITON,
)


def test_fused_mos_linear_numerical_equivalence_2d():
    torch.manual_seed(42)
    M, K, N, R = 32, 256, 512, 16

    x = torch.randn(M, K)
    W_base = torch.randn(N, K)
    A = torch.randn(R, K)
    B = torch.randn(N, R)

    # Referência desacoplada
    out_ref = torch.nn.functional.linear(x, W_base) + torch.nn.functional.linear(
        torch.nn.functional.linear(x, A), B
    )

    # Fused PyTorch addmm_
    out_fused = pytorch_fused_mos_linear(x, W_base, A, B)

    rel_error = (torch.norm(out_ref - out_fused) / torch.norm(out_ref)).item()
    assert rel_error < 1e-6, f"Erro relativo {rel_error:.2e} excedeu o limite em 2D"


def test_fused_mos_linear_3d_and_scaling():
    torch.manual_seed(42)
    batch, seq, in_dim, out_dim, rank = 2, 8, 128, 256, 16
    scaling = 0.5

    x = torch.randn(batch, seq, in_dim)
    base = nn.Linear(in_dim, out_dim, bias=False)

    mos_proj = MoSResidualLinear(
        base_linear=base,
        in_features=in_dim,
        out_features=out_dim,
        rank=rank,
        scaling=scaling,
    )

    y_unfused = mos_proj(x, use_fused=False)
    y_fused = mos_proj(x, use_fused=True)

    rel_error = (torch.norm(y_unfused - y_fused) / torch.norm(y_unfused)).item()
    assert rel_error < 1e-6, f"Erro relativo {rel_error:.2e} excedeu o limite em 3D com scaling"


def test_fused_interface_fallback():
    """Valida que a interface unificada funciona de ponta a ponta sem erros em qualquer hardware."""
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    W_base = torch.randn(128, 64)
    A = torch.randn(8, 64)
    B = torch.randn(128, 8)

    out = fused_mos_linear(x, W_base, A, B)
    assert out.shape == (4, 128)
    assert not torch.isnan(out).any()


if __name__ == "__main__":
    test_fused_mos_linear_numerical_equivalence_2d()
    test_fused_mos_linear_3d_and_scaling()
    test_fused_interface_fallback()
    print("✅ TODOS OS TESTES DO KERNEL FUSED MoS (FASE 3) PASSARAM COM SUCESSO!")
