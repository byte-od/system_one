"""Testes Unitários e Validação Numérica da Arquitetura MoS Real 70B."""

import math
import pytest
import torch
import torch.nn as nn
from scar_cache.real_70b import (
    MoSResidualLinear,
    MoS70BConfig,
    calculate_real_70b_memory_footprint,
)
from scar_cache.convert_qwen import factorize_svd


def test_mos_residual_linear_forward():
    torch.manual_seed(42)
    in_feat = 256
    out_feat = 512
    rank = 16
    batch = 2
    seq = 8

    # Base compartilhada
    base_linear = nn.Linear(in_feat, out_feat, bias=False)
    mos_proj = MoSResidualLinear(base_linear=base_linear, in_features=in_feat, out_features=out_feat, rank=rank)

    # Verifica que aponta para o mesmo ponteiro de dados
    assert mos_proj.base_weight.data_ptr() == base_linear.weight.data_ptr(), "Base weight deve ser compartilhada por referência!"

    x = torch.randn(batch, seq, in_feat)
    y = mos_proj(x)
    assert y.shape == (batch, seq, out_feat)


def test_factorize_svd_reconstruction_fidelity():
    """Valida a precisão da aproximação de baixo rank no resíduo entre duas camadas densas."""
    torch.manual_seed(42)
    in_feat = 512
    out_feat = 1024
    rank = 32

    # Simula camada base e resíduo intrinsecamente de baixo rank (como em LLMs)
    w_base = torch.randn(out_feat, in_feat)
    u_true = torch.randn(out_feat, rank)
    v_true = torch.randn(in_feat, rank)
    diff = (u_true @ v_true.T) / math.sqrt(rank)
    w_adapted = w_base + diff

    A, B = factorize_svd(diff, rank=rank)

    # 1. Validação matemática direta
    x = torch.randn(4, in_feat)
    target = x @ diff.T
    approx = (x @ A) @ B

    rel_error = torch.norm(target - approx) / torch.norm(target)
    assert rel_error < 1e-4, f"Erro relativo {rel_error:.2e} excedeu o limite de tolerância SVD"

    # 2. Validação através do módulo MoSResidualLinear
    base_linear = nn.Linear(in_feat, out_feat, bias=False)
    base_linear.weight.data.copy_(w_base)
    mos_proj = MoSResidualLinear(base_linear=base_linear, in_features=in_feat, out_features=out_feat, rank=rank)
    mos_proj.load_scar_weights(A, B)

    y_dense = x @ w_adapted.T
    y_mos = mos_proj(x)
    total_rel_error = torch.norm(y_dense - y_mos) / torch.norm(y_dense)
    assert total_rel_error < 1e-4, f"Erro MoS total {total_rel_error:.2e} excedeu tolerância"


def test_memory_budget_qwen72b():
    """Valida os números teóricos de consumo de memória para o Qwen 2.5 72B."""
    config = MoS70BConfig(
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        group_size=8,
        rank=64,
    )

    # Cenário 1: Base e Scar em FP16 / BF16 (2 bytes)
    fp16_mem = calculate_real_70b_memory_footprint(config, base_dtype_bytes=2, scar_dtype_bytes=2)
    assert fp16_mem["dense_70b_weights_fp16_gb"] > 130.0, "Modelo denso deve ter ~140 GB"
    assert fp16_mem["compression_ratio_mlp"] > 6.0, f"Compressão MLP deve ser > 6x (foi {fp16_mem['compression_ratio_mlp']:.2f}x)"
    assert fp16_mem["scache_kv_cache_128k_mb"] < 3.0, "SCache deve ter menos de 3 MB para 128k context"

    # Cenário 2: Base e Atenção quantizados em INT4 (0.5 byte) e Scar em FP16 (2 bytes)
    int4_mem = calculate_real_70b_memory_footprint(config, base_dtype_bytes=0.5, scar_dtype_bytes=2)
    assert int4_mem["total_runtime_vram_gb"] < 16.0, f"Modelo total deve caber em 16 GB VRAM! Foi {int4_mem['total_runtime_vram_gb']:.2f} GB"


if __name__ == "__main__":
    test_mos_residual_linear_forward()
    test_factorize_svd_reconstruction_fidelity()
    test_memory_budget_qwen72b()
    print("✅ TODOS OS TESTES DA ARQUITETURA MoS REAL 70B PASSARAM COM SUCESSO!")
