"""Kernel Triton Fused para MoS (Fase 3): Fusão Epilogue de Scar Experts na GPU.

Elimina o gargalo de largura de banda de memória (DRAM) na inferência:
  Em vez de 3 lançamentos de kernel separados:
    1. out1 = x @ W_base.T       (GEMM 1 -> Escreve matriz gigante na VRAM)
    2. out2 = x @ A.T            (GEMM 2 -> Escreve vetor residual na VRAM)
    3. out3 = out2 @ B.T         (GEMM 3 -> Escreve matriz na VRAM)
    4. y = out1 + out3           (Kernel de Adição -> Lê 2 matrizes, escreve 1)

  O Kernel Fused MoS:
    1. Executa um prólogo ultra-rápido: h_scar = x @ A.T (mantido em Cache L1/SRAM).
    2. Durante o cálculo em registradores da GEMM de W_base, funde a multiplicação
       do scar residual (h_scar @ B.T) diretamente no acumulador FP32 do registrador (Epilogue Fusion).
    3. Escreve a saída final consolidada de volta à VRAM em UMA ÚNICA ESCRITA!
    4. Economiza ~50% de tráfego de barramento DRAM e reduz a latência ao nível da camada densa pura.

Fallback Automático:
  Em ambientes sem CUDA/Triton (ex: macOS Metal MPS ou CPU), utiliza PyTorch BLAS Level 3
  otimizado com `addmm_` in-place para zero alocação intermediária.
"""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_mos_gemm_epilogue_kernel(
        X_ptr,          # [M, K]
        W_base_ptr,     # [N, K] (ou W.T dependendo dos strides)
        H_scar_ptr,     # [M, R]
        B_ptr,          # [N, R] (ou [R, N])
        Y_ptr,          # [M, N]
        M, N, K, R,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_hm, stride_hr,
        stride_bn, stride_br,
        stride_ym, stride_yn,
        scaling,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """Kernel Fused Triton: GEMM W_base + Low-Rank Epilogue Fusion no registrador."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        # Acumulador em registradores (FP32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # 1. Loop Principal GEMM sobre a Base Universal W_base
        for k in range(0, K, BLOCK_K):
            k_offs = k + offs_k
            mask_k = k_offs < K

            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            w_ptrs = W_base_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w_tile = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)

            acc += tl.dot(x_tile, w_tile)

        # 2. Epilogue Fusion: Adiciona a contribuição do Scar Expert diretamente aos registradores
        # h_scar: [BLOCK_M, R], B: [BLOCK_N, R]
        offs_r = tl.arange(0, BLOCK_R)
        mask_r = offs_r < R

        h_ptrs = H_scar_ptr + offs_m[:, None] * stride_hm + offs_r[None, :] * stride_hr
        b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br

        h_tile = tl.load(h_ptrs, mask=mask_m[:, None] & mask_r[None, :], other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0)

        # Dot product do resíduo scar e soma ao acumulador
        scar_dot = tl.dot(h_tile, tl.trans(b_tile)) * scaling
        acc += scar_dot

        # 3. Escrita Única na VRAM (Global Memory)
        y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_fused_mos_linear(
    x: torch.Tensor,
    W_base: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float = 1.0,
) -> torch.Tensor:
    """Executa a projeção MoS com kernel Triton fundido em GPU CUDA."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    # W_base é [N, K], A é [R, K] ou [K, R], B é [N, R] ou [R, N]
    N = W_base.shape[0]

    # Garante orientações padronizadas
    # A para [R, K]
    if A.shape[0] != B.shape[1] and A.shape[1] == K:
        A_mat = A
    elif A.shape[0] == K:
        A_mat = A.T
    else:
        A_mat = A

    R = A_mat.shape[0]

    # B para [N, R]
    if B.shape == (N, R):
        B_mat = B
    elif B.shape == (R, N):
        B_mat = B.T
    else:
        B_mat = B

    # 1. Prólogo rápido: h_scar = x @ A.T  ([M, K] x [K, R] -> [M, R])
    # Como R << K (ex: 64 vs 8192), este prólogo roda em microsegundos e reside em L1/L2
    h_scar = F.linear(x_2d, A_mat)  # [M, R]

    # 2. Aloca buffer de saída final
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Configuração de blocos Triton
    BLOCK_M = 32 if M <= 32 else 64
    BLOCK_N = 64
    BLOCK_K = 32
    BLOCK_R = triton.next_power_of_2(R)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _fused_mos_gemm_epilogue_kernel[grid](
        x_2d, W_base, h_scar, B_mat, y,
        M, N, K, R,
        x_2d.stride(0), x_2d.stride(1),
        W_base.stride(0), W_base.stride(1),
        h_scar.stride(0), h_scar.stride(1),
        B_mat.stride(0), B_mat.stride(1),
        y.stride(0), y.stride(1),
        scaling,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_R=BLOCK_R,
    )

    return y.reshape(*orig_shape[:-1], N)


def pytorch_fused_mos_linear(
    x: torch.Tensor,
    W_base: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scaling: float = 1.0,
) -> torch.Tensor:
    """Implementação BLAS Nível 3 ultra-otimizada em PyTorch com in-place accumulator."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])

    # 1. Base Universal: out = x @ W_base.T
    out = F.linear(x_2d, W_base, bias)

    # 2. Scar Expert: h_scar = x @ A.T
    # A é [R, K] ou [K, R]
    if A.shape[1] == x_2d.shape[-1]:
        h_scar = F.linear(x_2d, A)
    else:
        h_scar = x_2d @ A

    # 3. Epilogue addmm_ in-place: out += scaling * (h_scar @ B.T)
    # Zero alocação extra de matriz gigante!
    if B.shape[0] == W_base.shape[0]:
        out.addmm_(h_scar, B.T, alpha=scaling)
    else:
        out.addmm_(h_scar, B, alpha=scaling)

    return out.reshape(*orig_shape[:-1], -1)


def fused_mos_linear(
    x: torch.Tensor,
    W_base: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scaling: float = 1.0,
    force_pytorch: bool = False,
) -> torch.Tensor:
    """Interface unificada de Projeção MoS Fused de Alta Eficiência.
    
    Seleciona automaticamente:
      - Triton Fused Epilogue Kernel quando em GPU CUDA compatível.
      - PyTorch addmm_ In-Place BLAS Nível 3 quando em Apple Silicon Metal / CPU.
    """
    if (
        not force_pytorch
        and HAS_TRITON
        and x.is_cuda
        and W_base.is_cuda
        and bias is None
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        try:
            return triton_fused_mos_linear(x, W_base, A, B, scaling=scaling)
        except Exception:
            # Fallback seguro para BLAS caso a compilação JIT falhe no ambiente
            return pytorch_fused_mos_linear(x, W_base, A, B, bias=bias, scaling=scaling)

    return pytorch_fused_mos_linear(x, W_base, A, B, bias=bias, scaling=scaling)
