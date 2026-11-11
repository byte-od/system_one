"""Triton Kernel para SCache: Regra Delta de Widrow-Hoff Compilada para GPU (Nvidia Tesla T4).

Elimina 100% de laços em Python e roundtrips com a memória global:
  1. Carrega blocos de K e V nos registradores rápidos SRAM (Shared Memory).
  2. Executa a atualização da Regra Delta W_{t} = W_{t-1} + eta (v_t - W k_t) k_t^T via fused GEMM.
  3. Despeja a matriz de memória associativa W final de volta à VRAM em um único kernel launch.

No macOS ou ambientes sem Triton/CUDA, possui fallback automático e transparente
para a implementação PyTorch em bloco triangular (chunked_delta_rule_update).
"""

from __future__ import annotations

import math
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
    def _triton_delta_rule_fwd_kernel(
        W_ptr,          # [B, H, D, D]
        K_ptr,          # [B, H, T, D]
        V_ptr,          # [B, H, T, D]
        W_out_ptr,      # [B, H, D, D]
        decay,
        stride_wb, stride_wh, stride_wd1, stride_wd2,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        T: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Kernel Fused Triton para atualização associativa Delta Rule no SRAM."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        # Offsets
        w_offset = pid_b * stride_wb + pid_h * stride_wh
        k_offset = pid_b * stride_kb + pid_h * stride_kh
        v_offset = pid_b * stride_vb + pid_h * stride_vh

        d_row = tl.arange(0, BLOCK_D)
        d_col = tl.arange(0, BLOCK_D)
        mask_d = (d_row[:, None] < D) & (d_col[None, :] < D)

        # 1. Carrega matriz de memória inicial W_0 no SRAM
        w_ptrs = W_ptr + w_offset + d_row[:, None] * stride_wd1 + d_col[None, :] * stride_wd2
        W = tl.load(w_ptrs, mask=mask_d, other=0.0)

        # 2. Varre o chunk de tokens totalmente nos registradores locais (zero kernel launches)
        for t in range(T):
            k_ptrs = K_ptr + k_offset + t * stride_kt + d_col * stride_kd
            v_ptrs = V_ptr + v_offset + t * stride_vt + d_row * stride_vd

            mask_k = d_col < D
            mask_v = d_row < D

            k_t = tl.load(k_ptrs, mask=mask_k, other=0.0)
            v_t = tl.load(v_ptrs, mask=mask_v, other=0.0)

            # Normalização de k_t
            k_norm_sq = tl.sum(k_t * k_t) + 1e-6
            inv_norm = 1.0 / tl.sqrt(k_norm_sq)
            k_norm = k_t * inv_norm

            # Decaimento temporal
            W = W * decay

            # Predição: v_pred = W @ k_norm
            v_pred = tl.sum(W * k_norm[None, :], axis=1)
            err = v_t - v_pred

            # Regra Delta: W += err * k_norm^T
            dW = err[:, None] * k_norm[None, :]
            W = W + dW

        # 3. Salva a matriz final consolidada de volta à VRAM
        w_out_ptrs = W_out_ptr + w_offset + d_row[:, None] * stride_wd1 + d_col[None, :] * stride_wd2
        tl.store(w_out_ptrs, W, mask=mask_d)


def triton_chunk_delta_rule(
    W: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    decay: float = 0.999,
) -> torch.Tensor:
    """Executa a atualização Delta Rule compilada via Triton na GPU (ou fallback PyTorch).

    Args:
        W: Tensor de memória associativa [B, H, D, D].
        keys: Chaves de consulta do bloco [B, H, T, D].
        values: Valores esperados do bloco [B, H, T, D].
        decay: Fator de decaimento por token.

    Returns:
        W_new: Matriz associativa atualizada [B, H, D, D].
    """
    if not HAS_TRITON or not keys.is_cuda:
        # Fallback ultra-rápido via PyTorch Triangular Chunk Solver
        from scar_cache.projections import chunked_delta_rule_update
        W_new, _ = chunked_delta_rule_update(W, keys, values, decay=decay)
        return W_new

    B, H, T, D = keys.shape
    assert W.shape == (B, H, D, D)
    W_out = torch.empty_like(W)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (B, H)

    _triton_delta_rule_fwd_kernel[grid](
        W, keys, values, W_out,
        decay,
        W.stride(0), W.stride(1), W.stride(2), W.stride(3),
        keys.stride(0), keys.stride(1), keys.stride(2), keys.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        T=T, D=D, BLOCK_D=BLOCK_D,
    )
    return W_out
