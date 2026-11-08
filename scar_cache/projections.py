"""Módulo de projeções geométricas e álgebra Hebbiana para Scar-Cache.

Implementa:
1. Projeção Ortogonal de Conflito Hebbiano (PCGrad na Cicatriz):
   Impede que gradientes em direções opostas de manobras distintas canibalizem
   a matriz associativa histórica.
2. Produto Externo Hebbiano Estável:
   Computa G^T · Z preservando dimensionalidades e normalização de slots.
"""

from __future__ import annotations

import torch


def project_conflicting_gradient(
    g_new: torch.Tensor,
    g_target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Projeta o gradiente novo no hiperplano ortogonal se houver conflito direcional.

    Matemática:
        Se <g_new, g_target> < 0:
            g* = g_new - (<g_new, g_target> / ||g_target||^2) * g_target
        Caso contrário:
            g* = g_new

    Garante formalmente que <g*, g_target> >= 0, eliminando cancelamentos destrutivos.
    """
    norm_t_sq = torch.sum(g_target * g_target)
    if norm_t_sq < eps:
        return g_new

    dot = torch.sum(g_new * g_target)
    if dot < 0.0:
        # Projeção no plano ortogonal
        proj = (dot / (norm_t_sq + eps)) * g_target
        return g_new - proj
    return g_new


def hebbian_outer_product(
    norm_g: torch.Tensor,
    norm_z: torch.Tensor,
    num_slots: int = 1,
) -> torch.Tensor:
    """Calcula o produto externo Hebbiano associativo entre gradientes e representações latentes.

    Args:
        norm_g: Gradientes normalizados [..., slot_dim] ou [num_slots, slot_dim].
        norm_z: Representações normalizadas [..., slot_dim] ou [num_slots, slot_dim].
        num_slots: Número de slots relacionais / cabeças.

    Returns:
        Matriz associativa dW de dimensão [slot_dim, slot_dim].
    """
    if norm_g.dim() == 1 and norm_z.dim() == 1:
        return torch.outer(norm_g, norm_z)
    elif norm_g.dim() == 2 and norm_z.dim() == 2:
        # norm_g.t() [slot_dim, num_slots] @ norm_z [num_slots, slot_dim] -> [slot_dim, slot_dim]
        return torch.matmul(norm_g.t(), norm_z) / float(max(1, num_slots))
    else:
        g_flat = norm_g.reshape(-1, norm_g.shape[-1])
        z_flat = norm_z.reshape(-1, norm_z.shape[-1])
        return torch.matmul(g_flat.t(), z_flat) / float(max(1, g_flat.shape[0]))
