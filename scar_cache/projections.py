"""Módulo de projeções geométricas e álgebra Hebbiana para Scar-Cache.

Implementa:
1. Projeção Ortogonal de Conflito Hebbiano (PCGrad na Cicatriz):
   Impede que gradientes em direções opostas de manobras distintas canibalizem
   a matriz associativa histórica.
2. Produto Externo Hebbiano Estável:
   Computa G^T · Z preservando dimensionalidades e normalização de slots.
"""

from __future__ import annotations

from typing import Tuple
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


def project_conflicting_gradients_per_slot(
    g_news: torch.Tensor,
    g_targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Aplica PCGrad vetorizado de forma independente para cada slot K [K, D].

    Cada slot opera em sua própria jurisdição: o conflito no Slot 0 não afeta
    nem restringe o Slot 1.
    """
    assert g_news.shape == g_targets.shape, f"Dimensões incompatíveis: {g_news.shape} vs {g_targets.shape}"
    # dots: [K], norm_sq: [K]
    dots = torch.sum(g_news * g_targets, dim=-1, keepdim=True)
    norm_sq = torch.sum(g_targets * g_targets, dim=-1, keepdim=True)

    # Identifica onde há conflito (< 0) e onde a norma alvo é válida
    conflict_mask = (dots < 0.0) & (norm_sq > eps)

    # Projeção vetorizada no hiperplano ortogonal
    proj = (dots / (norm_sq + eps)) * g_targets
    return torch.where(conflict_mask, g_news - proj, g_news)


def delta_rule_update(
    current_W: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    learning_rate: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcula a atualização da Regra Delta (Widrow-Hoff) para sobrescrita exata sem fantasmas.

    Matemática:
        y_pred = W · k
        error = v - y_pred
        ΔW = β · (error ⊗ k^T) / (||k||^2 + eps)

    Args:
        current_W: Tensor de memória [K, D, D] ou [D, D].
        keys: Chaves de consulta normalizadas [K, D] ou [D].
        values: Valores alvo esperados [K, D] ou [D].
        learning_rate: Taxa de aprendizado da regra delta (1.0 = sobrescrita imediata).

    Returns:
        (delta_W, error): O tensor de ajuste de peso e o vetor de erro residual.
    """
    if current_W.dim() == 2:
        # Modo 2D simples [D, D]
        pred = torch.matmul(current_W, keys)
        error = values - pred
        k_norm_sq = torch.sum(keys * keys) + eps
        dW = (learning_rate / k_norm_sq) * torch.outer(error, keys)
        return dW, error
    elif current_W.dim() == 3:
        # Modo Multi-Slot Tensorial [K, D, D]
        # pred: [K, D] = bmm([K, D, D], [K, D, 1])
        pred = torch.bmm(current_W, keys.unsqueeze(-1)).squeeze(-1)
        error = values - pred
        k_norm_sq = torch.sum(keys * keys, dim=-1, keepdim=True).unsqueeze(-1) + eps
        # outer product por slot: [K, D, 1] x [K, 1, D] -> [K, D, D]
        dW = (learning_rate / k_norm_sq) * torch.bmm(error.unsqueeze(-1), keys.unsqueeze(1))
        return dW, error
    else:
        raise ValueError(f"Dimensão de W inválida: {current_W.shape}")


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
