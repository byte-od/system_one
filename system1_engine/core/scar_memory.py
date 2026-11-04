"""Memória de Cicatriz de Gradiente (Gradient Scar Memory) para o Sistema 1.

Substitui buffers episódicos infinitos de centenas de megabytes por representações
vetorizadas compactas de posto rápido e tamanho rigorosamente constante O(1).

Implementa as três abordagens proprietárias:
  1. mode="fast_weights" (Mecanismo 1):
     Matriz associativa de cicatrizes W_scar in R^(D x D) acumulada via produto externo
     de gradientes de vitória ponderados pela surpresa de Shannon. Modulação direta
     de slots em tempo zero de backward pass.
  2. mode="dual_gradient" (Mecanismo 2):
     Vetor acumulado g_alvo* in R^D de direções de sucesso. Avalia gradiente local
     g_local = grad_z V(z) em tempo de decisão e computa produto cruzado / cosseno.
  3. mode="hybrid" (Opção Híbrida v2 Refinada):
     Combina W_scar com g_alvo* e z_alvo*. Modula o espaço de slots via W_scar condicional
     à similaridade com o manifold de vitórias (Gated Activation) e calcula a
     ressonância escalar diretamente a partir de Delta z sem exigir backward no rollout.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientScarMemory(nn.Module):
    """Módulo Universal de Memória de Cicatriz de Gradiente (O(1) Space)."""

    VALID_MODES = ("none", "fast_weights", "dual_gradient", "hybrid")

    def __init__(
        self,
        slot_dim: int = 32,
        num_slots: int = 8,
        mode: str = "fast_weights",
        decay: float = 0.995,
        scale: float = 0.15,
        resonance_beta: float = 0.20,
    ) -> None:
        super().__init__()
        assert mode in self.VALID_MODES, f"Modo desconhecido {mode}. Válidos: {self.VALID_MODES}"
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.mode = mode
        self.decay = float(decay)
        self.scale = float(scale)
        self.resonance_beta = float(resonance_beta)

        # Buffers constantes O(1)
        self.register_buffer("W_scar", torch.zeros(slot_dim, slot_dim))
        self.register_buffer("g_target", torch.zeros(slot_dim))
        self.register_buffer("z_target", torch.zeros(slot_dim))
        self.register_buffer("scars_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_surprise_accum", torch.tensor(0.0, dtype=torch.float32))

    @property
    def memory_bytes(self) -> int:
        """Retorna o consumo exato de bytes na memória RAM."""
        if self.mode == "none":
            return 0
        if self.mode == "fast_weights":
            return int(self.W_scar.numel() * 4)  # 4.096 bytes (4.1 KB)
        if self.mode == "dual_gradient":
            return int(self.g_target.numel() * 4)  # 128 bytes
        # hybrid v2: W_scar (4096) + g_target (128) + z_target (128) = 4352 bytes (4.25 KB)
        return int((self.W_scar.numel() + self.g_target.numel() + self.z_target.numel()) * 4)

    @property
    def has_fast_weights(self) -> bool:
        return self.mode in ("fast_weights", "hybrid")

    @property
    def has_resonance(self) -> bool:
        return self.mode in ("dual_gradient", "hybrid")

    def record_scar(
        self,
        slots: torch.Tensor,
        grad_slots: torch.Tensor,
        surprise: float,
    ) -> None:
        """Acumula a cicatriz de uma vitória com ponderação por surpresa de Shannon.
        
        Solução Fricção A: Usa Produto Tensorial Direto (G^T @ Z) entre slots para
        eliminar o cancelamento destrutivo de gradientes opostos e projeção construtiva.
        """
        if self.mode == "none":
            return

        with torch.no_grad():
            # slots: [..., num_slots, slot_dim] -> achata batch/tempo se presentes
            if slots.dim() > 2:
                z_slots = slots.reshape(-1, self.num_slots, self.slot_dim).mean(dim=0)
                g_slots = grad_slots.reshape(-1, self.num_slots, self.slot_dim).mean(dim=0)
            elif slots.dim() == 2:
                z_slots = slots
                g_slots = grad_slots
            else:
                z_slots = slots.unsqueeze(0)
                g_slots = grad_slots.unsqueeze(0)

            # Normalização L2 por slot: shape [num_slots, slot_dim]
            norm_z = z_slots / (torch.norm(z_slots, dim=-1, keepdim=True) + 1e-6)
            norm_g = g_slots / (torch.norm(g_slots, dim=-1, keepdim=True) + 1e-6)

            # Ponderação pela surpresa de Shannon (bits de informação)
            weight = float(min(max(surprise, 0.1), 5.0) * 0.05)

            if self.has_fast_weights:
                # Solução Fricção A: dW = (1/K) * G^T @ Z
                # norm_g.t() [32, 8] @ norm_z [8, 32] -> dW [32, 32]
                # Preserva o contraste relacional individual de cada slot sem cancelamento mútuo!
                dW = torch.matmul(norm_g.t(), norm_z) / float(self.num_slots)
                self.W_scar.mul_(self.decay).add_(dW, alpha=weight)

            if self.has_resonance:
                # Projeção construtiva (estilo PCGrad) dos slots para o vetor consenso g_target
                norm_current_t = self.g_target / (torch.norm(self.g_target) + 1e-6)
                g_consensus = torch.zeros(self.slot_dim, device=slots.device, dtype=torch.float32)
                for k in range(self.num_slots):
                    gk = norm_g[k]
                    if int(self.scars_count.item()) > 0:
                        cos_sim = torch.dot(gk, norm_current_t)
                        if cos_sim < 0.0:
                            # Projeta ortogonalmente para não subtrair da direção consensual de vitória
                            gk = gk - cos_sim * norm_current_t
                    g_consensus.add_(gk)
                g_consensus = g_consensus / float(self.num_slots)
                norm_g_c = g_consensus / (torch.norm(g_consensus) + 1e-6)
                self.g_target.mul_(self.decay).add_(norm_g_c, alpha=weight)

            # Solução Fricção B: Atualiza o centróide do manifold de estado de vitória z_target
            z_consensus = norm_z.mean(dim=0)
            norm_z_c = z_consensus / (torch.norm(z_consensus) + 1e-6)
            self.z_target.mul_(self.decay).add_(norm_z_c, alpha=weight)

            self.scars_count.add_(1)
            self.total_surprise_accum.add_(float(surprise))

    def modulate_slots(self, slots: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Aplica modulação Fast-Weights condicional ao reconhecimento do manifold de vitória.
        
        Solução Fricção B: Se o estado atual não apresenta afinidade com o manifold de vitórias
        (afinidade <= 0), a injeção é 0 (slots permanecem 100% puros e blindados contra OOD).
        Se afinidade > 0, injeta Fast-Weights proporcionalmente à proximidade.
        """
        if not self.has_fast_weights or int(self.scars_count.item()) == 0:
            return slots, None

        # Projeção linear rápida O(1): slots @ W_scar^T -> [..., num_slots, slot_dim]
        delta = torch.matmul(slots, self.W_scar.t())

        # Gating de Manifold (Fricção B): Afinidade com o centróide z_target
        if self.mode == "hybrid" and torch.norm(self.z_target).item() > 1e-6:
            norm_z_ref = self.z_target / (torch.norm(self.z_target) + 1e-6)
            norm_s = slots / (torch.norm(slots, dim=-1, keepdim=True) + 1e-6)
            # Similaridade cosseno por slot: [..., num_slots]
            affinity = torch.matmul(norm_s, norm_z_ref)
            # Gate ReLU suave: só ativa quando a afinidade for positiva (> 0)
            gate = torch.clamp(affinity, min=0.0).unsqueeze(-1)  # [..., num_slots, 1]
            slots_guided = slots + self.scale * gate * delta
        else:
            slots_guided = slots + self.scale * delta

        return slots_guided, delta

    def compute_resonance_hybrid(self, delta_slots: torch.Tensor) -> torch.Tensor:
        """Calcula ressonância escalar analítica sem necessidade de autograd no rollout."""
        if not self.has_resonance or int(self.scars_count.item()) == 0:
            return torch.zeros(delta_slots.shape[:-1], device=delta_slots.device)

        norm_target = self.g_target / (torch.norm(self.g_target) + 1e-6)
        norm_delta = delta_slots / (torch.norm(delta_slots, dim=-1, keepdim=True) + 1e-6)
        # Produto escalar ao longo da última dimensão
        cos_res = torch.matmul(norm_delta, norm_target)
        return cos_res

    def modulate_logits(
        self,
        logits: torch.Tensor,
        resonance: torch.Tensor,
    ) -> torch.Tensor:
        """Modula a distribuição de logits com base no sinal de ressonância direcional (Intuição Sistema 1)."""
        if not self.has_resonance or int(self.scars_count.item()) == 0:
            return logits

        # Se resonance > 0: aumenta a nitidez (temperatura mais fria / certeza sem hesitação)
        # Se resonance < 0: aumenta a incerteza (temperatura mais quente / estimula exploração)
        res_scalar = resonance.mean() if resonance.dim() > 0 else resonance
        clamped_res = torch.clamp(res_scalar, -1.0, 1.0)
        
        # Modulação suave de escala
        scale_factor = 1.0 + self.resonance_beta * clamped_res
        return logits * scale_factor
