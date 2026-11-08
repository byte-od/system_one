"""MultiSlotSCache: Tensorized Multi-Slot O(1) Associative Memory (34 KB in L1).

Implementa os 5 pilares para substituição de KV Cache:
1. Tensorização Multi-Slot (8 x 32 x 32) com PCGrad Local por Slot.
2. Regra Delta de Sobrescrita Exata (Widrow-Hoff Anti-Ghosting).
3. Roteamento Competitivo por Slot (Slot Attention / Ingestion).
4. Decaimento Temporal Heterogêneo Multi-Escala (γ diferenciado por slot).
5. Inferência Recorrente O(1) em < 5 µs e modo de acumulação sequencial.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.projections import (
    delta_rule_update,
    hebbian_outer_product,
    project_conflicting_gradients_per_slot,
)


class MultiSlotSCache(nn.Module):
    """Módulo Tensorial Multi-Slot de Memória Associativa de Gradiente O(1).

    Configuração Padrão:
        num_slots = 8
        slot_dim = 32
        Memória total em float32:
            W_scar: [8, 32, 32] = 32 KB
            g_target: [8, 32] = 1 KB
            z_target: [8, 32] = 1 KB
            Total = 34 KB (100% no Cache L1)
    """

    def __init__(
        self,
        slot_dim: int = 32,
        num_slots: int = 8,
        scale: float = 0.20,
        resonance_beta: float = 0.25,
        base_decay: float = 0.995,
        use_delta_rule: bool = True,
        multi_scale_decay: bool = True,
    ) -> None:
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.scale = float(scale)
        self.resonance_beta = float(resonance_beta)
        self.use_delta_rule = bool(use_delta_rule)

        # 1. Buffers Estritamente O(1) no L1 Cache
        self.register_buffer(
            "W_scar",
            torch.zeros(self.num_slots, self.slot_dim, self.slot_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "g_target",
            torch.zeros(self.num_slots, self.slot_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "z_target",
            torch.zeros(self.num_slots, self.slot_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "scars_count",
            torch.zeros(self.num_slots, dtype=torch.long),
        )

        # 2. Decaimento Temporal Heterogêneo Multi-Escala (Pilar 4)
        if multi_scale_decay:
            # Escalonamento geométrico de curto prazo (0.50) a quase permanente (0.9999)
            decays = torch.tensor(
                [
                    0.5000,  # Slot 0: Sintaxe imediata / últimos 3-5 tokens
                    0.7500,  # Slot 1: Frase curta
                    0.9000,  # Slot 2: Oração
                    0.9500,  # Slot 3: Parágrafo
                    0.9800,  # Slot 4: Seção
                    0.9950,  # Slot 5: Contexto local
                    0.9990,  # Slot 6: Contexto global
                    0.9999,  # Slot 7: Meta da conversa / Persona / System Prompt
                ][: self.num_slots],
                dtype=torch.float32,
            )
            # Se num_slots != 8, interpola logaritmicamente
            if self.num_slots != 8:
                decays = torch.exp(
                    torch.linspace(math.log(0.5), math.log(0.9999), self.num_slots)
                )
        else:
            decays = torch.full((self.num_slots,), float(base_decay), dtype=torch.float32)

        self.register_buffer("decay_rates", decays)

        # 3. Consultas de Roteamento Competitivo (Pilar 3)
        # Inicializadas ortogonalmente para incentivar especialização imediata
        router_init = torch.empty(self.num_slots, self.slot_dim)
        nn.init.orthogonal_(router_init)
        self.slot_queries = nn.Parameter(router_init)

    @property
    def memory_bytes(self) -> int:
        """Retorna o consumo estrito de bytes na memória RAM."""
        w_bytes = self.W_scar.numel() * 4
        g_bytes = self.g_target.numel() * 4
        z_bytes = self.z_target.numel() * 4
        return int(w_bytes + g_bytes + z_bytes)

    def reset_scars(self) -> None:
        """Limpa todos os buffers tensoriais para reinício Tabula Rasa."""
        with torch.no_grad():
            self.W_scar.zero_()
            self.g_target.zero_()
            self.z_target.zero_()
            self.scars_count.zero_()

    def route_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roteamento Competitivo por Slot (Pilar 3).

        Args:
            x: Tensor de entrada [..., slot_dim] (ex: [B, D] ou [B, T, D]).

        Returns:
            (routing_weights, projected_slots):
                routing_weights: Afinidade softmax [..., num_slots].
                projected_slots: Entrada replicada/projetada por slot [..., num_slots, slot_dim].
        """
        # score_k = (x · q_k) / sqrt(D)
        # x: [..., D], slot_queries: [K, D] -> scores: [..., K]
        scale = 1.0 / math.sqrt(self.slot_dim)
        scores = torch.matmul(x, self.slot_queries.t()) * scale
        weights = F.softmax(scores, dim=-1)

        # Repete x para os slots: [..., K, D]
        if x.dim() == 2:
            slots = x.unsqueeze(1).repeat(1, self.num_slots, 1)
        elif x.dim() == 3:
            slots = x.unsqueeze(2).repeat(1, 1, self.num_slots, 1)
        else:
            slots = x.unsqueeze(-2).repeat(*([1] * (x.dim() - 1)), self.num_slots, 1)

        return weights, slots

    def modulate(
        self,
        x: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Leitura e Modulação Recorrente O(1) em < 5 µs (Pilar 1 e 5).

        Args:
            x: Entrada [Batch, slot_dim] ou [Batch, num_slots, slot_dim].
            routing_weights: Pesos opcionais de despacho [Batch, num_slots].

        Returns:
            (x_guided, deltas, affinities):
                x_guided: Saída modulada com Fast-Weights.
                deltas: Deltas injetados por slot.
                affinities: Similaridades de cosseno com centróides z_target.
        """
        if int(self.scars_count.sum().item()) == 0:
            zeros_aff = torch.zeros(
                (*x.shape[:-1], self.num_slots), device=x.device, dtype=x.dtype
            )
            return x, torch.zeros_like(x), zeros_aff

        if x.dim() == 2 and x.shape[-1] == self.slot_dim:
            if routing_weights is None:
                routing_weights, slots = self.route_tokens(x)
            else:
                slots = x.unsqueeze(1).repeat(1, self.num_slots, 1)
        else:
            slots = x
            if routing_weights is None:
                routing_weights = torch.full(
                    (*x.shape[:-2], self.num_slots),
                    1.0 / float(self.num_slots),
                    device=x.device,
                    dtype=x.dtype,
                )

        # Projeção tensorial paralela em batch: slots @ W_scar^T
        # slots: [B, K, D], W_scar: [K, D, D] -> deltas: [B, K, D]
        # Para cada slot k: delta[:, k, :] = slots[:, k, :] @ W_scar[k, :, :].t()
        deltas = torch.einsum("bki,kij->bkj", slots, self.W_scar)

        # Gating de Manifold Imunológico por Slot:
        # affinity_k = cos(slots_k, z_target_k)
        norm_s = slots / (torch.norm(slots, dim=-1, keepdim=True) + 1e-6)
        norm_z = self.z_target / (torch.norm(self.z_target, dim=-1, keepdim=True) + 1e-6)
        # affinities: [B, K]
        affinities = torch.einsum("bki,ki->bk", norm_s, norm_z)

        # Gate ReLU suave por slot: max(0, affinity_k)
        gates = torch.clamp(affinities, min=0.0).unsqueeze(-1)  # [B, K, 1]

        # Injeção ponderada por roteamento competitivo:
        # slot_delta = routing_weights * gates * deltas
        w_expanded = routing_weights.unsqueeze(-1)  # [B, K, 1]
        weighted_delta = w_expanded * gates * deltas  # [B, K, D]

        if x.dim() == 2:
            # Recombina a contribuição de todos os 8 slots no vetor original [B, D]
            total_injection = weighted_delta.sum(dim=1)  # [B, D]
            x_guided = x + self.scale * total_injection
        else:
            x_guided = slots + self.scale * weighted_delta

        return x_guided, deltas, affinities

    def record_gradient_scar(
        self,
        slots: torch.Tensor,
        grad_slots: torch.Tensor,
        surprise: float = 1.0,
        learning_rate: float = 1.0,
        orthogonal_projection: bool = True,
    ) -> None:
        """Gravação de Cicatriz de Gradiente com PCGrad Local por Slot (Pilar 1 e 2).

        Args:
            slots: Ativações por slot [..., num_slots, slot_dim].
            grad_slots: Gradientes de vitória por slot [..., num_slots, slot_dim].
            surprise: Surpresa temporal escalar (Shannon + Bellman).
            learning_rate: Taxa de aprendizado para a regra delta.
            orthogonal_projection: Se True, aplica PCGrad local em cada slot.
        """
        if float(surprise) <= 1e-6:
            return

        with torch.no_grad():
            s = slots.reshape(-1, self.num_slots, self.slot_dim).mean(dim=0)  # [K, D]
            g = grad_slots.reshape(-1, self.num_slots, self.slot_dim).mean(dim=0)  # [K, D]

            norm_s = s / (torch.norm(s, dim=-1, keepdim=True) + 1e-6)
            norm_g = g / (torch.norm(g, dim=-1, keepdim=True) + 1e-6)

            # PCGrad Cirúrgico por Slot (Pilar 1)
            if orthogonal_projection:
                norm_g = project_conflicting_gradients_per_slot(norm_g, self.g_target)

            # Fator de surpresa adaptativo
            surp_factor = min(max(float(surprise), 0.05), 5.0) / 2.5
            eta_base = 0.05 * surp_factor

            for k in range(self.num_slots):
                is_first = int(self.scars_count[k].item()) == 0
                eta = 1.0 if is_first else min(0.10, eta_base)
                decay = 0.0 if is_first else float(self.decay_rates[k].item())

                if self.use_delta_rule and not is_first:
                    # Pilar 2: Regra Delta (v_k - W_k @ s_k) s_k^T
                    # target_v é norm_g[k] (direção de descida para a vitória)
                    target_v = norm_g[k]
                    dW_k, _ = delta_rule_update(
                        self.W_scar[k],
                        keys=norm_s[k],
                        values=target_v,
                        learning_rate=learning_rate * eta,
                    )
                    self.W_scar[k].mul_(decay).add_(dW_k)
                else:
                    # Acumulação Hebbiana inicial: g_k ⊗ s_k^T
                    dW_k = torch.outer(norm_g[k], norm_s[k])
                    self.W_scar[k].mul_(decay).add_(dW_k, alpha=eta)

                # Atualiza centróide e bússola daquele slot
                self.g_target[k].mul_(decay).add_(norm_g[k], alpha=eta)
                self.z_target[k].mul_(decay).add_(norm_s[k], alpha=eta)
                self.scars_count[k].add_(1)

    def write_key_value(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        slot_index: Optional[int] = None,
        learning_rate: float = 1.0,
    ) -> torch.Tensor:
        """Grava diretamente pares Chave-Valor via Regra Delta com Sobrescrita Exata (Pilar 2).

        Garante que fatos antigos sejam apagados sem alucinações ou fantasmas aditivos.

        Args:
            keys: Chaves de consulta [slot_dim] ou [num_slots, slot_dim].
            values: Valores esperados [slot_dim] ou [num_slots, slot_dim].
            slot_index: Se fornecido, grava apenas no slot especificado.
            learning_rate: 1.0 garante convergência imediata para o novo valor.

        Returns:
            O erro residual antes da sobrescrita.
        """
        with torch.no_grad():
            if slot_index is not None:
                k = int(slot_index)
                keys_k = keys / (torch.norm(keys) + 1e-6)
                decay = float(self.decay_rates[k].item())
                # Aplica decaimento primeiro
                self.W_scar[k].mul_(decay)
                # Calcula a correção exata sobre a matriz decaída
                dW, err = delta_rule_update(
                    self.W_scar[k],
                    keys=keys_k,
                    values=values,
                    learning_rate=learning_rate,
                )
                self.W_scar[k].add_(dW)
                self.scars_count[k].add_(1)
                self.z_target[k].copy_(keys_k)
                return err
            else:
                if keys.dim() == 1:
                    keys = keys.unsqueeze(0).repeat(self.num_slots, 1)
                if values.dim() == 1:
                    values = values.unsqueeze(0).repeat(self.num_slots, 1)

                norm_k = keys / (torch.norm(keys, dim=-1, keepdim=True) + 1e-6)
                decays = self.decay_rates.view(self.num_slots, 1, 1)
                # Aplica decaimento primeiro
                self.W_scar.mul_(decays)
                # Calcula a correção delta sobre a matriz decaída
                dW, err = delta_rule_update(
                    self.W_scar,
                    keys=norm_k,
                    values=values,
                    learning_rate=learning_rate,
                )
                self.W_scar.add_(dW)
                self.scars_count.add_(1)
                self.z_target.copy_(norm_k)
                return err

    def read_key(
        self,
        key: torch.Tensor,
        slot_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Recupera o valor associado a uma chave na memória (Pilar 2).

        Args:
            key: Vetor de consulta [slot_dim].
            slot_index: Se fornecido, lê exclusivamente daquele slot.

        Returns:
            O valor recuperado da memória associativa.
        """
        with torch.no_grad():
            norm_k = key / (torch.norm(key, dim=-1, keepdim=True) + 1e-6)
            if slot_index is not None:
                return torch.matmul(self.W_scar[int(slot_index)], norm_k)
            else:
                # Retorna a média ou valores por slot: [K, D]
                # W_scar: [K, D, D], norm_k: [D] -> [K, D]
                return torch.einsum("kij,j->ki", self.W_scar, norm_k)
