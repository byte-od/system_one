"""SCache: Gradient Scar Memory Core (O(1) Constant Space).

Substrato de Memória Associativa de Pesos Rápidos (Fast-Weights) guiada por Gradientes
de Vitória, Gating de Manifold Imunológico e Modulação de Ressonância Direcional.

Opera em tamanho estritamente constante O(1) (~4.25 KB para dimensão 32)
eliminando buffers de replay episódico de gigabytes e inferência lenta com árvore de busca.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.projections import hebbian_outer_product, project_conflicting_gradient


class SCache(nn.Module):
    """Módulo Universal de Memória de Cicatriz de Gradiente (Gradient Scar Cache - O(1)).

    Atributos Principais:
        W_scar: Matriz associativa Hebbiana de Fast-Weights [slot_dim, slot_dim].
        z_target: Vetor centróide geométrico do território de vitórias [slot_dim].
        g_target: Bússola consensual de direção de gradientes [slot_dim].
        scars_count: Contador total de cicatrizes consolidadas.
        total_surprise_accum: Acumulador de surpresa temporal (Shannon + Bellman).
    """

    VALID_MODES = ("none", "fast_weights", "dual_gradient", "hybrid")

    def __init__(
        self,
        slot_dim: int = 32,
        num_slots: int = 8,
        mode: str = "hybrid",
        decay: float = 0.995,
        scale: float = 0.15,
        resonance_beta: float = 0.20,
    ) -> None:
        super().__init__()
        assert mode in self.VALID_MODES, f"Modo '{mode}' inválido. Válidos: {self.VALID_MODES}"
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.mode = str(mode).lower()
        self.decay = float(decay)
        self.scale = float(scale)
        self.resonance_beta = float(resonance_beta)

        self._batch_scale_override: Optional[torch.Tensor] = None
        self._batch_beta_override: Optional[torch.Tensor] = None
        self._last_affinity: Optional[torch.Tensor] = None

        # Buffers estritamente constantes O(1)
        self.register_buffer("W_scar", torch.zeros(self.slot_dim, self.slot_dim, dtype=torch.float32))
        self.register_buffer("g_target", torch.zeros(self.slot_dim, dtype=torch.float32))
        self.register_buffer("z_target", torch.zeros(self.slot_dim, dtype=torch.float32))
        self.register_buffer("scars_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_surprise_accum", torch.tensor(0.0, dtype=torch.float32))

    def reset_scars(self) -> None:
        """Limpa integralmente a memória de cicatriz para reinício Tabula Rasa limpo."""
        with torch.no_grad():
            self.W_scar.zero_()
            self.g_target.zero_()
            self.z_target.zero_()
            self.scars_count.zero_()
            self.total_surprise_accum.zero_()
            self._last_affinity = None

    @contextmanager
    def batch_modulation_scope(
        self,
        scale_vec: Optional[torch.Tensor] = None,
        beta_vec: Optional[torch.Tensor] = None,
    ) -> Iterator[None]:
        """Aplica parâmetros estratificados de injeção por elemento do batch (ex: PPO vs Batedores)."""
        prev_scale = self._batch_scale_override
        prev_beta = self._batch_beta_override
        self._batch_scale_override = scale_vec
        self._batch_beta_override = beta_vec
        try:
            yield
        finally:
            self._batch_scale_override = prev_scale
            self._batch_beta_override = prev_beta

    @property
    def memory_bytes(self) -> int:
        """Retorna o consumo estrito de bytes na memória RAM."""
        if self.mode == "none":
            return 0
        if self.mode == "fast_weights":
            return int(self.W_scar.numel() * 4)
        if self.mode == "dual_gradient":
            return int(self.g_target.numel() * 4)
        # Modo hybrid: W_scar (dim^2) + g_target (dim) + z_target (dim)
        return int((self.W_scar.numel() + self.g_target.numel() + self.z_target.numel()) * 4)

    @property
    def has_fast_weights(self) -> bool:
        return self.mode in ("fast_weights", "hybrid")

    @property
    def has_resonance(self) -> bool:
        return self.mode in ("dual_gradient", "hybrid")

    @property
    def scar_norm(self) -> float:
        """Retorna a norma de Frobenius da matriz associativa W_scar."""
        if self.has_fast_weights:
            return float(torch.norm(self.W_scar).item())
        return float(torch.norm(self.g_target).item())

    def record_scar(
        self,
        slots: torch.Tensor,
        grad_slots: torch.Tensor,
        surprise: float = 1.0,
        orthogonal_projection: bool = True,
    ) -> None:
        """Acumula a cicatriz sináptica de uma vitória inédita com ponderação de surpresa.

        Args:
            slots: Representação latente dos slots no episódio de vitória [..., num_slots, slot_dim].
            grad_slots: Gradiente de perda nos slots de vitória [..., num_slots, slot_dim].
            surprise: Peso escalar de novidade temporal (Shannon + Bellman).
            orthogonal_projection: Se True, aplica PCGrad na direção consensual g_target.
        """
        if self.mode == "none" or float(surprise) <= 1e-6:
            return

        with torch.no_grad():
            if slots.dim() > 2:
                z_slots = slots.reshape(-1, self.num_slots, self.slot_dim).mean(dim=0)
                g_slots = grad_slots.reshape(-1, self.num_slots, self.slot_dim).mean(dim=0)
            elif slots.dim() == 2 and slots.shape[0] == self.num_slots:
                z_slots = slots
                g_slots = grad_slots
            else:
                z_slots = slots.reshape(-1, self.slot_dim)
                g_slots = grad_slots.reshape(-1, self.slot_dim)

            # Normalização L2 por slot
            norm_z = z_slots / (torch.norm(z_slots, dim=-1, keepdim=True) + 1e-6)
            norm_g = g_slots / (torch.norm(g_slots, dim=-1, keepdim=True) + 1e-6)

            # Taxa adaptativa do EMA Convexo
            base_rate = max(5e-4, min(0.05, 1.0 - float(self.decay)))
            surp_factor = min(max(float(surprise), 0.05), 5.0) / 2.5
            is_first = int(self.scars_count.item()) == 0
            eta = 1.0 if is_first else float(min(0.05, base_rate * surp_factor))

            if self.has_fast_weights:
                dW = hebbian_outer_product(norm_g, norm_z, num_slots=self.num_slots)
                self.W_scar.mul_(1.0 - eta).add_(dW, alpha=eta)

            if self.has_resonance:
                g_consensus = torch.zeros(self.slot_dim, device=slots.device, dtype=torch.float32)
                for k in range(norm_g.shape[0]):
                    gk = norm_g[k]
                    if orthogonal_projection and not is_first:
                        gk = project_conflicting_gradient(gk, self.g_target)
                    g_consensus.add_(gk)
                g_consensus = g_consensus / float(max(1, norm_g.shape[0]))
                norm_g_c = g_consensus / (torch.norm(g_consensus) + 1e-6)
                self.g_target.mul_(1.0 - eta).add_(norm_g_c, alpha=eta)

            # Atualiza o centróide do manifold z_target
            z_consensus = norm_z.mean(dim=0)
            norm_z_c = z_consensus / (torch.norm(z_consensus) + 1e-6)
            self.z_target.mul_(1.0 - eta).add_(norm_z_c, alpha=eta)

            self.scars_count.add_(1)
            self.total_surprise_accum.add_(float(surprise))

    def modulate(
        self,
        slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Aplica a modulação Hebbiana Fast-Weights condicionada ao Gating de Manifold.

        Args:
            slots: Tensor de entrada [..., slot_dim].

        Returns:
            slots_guided: Tensor modulado com Fast-Weights.
            delta: O delta de injeção z · W^T.
            affinity: Similaridade de cosseno com o território de vitórias z_target.
        """
        if not self.has_fast_weights or int(self.scars_count.item()) == 0:
            self._last_affinity = None
            return slots, None, None

        # Projeção linear rápida O(1): slots @ W_scar^T
        delta = torch.matmul(slots, self.W_scar.t())

        eff_scale: Any = self.scale
        if (
            self._batch_scale_override is not None
            and slots.dim() >= 2
            and slots.shape[0] == self._batch_scale_override.shape[0]
        ):
            eff_scale = self._batch_scale_override.to(device=slots.device, dtype=slots.dtype).view(
                slots.shape[0], *([1] * (slots.dim() - 1))
            )

        # Gating de Manifold Imunológico
        affinity: Optional[torch.Tensor] = None
        if self.mode == "hybrid" and torch.norm(self.z_target).item() > 1e-6:
            norm_z_ref = self.z_target / (torch.norm(self.z_target) + 1e-6)
            norm_s = slots / (torch.norm(slots, dim=-1, keepdim=True) + 1e-6)
            affinity = torch.matmul(norm_s, norm_z_ref)
            self._last_affinity = affinity
            # Gate ReLU suave: só injeta se afinidade for positiva (> 0)
            gate = torch.clamp(affinity, min=0.0).unsqueeze(-1)
            slots_guided = slots + eff_scale * gate * delta
        else:
            self._last_affinity = None
            slots_guided = slots + eff_scale * delta

        return slots_guided, delta, affinity

    def compute_resonance(self, delta_slots: torch.Tensor) -> torch.Tensor:
        """Calcula a similaridade direcional analítica entre o passo atual e a bússola g_target."""
        if not self.has_resonance or int(self.scars_count.item()) == 0 or delta_slots is None:
            return torch.zeros(delta_slots.shape[:-1], device=delta_slots.device)

        norm_target = self.g_target / (torch.norm(self.g_target) + 1e-6)
        norm_delta = delta_slots / (torch.norm(delta_slots, dim=-1, keepdim=True) + 1e-6)
        return torch.matmul(norm_delta, norm_target)

    def modulate_logits(
        self,
        logits: torch.Tensor,
        resonance: torch.Tensor,
    ) -> torch.Tensor:
        """Modula a nitidez/temperatura da distribuição de logits com base na ressonância."""
        if not self.has_resonance or int(self.scars_count.item()) == 0 or resonance is None:
            return logits

        if resonance.dim() > 0:
            res_per_sample = resonance.mean(dim=-1, keepdim=True)
            if logits.dim() == 1:
                res_per_sample = res_per_sample.squeeze()
            elif logits.dim() == 2 and res_per_sample.dim() == 3:
                res_per_sample = res_per_sample.squeeze(1)
            while res_per_sample.dim() < logits.dim():
                res_per_sample = res_per_sample.unsqueeze(-1)
        else:
            res_per_sample = resonance
        clamped_res = torch.clamp(res_per_sample, -1.0, 1.0)

        eff_beta: Any = self.resonance_beta
        if (
            self._batch_beta_override is not None
            and logits.dim() >= 2
            and logits.shape[0] == self._batch_beta_override.shape[0]
        ):
            eff_beta = self._batch_beta_override.to(device=logits.device, dtype=logits.dtype).view(
                logits.shape[0], *([1] * (logits.dim() - 1))
            )

        scale_factor = 1.0 + eff_beta * clamped_res
        return logits * scale_factor

    def modulate_scale(
        self,
        scale: torch.Tensor,
        resonance: torch.Tensor,
    ) -> torch.Tensor:
        """Modula a dispersão (sigma) de distribuições contínuas de forma simétrica."""
        if not self.has_resonance or int(self.scars_count.item()) == 0 or resonance is None:
            return scale

        if resonance.dim() == scale.dim() - 1 and resonance.shape == scale.shape[:-1]:
            res_per_sample = resonance.unsqueeze(-1)
        elif resonance.dim() > 0:
            res_per_sample = resonance.mean(dim=-1, keepdim=True)
            if scale.dim() == 1:
                res_per_sample = res_per_sample.squeeze()
            elif scale.dim() == 2 and res_per_sample.dim() == 3:
                res_per_sample = res_per_sample.squeeze(1)
            while res_per_sample.dim() < scale.dim():
                res_per_sample = res_per_sample.unsqueeze(-1)
        else:
            res_per_sample = resonance
        clamped_res = torch.clamp(res_per_sample, -1.0, 1.0)

        eff_beta: Any = self.resonance_beta
        if (
            self._batch_beta_override is not None
            and scale.dim() >= 2
            and scale.shape[0] == self._batch_beta_override.shape[0]
        ):
            eff_beta = self._batch_beta_override.to(device=scale.device, dtype=scale.dtype).view(
                scale.shape[0], *([1] * (scale.dim() - 1))
            )

        return (scale * torch.exp(-eff_beta * clamped_res)).clamp(min=1e-5)


# Alias de conveniência
GradientScarMemory = SCache
