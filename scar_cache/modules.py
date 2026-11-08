"""High-level PyTorch Modules integrated with SCache (Gradient Scar Memory).

Drop-in replacements e wrappers para injetar memória associativa O(1) diretamente
em camadas lineares, blocos de atenção e redes neurais pré-existentes.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.core import SCache


class ScarLinear(nn.Module):
    """Drop-in replacement para nn.Linear com memória de cicatriz de gradiente O(1).

    Aplica a modulação Hebbiana Fast-Weights e Gating de Manifold diretamente
    no espaço latente de entrada antes da projeção linear.

    Exemplo de uso:
        >>> layer = ScarLinear(in_features=64, out_features=10, mode="hybrid")
        >>> x = torch.randn(8, 64)
        >>> out = layer(x) # [8, 10]
        >>> # Ao obter um episódio de alta recompensa/vitória:
        >>> layer.record_scar(x, grad_x, surprise=1.0)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        slot_dim: Optional[int] = None,
        num_slots: int = 1,
        mode: str = "hybrid",
        decay: float = 0.995,
        scale: float = 0.15,
        resonance_beta: float = 0.20,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.slot_dim = int(slot_dim or in_features)
        self.num_slots = int(num_slots)

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # Se in_features != slot_dim, usa projeção adaptativa interna
        self.needs_projection = self.slot_dim != self.in_features
        if self.needs_projection:
            self.proj_in = nn.Linear(self.in_features, self.slot_dim, bias=False, device=device, dtype=dtype)
            self.proj_out = nn.Linear(self.slot_dim, self.in_features, bias=False, device=device, dtype=dtype)
        else:
            self.proj_in = None
            self.proj_out = None

        self.cache = SCache(
            slot_dim=self.slot_dim,
            num_slots=self.num_slots,
            mode=mode,
            decay=decay,
            scale=scale,
            resonance_beta=resonance_beta,
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        return_affinity: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward pass modulado por Fast-Weights.

        Args:
            x: Tensor de entrada [..., in_features].
            return_affinity: Se True, retorna (out, affinity_manifold).
        """
        if self.needs_projection:
            z = self.proj_in(x)
            z_guided, _, affinity = self.cache.modulate(z)
            x_guided = x + self.proj_out(z_guided - z)
        else:
            x_guided, _, affinity = self.cache.modulate(x)

        out = F.linear(x_guided, self.weight, self.bias)

        if return_affinity:
            return out, affinity
        return out

    def record_scar(
        self,
        slots: torch.Tensor,
        grad_slots: torch.Tensor,
        surprise: float = 1.0,
        orthogonal_projection: bool = True,
    ) -> None:
        """Registra a cicatriz sináptica no SCache encapsulado."""
        if self.needs_projection:
            with torch.no_grad():
                slots = self.proj_in(slots)
                grad_slots = self.proj_in(grad_slots)
        self.cache.record_scar(
            slots=slots,
            grad_slots=grad_slots,
            surprise=surprise,
            orthogonal_projection=orthogonal_projection,
        )

    def record_from_loss(
        self,
        loss: torch.Tensor,
        slots: torch.Tensor,
        surprise: float = 1.0,
        retain_graph: bool = True,
        orthogonal_projection: bool = True,
    ) -> None:
        """Calcula gradiente autônomo autograd e consolida cicatriz sem atualizar pesos lentos.

        Args:
            loss: Perda escalar do episódio ou critério de vitória.
            slots: Ativações que devem ser associadas à vitória (devem ter requires_grad=True).
            surprise: Peso escalar da surpresa temporal.
            retain_graph: Se True, preserva o grafo computacional para outros backward passes.
            orthogonal_projection: Se True, aplica PCGrad na direção consensual.
        """
        if not slots.requires_grad:
            raise ValueError("slots deve ter requires_grad=True para calcular grad_slots automaticamente.")
        grads = torch.autograd.grad(loss, slots, retain_graph=retain_graph, create_graph=False)[0]
        # Invertemos o sinal do gradiente para seguir a descida de gradiente (-grad) em direção ao mínimo
        grad_slots = -grads
        self.record_scar(
            slots=slots,
            grad_slots=grad_slots,
            surprise=surprise,
            orthogonal_projection=orthogonal_projection,
        )

    def reset_scars(self) -> None:
        self.cache.reset_scars()

    @property
    def scars_count(self) -> int:
        return int(self.cache.scars_count.item())


class ScarGatedLayer(nn.Module):
    """Wrapper universal para adicionar SCache na saída de qualquer módulo PyTorch.

    Envolve qualquer camada existente (ex: nn.Linear, MultiheadAttention, MLP block)
    interceptando suas ativações de saída com o mecanismo de cicatriz O(1).
    """

    def __init__(
        self,
        layer: nn.Module,
        slot_dim: int,
        num_slots: int = 1,
        mode: str = "hybrid",
        decay: float = 0.995,
        scale: float = 0.15,
        resonance_beta: float = 0.20,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)

        self.cache = SCache(
            slot_dim=self.slot_dim,
            num_slots=self.num_slots,
            mode=mode,
            decay=decay,
            scale=scale,
            resonance_beta=resonance_beta,
        )

    def forward(
        self,
        *args: Any,
        return_affinity: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        out = self.layer(*args, **kwargs)
        out_guided, _, affinity = self.cache.modulate(out)

        if return_affinity:
            return out_guided, affinity
        return out_guided

    def record_scar(
        self,
        slots: torch.Tensor,
        grad_slots: torch.Tensor,
        surprise: float = 1.0,
        orthogonal_projection: bool = True,
    ) -> None:
        self.cache.record_scar(
            slots=slots,
            grad_slots=grad_slots,
            surprise=surprise,
            orthogonal_projection=orthogonal_projection,
        )

    def reset_scars(self) -> None:
        self.cache.reset_scars()

    @property
    def scars_count(self) -> int:
        return int(self.cache.scars_count.item())

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)
