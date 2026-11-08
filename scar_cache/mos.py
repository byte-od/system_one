"""Mixture of Scars (MoS): Compressão Radical de Pesos Estáticos para LLMs Gigantes.

Substitui camadas densas independentes e blocos MoE tradicionais (onde cada especialista
consome dezenas de megabytes de VRAM) por uma arquitetura de dois níveis:
  1. Base Universal Compartilhada (W_base): Matriz densa de alta capacidade compartilhada
     entre múltiplas camadas ou blocos de atenção.
  2. Especialistas em Cicatriz (Scar Experts): Micro-especialistas associativos de baixo rank
     (32 KB a 64 KB cada) residentes no L1/L2 Cache, ativados por roteamento competitivo
     Top-K e gating direcional de manifold max(0, cos(x, z_target)).
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScarExpert(nn.Module):
    """Micro-especialista de Cicatriz de baixo rank com gating de manifold direcional.

    Ocupa dezenas de kilobytes (em vez de dezenas de megabytes de uma MLP completa),
    armazenando uma perturbação direcional factorizada A @ B^T e um centróide semântico z_target.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        scale: float = 0.20,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.scale = float(scale)

        # Decomposição Low-Rank: [in_features, rank] e [rank, out_features]
        # Para in=1024, out=1024, rank=8: (1024*8 + 8*1024)*4 bytes = 65.5 KB!
        self.A = nn.Parameter(
            torch.empty((self.in_features, self.rank), device=device, dtype=dtype)
        )
        self.B = nn.Parameter(
            torch.empty((self.rank, self.out_features), device=device, dtype=dtype)
        )

        # Centróide Direcional Semântico (bússola de especialização no L1 Cache)
        self.register_buffer(
            "z_target",
            torch.zeros(self.in_features, device=device, dtype=dtype or torch.float32),
        )
        self.register_buffer("scars_count", torch.zeros(1, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Inicialização Kaiming / LoRA style: A com normal, B com zeros (início identity)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.z_target.zero_()
        self.scars_count.zero_()

    @property
    def memory_bytes(self) -> int:
        """Memória real consumida pelos tensores do especialista."""
        param_bytes = (self.A.numel() + self.B.numel()) * self.A.element_size()
        buffer_bytes = self.z_target.numel() * self.z_target.element_size()
        return int(param_bytes + buffer_bytes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projeção de cicatriz com modulação de manifold direcional."""
        # x: [..., in_features]
        # 1. Projeção Low-Rank: (x @ A) @ B
        delta = torch.matmul(torch.matmul(x, self.A), self.B) * self.scale

        # 2. Gating Direcional de Manifold: max(0, cos(x, z_target))
        if self.scars_count.item() > 0:
            norm_x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)
            norm_z = self.z_target / (torch.norm(self.z_target) + 1e-6)
            cosine = torch.matmul(norm_x, norm_z.unsqueeze(-1))  # [..., 1]
            gate = torch.clamp(cosine, min=0.0)
            delta = delta * gate

        return delta

    def calibrate_direction(self, x_centroid: torch.Tensor) -> None:
        """Calibra a bússola direcional do especialista com o centróide do cluster."""
        with torch.no_grad():
            flat = x_centroid.view(-1, self.in_features).mean(dim=0)
            norm_c = flat / (torch.norm(flat) + 1e-6)
            self.z_target.copy_(norm_c)
            self.scars_count.add_(1)


class MixtureOfScarsLinear(nn.Module):
    """Camada Linear com Mixture of Scars (MoS).

    Combina um núcleo denso compartilhado (W_base) com múltiplos especialistas
    em cicatriz ativados por roteamento Top-K competitivo.

    Exemplo de uso:
        >>> # Duas camadas compartilhando a MESMA matriz base (zero memória extra):
        >>> base = nn.Linear(512, 512)
        >>> layer1 = MixtureOfScarsLinear(512, 512, num_experts=8, top_k=2, shared_base=base)
        >>> layer2 = MixtureOfScarsLinear(512, 512, num_experts=8, top_k=2, shared_base=base)
        >>> x = torch.randn(4, 512)
        >>> out1 = layer1(x) # [4, 512]
        >>> out2 = layer2(x) # [4, 512]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 8,
        top_k: int = 2,
        rank: int = 8,
        bias: bool = True,
        shared_base: Optional[nn.Linear] = None,
        scale: float = 0.20,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)
        self.rank = int(rank)
        self.scale = float(scale)

        # 1. Base Universal Compartilhada (W_base)
        if shared_base is not None:
            self.base = shared_base
            self._shares_base = True
            dtype = shared_base.weight.dtype
            device = shared_base.weight.device
        else:
            self.base = nn.Linear(
                in_features, out_features, bias=bias, device=device, dtype=dtype
            )
            self._shares_base = False
            dtype = self.base.weight.dtype
            device = self.base.weight.device

        # 2. Roteador de Especialistas (Linear Router: in_features -> num_experts)
        self.router = nn.Linear(
            self.in_features, self.num_experts, bias=False, device=device, dtype=dtype
        )

        # 3. Conjunto de Especialistas em Cicatriz (Scar Experts de Kilobytes)
        self.experts = nn.ModuleList(
            [
                ScarExpert(
                    in_features=in_features,
                    out_features=out_features,
                    rank=self.rank,
                    scale=self.scale,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.num_experts)
            ]
        )

    @property
    def base_bytes(self) -> int:
        """Memória consumida pela base linear (0 se for compartilhada externamente)."""
        if self._shares_base:
            return 0
        w_bytes = self.base.weight.numel() * self.base.weight.element_size()
        b_bytes = (
            self.base.bias.numel() * self.base.bias.element_size()
            if self.base.bias is not None
            else 0
        )
        return int(w_bytes + b_bytes)

    @property
    def scars_bytes(self) -> int:
        """Memória consumida por todos os especialistas de cicatriz somados."""
        router_bytes = self.router.weight.numel() * self.router.weight.element_size()
        return router_bytes + sum(e.memory_bytes for e in self.experts)

    @property
    def total_bytes(self) -> int:
        return self.base_bytes + self.scars_bytes

    def forward(
        self, x: torch.Tensor, return_routing: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass com MoS (Mixture of Scars).

        Args:
            x: Tensor de entrada [..., in_features].
            return_routing: Se True, retorna também routing_weights e topk_indices.
                            Se False (padrão drop-in para nn.Linear), retorna apenas o tensor de saída.
        """
        orig_dtype = x.dtype
        # Converte x para o dtype dos parâmetros se necessário
        target_dtype = self.router.weight.dtype
        if x.dtype != target_dtype:
            x_in = x.to(target_dtype)
        else:
            x_in = x

        # 1. Projeção Base Universal
        h_base = self.base(x_in)

        # 2. Roteamento Competitivo: logits = x @ W_router^T / sqrt(D)
        router_logits = self.router(x_in) / math.sqrt(self.in_features)

        # 3. Seleção Top-K
        topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        routing_weights = F.softmax(topk_logits, dim=-1)  # [..., top_k]

        # 4. Injeção dos Especialistas em Cicatriz (apenas os Top-K são computados)
        delta_total = torch.zeros_like(h_base)

        for k_idx in range(self.top_k):
            weight_k = routing_weights[..., k_idx].unsqueeze(-1)  # [..., 1]
            idx_k = topk_indices[..., k_idx]

            for expert_id, expert in enumerate(self.experts):
                mask = (idx_k == expert_id).unsqueeze(-1)  # [..., 1]
                if mask.any():
                    expert_out = expert(x_in)
                    delta_total = delta_total + mask * (weight_k * expert_out)

        out = (h_base + delta_total).to(orig_dtype)

        if return_routing:
            return out, routing_weights, topk_indices
        return out
