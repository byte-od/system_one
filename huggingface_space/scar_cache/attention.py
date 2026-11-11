"""SlotScarAttention: Attention Module sem KV Cache (Memória Estritamente O(1)).

Substitui a atenção Softmax com KV Cache acumulativo O(T) por uma projeção
associativa multi-slot no L1 Cache com Regra Delta de sobrescrita exata.
"""

from __future__ import annotations

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.multi_slot import MultiSlotSCache


class SlotScarAttention(nn.Module):
    """Módulo de Atenção Auto-Regressiva sem KV Cache.

    Em vez de armazenar todos os tensores K e V passados em listas que crescem
    sem limites (causando estouro de VRAM), o SlotScarAttention mantém o estado
    associativo em uma matriz constante O(1) de 34 KB no L1 Cache.

    Equações:
        q_t = x_t W_q, k_t = x_t W_k, v_t = x_t W_v
        out_t = MultiSlotSCache.read(q_t)
        MultiSlotSCache.write_delta(k_t, v_t)
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 2,
        num_slots: int = 8,
        head_dim: Optional[int] = None,
        use_delta_rule: bool = True,
        multi_scale_decay: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim or (d_model // num_heads))
        self.num_slots = int(num_slots)

        assert self.d_model == self.num_heads * self.head_dim, "d_model deve ser divisível por num_heads"

        # Projeções padrão Q, K, V e Out
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Um cache multi-slot O(1) por cabeça de atenção
        self.caches = nn.ModuleList([
            MultiSlotSCache(
                slot_dim=self.head_dim,
                num_slots=self.num_slots,
                use_delta_rule=use_delta_rule,
                multi_scale_decay=multi_scale_decay,
            )
            for _ in range(self.num_heads)
        ])

    def reset_cache(self) -> None:
        """Limpa o estado associativo para uma nova sequência."""
        for c in self.caches:
            c.reset_scars()

    @property
    def total_memory_bytes(self) -> int:
        """Retorna o consumo estrito de bytes de todos os slots de todas as cabeças."""
        return sum(c.memory_bytes for c in self.caches)

    def forward_step(
        self,
        x_t: torch.Tensor,
        update_state: bool = True,
    ) -> torch.Tensor:
        """Processa um único token t durante a geração auto-regressiva em O(1).

        Args:
            x_t: Tensor do token atual [Batch, d_model].
            update_state: Se True, atualiza a memória com o par (k_t, v_t).

        Returns:
            out_t: Tensor de saída [Batch, d_model].
        """
        B = x_t.shape[0]

        # 1. Projeções lineares
        q = self.q_proj(x_t).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(x_t).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(x_t).view(B, self.num_heads, self.head_dim)

        head_outputs = []

        # 2. Leitura e Escrita O(1) por cabeça de atenção
        for h in range(self.num_heads):
            cache_h = self.caches[h]
            q_h = q[:, h, :]  # [B, head_dim]
            k_h = k[:, h, :]  # [B, head_dim]
            v_h = v[:, h, :]  # [B, head_dim]

            # Leitura associativa da memória passada (Atenção Linear O(1))
            out_h = cache_h.read(q_h)
            head_outputs.append(out_h)

            # Escrita atômica com Regra Delta (sem alocar novos tensores na memória!)
            if update_state:
                # Usa a média do batch desconectada do grafo para atualizar a memória associativa
                mean_k = k_h.mean(dim=0).detach()
                mean_v = v_h.mean(dim=0).detach()
                cache_h.write_key_value(mean_k, mean_v)

        concat_out = torch.cat(head_outputs, dim=-1)  # [B, d_model]
        return self.out_proj(concat_out)
