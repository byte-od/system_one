"""HuggingFace SCache Integration: Drop-in Cache para Modelos Reais (Llama, SmolLM, Qwen).

Substitui o DynamicCache do HuggingFace por uma arquitetura com janela deslizante
congelada e compressão em MultiSlotSCache via Regra Delta O(1).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import torch
from transformers.cache_utils import CacheLayerMixin, DynamicCache, DynamicLayer

from scar_cache.multi_slot import MultiSlotSCache


class SCacheDynamicLayer(DynamicLayer):
    """Camada de Cache do HuggingFace com tamanho físico estritamente limitado O(1).

    Combina:
      1. Attention Sinks (os primeiros `num_sink_tokens` preservados para ancoragem estável do Softmax)
      2. Janela deslizante local (os últimos `window_size - num_sink_tokens` tokens mais recentes)
      3. MultiSlotSCache (todos os tokens intermediários desalojados são comprimidos via Regra Delta)
    """

    def __init__(self, window_size: int = 64, num_slots: int = 8, num_sink_tokens: int = 4):
        super().__init__()
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
        self.num_sink_tokens = min(int(num_sink_tokens), self.window_size // 2) if num_sink_tokens > 0 else 0
        self.scache: Optional[MultiSlotSCache] = None
        self.total_tokens_absorbed: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            head_dim = key_states.shape[-1]
            self.scache = MultiSlotSCache(slot_dim=head_dim, num_slots=self.num_slots)

        # Concatena os novos tokens recebidos neste passo
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        # Se ultrapassou a janela máxima, comprime o excedente intermediário no SCache O(1)
        if self.keys.shape[-2] > self.window_size:
            slide_len = self.window_size - self.num_sink_tokens
            excess_k = self.keys[..., self.num_sink_tokens : -slide_len, :]
            excess_v = self.values[..., self.num_sink_tokens : -slide_len, :]
            num_excess = excess_k.shape[-2]

            # Comprime no MultiSlotSCache via Regra Delta em float32
            with torch.no_grad():
                for b in range(excess_k.shape[0]):
                    for h in range(excess_k.shape[1]):
                        for t in range(num_excess):
                            k_vec = excess_k[b, h, t].float()
                            v_vec = excess_v[b, h, t].float()
                            self.scache.write_key_value(k_vec, v_vec)

            self.total_tokens_absorbed += num_excess

            # Reconstrói os tensores: Sinks (início) + Slide (fim): MEMÓRIA CONGELADA EM O(1)!
            if self.num_sink_tokens > 0:
                sink_k = self.keys[..., : self.num_sink_tokens, :]
                sink_v = self.values[..., : self.num_sink_tokens, :]
                slide_k = self.keys[..., -slide_len :, :]
                slide_v = self.values[..., -slide_len :, :]
                self.keys = torch.cat([sink_k, slide_k], dim=-2)
                self.values = torch.cat([sink_v, slide_v], dim=-2)
            else:
                self.keys = self.keys[..., -self.window_size :, :]
                self.values = self.values[..., -self.window_size :, :]

        return self.keys, self.values

    @property
    def physical_bytes(self) -> int:
        """Memória física consumida pelos tensores da janela deslizante."""
        if not self.is_initialized or self.keys is None:
            return 0
        return int((self.keys.numel() + self.values.numel()) * self.keys.element_size())

    @property
    def scache_bytes(self) -> int:
        """Memória do MultiSlotSCache associativo residente."""
        return self.scache.memory_bytes if self.scache is not None else 0


class SCacheDynamicCache(DynamicCache):
    """Drop-in Replacement para DynamicCache do HuggingFace compatível com qualquer LLM.

    Exemplo de uso:
        >>> from transformers import AutoModelForCausalLM
        >>> from scar_cache import SCacheDynamicCache
        >>> model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        >>> cache = SCacheDynamicCache(window_size=64, num_slots=8, num_sink_tokens=4)
        >>> out = model.generate(**inputs, past_key_values=cache)
    """

    def __init__(self, window_size: int = 64, num_slots: int = 8, num_sink_tokens: int = 4):
        super().__init__()
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
        self.num_sink_tokens = int(num_sink_tokens)
        self.layer_class_to_replicate = lambda: SCacheDynamicLayer(
            window_size=self.window_size,
            num_slots=self.num_slots,
            num_sink_tokens=self.num_sink_tokens,
        )

    @property
    def total_physical_bytes(self) -> int:
        """Soma da memória física de todas as camadas."""
        return sum(
            layer.physical_bytes for layer in self.layers if isinstance(layer, SCacheDynamicLayer)
        )

    @property
    def total_scache_bytes(self) -> int:
        """Soma da memória associativa de todas as camadas."""
        return sum(
            layer.scache_bytes for layer in self.layers if isinstance(layer, SCacheDynamicLayer)
        )

    @property
    def total_memory_bytes(self) -> int:
        return self.total_physical_bytes + self.total_scache_bytes

    def get_max_physical_length(self) -> int:
        """Retorna o maior tamanho de tensor observado nas camadas (limitado por window_size)."""
        lengths = [
            layer.keys.shape[-2]
            for layer in self.layers
            if isinstance(layer, SCacheDynamicLayer) and layer.keys is not None
        ]
        return max(lengths) if lengths else 0
