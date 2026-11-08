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

    Mantém no máximo `window_size` tokens físicos em memória para atenção local exata.
    Tokens mais antigos que deslizam para fora da janela são comprimidos atômica e
    continuamente no MultiSlotSCache (no L1 Cache) usando a Regra Delta.
    """

    def __init__(self, window_size: int = 64, num_slots: int = 8):
        super().__init__()
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
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

        # Se ultrapassou a janela máxima, comprime o excedente antigo no SCache O(1)
        if self.keys.shape[-2] > self.window_size:
            excess_k = self.keys[..., : -self.window_size, :]
            excess_v = self.values[..., : -self.window_size, :]
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

            # Trunca fisicamente o tensor: MEMÓRIA CONGELADA EM O(1)!
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
        >>> cache = SCacheDynamicCache(window_size=64, num_slots=8)
        >>> out = model.generate(**inputs, past_key_values=cache)
    """

    def __init__(self, window_size: int = 64, num_slots: int = 8):
        super().__init__()
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
        self.layer_class_to_replicate = lambda: SCacheDynamicLayer(
            window_size=self.window_size, num_slots=self.num_slots
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
