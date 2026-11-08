"""Testes unitários para integração do SCache com HuggingFace Cache (HFSScarCache)."""

import torch
import pytest

from scar_cache.hf_cache import SCacheDynamicCache, SCacheDynamicLayer


def test_scache_dynamic_layer_capped_window():
    """Verifica que o tensor físico é congelado em window_size e o excedente é absorvido."""
    window_size = 16
    layer = SCacheDynamicLayer(window_size=window_size, num_slots=4)

    # Simula geração de 50 tokens passo a passo: [Batch=1, Heads=2, Seq=1, Dim=32]
    for t in range(50):
        k_t = torch.randn(1, 2, 1, 32)
        v_t = torch.randn(1, 2, 1, 32)
        keys, values = layer.update(k_t, v_t)

    # O comprimento físico NUNCA deve ultrapassar window_size (16)
    assert keys.shape[-2] == window_size
    assert values.shape[-2] == window_size
    assert layer.total_tokens_absorbed == (50 - window_size)
    assert layer.scache is not None
    assert layer.scache.memory_bytes > 0


def test_scache_dynamic_cache_multi_layer():
    """Verifica o cache multi-camada compatível com HuggingFace."""
    cache = SCacheDynamicCache(window_size=8, num_slots=4)

    # Simula 4 camadas de atenção
    for layer_idx in range(4):
        k = torch.randn(1, 2, 20, 16)
        v = torch.randn(1, 2, 20, 16)
        keys, values = cache.update(k, v, layer_idx=layer_idx)
        assert keys.shape[-2] == 8  # Capped at window_size=8

    assert len(cache.layers) == 4
    assert cache.get_max_physical_length() == 8
    assert cache.total_physical_bytes > 0
    assert cache.total_scache_bytes > 0
