"""Testes unitários para integração do SCache com HuggingFace Cache (SCacheDynamicCache)."""

from typing import Any
import torch

from scar_cache.hf_cache import SCacheDynamicCache, SCacheDynamicLayer


def _recursive_tensor_storage_bytes(obj: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    if isinstance(obj, torch.Tensor):
        st = obj.untyped_storage()
        sid = int(st.data_ptr())
        if sid in seen:
            return 0
        seen.add(sid)
        return int(st.nbytes())
    if isinstance(obj, dict):
        return sum(_recursive_tensor_storage_bytes(k, seen) + _recursive_tensor_storage_bytes(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return sum(_recursive_tensor_storage_bytes(x, seen) for x in obj)
    if hasattr(obj, "__dict__"):
        return _recursive_tensor_storage_bytes(vars(obj), seen)
    return 0


def test_scache_dynamic_layer_capped_window():
    """Verifica que o tensor físico é congelado em window_size e o excedente é absorvido."""
    window_size = 16
    layer = SCacheDynamicLayer(window_size=window_size, num_slots=4)

    for _ in range(50):
        k_t = torch.randn(1, 2, 1, 32)
        v_t = torch.randn(1, 2, 1, 32)
        keys, values = layer.update(k_t, v_t)

    assert keys.shape[-2] == window_size
    assert values.shape[-2] == window_size
    assert layer.total_tokens_absorbed == (50 - window_size)
    assert layer.scache is not None
    assert layer.scache.memory_bytes > 0


def test_scache_dynamic_cache_multi_layer():
    """Verifica o cache multi-camada compatível com HuggingFace."""
    cache = SCacheDynamicCache(window_size=8, num_slots=4)

    for layer_idx in range(4):
        k = torch.randn(1, 2, 20, 16)
        v = torch.randn(1, 2, 20, 16)
        keys, values = cache.update(k, v, layer_idx=layer_idx)
        assert keys.shape[-2] == 8

    assert len(cache.layers) == 4
    assert cache.get_max_physical_length() == 8
    assert cache.total_physical_bytes > 0
    assert cache.total_scache_bytes > 0


def test_scache_mask_alignment_and_zero_byte_discrepancy():
    """Verifica alinhamento causal exato de `get_mask_sizes` pós-evicção e discrepância física de 0 B."""
    cache = SCacheDynamicCache(window_size=32, num_slots=16, num_sink_tokens=4, project_slots=True)

    # 1. Single-shot prefill com evicção (60 tokens > window_size=32)
    for l_idx in range(3):
        pred_kv, _ = cache.layers[l_idx].get_mask_sizes(60) if l_idx < len(cache.layers) else (60, 0)
        rk, _ = cache.update(torch.randn(1, 4, 60, 32), torch.randn(1, 4, 60, 32), layer_idx=l_idx)
        if l_idx < len(cache.layers) - 1:
            assert rk.shape[-2] == 60  # Single-shot prefill não duplica slots futuros!

    b_prefill_real = _recursive_tensor_storage_bytes(cache)
    b_prefill_rep = cache.total_memory_bytes
    assert b_prefill_real == b_prefill_rep, f"Discrepancy: real={b_prefill_real} vs rep={b_prefill_rep}"

    # 2. Continuação multi-turno (chunk de 24 tokens após evicção prévia) + 12 passos de decode
    for l_idx in range(3):
        pred_kv, _ = cache.layers[l_idx].get_mask_sizes(24)
        rk, _ = cache.update(torch.randn(1, 4, 24, 32), torch.randn(1, 4, 24, 32), layer_idx=l_idx)
        assert rk.shape[-2] == pred_kv, f"Mask/KV mismatch on chunk continuation: {rk.shape[-2]} vs {pred_kv}"

    for _ in range(12):
        for l_idx in range(3):
            pred_kv, _ = cache.layers[l_idx].get_mask_sizes(1)
            rk, _ = cache.update(torch.randn(1, 4, 1, 32), torch.randn(1, 4, 1, 32), layer_idx=l_idx)
            assert rk.shape[-2] == pred_kv, f"Mask/KV mismatch on decode: {rk.shape[-2]} vs {pred_kv}"

    b_after_real = _recursive_tensor_storage_bytes(cache)
    assert b_after_real == b_prefill_real, f"O(1) storage drift detected: {b_prefill_real} -> {b_after_real}"
    assert b_after_real == cache.total_memory_bytes
