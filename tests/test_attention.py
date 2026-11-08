"""Testes unitários para SlotScarAttention (Módulo de Atenção sem KV Cache)."""

import torch
import pytest

from scar_cache.attention import SlotScarAttention


def test_slot_scar_attention_step():
    """Verifica que a atenção sem KV Cache processa tokens auto-regressivos sem erros."""
    attn = SlotScarAttention(d_model=32, num_heads=2, num_slots=4)
    x = torch.randn(1, 32)

    # Passo 1
    out1 = attn.forward_step(x, update_state=True)
    assert out1.shape == (1, 32)

    # Passo 2
    x2 = torch.randn(1, 32)
    out2 = attn.forward_step(x2, update_state=True)
    assert out2.shape == (1, 32)

    # A memória deve permanecer estritamente constante
    mem_init = attn.total_memory_bytes
    for _ in range(50):
        _ = attn.forward_step(torch.randn(1, 32), update_state=True)

    assert attn.total_memory_bytes == mem_init


def test_slot_scar_attention_reset():
    """Verifica a limpeza de estado entre sequências."""
    attn = SlotScarAttention(d_model=16, num_heads=1, num_slots=2)
    attn.forward_step(torch.randn(1, 16), update_state=True)

    assert attn.caches[0].scars_count.sum().item() > 0
    attn.reset_cache()
    assert attn.caches[0].scars_count.sum().item() == 0
