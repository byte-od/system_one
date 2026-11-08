"""Testes unitários para o núcleo SCache (GradientScarMemory)."""

import torch
import pytest

from scar_cache.core import SCache


def test_o1_memory_invariant():
    """Garante que a memória é estritamente constante independente de quantas cicatrizes são salvas."""
    cache = SCache(slot_dim=32, num_slots=8, mode="hybrid")
    initial_bytes = cache.memory_bytes
    # 32*32*4 (W_scar) + 32*4 (g_target) + 32*4 (z_target) = 4096 + 128 + 128 = 4352 bytes (~4.25 KB)
    assert initial_bytes == 4352

    # Registra 500 cicatrizes sucessivas
    for i in range(500):
        slots = torch.randn(8, 32)
        grads = torch.randn(8, 32)
        cache.record_scar(slots, grads, surprise=1.5)

    assert cache.scars_count.item() == 500
    # O consumo de memória NÃO pode crescer
    assert cache.memory_bytes == initial_bytes


def test_manifold_gating():
    """Verifica que o gating só injeta modulação se o estado estiver no mesmo semi-espaço de z_target."""
    cache = SCache(slot_dim=16, num_slots=1, mode="hybrid", scale=0.5)

    # Cria uma cicatriz conhecida com z=[1, 0, 0, ...]
    z_known = torch.zeros(1, 16)
    z_known[0, 0] = 1.0
    g_known = torch.zeros(1, 16)
    g_known[0, 1] = 1.0

    cache.record_scar(z_known, g_known, surprise=2.0)

    # Estado alinhado com a vitória anterior
    z_aligned = torch.zeros(1, 16)
    z_aligned[0, 0] = 1.0
    guided_aligned, delta, affinity = cache.modulate(z_aligned)
    assert affinity is not None and affinity.item() > 0.99
    # Deve sofrer injeção (delta != 0)
    assert not torch.allclose(guided_aligned, z_aligned)

    # Estado oposto (em outra direção / semi-espaço negativo)
    z_opposite = torch.zeros(1, 16)
    z_opposite[0, 0] = -1.0
    guided_opposite, delta_opp, affinity_opp = cache.modulate(z_opposite)
    assert affinity_opp is not None and affinity_opp.item() < -0.99
    # O gate max(0, cos) bloqueia a injeção espúria
    assert torch.allclose(guided_opposite, z_opposite, atol=1e-6)


def test_batch_modulation_scope():
    """Verifica o context manager para modulação seletiva por batedor/explorador."""
    cache = SCache(slot_dim=8, num_slots=1, mode="hybrid", scale=0.2)
    z = torch.randn(1, 8)
    g = torch.randn(1, 8)
    cache.record_scar(z, g, surprise=1.0)

    batch_x = torch.randn(4, 8)
    # Batedor 0 tem escala 0.0 (exploração pura), Batedor 3 tem escala 1.0 (aproveitamento total)
    scales = torch.tensor([0.0, 0.3, 0.6, 1.0])

    with cache.batch_modulation_scope(scale_vec=scales):
        guided, _, _ = cache.modulate(batch_x)
        # Batedor 0 não deve ter nenhuma alteração
        assert torch.allclose(guided[0], batch_x[0], atol=1e-6)
        # Batedor 3 deve ter alteração significativa
        assert not torch.allclose(guided[3], batch_x[3], atol=1e-4)

    # Fora do escopo, retorna à escala original
    assert cache._batch_scale_override is None


def test_resonance_logits():
    """Verifica o cálculo analítico de ressonância e a modulação de nitidez dos logits."""
    cache = SCache(slot_dim=8, num_slots=1, mode="hybrid", resonance_beta=0.5)
    z = torch.randn(1, 8)
    g = torch.randn(1, 8)
    cache.record_scar(z, g)

    # Delta perfeitamente alinhado com g_target
    delta_step = cache.g_target.unsqueeze(0).clone()
    resonance = cache.compute_resonance(delta_step)
    assert resonance.item() > 0.99

    logits = torch.tensor([1.0, 2.0, 3.0])
    sharpened_logits = cache.modulate_logits(logits, resonance)
    # Com ressonância positiva, a escala multiplica os logits, aumentando a confiança
    assert sharpened_logits[2].item() > logits[2].item()


def test_reset_scars():
    """Garante a limpeza completa do estado para Tabula Rasa."""
    cache = SCache(slot_dim=8, num_slots=1, mode="hybrid")
    cache.record_scar(torch.randn(1, 8), torch.randn(1, 8))
    assert cache.scars_count.item() == 1
    assert torch.norm(cache.W_scar).item() > 0

    cache.reset_scars()
    assert cache.scars_count.item() == 0
    assert torch.norm(cache.W_scar).item() == 0.0
    assert torch.norm(cache.g_target).item() == 0.0
    assert torch.norm(cache.z_target).item() == 0.0
