"""Testes unitários rigorosos para a arquitetura MultiSlotSCache (Substituição de KV Cache)."""

import torch
import pytest

from scar_cache.multi_slot import MultiSlotSCache
from scar_cache.modules import SlotScarLinear
from scar_cache.projections import project_conflicting_gradients_per_slot, delta_rule_update


def test_multi_slot_memory_bytes_o1():
    """Garante que a pegada de memória do MultiSlotSCache (8x32x32) é exatamente 34 KB em L1."""
    cache = MultiSlotSCache(slot_dim=32, num_slots=8)
    # W_scar: 8 * 32 * 32 * 4 = 32.768 bytes
    # g_target: 8 * 32 * 4 = 1.024 bytes
    # z_target: 8 * 32 * 4 = 1.024 bytes
    # Total = 34.816 bytes (34 KB)
    assert cache.memory_bytes == 34816

    # Realiza 200 escritas
    for _ in range(200):
        k = torch.randn(32)
        v = torch.randn(32)
        cache.write_key_value(k, v)

    # Memória estritamente constante O(1)
    assert cache.memory_bytes == 34816


def test_pcgrad_per_slot_independence():
    """Verifica que o PCGrad no Slot 0 não restringe nem interfere no Slot 1."""
    # Slot 0 em conflito (< 0), Slot 1 perfeitamente alinhado (> 0)
    g_targets = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    g_news = torch.tensor([
        [-0.5, 0.8, 0.0],  # Conflito no Slot 0 (<g_new, g_t> = -0.5)
        [0.0, 0.9, 0.1],   # Alinhado no Slot 1 (<g_new, g_t> = 0.9 > 0)
    ])

    proj = project_conflicting_gradients_per_slot(g_news, g_targets)

    # Slot 0 deve ter a componente de conflito removida -> [0.0, 0.8, 0.0]
    assert torch.allclose(proj[0], torch.tensor([0.0, 0.8, 0.0]), atol=1e-5)
    # Slot 1 deve permanecer idêntico ao original -> [0.0, 0.9, 0.1]
    assert torch.allclose(proj[1], g_news[1], atol=1e-6)


def test_delta_rule_fact_overwriting_no_ghosts():
    """Pilar 2: Prova que a Regra Delta sobrescreve fatos antigos sem fantasmas aditivos."""
    cache = MultiSlotSCache(slot_dim=16, num_slots=1, multi_scale_decay=False)

    key_city = torch.zeros(16)
    key_city[0] = 1.0  # Chave canônica "Cidade"

    val_curitiba = torch.zeros(16)
    val_curitiba[1] = 2.0  # Valor antigo "Curitiba"

    val_madri = torch.zeros(16)
    val_madri[2] = 5.0  # Valor retificado "Madri"

    # 1. Escreve o fato inicial
    cache.write_key_value(key_city, val_curitiba, slot_index=0, learning_rate=1.0)
    rec1 = cache.read_key(key_city, slot_index=0)
    assert torch.allclose(rec1, val_curitiba, atol=1e-4)

    # 2. Retifica o fato: agora Cidade = Madri
    cache.write_key_value(key_city, val_madri, slot_index=0, learning_rate=1.0)
    rec2 = cache.read_key(key_city, slot_index=0)

    # 3. VERIFICAÇÃO CRÍTICA ANTI-GHOSTING:
    # A resposta deve ser Madri (índice 2 == 5.0) e NÃO deve conter Curitiba (índice 1 deve ser ~0.0)
    assert abs(rec2[2].item() - 5.0) < 1e-3
    assert abs(rec2[1].item()) < 1e-3, "FANTASMA DETECTADO! A memória antiga não foi subtraída!"


def test_competitive_routing():
    """Pilar 3: Verifica que tokens são roteados preferencialmente para slots afins."""
    cache = MultiSlotSCache(slot_dim=16, num_slots=4)
    # Token exatamente alinhado com a consulta do slot 2
    token = cache.slot_queries[2].clone().unsqueeze(0)

    weights, slots = cache.route_tokens(token)
    assert weights.shape == (1, 4)
    # O Slot 2 deve receber a maior probabilidade de roteamento
    best_slot = torch.argmax(weights[0]).item()
    assert best_slot == 2


def test_multi_scale_decay():
    """Pilar 4: Verifica que o Slot 0 esquece rápido enquanto o Slot 7 retém longo prazo."""
    cache = MultiSlotSCache(slot_dim=8, num_slots=8, multi_scale_decay=True)
    # Taxa de decaimento do Slot 0 deve ser muito menor que a do Slot 7
    gamma_0 = cache.decay_rates[0].item()
    gamma_7 = cache.decay_rates[7].item()

    assert gamma_0 == 0.50
    assert gamma_7 > 0.999

    k = torch.randn(8)
    v = torch.randn(8)
    # Grava simultaneamente nos dois slots
    cache.write_key_value(k, v, slot_index=0)
    cache.write_key_value(k, v, slot_index=7)

    # Passam 10 passos vazios (sem nova escrita)
    for _ in range(10):
        cache.W_scar[0].mul_(gamma_0)
        cache.W_scar[7].mul_(gamma_7)

    # No Slot 0, a informação foi praticamente zerada: (0.5)^10 ≈ 0.00097
    assert torch.norm(cache.read_key(k, slot_index=0)).item() < 0.01
    # No Slot 7, a informação ainda está quase 100% intacta: (0.9999)^10 ≈ 0.999
    assert torch.norm(cache.read_key(k, slot_index=7)).item() > 0.80


def test_slot_scar_linear_module():
    """Testa o módulo SlotScarLinear integrado em um forward/backward pass do PyTorch."""
    layer = SlotScarLinear(in_features=32, out_features=10, slot_dim=32, num_slots=8)
    x = torch.randn(4, 32, requires_grad=True)

    out = layer(x)
    assert out.shape == (4, 10)

    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert x.grad is not None

    # Teste de escrita e leitura de chave
    k = torch.randn(32)
    v = torch.randn(32)
    layer.write_key_value(k, v, slot_index=3)
    rec = layer.read_key(k, slot_index=3)
    assert torch.allclose(rec, v, atol=1e-3)
