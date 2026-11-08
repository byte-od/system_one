"""Testes unitários para projeções ortogonais e produto externo Hebbiano."""

import torch
import pytest

from scar_cache.projections import hebbian_outer_product, project_conflicting_gradient


def test_project_conflicting_gradient_aligned():
    # Vetores apontando na mesma direção (produto escalar > 0)
    g_target = torch.tensor([1.0, 0.0, 0.0])
    g_new = torch.tensor([0.8, 0.2, 0.0])

    proj = project_conflicting_gradient(g_new, g_target)
    # Não deve alterar pois não há conflito
    assert torch.allclose(proj, g_new, atol=1e-6)


def test_project_conflicting_gradient_orthogonal_conflict():
    # Vetores em oposição (produto escalar < 0)
    g_target = torch.tensor([1.0, 0.0, 0.0])
    g_new = torch.tensor([-0.6, 0.8, 0.0])

    assert torch.dot(g_new, g_target).item() < 0

    proj = project_conflicting_gradient(g_new, g_target)
    # O componente conflitante (-0.6 * [1, 0, 0]) deve ser zerado, restando [0.0, 0.8, 0.0]
    expected = torch.tensor([0.0, 0.8, 0.0])
    assert torch.allclose(proj, expected, atol=1e-5)
    # Produto escalar com g_target agora deve ser 0.0 (ortogonal)
    assert abs(torch.dot(proj, g_target).item()) < 1e-6


def test_hebbian_outer_product():
    num_slots = 4
    slot_dim = 16
    g = torch.randn(num_slots, slot_dim)
    z = torch.randn(num_slots, slot_dim)

    dW = hebbian_outer_product(g, z, num_slots=num_slots)
    assert dW.shape == (slot_dim, slot_dim)
    assert not torch.isnan(dW).any()
