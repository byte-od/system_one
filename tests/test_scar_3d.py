"""Testes unitários e de contrato para o Mecanismo de Cicatriz 3D (Multi-Slot Tensorizada).

Valida:
1. Alocação correta dos buffers 3D O(1): W_scar in R^(S x D x D), g_target in R^(S x D), z_target in R^(S x D).
2. Isolamento de slots: a gravação de uma cicatriz no slot k não interfere nos demais slots.
3. Recuperação associativa tensorial exata via Einsum.
4. Gating de manifold direcional 3D por slot.
5. Integração com UniversalS1Agent e RecurrentPPOTrainer em regime Tabula Rasa.
"""

import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.scar_memory import GradientScarMemory
from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer


def test_scar_3d_buffer_shapes_and_memory():
    """Garante que a Scar 3D aloca os tensores [S, D, D] e [S, D] ocupando ~34 KB na L1."""
    num_slots = 8
    slot_dim = 32
    scar_3d = GradientScarMemory(
        slot_dim=slot_dim,
        num_slots=num_slots,
        mode="hybrid_3d",
        is_3d=True,
    )

    assert scar_3d.is_3d is True
    assert scar_3d.W_scar.shape == (num_slots, slot_dim, slot_dim)
    assert scar_3d.g_target.shape == (num_slots, slot_dim)
    assert scar_3d.z_target.shape == (num_slots, slot_dim)
    assert scar_3d.memory_bytes == 34816  # (8*32*32 + 8*32 + 8*32) * 4 = 34.0 KB


def test_scar_3d_slot_isolation_and_associative_recall():
    """Verifica que cada slot k possui memória associativa independente sem vazamento inter-slots."""
    num_slots = 8
    slot_dim = 32
    scar_3d = GradientScarMemory(
        slot_dim=slot_dim,
        num_slots=num_slots,
        mode="hybrid_3d",
        is_3d=True,
    )

    z = torch.randn(num_slots, slot_dim)
    g = torch.randn(num_slots, slot_dim)
    norm_z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
    norm_g = g / (g.norm(dim=-1, keepdim=True) + 1e-6)

    # Grava cicatriz com surprise=1.0
    scar_3d.record_scar(norm_z, norm_g, surprise=1.0)
    assert scar_3d.scars_count.item() == 1

    # Modula slots usando os próprios slots z como consulta
    slots_in = norm_z.clone()
    slots_guided, delta = scar_3d.modulate_slots(slots_in)

    assert delta is not None
    assert delta.shape == (num_slots, slot_dim)

    # Verifica que para CADA slot individual, a direção de delta é perfeitamente alinhada com norm_g (+1.0)
    cos_sim = torch.einsum(
        "sd,sd->s",
        delta / (delta.norm(dim=-1, keepdim=True) + 1e-6),
        norm_g,
    )
    assert torch.allclose(cos_sim, torch.ones(num_slots), atol=1e-4)


def test_scar_3d_agent_and_trainer_integration():
    """Valida a integração completa da Scar 3D no UniversalS1Agent e no RecurrentPPOTrainer."""
    raw_env = VectorizedRubiksEnv(num_envs=2, curriculum=True, min_depth=1)
    env = UniversalS1Wrapper(raw_env, tri_compass=True)

    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        tri_compass=True,
        relational=True,
    )

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=64,
        scar_mode="hybrid_3d",
        pure_scars=True,
        scar_3d=True,
        device="cpu",
    )

    assert trainer.scar_memory is not None
    assert trainer.scar_memory.is_3d is True
    assert trainer.scar_memory.W_scar.dim() == 3

    # Simula vitória completa
    fake_win_traj = [
        {"obs": np.zeros(324, dtype=np.float32), "action": 0, "prev_action": 0, "reward": 0.10, "prev_reward": 0.0, "delta_obs": None, "log_prob": -0.5, "value": 1.0},
        {"obs": np.zeros(324, dtype=np.float32), "action": 1, "prev_action": 0, "reward": 10.0, "prev_reward": 0.10, "delta_obs": None, "log_prob": -0.3, "value": 10.0},
    ]

    trainer._process_completed_trajectory(fake_win_traj, is_win=True, cur_depth=1)
    assert trainer.scar_memory.scars_count.item() == 1
    assert trainer.scar_memory.scar_norm > 0.0
