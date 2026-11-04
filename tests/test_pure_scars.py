"""Testes de contrato para o Mecanismo de Cicatriz Puríssimo (pure_scars=True).

Verifica que:
1. Cicatrizes são registradas ESTRITAMENTE para vitórias completas (is_win=True).
2. Trajetórias com Peak Slicing (ganho parcial sem vitória completa) NUNCA incrementam o scar_memory quando pure_scars=True.
3. Quando pure_scars=False (modo legado), o peak slicing grava cicatrizes parciais.
"""

import numpy as np
import torch
import pytest

from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.core.agent import UniversalS1Agent
from system1_engine.training.ppo import RecurrentPPOTrainer


def test_pure_scars_contract_ignores_peak_slicing():
    """Garante que com pure_scars=True, peak slices de fracassos são estritamente rejeitados."""
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
        scar_mode="hybrid",
        pure_scars=True,
        device="cpu",
    )
    assert trainer.pure_scars is True
    assert trainer.scar_memory.scars_count.item() == 0

    # Simula episódio com pico de ganho parcial (Peak Slicing clássico), mas SEM vitória (is_win=False)
    fake_failed_traj = [
        {"obs": np.zeros(324, dtype=np.float32), "action": 0, "prev_action": 0, "reward": 0.20, "prev_reward": 0.0, "delta_obs": None, "log_prob": -1.0, "value": 0.5},
        {"obs": np.zeros(324, dtype=np.float32), "action": 1, "prev_action": 0, "reward": 0.10, "prev_reward": 0.20, "delta_obs": None, "log_prob": -1.2, "value": 0.7},
        {"obs": np.zeros(324, dtype=np.float32), "action": 2, "prev_action": 1, "reward": -0.02, "prev_reward": 0.10, "delta_obs": None, "log_prob": -1.5, "value": 0.4},
    ]

    # Processa trajetória não resolvida
    trainer._process_completed_trajectory(fake_failed_traj, is_win=False, cur_depth=10)

    # Scars count DEVE PERMANECER RIGOROSAMENTE ZERO!
    assert trainer.scar_memory.scars_count.item() == 0, "Falha de isolamento: Peak Slicing contaminou o scar_memory puro!"

    # Agora simula uma vitória real (is_win=True)
    fake_win_traj = [
        {"obs": np.zeros(324, dtype=np.float32), "action": 0, "prev_action": 0, "reward": 0.10, "prev_reward": 0.0, "delta_obs": None, "log_prob": -0.5, "value": 1.0},
        {"obs": np.zeros(324, dtype=np.float32), "action": 1, "prev_action": 0, "reward": 10.0, "prev_reward": 0.10, "delta_obs": None, "log_prob": -0.3, "value": 10.0},
    ]

    trainer._process_completed_trajectory(fake_win_traj, is_win=True, cur_depth=10)

    # Scars count DEVE TER SIDO INCREMENTADO EXATAMENTE EM 1!
    assert trainer.scar_memory.scars_count.item() == 1, "Vitória genuína deve gravar cicatriz!"


def test_legacy_pure_scars_false_allows_peak_slicing():
    """Garante compatibilidade retroativa: com pure_scars=False, peak slices continuam gravando."""
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
        scar_mode="hybrid",
        pure_scars=False,
        device="cpu",
    )
    assert trainer.pure_scars is False
    assert trainer.scar_memory.scars_count.item() == 0

    fake_failed_traj = [
        {"obs": np.zeros(324, dtype=np.float32), "action": 0, "prev_action": 0, "reward": 0.20, "prev_reward": 0.0, "delta_obs": None, "log_prob": -1.0, "value": 0.5},
        {"obs": np.zeros(324, dtype=np.float32), "action": 1, "prev_action": 0, "reward": 0.10, "prev_reward": 0.20, "delta_obs": None, "log_prob": -1.2, "value": 0.7},
        {"obs": np.zeros(324, dtype=np.float32), "action": 2, "prev_action": 1, "reward": -0.02, "prev_reward": 0.10, "delta_obs": None, "log_prob": -1.5, "value": 0.4},
    ]

    trainer._process_completed_trajectory(fake_failed_traj, is_win=False, cur_depth=10)
    assert trainer.scar_memory.scars_count.item() == 1, "No modo legado (pure_scars=False), peak slicing deve gravar cicatriz."
