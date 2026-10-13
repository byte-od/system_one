"""Testes formais para Hindsight Experience Replay (HER) / Reversão Retrospectiva On-Policy (B.2)."""

import pytest
import numpy as np
import gymnasium as gym
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.env.adapters.rubiks import RubiksCubeEnv, VectorizedRubiksEnv


def test_hindsight_relabelling_boosts_peak_progress_advantage():
    """Valida que o relabelling retrospectivo detecta o ápice intermediário t* e injeta bônus."""
    env = RubiksCubeEnv(scramble_depth=2)
    wrapper_env = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(obs_space=wrapper_env.observation_space, action_space=wrapper_env.action_space)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapper_env,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
        hindsight_replay=True,
        hindsight_bonus=3.0,
        hindsight_min_gain=0.05,
    )

    # Simula 1 ambiente com 5 passos
    rew_es = np.array([[-0.1, -0.1, -0.1, -0.1, -0.1]], dtype=np.float32)
    done_es = np.array([[False, False, False, False, True]], dtype=bool)
    val_es = np.zeros((1, 5), dtype=np.float32)

    # Score sobe até o passo 2 (0.65) e decai até o final
    step_infos = [
        [{"score": 0.20, "is_solved": False}],
        [{"score": 0.40, "is_solved": False}],
        [{"score": 0.65, "is_solved": False}],  # Ápice t* = 2, delta = 0.45
        [{"score": 0.35, "is_solved": False}],
        [{"score": 0.15, "is_solved": False}],
    ]

    trainer._apply_hindsight_relabelling(rew_es, done_es, val_es, step_infos)

    # O passo 2 deve ter recebido bônus: 3.0 * 0.45 = 1.35 (rew_es original era -0.1 -> ~ 1.25)
    assert rew_es[0, 2] > 1.0, f"Bônus retrospectivo não aplicado no ápice: {rew_es[0, 2]}"
    # Passos pós-ápice (3 e 4) receberam penalidade de dispersão
    assert rew_es[0, 3] < -0.1
    assert rew_es[0, 4] < -0.1


def test_hindsight_skips_already_solved_episodes():
    """Garante que episódios já resolvidos com sucesso não sofrem alteração retrospectiva."""
    env = RubiksCubeEnv(scramble_depth=2)
    wrapper_env = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(obs_space=wrapper_env.observation_space, action_space=wrapper_env.action_space)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapper_env,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
        hindsight_replay=True,
    )

    rew_orig = np.array([[-0.1, -0.1, 10.0]], dtype=np.float32)
    rew_es = rew_orig.copy()
    done_es = np.array([[False, False, True]], dtype=bool)
    val_es = np.zeros((1, 3), dtype=np.float32)

    step_infos = [
        [{"score": 0.20, "is_solved": False}],
        [{"score": 0.50, "is_solved": False}],
        [{"score": 1.00, "is_solved": True}],  # Resolvido com sucesso!
    ]

    trainer._apply_hindsight_relabelling(rew_es, done_es, val_es, step_infos)

    # As recompensas devem permanecer idênticas
    np.testing.assert_allclose(rew_es, rew_orig)


def test_hindsight_fallback_uses_critic_value_when_no_score_info():
    """Valida o fallback 100% universal: quando não há info['score'], usa V(s) do crítico."""
    env = RubiksCubeEnv(scramble_depth=2)
    wrapper_env = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(obs_space=wrapper_env.observation_space, action_space=wrapper_env.action_space)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapper_env,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
        hindsight_replay=True,
        hindsight_bonus=2.0,
        hindsight_min_gain=0.10,
    )

    rew_es = np.zeros((1, 4), dtype=np.float32)
    done_es = np.array([[False, False, False, True]], dtype=bool)
    # V(s) tem pico no passo 1 (0.8) vs inicial (0.2) -> delta = 0.6
    val_es = np.array([[0.20, 0.80, 0.30, 0.10]], dtype=np.float32)

    # Sem step_infos (ambiente arbitrário Gymnasium genérico)
    trainer._apply_hindsight_relabelling(rew_es, done_es, val_es, step_infos=None)

    # Passo 1 deve ter recebido o bônus através de V(s)
    assert rew_es[0, 1] > 1.0, f"Fallback com V(s) falhou: {rew_es[0, 1]}"


def test_ppo_vectorized_and_serial_rollout_with_hindsight():
    """Garante estabilidade completa do rollout vetorizado e serial com hindsight ativo."""
    # 1. Vetorizado
    vec_env = VectorizedRubiksEnv(num_envs=4, scramble_depth=2)
    wrapper_vec = UniversalS1Wrapper(vec_env)
    agent_vec = UniversalS1Agent(obs_space=wrapper_vec.observation_space, action_space=wrapper_vec.action_space)

    trainer_vec = RecurrentPPOTrainer(
        agent=agent_vec,
        env=wrapper_vec,
        rollout_steps=32,
        chunk_length=8,
        chunk_batch_size=4,
        hindsight_replay=True,
    )

    obs_dict, _ = wrapper_vec.reset(seed=42)
    obs_dict, hx, ep_start, mean_ret = trainer_vec.collect_rollouts(
        current_obs_dict=obs_dict,
        current_hx=None,
        episode_start=True,
    )

    # Vantagens GAE e retornos calculados sem NaN ou inf
    adv = trainer_vec.buffer.advantages
    assert not np.isnan(adv).any(), "Vantagens contêm NaN após Hindsight!"
    assert not np.isinf(adv).any(), "Vantagens contêm Inf após Hindsight!"

    # 2. Serial
    env_serial = RubiksCubeEnv(scramble_depth=2)
    wrapper_serial = UniversalS1Wrapper(env_serial)
    agent_serial = UniversalS1Agent(obs_space=wrapper_serial.observation_space, action_space=wrapper_serial.action_space)

    trainer_serial = RecurrentPPOTrainer(
        agent=agent_serial,
        env=wrapper_serial,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
        hindsight_replay=True,
    )

    obs_dict_s, _ = wrapper_serial.reset(seed=42)
    obs_dict_s, hx_s, ep_start_s, mean_ret_s = trainer_serial.collect_rollouts(
        current_obs_dict=obs_dict_s,
        current_hx=None,
        episode_start=True,
    )

    adv_s = trainer_serial.buffer.advantages
    assert not np.isnan(adv_s).any()
    assert not np.isinf(adv_s).any()
