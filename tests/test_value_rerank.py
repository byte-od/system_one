import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent, ReflexDecision
from system1_engine.env.adapters.rubiks import RubiksCubeEnv, RubiksCubeMacroEnv, VectorizedRubiksEnv
from system1_engine.env.wrapper import UniversalS1Wrapper


def test_rubiks_env_simulate_step():
    env = RubiksCubeEnv(curriculum=False, scramble_depth=3)
    obs, info = env.reset(seed=42)
    orig_state = env.core.state.copy()
    orig_steps = env._steps

    # Testa simulação em múltiplos movimentos atômicos
    for act in range(12):
        sim_obs, rew, is_solved, s_info = env.simulate_step(act)
        assert sim_obs.shape == (324,)
        assert isinstance(rew, float)
        assert isinstance(is_solved, bool)
        # O estado do cubo DEVE permanecer rigorosamente inalterado após simulate_step
        assert np.array_equal(env.core.state, orig_state), f"Falha na reversão exata para ação {act}"
        assert env._steps == orig_steps


def test_rubiks_macro_env_simulate_step():
    env = RubiksCubeMacroEnv(curriculum=False, scramble_depth=2)
    obs, info = env.reset(seed=123)
    orig_state = env.core.state.copy()

    for act in range(len(env.core.MACRO_NAMES)):
        sim_obs, rew, is_solved, s_info = env.simulate_step(act)
        assert sim_obs.shape == (324,)
        assert np.array_equal(env.core.state, orig_state)


def test_vectorized_rubiks_env_simulate_step():
    vec_env = VectorizedRubiksEnv(num_envs=4, scramble_depth=2)
    obs, info = vec_env.reset(seed=99)
    orig_state = vec_env.envs[0].core.state.copy()

    sim_obs, rew, is_solved, s_info = vec_env.simulate_step(3, env_idx=0)
    assert sim_obs.shape == (324,)
    assert np.array_equal(vec_env.envs[0].core.state, orig_state)


def test_wrapper_simulate_step():
    raw_env = RubiksCubeEnv(curriculum=False, scramble_depth=2)
    wrapped = UniversalS1Wrapper(raw_env)
    obs_dict, _ = wrapped.reset(seed=42)

    init_obs_copy = wrapped.init_obs.copy()
    prev_act_copy = wrapped.prev_action
    prev_rew_copy = wrapped.prev_reward

    sim_dict, rew, done, info = wrapped.simulate_step(0)
    assert isinstance(sim_dict, dict)
    assert "obs" in sim_dict
    assert "delta_obs" in sim_dict
    assert sim_dict["obs"].shape == (324,)
    assert sim_dict["delta_obs"].shape == (324,)

    # Estado persistente do wrapper não pode sofrer mutação
    assert np.array_equal(wrapped.init_obs, init_obs_copy)
    assert wrapped.prev_action == prev_act_copy
    assert wrapped.prev_reward == prev_rew_copy


def test_agent_act_fast_value_rerank():
    raw_env = RubiksCubeEnv(curriculum=False, scramble_depth=3)
    wrapped = UniversalS1Wrapper(raw_env)
    obs_dict, _ = wrapped.reset(seed=42)

    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
    )
    agent.eval()

    # 1. Execução sem value_rerank
    dec_plain = agent.act_with_confidence(obs_dict, value_rerank=False)
    assert isinstance(dec_plain, ReflexDecision)
    assert dec_plain.value_rerank_triggered is False

    # 2. Execução com value_rerank e env fornecido diretamente
    dec_rerank = agent.act_with_confidence(
        obs_dict,
        value_rerank=True,
        value_rerank_top_k=2,
        value_rerank_threshold=1.0, # força o disparo do re-ranking
        env=wrapped,
    )
    assert isinstance(dec_rerank, ReflexDecision)
    assert dec_rerank.value_rerank_triggered is True
    assert 0 <= dec_rerank.action < 12


def test_agent_set_env_convenience():
    raw_env = RubiksCubeEnv(curriculum=False, scramble_depth=3)
    wrapped = UniversalS1Wrapper(raw_env)
    obs_dict, _ = wrapped.reset(seed=100)

    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
    )
    agent.set_env(wrapped)
    assert agent.default_env is wrapped

    # Chama sem passar env explicitamente no act_fast
    dec = agent.act_fast(
        obs_dict,
        value_rerank=True,
        value_rerank_top_k=2,
        value_rerank_threshold=1.0,
        return_decision=True,
    )
    assert isinstance(dec, ReflexDecision)
    assert dec.value_rerank_triggered is True


def test_value_rerank_immediate_solve_d1():
    raw_env = RubiksCubeEnv(curriculum=False, scramble_depth=1)
    wrapped = UniversalS1Wrapper(raw_env)
    obs_dict, _ = wrapped.reset(seed=42)

    # Identifica o movimento que resolve imediatamente o cubo D1
    last_scramble = raw_env._last_scramble_moves[0]
    inv_scramble_name = raw_env.core.INVERSE_ATOMIC[last_scramble]
    target_solve_action = raw_env.core.ATOMIC_MOVES.index(inv_scramble_name)

    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
    )
    agent.eval()

    # Passa simulador com candidatos incluindo o movimento de solução imediata
    decision = agent.act_with_confidence(
        obs_dict,
        value_rerank=True,
        value_rerank_top_k=12, # cobre todas para garantir que encontra a solução
        value_rerank_threshold=1.0,
        env=wrapped,
    )
    assert decision.value_rerank_triggered is True
    assert decision.action == target_solve_action
