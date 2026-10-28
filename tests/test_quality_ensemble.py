"""Testes formais do mecanismo de garantia de qualidade e ensemble multicalibrado."""

import pytest
import numpy as np
import gymnasium as gym
import torch

from system1_engine.core.agent import UniversalS1Agent, ReflexDecision
from system1_engine.core.ensemble import ConfidenceGatedEnsemble, compute_purity_score
from system1_engine.env.adapters.rubiks import RubiksCubeEnv
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.env.wrapper import UniversalS1Wrapper


def test_action_quality_filter_prevents_immediate_inverse():
    """Garante que o Action Quality Filter impede a anulação imediata do movimento anterior no cubo atômico."""
    env = RubiksCubeEnv(scramble_depth=2)
    obs, _ = env.reset(seed=42)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    obs_dict = {
        "obs": obs,
        "prev_action": 0,
        "prev_reward": 0.0,
        "delta_obs": None,
    }

    # Executa primeiro passo: suponha que escolheu ação A
    dec1 = agent.act_with_confidence(obs_dict, quality_filter=True)
    first_act = dec1.action

    # O inverso de first_act
    inv_act = (first_act + 1) if (first_act % 2 == 0) else (first_act - 1)

    # Executa segundo passo com o mesmo agente (histórico ativo)
    dec2 = agent.act_with_confidence(obs_dict, quality_filter=True)
    second_act = dec2.action

    # O segundo passo NÃO deve ser o inverso do primeiro, pois desfaria o cubo
    assert second_act != inv_act, f"Action Quality Filter falhou: escolheu inverso {second_act} de {first_act}"


def test_confidence_gated_ensemble_multicalibration_resolution():
    """Valida que o ConfidenceGatedEnsemble avalia e desempata ações sob incerteza em tempo sub-milissegundo."""
    env = RubiksCubeEnv(scramble_depth=1)
    obs, _ = env.reset(seed=123)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    obs_dict = {
        "obs": obs,
        "prev_action": 0,
        "prev_reward": 0.0,
        "delta_obs": None,
    }

    # Força incerteza para testar desempate térmico por ensemble no espaço latente
    decision = agent.act_fast(
        obs_dict,
        confidence_threshold=0.99,
        ensemble_fallback=True,
        return_decision=True,
    )
    assert decision.is_uncertain is True
    assert decision.ensemble_fallback_triggered is True
    assert decision.latency_ms < 1.0
    assert 0 <= decision.action < 12


def test_ppo_vectorized_exploration_scale_does_not_collapse():
    """Garante que a escala de exploração adaptativa no treino PPO vetorizado não colapsa para 0.15."""
    from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv

    vec_env = VectorizedRubiksEnv(num_envs=4, scramble_depth=2)
    wrapper_env = UniversalS1Wrapper(vec_env)

    agent = UniversalS1Agent(obs_space=wrapper_env.observation_space, action_space=wrapper_env.action_space)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapper_env,
        rollout_steps=32,
        chunk_length=8,
        chunk_batch_size=4,
        exploration_scale="auto",
    )

    assert trainer.is_auto_exploration is True
    assert trainer._adaptive_scale == 0.80

    obs_dict, _ = wrapper_env.reset(seed=42)

    # Executa múltiplos rollouts
    for _ in range(3):
        obs_dict, hx, ep_start, ret = trainer.collect_rollouts(
            current_obs_dict=obs_dict,
            current_hx=None,
            episode_start=True,
        )

    # A escala deve se manter saudável e NÃO cair abaixo do piso configurado (0.40)
    assert trainer._adaptive_scale >= 0.40, f"Escala colapsou para {trainer._adaptive_scale}"
    assert trainer.exploration_scale >= 0.40


def test_confidence_gated_ensemble_triggered_on_uncertainty():
    """Valida que o ConfidenceGatedEnsemble é acionado como fallback exclusivamente sob incerteza."""
    env = RubiksCubeEnv(scramble_depth=2)
    obs, _ = env.reset(seed=77)
    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    obs_dict = {
        "obs": obs,
        "prev_action": 0,
        "prev_reward": 0.0,
        "delta_obs": None,
    }

    # 1. Quando ensemble_fallback=False, nunca deve acionar
    dec_no_fallback = agent.act_with_confidence(
        obs_dict,
        confidence_threshold=0.99,  # Força is_uncertain = True
        ensemble_fallback=False,
    )
    assert dec_no_fallback.is_uncertain is True
    assert dec_no_fallback.ensemble_fallback_triggered is False

    # 2. Quando ensemble_fallback=True e sob incerteza, deve acionar o fallback instantâneo
    dec_with_fallback = agent.act_with_confidence(
        obs_dict,
        confidence_threshold=0.99,  # Força is_uncertain = True
        ensemble_fallback=True,
    )
    assert dec_with_fallback.is_uncertain is True
    assert dec_with_fallback.ensemble_fallback_triggered is True
    # Latência sub-milissegundo preservada
    assert dec_with_fallback.latency_ms < 1.0, f"Latência excedeu orçamento: {dec_with_fallback.latency_ms} ms"


def test_continuous_purity_score_penalizes_jitter_and_reversals():
    """Garante que a pureza contínua generalizada penaliza oscilações de cosseno e alto jerk."""
    # 1. Trajetória suave unidirecional
    smooth_traj = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.95, 0.05], dtype=np.float32),
        np.array([0.90, 0.10], dtype=np.float32),
        np.array([0.85, 0.15], dtype=np.float32),
    ]
    purity_smooth = compute_purity_score(smooth_traj, is_discrete=False)
    assert purity_smooth >= 0.95, f"Pureza suave muito baixa: {purity_smooth}"

    # 2. Trajetória com reversões bruscas de direção (cosine similarity negativa)
    reversing_traj = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0], dtype=np.float32),
    ]
    purity_reversing = compute_purity_score(reversing_traj, is_discrete=False)
    assert purity_reversing < purity_smooth, f"Pureza não penalizou reversões: {purity_reversing}"
    assert purity_reversing <= 0.60, f"Pureza de reversão esperada <= 0.60, obteve {purity_reversing}"


def test_discrete_purity_score_generalized_for_any_environment():
    """Valida o cálculo de pureza discreta agnóstico ao domínio (com e sem regras de cubo)."""
    # Trajetória suave sem repetições
    pure_discrete = [0, 1, 2, 3, 4, 5]
    purity_pure = compute_purity_score(pure_discrete, is_atomic_cube=False)
    assert purity_pure == 1.0

    # Trajetória com oscilação ping-pong 2-passos (A -> B -> A -> B)
    oscillating_discrete = [0, 1, 0, 1, 0]
    purity_osc = compute_purity_score(oscillating_discrete, is_atomic_cube=False)
    assert purity_osc < 1.0
    assert purity_osc <= 0.65

