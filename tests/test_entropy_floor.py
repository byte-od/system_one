"""Testes formais para Opção 2: Dynamic Entropy Floor (Piso Dinâmico de Entropia e Re-aquecimento Automático)."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry.tracker import LiveStatsTracker
from system1_engine.training.ppo import RecurrentPPOTrainer


def _create_test_env_and_trainer(
    entropy_floor_enabled: bool = True,
    entropy_floor: float = 0.20,
    entropy_floor_coef: float = 0.50,
    adaptive_entropy_coef: bool = True,
    base_entropy_coef: float = 0.01,
):
    raw_env = gym.make("CartPole-v1")
    wrapped_env = UniversalS1Wrapper(raw_env, tri_compass=False)
    tracker = LiveStatsTracker()
    agent = UniversalS1Agent(
        obs_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        trunk_type="lean",
    )
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped_env,
        rollout_steps=32,
        chunk_length=8,
        chunk_batch_size=2,
        entropy_coef=base_entropy_coef,
        tracker=tracker,
        entropy_floor_enabled=entropy_floor_enabled,
        entropy_floor=entropy_floor,
        entropy_floor_coef=entropy_floor_coef,
        adaptive_entropy_coef=adaptive_entropy_coef,
        min_entropy_coef=0.003,
        max_entropy_coef=0.080,
    )
    return wrapped_env, agent, trainer, tracker


def test_entropy_floor_initialization():
    """Valida inicialização dos parâmetros do Dynamic Entropy Floor e propagação à telemetria."""
    _, _, trainer, tracker = _create_test_env_and_trainer(
        entropy_floor_enabled=True,
        entropy_floor=0.25,
        entropy_floor_coef=0.60,
        adaptive_entropy_coef=True,
    )

    assert trainer.entropy_floor_enabled is True
    assert trainer.entropy_floor == 0.25
    assert trainer.entropy_floor_coef == 0.60
    assert trainer.adaptive_entropy_coef is True
    assert trainer.base_entropy_coef == 0.01
    assert tracker.metrics.entropy_floor == 0.25


def test_entropy_floor_barrier_inactive_when_healthy():
    """Valida que quando a entropia está acima do piso saudável, a perda de barreira é exatamente 0.0."""
    wrapped_env, agent, trainer, tracker = _create_test_env_and_trainer(
        entropy_floor_enabled=True,
        entropy_floor=0.20,
    )

    # Coleta rollouts iniciais (política aleatória/inicial com entropia alta ~ 0.69)
    obs_dict, _ = wrapped_env.reset()
    trainer.collect_rollouts(obs_dict, current_hx=None, episode_start=True)

    epoch_res = trainer.train_epoch()
    # Entropia de CartPole com 2 ações é ~ 0.69 > 0.20
    assert epoch_res["entropy_loss"] < -0.20  # entropy_loss = -H
    assert epoch_res["entropy_floor_loss"] == 0.0

    snap = tracker.snapshot()
    assert snap["entropy_floor_active"] is False
    assert snap["entropy_floor_loss"] == 0.0


def test_entropy_floor_barrier_active_when_entropy_drops_below_floor():
    """Valida que a barreira quadrática é ativada quando a entropia cai abaixo do piso, gerando perda restauradora."""
    wrapped_env, agent, trainer, tracker = _create_test_env_and_trainer(
        entropy_floor_enabled=True,
        entropy_floor=0.40,  # Piso alto para forçar ativação sob política semi-determinística
        entropy_floor_coef=1.0,
    )

    # Força a cabeça de política a emitir logits quase determinísticos (baixa entropia)
    with torch.no_grad():
        agent.policy_head.linear.weight[0].fill_(10.0)
        agent.policy_head.linear.weight[1].fill_(-10.0)

    obs_dict, _ = wrapped_env.reset()
    trainer.collect_rollouts(obs_dict, current_hx=None, episode_start=True)

    epoch_res = trainer.train_epoch()

    # Com logits [10, -10], a entropia é praticamente zero (< 0.01)
    assert epoch_res["entropy_floor_loss"] > 0.0
    snap = tracker.snapshot()
    assert snap["entropy_floor_active"] is True
    assert snap["entropy_floor_loss"] > 0.0


def test_adaptive_entropy_coef_reheating_and_relaxation():
    """Valida re-aquecimento dinâmico do coeficiente de entropia sob inanição e relaxamento posterior."""
    wrapped_env, agent, trainer, tracker = _create_test_env_and_trainer(
        entropy_floor_enabled=True,
        entropy_floor=0.30,
        adaptive_entropy_coef=True,
        base_entropy_coef=0.01,
    )

    initial_coef = trainer.entropy_coef
    assert initial_coef == 0.01

    # Simula colapso de entropia nos logits
    with torch.no_grad():
        agent.policy_head.linear.weight[0].fill_(8.0)
        agent.policy_head.linear.weight[1].fill_(-8.0)

    obs_dict, _ = wrapped_env.reset()
    trainer.collect_rollouts(obs_dict, current_hx=None, episode_start=True)

    # Época com baixa entropia deve re-aquecer o coeficiente
    trainer.train_epoch()
    assert trainer.entropy_coef > initial_coef

    # Restaura pesos equilibrados (alta entropia)
    with torch.no_grad():
        agent.policy_head.linear.weight.zero_()
        agent.policy_head.linear.bias.zero_()

    # Executa épocas saudáveis: coeficiente deve relaxar gradualmente em direção ao base_entropy_coef
    heated_coef = trainer.entropy_coef
    for _ in range(5):
        obs_dict, _ = wrapped_env.reset()
        trainer.collect_rollouts(obs_dict, current_hx=None, episode_start=True)
        trainer.train_epoch()

    assert trainer.entropy_coef < heated_coef
    assert trainer.entropy_coef >= trainer.base_entropy_coef


def test_rollout_thermal_reheating_on_entropy_starvation():
    """Valida o re-aquecimento térmico imediato do rollout sampler quando a entropia média recente é crítica."""
    _, _, trainer, tracker = _create_test_env_and_trainer(
        entropy_floor_enabled=True,
        entropy_floor=0.25,
    )

    # Configura escala de exploração em valor baixo
    trainer._adaptive_scale = 0.20
    trainer.exploration_scale = 0.20

    # Injeta entropias recentes muito baixas na telemetria (simulando colapso de exploração em rollout)
    for _ in range(60):
        tracker.record_inference(latency_us=100.0, uncertainty=0.01, confidence=0.99, entropy=0.03)

    # Chama atualização adaptativa de exploração
    trainer._update_adaptive_exploration(mean_return=1.0, env_infos={})

    # O mecanismo deve forçar o re-aquecimento da escala adaptativa para pelo menos 0.85
    assert trainer._adaptive_scale >= 0.85
    assert trainer.exploration_scale >= 0.85


def test_entropy_floor_disabled_behavior():
    """Valida que quando desativado, o piso não gera perda e o coeficiente permanece estático."""
    wrapped_env, agent, trainer, tracker = _create_test_env_and_trainer(
        entropy_floor_enabled=False,
        adaptive_entropy_coef=False,
        base_entropy_coef=0.01,
    )

    # Logits com baixa entropia
    with torch.no_grad():
        agent.policy_head.linear.weight[0].fill_(10.0)
        agent.policy_head.linear.weight[1].fill_(-10.0)

    obs_dict, _ = wrapped_env.reset()
    trainer.collect_rollouts(obs_dict, current_hx=None, episode_start=True)

    epoch_res = trainer.train_epoch()
    assert epoch_res["entropy_floor_loss"] == 0.0
    assert trainer.entropy_coef == 0.01
