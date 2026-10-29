"""Testes formais para Contrastive Online DPO-SIL (c-SIL) e Buffer com Capacidade Progressiva."""

import pytest
import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry.tracker import LiveStatsTracker
from system1_engine.training.buffer import (
    FrontierTransition,
    FrontierTransitionBuffer,
    SuccessReplayBuffer,
    SuccessfulTrajectory,
)
from system1_engine.training.ppo import RecurrentPPOTrainer


def test_successful_trajectory_log_probs():
    """Valida armazenamento e preservação de log_probs na trajetória."""
    obs = np.ones((5, 10), dtype=np.float32)
    actions = np.zeros(5, dtype=np.int64)
    returns = np.ones(5, dtype=np.float32) * 10.0
    log_probs = np.array([-0.5, -0.4, -0.6, -0.2, -0.3], dtype=np.float32)

    traj = SuccessfulTrajectory(
        obs=obs,
        delta_obs=None,
        prev_actions=np.zeros(5, dtype=np.int64),
        prev_rewards=np.zeros(5, dtype=np.float32),
        actions=actions,
        returns=returns,
        depth=3,
        length=5,
        log_probs=log_probs,
    )
    assert traj.log_probs is not None
    assert len(traj.log_probs) == 5
    np.testing.assert_allclose(traj.log_probs, log_probs)


def test_progressive_buffer_capacity():
    """Valida a escala progressiva de capacidade por estrato curricular."""
    buf = SuccessReplayBuffer(
        max_trajectories=2048,
        max_per_depth=512,
        progressive_capacity=True,
    )
    # D1 deve ter cota reduzida (32), liberando slots para profundidades mais altas
    assert buf.get_capacity_for_depth(1) == 32
    # D2: 64
    assert buf.get_capacity_for_depth(2) == 64
    # D3: 128
    assert buf.get_capacity_for_depth(3) == 128
    # D4: 256
    assert buf.get_capacity_for_depth(4) == 256
    # D5+: limitado a max_per_depth (512)
    assert buf.get_capacity_for_depth(5) == 512
    assert buf.get_capacity_for_depth(7) == 512
    assert buf.get_capacity_for_depth(10) == 512

    # Se progressive_capacity for desativado, retorna max_per_depth fixo
    buf_fixed = SuccessReplayBuffer(
        max_trajectories=2048,
        max_per_depth=512,
        progressive_capacity=False,
    )
    assert buf_fixed.get_capacity_for_depth(1) == 512
    assert buf_fixed.get_capacity_for_depth(4) == 512


def test_progressive_buffer_enforces_quota():
    """Garante que o estrato D1 respeita a cota de 32 no modo progressivo."""
    buf = SuccessReplayBuffer(
        max_trajectories=2048,
        max_per_depth=512,
        progressive_capacity=True,
    )
    # Insere 40 trajetórias no nível D1 (comprimento 2, permitido para D1)
    for i in range(40):
        traj = SuccessfulTrajectory(
            obs=np.zeros((2, 10), dtype=np.float32),
            delta_obs=None,
            prev_actions=np.zeros(2, dtype=np.int64),
            prev_rewards=np.zeros(2, dtype=np.float32),
            actions=np.zeros(2, dtype=np.int64),
            returns=np.ones(2, dtype=np.float32) * (10.0 + i),
            depth=1,
            length=2,
        )
        buf.add(traj)

    # Capacidade do estrato 1 no modo progressivo é exatamente 32
    assert len(buf._depth_buffers[1]) == 32
    # E deve ter retido as 32 melhores (com retornos mais altos: 18.0 até 49.0)
    assert min(t.returns[0] for t in buf._depth_buffers[1]) >= 18.0


def test_failure_buffer_ops():
    """Valida armazenamento de trajetórias negativas (falhas) por estrato."""
    buf = SuccessReplayBuffer(
        max_trajectories=128,
        max_per_depth=64,
        max_failures_per_depth=32,
    )
    assert len(buf.failures) == 0
    assert buf.get_failure_depth_distribution() == {}

    # Adiciona 10 falhas em D2 e 5 falhas em D4
    for i in range(10):
        buf.add_failure(
            SuccessfulTrajectory(
                obs=np.zeros((6, 10), dtype=np.float32),
                delta_obs=None,
                prev_actions=np.zeros(6, dtype=np.int64),
                prev_rewards=np.zeros(6, dtype=np.float32),
                actions=np.zeros(6, dtype=np.int64),
                returns=np.ones(6, dtype=np.float32) * -1.0,
                depth=2,
                length=6,
            )
        )
    for i in range(5):
        buf.add_failure(
            SuccessfulTrajectory(
                obs=np.zeros((8, 10), dtype=np.float32),
                delta_obs=None,
                prev_actions=np.zeros(8, dtype=np.int64),
                prev_rewards=np.zeros(8, dtype=np.float32),
                actions=np.zeros(8, dtype=np.int64),
                returns=np.ones(8, dtype=np.float32) * -2.0,
                depth=4,
                length=8,
            )
        )

    assert len(buf.failures) == 15
    dist = buf.get_failure_depth_distribution()
    assert dist == {2: 10, 4: 5}

    # Limpeza zera falhas
    buf.clear()
    assert len(buf.failures) == 0
    assert buf.get_failure_depth_distribution() == {}


def test_sample_contrastive_pairs_matching_stratum():
    """Valida que pares contrastivos (vitória, falha) são estritamente emparelhados pelo mesmo estrato."""
    buf = SuccessReplayBuffer(
        max_trajectories=128,
        obs_shape=(10,),
        action_shape=(),
        device=torch.device("cpu"),
        hidden_dim=64,
    )

    # Cenário 1: apenas vitórias em D2 e falhas em D3 (sem estrato comum) -> Deve retornar None
    buf.add(
        SuccessfulTrajectory(
            obs=np.zeros((4, 10), dtype=np.float32),
            delta_obs=None,
            prev_actions=np.zeros(4, dtype=np.int64),
            prev_rewards=np.zeros(4, dtype=np.float32),
            actions=np.zeros(4, dtype=np.int64),
            returns=np.ones(4, dtype=np.float32) * 10.0,
            depth=2,
            length=4,
        )
    )
    buf.add_failure(
        SuccessfulTrajectory(
            obs=np.zeros((6, 10), dtype=np.float32),
            delta_obs=None,
            prev_actions=np.zeros(6, dtype=np.int64),
            prev_rewards=np.zeros(6, dtype=np.float32),
            actions=np.zeros(6, dtype=np.int64),
            returns=np.ones(6, dtype=np.float32) * -1.0,
            depth=3,
            length=6,
        )
    )
    assert buf.sample_contrastive_pairs(batch_size=4) is None

    # Cenário 2: adiciona vitória em D3 e falha em D2 (ambos os níveis agora têm par)
    buf.add(
        SuccessfulTrajectory(
            obs=np.ones((5, 10), dtype=np.float32) * 3,
            delta_obs=None,
            prev_actions=np.zeros(5, dtype=np.int64),
            prev_rewards=np.zeros(5, dtype=np.float32),
            actions=np.zeros(5, dtype=np.int64),
            returns=np.ones(5, dtype=np.float32) * 12.0,
            depth=3,
            length=5,
            log_probs=np.full(5, -0.5, dtype=np.float32),
        )
    )
    buf.add_failure(
        SuccessfulTrajectory(
            obs=np.ones((7, 10), dtype=np.float32) * 2,
            delta_obs=None,
            prev_actions=np.zeros(7, dtype=np.int64),
            prev_rewards=np.zeros(7, dtype=np.float32),
            actions=np.zeros(7, dtype=np.int64),
            returns=np.ones(7, dtype=np.float32) * -2.0,
            depth=2,
            length=7,
            log_probs=np.full(7, -0.7, dtype=np.float32),
        )
    )

    pairs = buf.sample_contrastive_pairs(batch_size=4)
    assert pairs is not None
    batch_win, batch_loss = pairs

    assert "obs" in batch_win and "obs" in batch_loss
    assert "actions" in batch_win and "actions" in batch_loss
    assert "old_log_probs" in batch_win and "old_log_probs" in batch_loss
    assert "mask" in batch_win and "mask" in batch_loss
    assert batch_win["obs"].shape[0] == batch_loss["obs"].shape[0]


def test_dpo_loss_gradient_flow():
    """Valida o cálculo do loss DPO e a propagação correta de gradientes."""
    import gymnasium as gym

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    act_space = gym.spaces.Discrete(4)
    agent = UniversalS1Agent(obs_space=obs_space, action_space=act_space, trunk_type="lean")

    buf = SuccessReplayBuffer(
        max_trajectories=32,
        obs_shape=(10,),
        action_shape=(),
        device=torch.device("cpu"),
        hidden_dim=agent.trunk.hidden_dim,
    )

    # Trajetória vitoriosa curta (4 passos)
    obs_w = np.random.randn(4, 10).astype(np.float32)
    buf.add(
        SuccessfulTrajectory(
            obs=obs_w,
            delta_obs=np.zeros_like(obs_w),
            prev_actions=np.zeros(4, dtype=np.int64),
            prev_rewards=np.zeros(4, dtype=np.float32),
            actions=np.array([0, 1, 2, 3], dtype=np.int64),
            returns=np.ones(4, dtype=np.float32) * 10.0,
            depth=2,
            length=4,
            log_probs=np.full(4, -1.38, dtype=np.float32),
        )
    )

    # Trajetória derrotada longa (8 passos)
    obs_l = np.random.randn(8, 10).astype(np.float32)
    buf.add_failure(
        SuccessfulTrajectory(
            obs=obs_l,
            delta_obs=np.zeros_like(obs_l),
            prev_actions=np.zeros(8, dtype=np.int64),
            prev_rewards=np.zeros(8, dtype=np.float32),
            actions=np.array([3, 2, 1, 0, 3, 2, 1, 0], dtype=np.int64),
            returns=np.ones(8, dtype=np.float32) * -1.0,
            depth=2,
            length=8,
            log_probs=np.full(8, -1.38, dtype=np.float32),
        )
    )

    pairs = buf.sample_contrastive_pairs(batch_size=2)
    assert pairs is not None
    win_b, loss_b = pairs

    # Forward pass em ambas
    win_in = {
        "obs": win_b["obs"],
        "delta_obs": win_b["delta_obs"],
        "prev_action": win_b["prev_actions"],
        "prev_reward": win_b["prev_rewards"],
    }
    loss_in = {
        "obs": loss_b["obs"],
        "delta_obs": loss_b["delta_obs"],
        "prev_action": loss_b["prev_actions"],
        "prev_reward": loss_b["prev_rewards"],
    }

    win_dist, _, _ = agent(win_in, hx=win_b["hx"], dones=win_b["dones"])
    loss_dist, _, _ = agent(loss_in, hx=loss_b["hx"], dones=loss_b["dones"])

    win_logp = win_dist.log_prob(win_b["actions"])
    loss_logp = loss_dist.log_prob(loss_b["actions"])

    win_len = win_b["mask"].sum(dim=-1).clamp(min=1.0)
    loss_len = loss_b["mask"].sum(dim=-1).clamp(min=1.0)

    # Length normalization
    h_w = ((win_logp - win_b["old_log_probs"]) * win_b["mask"]).sum(dim=-1) / win_len
    h_l = ((loss_logp - loss_b["old_log_probs"]) * loss_b["mask"]).sum(dim=-1) / loss_len

    beta = 0.10
    dpo_logits = beta * (h_w - h_l)
    dpo_loss = -torch.nn.functional.logsigmoid(dpo_logits).mean()

    assert torch.isfinite(dpo_loss)
    assert dpo_loss.item() > 0.0

    dpo_loss.backward()

    # Verifica que gradientes fluem normalmente
    grad_norms = [p.grad.norm().item() for p in agent.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(np.isfinite(g) for g in grad_norms)
    assert any(g > 0.0 for g in grad_norms)


def test_recurrent_ppo_trainer_with_dpo_enabled():
    """Valida integração ponta a ponta do RecurrentPPOTrainer com sil_dpo_enabled=True."""
    import gymnasium as gym

    raw_env = gym.make("CartPole-v1")
    wrapped_env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(obs_space=wrapped_env.observation_space, action_space=wrapped_env.action_space, trunk_type="lean")

    tracker = LiveStatsTracker()
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped_env,
        learning_rate=1e-3,
        rollout_steps=128,
        chunk_length=16,
        chunk_batch_size=4,
        sil_enabled=True,
        sil_dpo_enabled=True,
        dpo_coef=0.08,
        dpo_beta=0.10,
        dpo_batch_size=4,
        sil_min_samples=2,
        progressive_buffer_capacity=True,
        tracker=tracker,
    )
    assert trainer.sil_dpo_enabled is True
    assert trainer.dpo_coef == 0.08

    # Executa 1 ciclo de coleta e 1 ciclo de treino
    init_obs, _ = wrapped_env.reset()
    trainer.collect_rollouts(current_obs_dict=init_obs, current_hx=None, episode_start=True)
    # Adiciona sinteticamente 1 par para garantir execução do passo DPO
    obs_dim = wrapped_env.observation_space["obs"].shape[0]
    delta_dim = wrapped_env.observation_space["delta_obs"].shape[0]
    trainer.sil_buffer.add(
        SuccessfulTrajectory(
            obs=np.zeros((6, obs_dim), dtype=np.float32),
            delta_obs=np.zeros((6, delta_dim), dtype=np.float32),
            prev_actions=np.zeros(6, dtype=np.int64),
            prev_rewards=np.zeros(6, dtype=np.float32),
            actions=np.zeros(6, dtype=np.int64),
            returns=np.ones(6, dtype=np.float32) * 5.0,
            depth=1,
            length=6,
            log_probs=np.full(6, -0.69, dtype=np.float32),
        )
    )
    trainer.sil_buffer.add_failure(
        SuccessfulTrajectory(
            obs=np.zeros((6, obs_dim), dtype=np.float32),
            delta_obs=np.zeros((6, delta_dim), dtype=np.float32),
            prev_actions=np.zeros(6, dtype=np.int64),
            prev_rewards=np.zeros(6, dtype=np.float32),
            actions=np.zeros(6, dtype=np.int64),
            returns=np.ones(6, dtype=np.float32) * -1.0,
            depth=1,
            length=6,
            log_probs=np.full(6, -0.69, dtype=np.float32),
        )
    )

    metrics = trainer.train_epoch()
    assert "sil_dpo_loss" in metrics
    assert "sil_failures" in metrics
    assert metrics["sil_failures"] >= 1
    assert metrics["sil_dpo_loss"] >= 0.0

    snap = tracker.snapshot()
    assert "sil_dpo_loss" in snap
    assert "sil_failures" in snap


def test_progressive_buffer_capacity_scaling_to_2048():
    """Valida o escalonamento exponencial suave ate C_max=2048 nos estratos de fronteira."""
    buf = SuccessReplayBuffer(
        max_trajectories=10240,
        max_per_depth=2048,
        min_capacity_per_depth=32,
        progressive_capacity=True,
    )
    # D1..D4 crescem exponencialmente
    assert buf.get_capacity_for_depth(1) == 32
    assert buf.get_capacity_for_depth(2) == 64
    assert buf.get_capacity_for_depth(3) == 128
    assert buf.get_capacity_for_depth(4) == 256
    assert buf.get_capacity_for_depth(5) == 512
    assert buf.get_capacity_for_depth(6) == 1024
    # D7 em diante atingem e mantem o teto C_max=2048
    assert buf.get_capacity_for_depth(7) == 2048
    assert buf.get_capacity_for_depth(10) == 2048
    assert buf.get_capacity_for_depth(20) == 2048


def test_progressive_failure_capacity():
    """Valida cotas progressivas de falhas para DPO (16 no D1 ate 512 no D6+)."""
    buf = SuccessReplayBuffer(
        max_trajectories=10240,
        max_failures_per_depth=512,
        min_failures_per_depth=16,
        progressive_capacity=True,
    )
    assert buf.get_failure_capacity_for_depth(1) == 16
    assert buf.get_failure_capacity_for_depth(2) == 32
    assert buf.get_failure_capacity_for_depth(3) == 64
    assert buf.get_failure_capacity_for_depth(4) == 128
    assert buf.get_failure_capacity_for_depth(5) == 256
    assert buf.get_failure_capacity_for_depth(6) == 512
    assert buf.get_failure_capacity_for_depth(10) == 512

    # Insere 25 falhas em D1; deve reter apenas 16 (cota de D1)
    for i in range(25):
        buf.add_failure(
            SuccessfulTrajectory(
                obs=np.zeros((2, 4), dtype=np.float32),
                delta_obs=None,
                prev_actions=np.zeros(2, dtype=np.int64),
                prev_rewards=np.zeros(2, dtype=np.float32),
                actions=np.zeros(2, dtype=np.int64),
                returns=np.ones(2, dtype=np.float32) * -1.0,
                depth=1,
                length=2,
            )
        )
    assert len(buf._failure_buffers[1]) == 16


def test_anchor_floor_protection_during_global_eviction():
    """Garante que a expulsao global por excesso de max_trajectories nunca extingue estratos fundacionais (piso >= 8)."""
    buf = SuccessReplayBuffer(
        max_trajectories=20,  # Limite global pequeno para disparar eviction
        max_per_depth=2048,
        min_capacity_per_depth=32,
        progressive_capacity=True,
    )

    # Adiciona 12 trajetórias em D1 (piso de âncora é 8, portanto há 4 excedentes)
    for i in range(12):
        buf.add(
            SuccessfulTrajectory(
                obs=np.zeros((2, 4), dtype=np.float32),
                delta_obs=None,
                prev_actions=np.zeros(2, dtype=np.int64),
                prev_rewards=np.zeros(2, dtype=np.float32),
                actions=np.zeros(2, dtype=np.int64),
                returns=np.ones(2, dtype=np.float32) * (10.0 + i),
                depth=1,
                length=2,
            )
        )
    assert len(buf._depth_buffers[1]) == 12

    # Agora adiciona 15 trajetórias em D2. O total seria 12 + 15 = 27 > 20.
    # O eviction global deve expurgar 4 excedentes de D1 (deixando D1 no piso 8)
    # e 3 de D2 (deixando D2 com 12). D1 NUNCA deve ser zerado!
    for i in range(15):
        buf.add(
            SuccessfulTrajectory(
                obs=np.zeros((3, 4), dtype=np.float32),
                delta_obs=None,
                prev_actions=np.zeros(3, dtype=np.int64),
                prev_rewards=np.zeros(3, dtype=np.float32),
                actions=np.zeros(3, dtype=np.int64),
                returns=np.ones(3, dtype=np.float32) * (20.0 + i),
                depth=2,
                length=3,
            )
        )

    assert len(buf) == 20
    # D1 preservou seu piso de âncora de 8 trajetórias
    assert len(buf._depth_buffers[1]) == 8
    # D2 manteve 12 trajetórias
    assert len(buf._depth_buffers[2]) == 12
    # Ambas profundidades continuam presentes e limitadas pelo max_trajectories=20
    caps = buf.get_depth_capacities()
    assert caps[1] == 20
    assert caps[2] == 20


def test_full_win_priority_over_peak_slice():
    """Garante que vitórias totais têm prioridade estrita de retenção sobre fatias de pico parciais."""
    buf = SuccessReplayBuffer(
        max_trajectories=100,
        max_per_depth=2,
        progressive_capacity=False,
    )

    # Adiciona 2 fatias parciais em D5
    p1 = FrontierTransition(
        obs=np.zeros((3, 4), dtype=np.float32),
        delta_obs=None,
        prev_actions=np.zeros(3, dtype=np.int64),
        prev_rewards=np.zeros(3, dtype=np.float32),
        actions=np.zeros(3, dtype=np.int64),
        returns=np.ones(3, dtype=np.float32) * 5.0,
        depth=5,
        length=3,
        is_full_win=False,
    )
    p2 = FrontierTransition(
        obs=np.zeros((4, 4), dtype=np.float32),
        delta_obs=None,
        prev_actions=np.zeros(4, dtype=np.int64),
        prev_rewards=np.zeros(4, dtype=np.float32),
        actions=np.zeros(4, dtype=np.int64),
        returns=np.ones(4, dtype=np.float32) * 6.0,
        depth=5,
        length=4,
        is_full_win=False,
    )
    buf.add(p1)
    buf.add(p2)
    assert len(buf._depth_buffers[5]) == 2
    assert all(not getattr(t, "is_full_win", True) for t in buf._depth_buffers[5])

    # Agora adiciona uma vitória completa (is_full_win=True) com retorno mesmo menor (ex: 2.0)
    w1 = FrontierTransition(
        obs=np.zeros((5, 4), dtype=np.float32),
        delta_obs=None,
        prev_actions=np.zeros(5, dtype=np.int64),
        prev_rewards=np.zeros(5, dtype=np.float32),
        actions=np.zeros(5, dtype=np.int64),
        returns=np.ones(5, dtype=np.float32) * 2.0,
        depth=5,
        length=5,
        is_full_win=True,
    )
    buf.add(w1)

    # A vitória completa deve ter entrado no buffer, substituindo uma das parciais
    assert len(buf._depth_buffers[5]) == 2
    wins = [t for t in buf._depth_buffers[5] if getattr(t, "is_full_win", False)]
    assert len(wins) == 1
    assert wins[0] is w1


def test_process_completed_trajectory_peak_slicing_isolated_bellman():
    """Valida Peak Slicing com Bellman Isolado e Bootstrapping de Valor em PPOTrainer."""
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    wrapped_env = UniversalS1Wrapper(env, tri_compass=False)
    agent = UniversalS1Agent(
        obs_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        trunk_type="lean",
    )
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped_env,
        sil_enabled=True,
        sil_dpo_enabled=True,
        gamma=0.99,
    )

    # Cria passos simulados de um episódio incompleto (T=6 passos):
    # Passos 0 a 2 progridem (recompensas positivas), passos 3 a 5 degradam (recompensas negativas)
    traj_steps = [
        {"obs": np.zeros(4, dtype=np.float32), "delta_obs": None, "prev_action": 0, "prev_reward": 0.0, "action": 0, "reward": 0.20, "log_prob": -0.5, "value": 1.0},
        {"obs": np.ones(4, dtype=np.float32), "delta_obs": None, "prev_action": 0, "prev_reward": 0.20, "action": 1, "reward": 0.30, "log_prob": -0.6, "value": 1.5},
        {"obs": np.ones(4, dtype=np.float32) * 2, "delta_obs": None, "prev_action": 1, "prev_reward": 0.30, "action": 0, "reward": 0.50, "log_prob": -0.4, "value": 2.5},  # t* = 2 (pico: cumsum = 1.00)
        {"obs": np.ones(4, dtype=np.float32) * 3, "delta_obs": None, "prev_action": 0, "prev_reward": 0.50, "action": 1, "reward": -0.20, "log_prob": -0.7, "value": 2.0},
        {"obs": np.ones(4, dtype=np.float32) * 4, "delta_obs": None, "prev_action": 1, "prev_reward": -0.20, "action": 0, "reward": -0.30, "log_prob": -0.8, "value": 1.2},
        {"obs": np.ones(4, dtype=np.float32) * 5, "delta_obs": None, "prev_action": 0, "prev_reward": -0.30, "action": 1, "reward": -0.40, "log_prob": -0.9, "value": 0.5},
    ]

    trainer._process_completed_trajectory(traj_steps=traj_steps, is_win=False, cur_depth=4)

    # 1. Verifica se a fatia positiva de pico foi adicionada ao SIL
    assert len(trainer.sil_buffer) == 1
    stored_slice = trainer.sil_buffer.trajectories[0]
    # Comprimento deve ser exatamente t* + 1 = 3 passos
    assert stored_slice.length == 3
    assert not getattr(stored_slice, "is_full_win", True)

    # 2. Verifica se o retorno de Bellman foi calculado isoladamente com bootstrapping de V(s_{t*+1}) = 2.0
    # No passo t* = 2: G_2 = r_2 + gamma * V(s_3) = 0.50 + 0.99 * 2.0 = 2.48
    # No passo t = 1: G_1 = r_1 + gamma * G_2 = 0.30 + 0.99 * 2.48 = 2.7552
    # No passo t = 0: G_0 = r_0 + gamma * G_1 = 0.20 + 0.99 * 2.7552 = 2.9276
    expected_G2 = 0.50 + 0.99 * 2.0
    expected_G1 = 0.30 + 0.99 * expected_G2
    expected_G0 = 0.20 + 0.99 * expected_G1

    np.testing.assert_allclose(stored_slice.returns[2], expected_G2, rtol=1e-4)
    np.testing.assert_allclose(stored_slice.returns[1], expected_G1, rtol=1e-4)
    np.testing.assert_allclose(stored_slice.returns[0], expected_G0, rtol=1e-4)

    # 3. Verifica que a cauda de degradação pós-pico é descartada (armazenam-se apenas vitórias e fatias de sucesso)
    assert len(trainer.sil_buffer.failures) == 0


def test_process_completed_trajectory_no_progress_rejection():
    """Garante que episódios sem ganho líquido (apenas penalidades) não geram fatia positiva no SIL."""
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    wrapped_env = UniversalS1Wrapper(env, tri_compass=False)
    agent = UniversalS1Agent(
        obs_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        trunk_type="lean",
    )
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped_env,
        sil_enabled=True,
        sil_dpo_enabled=False,
    )

    # Episódio que só deu recompensas negativas desde o início
    traj_steps = [
        {"obs": np.zeros(4, dtype=np.float32), "delta_obs": None, "prev_action": 0, "prev_reward": 0.0, "action": 0, "reward": -0.05, "log_prob": -0.5, "value": 0.5},
        {"obs": np.ones(4, dtype=np.float32), "delta_obs": None, "prev_action": 0, "prev_reward": -0.05, "action": 1, "reward": -0.10, "log_prob": -0.6, "value": 0.3},
        {"obs": np.ones(4, dtype=np.float32) * 2, "delta_obs": None, "prev_action": 1, "prev_reward": -0.10, "action": 0, "reward": -0.20, "log_prob": -0.4, "value": 0.1},
    ]

    trainer._process_completed_trajectory(traj_steps=traj_steps, is_win=False, cur_depth=3)

    # Nenhuma fatia positiva adicionada
    assert len(trainer.sil_buffer) == 0


