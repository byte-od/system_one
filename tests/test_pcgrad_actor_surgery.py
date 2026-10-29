"""Testes formais da Opção 3: Cirurgia de Gradiente Simétrica por Curvatura no Caminho do Ator + Crítico 100% PPO Puro."""

import math
import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters.rubiks import RubiksCubeEnv
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.buffer import SuccessfulTrajectory
from system1_engine.training.ppo import RecurrentPPOTrainer


def _make_dummy_trainer(**kwargs) -> RecurrentPPOTrainer:
    env = RubiksCubeEnv(scramble_depth=2, curriculum=False)
    wrapped = UniversalS1Wrapper(env, tri_compass=True)
    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
        tri_compass=True,
    )
    defaults = dict(
        agent=agent,
        env=wrapped,
        rollout_steps=32,
        chunk_length=8,
        chunk_batch_size=2,
        sil_enabled=True,
        sil_coef=0.10,
        sil_batch_size=2,
        sil_min_samples=2,
        sil_dpo_enabled=True,
        dpo_coef=0.05,
        dpo_beta=0.10,
        dpo_batch_size=2,
        pcgrad_enabled=True,
        gradvac_max_alpha=0.25,
    )
    defaults.update(kwargs)
    return RecurrentPPOTrainer(**defaults)


def test_curvature_gated_projection_exact_pcgrad_at_alpha_zero():
    """Quando alpha_target == 0.0, a projeção reduz-se identicamente ao PCGrad 90° preservador de norma."""
    trainer = _make_dummy_trainer()

    p1 = nn.Parameter(torch.zeros(4))
    p2 = nn.Parameter(torch.zeros(4))
    # g = [-3, 4, 0, 0] (norm = 5.0), r = [2, 0, 0, 0] (norm = 2.0) -> cos = -0.6 < 0
    p1.grad = torch.tensor([-3.0, 4.0, 0.0, 0.0])
    p2.grad = torch.tensor([0.0, 0.0, 0.0, 0.0])
    r1 = torch.tensor([2.0, 0.0, 0.0, 0.0])
    r2 = torch.tensor([0.0, 0.0, 0.0, 0.0])

    orig_norm = math.sqrt(float((p1.grad ** 2).sum() + (p2.grad ** 2).sum()))
    projected, cos_obs = trainer._apply_curvature_gated_projection_inplace(
        [p1, p2], [r1, r2], alpha_target=0.0
    )

    assert projected is True
    assert cos_obs == pytest.approx(-0.6, abs=1e-5)

    g_after = torch.cat([p1.grad, p2.grad])
    r_cat = torch.cat([r1, r2])
    dot_after = float(torch.dot(g_after, r_cat).item())
    norm_after = float(torch.norm(g_after).item())

    # Ortogonalidade exata (<g_tilde, r> == 0) e preservação exata da norma (||g_tilde|| == 5.0)
    assert dot_after == pytest.approx(0.0, abs=1e-5)
    assert norm_after == pytest.approx(orig_norm, rel=1e-5)


def test_curvature_gated_projection_exact_gradvac_law_of_sines():
    """Quando alpha_target > 0.0, rotaciona exatamente até cos(g_tilde, r) == alpha_target preservando ||g||_2."""
    trainer = _make_dummy_trainer(gradvac_max_alpha=0.25)

    p = nn.Parameter(torch.zeros(6))
    # g com cos = -0.5 em relação a r, e componente ortogonal bem definida
    r = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p.grad = torch.tensor([-2.0, 2.0 * math.sqrt(3.0), 0.0, 0.0, 0.0, 0.0])  # norm = 4.0, cos = -0.5
    orig_norm = float(torch.norm(p.grad).item())

    target_alpha = 0.20
    projected, cos_obs = trainer._apply_curvature_gated_projection_inplace(
        [p], [r], alpha_target=target_alpha
    )

    assert projected is True
    assert cos_obs == pytest.approx(-0.5, abs=1e-5)

    new_norm = float(torch.norm(p.grad).item())
    new_cos = float(torch.dot(p.grad, r).item() / (new_norm * float(torch.norm(r).item())))
    assert new_norm == pytest.approx(orig_norm, rel=1e-5)
    assert new_cos == pytest.approx(target_alpha, abs=1e-5)


def test_curvature_gated_projection_antiparallel_collinear_safety():
    """Caso degenerado 1D perfeitamente antiparalelo (g = -k * r): cancela sem explodir ruído numérico."""
    trainer = _make_dummy_trainer()
    p = nn.Parameter(torch.zeros(3))
    r = torch.tensor([1.0, -2.0, 3.0])
    p.grad = -2.5 * r.clone()

    projected, cos_obs = trainer._apply_curvature_gated_projection_inplace(
        [p], [r], alpha_target=0.0
    )
    assert projected is True
    assert cos_obs == pytest.approx(-1.0, abs=1e-5)
    # Não pode apontar na direção negativa de r após a cirurgia
    assert float(torch.dot(p.grad, r).item()) >= -1e-5
    assert float(torch.norm(p.grad).item()) == pytest.approx(0.0, abs=1e-5)


def test_value_head_pure_on_policy_isolation_no_adam_leakage():
    """Garante isolamento bit-a-bit da ValueHead e do estado Adam durante consolidação SIL + DPO."""
    trainer = _make_dummy_trainer()
    obs_dim = trainer.env.observation_space["obs"].shape[0]
    delta_dim = trainer.env.observation_space["delta_obs"].shape[0]

    for _ in range(3):
        trainer.sil_buffer.add(
            SuccessfulTrajectory(
                obs=np.random.randn(5, obs_dim).astype(np.float32),
                delta_obs=np.random.randn(5, delta_dim).astype(np.float32),
                prev_actions=np.zeros(5, dtype=np.int64),
                prev_rewards=np.zeros(5, dtype=np.float32),
                actions=np.random.randint(0, 12, size=5),
                returns=np.linspace(8.0, 15.0, 5, dtype=np.float32),
                depth=2,
                length=5,
                log_probs=np.full(5, -2.4, dtype=np.float32),
            )
        )
        trainer.sil_buffer.add_failure(
            SuccessfulTrajectory(
                obs=np.random.randn(6, obs_dim).astype(np.float32),
                delta_obs=np.random.randn(6, delta_dim).astype(np.float32),
                prev_actions=np.zeros(6, dtype=np.int64),
                prev_rewards=np.zeros(6, dtype=np.float32),
                actions=np.random.randint(0, 12, size=6),
                returns=np.linspace(-2.0, -0.5, 6, dtype=np.float32),
                depth=2,
                length=6,
                log_probs=np.full(6, -2.4, dtype=np.float32),
            )
        )

    # Executa 1 rollout + train_epoch para inicializar buffers de momentum do Adam (exp_avg, exp_avg_sq)
    init_obs, _ = trainer.env.reset(seed=42)
    trainer.collect_rollouts(init_obs, current_hx=None, episode_start=True)
    first_metrics = trainer.train_epoch()
    assert first_metrics["sil_value_loss"] == 0.0

    # Captura snapshot exato dos pesos e estados do Adam da ValueHead
    _, value_params = trainer._get_actor_and_value_blocks()
    assert len(value_params) > 0
    val_weights_before = [vp.detach().clone() for vp in value_params]
    val_step_before = [
        trainer.optimizer.state[vp]["step"].clone()
        if isinstance(trainer.optimizer.state[vp]["step"], torch.Tensor)
        else trainer.optimizer.state[vp]["step"]
        for vp in value_params
    ]
    val_exp_avg_before = [trainer.optimizer.state[vp]["exp_avg"].detach().clone() for vp in value_params]

    # Simula isoladamente o passo de consolidação off-policy unificado (SIL + DPO)
    trainer.optimizer.zero_grad()
    mem_loss, sil_p, dpo_p = trainer._compute_unified_memory_loss(target_depth=2)
    assert mem_loss is not None
    assert sil_p >= 0.0
    assert dpo_p > 0.0
    mem_loss.backward()

    # Mesmo antes da limpeza explícita, sil_val_sq.detach() já não propaga gradiente para ValueHead
    for vp in value_params:
        assert vp.grad is None

    nn.utils.clip_grad_norm_(trainer.agent.parameters(), trainer.max_grad_norm)
    trainer.optimizer.step()

    # Pesos e estados do Adam da ValueHead DEVEM permanecer 100% idênticos bit-a-bit!
    for vp, w_b, s_b, m_b in zip(value_params, val_weights_before, val_step_before, val_exp_avg_before):
        assert torch.equal(vp.detach(), w_b)
        cur_s = trainer.optimizer.state[vp]["step"]
        if isinstance(cur_s, torch.Tensor):
            assert torch.equal(cur_s, s_b)
        else:
            assert cur_s == s_b
        assert torch.equal(trainer.optimizer.state[vp]["exp_avg"].detach(), m_b)


def test_curvature_gating_adapts_alpha_cap_to_on_policy_win_density():
    """Valida que alpha_cap escala suavemente de 0.0 (fronteira esparsa) até gradvac_max_alpha (domínio denso)."""
    trainer = _make_dummy_trainer(gradvac_max_alpha=0.25)
    obs_dim = trainer.env.observation_space["obs"].shape[0]
    delta_dim = trainer.env.observation_space["delta_obs"].shape[0]

    for _ in range(3):
        trainer.sil_buffer.add(
            SuccessfulTrajectory(
                obs=np.random.randn(4, obs_dim).astype(np.float32),
                delta_obs=np.random.randn(4, delta_dim).astype(np.float32),
                prev_actions=np.zeros(4, dtype=np.int64),
                prev_rewards=np.zeros(4, dtype=np.float32),
                actions=np.random.randint(0, 12, size=4),
                returns=np.linspace(5.0, 12.0, 4, dtype=np.float32),
                depth=2,
                length=4,
                log_probs=np.full(4, -2.0, dtype=np.float32),
            )
        )

    init_obs, _ = trainer.env.reset(seed=7)
    trainer.collect_rollouts(init_obs, current_hx=None, episode_start=True)

    # Caso 1: Fronteira extrema esparsa (0% de vitórias recentes) -> alpha_cap == 0.0 (PCGrad 90° puro)
    trainer._recent_frontier_wins.clear()
    for _ in range(20):
        trainer._recent_frontier_wins.append(0.0)
    m_sparse = trainer.train_epoch()
    assert m_sparse["pcgrad_win_density"] == pytest.approx(0.0, abs=1e-6)
    assert m_sparse["pcgrad_alpha_cap"] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= m_sparse["pcgrad_conflict_rate"] <= 1.0

    # Caso 2: Regime denso (80% de vitórias recentes) -> alpha_cap == 0.25 * 0.80 = 0.20
    trainer._recent_frontier_wins.clear()
    for i in range(20):
        trainer._recent_frontier_wins.append(1.0 if i < 16 else 0.0)
    m_dense = trainer.train_epoch()
    assert m_dense["pcgrad_win_density"] == pytest.approx(0.80, abs=1e-6)
    assert m_dense["pcgrad_alpha_cap"] == pytest.approx(0.20, abs=1e-6)
    assert 0.0 <= m_dense["pcgrad_conflict_rate"] <= 1.0


def test_option3_respects_frozen_trunk_and_continuous_action_space():
    """Garante compatibilidade universal da Opção 3 com ação contínua (Box) e Trunk congelado."""
    raw_env = gym.make("Pendulum-v1")
    wrapped = UniversalS1Wrapper(raw_env, tri_compass=False)
    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
        tri_compass=False,
    )
    # Congela o Trunk para validar que _get_actor_and_value_blocks ignora blocos sem requires_grad
    for p in agent.trunk.parameters():
        p.requires_grad = False
    trunk_before = [p.detach().clone() for p in agent.trunk.parameters()]

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped,
        rollout_steps=32,
        chunk_length=8,
        chunk_batch_size=2,
        sil_enabled=True,
        sil_coef=0.10,
        sil_min_samples=2,
        sil_batch_size=2,
        sil_dpo_enabled=True,
        dpo_coef=0.05,
        dpo_batch_size=2,
        pcgrad_enabled=True,
    )

    obs_dim = wrapped.observation_space["obs"].shape[0]
    delta_dim = wrapped.observation_space["delta_obs"].shape[0]
    for _ in range(3):
        trainer.sil_buffer.add(
            SuccessfulTrajectory(
                obs=np.random.randn(4, obs_dim).astype(np.float32),
                delta_obs=np.random.randn(4, delta_dim).astype(np.float32),
                prev_actions=np.zeros((4, 1), dtype=np.float32),
                prev_rewards=np.zeros(4, dtype=np.float32),
                actions=np.zeros((4, 1), dtype=np.float32),
                returns=np.ones(4, dtype=np.float32) * 3.0,
                depth=1,
                length=4,
                log_probs=np.full(4, -1.0, dtype=np.float32),
            )
        )
        trainer.sil_buffer.add_failure(
            SuccessfulTrajectory(
                obs=np.random.randn(4, obs_dim).astype(np.float32),
                delta_obs=np.random.randn(4, delta_dim).astype(np.float32),
                prev_actions=np.zeros((4, 1), dtype=np.float32),
                prev_rewards=np.zeros(4, dtype=np.float32),
                actions=np.ones((4, 1), dtype=np.float32),
                returns=np.ones(4, dtype=np.float32) * -5.0,
                depth=1,
                length=4,
                log_probs=np.full(4, -1.0, dtype=np.float32),
            )
        )

    init_obs, _ = wrapped.reset(seed=11)
    trainer.collect_rollouts(init_obs, current_hx=None, episode_start=True)
    metrics = trainer.train_epoch()

    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["sil_policy_loss"])
    assert np.isfinite(metrics["sil_dpo_loss"])
    assert metrics["sil_value_loss"] == 0.0
    for p, p_b in zip(agent.trunk.parameters(), trunk_before):
        assert torch.equal(p.detach(), p_b)


def test_verified_mastery_shield_excludes_partial_peak_slices_from_g_mem_ref():
    """Garante que fatias parciais (is_full_win=False) não bloqueiam os mini-batches do PPO (g_mem_ref=None),
    enquanto ainda participam da consolidação off-policy projetada contra g_ppo_mean ao final da época."""
    trainer = _make_dummy_trainer(sil_dpo_enabled=False, sil_min_samples=2)
    obs_dim = trainer.env.observation_space["obs"].shape[0]
    delta_dim = trainer.env.observation_space["delta_obs"].shape[0]

    # Adiciona apenas fatias de pico parciais não verificadas (is_full_win=False)
    for _ in range(3):
        trainer.sil_buffer.add(
            SuccessfulTrajectory(
                obs=np.random.randn(4, obs_dim).astype(np.float32),
                delta_obs=np.random.randn(4, delta_dim).astype(np.float32),
                prev_actions=np.zeros(4, dtype=np.int64),
                prev_rewards=np.zeros(4, dtype=np.float32),
                actions=np.random.randint(0, 12, size=4),
                returns=np.linspace(3.0, 6.0, 4, dtype=np.float32),
                depth=2,
                length=4,
                is_full_win=False,
            )
        )

    assert trainer.sil_buffer.count_full_wins() == 0
    mem_ref, _, _ = trainer._compute_unified_memory_loss(target_depth=2, only_full_wins=True)
    assert mem_ref is None

    mem_full, _, _ = trainer._compute_unified_memory_loss(target_depth=2, only_full_wins=False)
    assert mem_full is not None

    # Agora adiciona 2 vitórias completas verificadas (is_full_win=True) -> g_mem_ref ativa!
    for _ in range(2):
        trainer.sil_buffer.add(
            SuccessfulTrajectory(
                obs=np.random.randn(4, obs_dim).astype(np.float32),
                delta_obs=np.random.randn(4, delta_dim).astype(np.float32),
                prev_actions=np.zeros(4, dtype=np.int64),
                prev_rewards=np.zeros(4, dtype=np.float32),
                actions=np.random.randint(0, 12, size=4),
                returns=np.linspace(8.0, 14.0, 4, dtype=np.float32),
                depth=2,
                length=4,
                is_full_win=True,
            )
        )

    assert trainer.sil_buffer.count_full_wins() == 2
    mem_ref_active, _, _ = trainer._compute_unified_memory_loss(target_depth=2, only_full_wins=True)
    assert mem_ref_active is not None

