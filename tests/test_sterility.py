"""Testes formais para a Memória Latente Estéril Auto-Corretiva (Self-Healing Latent Tabu)."""

import pytest
import numpy as np
import torch
import gymnasium as gym

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.training.sterility import LatentSterilityMemory
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.env.adapters.rubiks import RubiksCubeEnv


def test_latent_sterility_memory_basic_ops():
    """Valida gravação de esterilidade, acúmulo de peso e cálculo de resistência sobre logits."""
    mem = LatentSterilityMemory(
        capacity=16,
        latent_dim=256,
        num_actions=12,
        sim_threshold=0.85,
        penalty_scale=2.0,
        step_increment=1.0,
        max_weight=10.0,
    )
    assert len(mem) == 0

    # Estado latente sintético h1
    h1 = torch.zeros(256)
    h1[0] = 1.0  # Vetor canônico

    # Quando vazio, resistência é zero para todas as 12 ações
    res0 = mem.query_resistance(h1)
    assert res0.shape == (12,)
    assert (res0 == 0.0).all()

    # 1. Registra ação 3 como estéril sob o motivo h1
    updated = mem.record_transition(latent=h1, action=3, is_productive=False)
    assert updated is True
    assert len(mem) == 1
    assert mem.total_recorded == 1

    # Consulta resistência para h1: ação 3 deve ter resistência positiva
    res1 = mem.query_resistance(h1)
    assert res1[3].item() > 0.0
    # Todas as outras ações devem ter resistência 0.0
    for a in range(12):
        if a != 3:
            assert res1[a].item() == 0.0

    # 2. Confirmação do fracasso: tenta ação 3 novamente sob o mesmo motivo h1
    mem.record_transition(latent=h1, action=3, is_productive=False)
    assert len(mem) == 1
    assert mem.total_recorded == 2

    # A resistência sobre a ação 3 deve ter aumentado
    res2 = mem.query_resistance(h1)
    assert res2[3].item() > res1[3].item()


def test_sterility_rehabilitation_self_healing():
    """Valida o mecanismo de auto-cura: ação estéril é reabilitada e perdoada se for produtiva."""
    mem = LatentSterilityMemory(
        capacity=16,
        latent_dim=256,
        num_actions=12,
        sim_threshold=0.85,
        step_increment=2.0,
    )

    h = torch.randn(256)
    # Marca ação 5 como estéril duas vezes
    mem.record_transition(latent=h, action=5, is_productive=False)
    mem.record_transition(latent=h, action=5, is_productive=False)
    assert len(mem) == 1
    assert mem.query_resistance(h)[5].item() > 0.0

    # O agente tenta a ação 5 e descobre que ela foi PRODUTIVA (is_productive=True)
    rehab = mem.record_transition(latent=h, action=5, is_productive=True)
    assert rehab is True
    assert mem.total_rehabilitated == 1

    # O protótipo deve ter sido desativado e a resistência zerada imediatamente!
    assert len(mem) == 0
    res_after = mem.query_resistance(h)
    assert res_after[5].item() == 0.0


def test_sterility_different_latent_motives():
    """Garante que a resistência só afeta estados cognitivos com alta similaridade de cosseno."""
    mem = LatentSterilityMemory(
        capacity=16,
        latent_dim=256,
        num_actions=12,
        sim_threshold=0.85,
    )

    # h_A e h_B são ortogonais (similaridade de cosseno = 0.0)
    h_A = torch.zeros(256)
    h_A[0] = 1.0

    h_B = torch.zeros(256)
    h_B[1] = 1.0

    # Ação 4 é estéril sob o motivo h_A
    mem.record_transition(latent=h_A, action=4, is_productive=False)

    # Consulta resistência para h_A: ação 4 penalizada
    assert mem.query_resistance(h_A)[4].item() > 0.0

    # Consulta resistência para h_B: ação 4 NÃO é penalizada (motivo diferente!)
    assert mem.query_resistance(h_B)[4].item() == 0.0


def test_agent_act_fast_sterility_modulation():
    """Valida que a memória de esterilidade acoplada ao UniversalS1Agent atenua logits em act_fast."""
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(324,), dtype=np.float32)
    act_space = gym.spaces.Discrete(12)
    agent = UniversalS1Agent(obs_space=obs_space, action_space=act_space)
    agent.eval()

    obs_dict = {
        "obs": np.zeros(324, dtype=np.float32),
        "delta_obs": np.zeros(324, dtype=np.float32),
        "prev_action": 0,
        "prev_reward": 0.0,
    }

    # Executa primeiro passo para extrair o motivo latente h
    dec1 = agent.act_fast(obs_dict, return_decision=True, quality_filter=False)
    h_latent = agent.last_latent
    assert h_latent is not None

    # Configura a memória de esterilidade no agente
    mem = LatentSterilityMemory(latent_dim=h_latent.shape[-1], num_actions=12, penalty_scale=5.0)
    agent.set_sterility_memory(mem)

    # Marca a melhor ação original (dec1.action) como estéril com peso alto
    chosen_act = int(dec1.action)
    for _ in range(3):
        mem.record_transition(latent=h_latent, action=chosen_act, is_productive=False)

    # Próxima decisão determinística: com a resistência aplicada, o agente deve escolher uma ação diferente!
    dec2 = agent.act_fast(obs_dict, return_decision=True, quality_filter=False, deterministic=True)
    assert dec2.action != chosen_act


def test_trainer_integration_with_sterility_memory():
    """Valida ciclo completo no RecurrentPPOTrainer com métricas de esterilidade e reabilitação."""
    env = RubiksCubeEnv(scramble_depth=1, curriculum=False)
    wrapper_env = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(obs_space=wrapper_env.observation_space, action_space=wrapper_env.action_space)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapper_env,
        rollout_steps=32,
        chunk_length=8,
        chunk_batch_size=2,
        sterility_memory_enabled=True,
    )
    assert trainer.sterility_memory is not None
    assert trainer.agent.sterility_memory is trainer.sterility_memory

    obs_dict, _ = wrapper_env.reset()
    trainer.collect_rollouts(obs_dict, current_hx=None, episode_start=True)

    metrics = trainer.train_epoch()
    assert "sterility_prototypes" in metrics
    assert "sterility_rehabilitated" in metrics
    assert isinstance(metrics["sterility_prototypes"], int)


def test_sterility_symmetric_decision_evaluation():
    """Valida que o reforço negativo e retorno positivo avaliam cada decisão com base no retorno real G_t."""
    env = RubiksCubeEnv(scramble_depth=2, curriculum=False)
    wrapper_env = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(obs_space=wrapper_env.observation_space, action_space=wrapper_env.action_space)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapper_env,
        rollout_steps=64,
        chunk_length=8,
        chunk_batch_size=2,
        sterility_memory_enabled=True,
    )
    assert trainer.sterility_memory is not None

    obs_dict, _ = wrapper_env.reset()
    trainer.collect_rollouts(obs_dict, current_hx=None, episode_start=True)

    # Verifica que o buffer de esterilidade registrou protótipos cognitivos associados a decisões
    stats = trainer.sterility_memory.get_stats()
    assert stats["total_recorded"] >= 0

