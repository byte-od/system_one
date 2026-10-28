"""Testes formais para Shadow EMA (Polyak Refined Weights) e Tethering Anti-Colapso."""

import copy
import os
import tempfile
import numpy as np
import pytest
import torch
import gymnasium as gym

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def _create_test_env_and_trainer(ema_enabled: bool = True, ema_decay: float = 0.90):
    raw_env = gym.make("CartPole-v1")
    wrapped_env = UniversalS1Wrapper(raw_env, tri_compass=False)
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
        ema_enabled=ema_enabled,
        ema_decay=ema_decay,
        ema_anti_collapse=True,
    )
    return wrapped_env, agent, trainer


def test_shadow_ema_initialization():
    """Valida inicialização do modelo sombra: pesos clonados, gradientes desativados e modo eval."""
    _, agent, trainer = _create_test_env_and_trainer(ema_enabled=True, ema_decay=0.99)
    assert trainer.ema_enabled is True
    assert trainer.ema_agent is not None
    assert trainer.ema_agent is not agent
    assert not trainer.ema_agent.training

    # Todos os parâmetros da EMA devem ter requires_grad=False
    assert all(not p.requires_grad for p in trainer.ema_agent.parameters())

    # Pesos iniciais devem ser idênticos
    for p_online, p_ema in zip(agent.parameters(), trainer.ema_agent.parameters()):
        torch.testing.assert_close(p_ema, p_online)

    # get_eval_agent() deve retornar o ema_agent
    eval_agent = trainer.get_eval_agent()
    assert eval_agent is trainer.ema_agent


def test_shadow_ema_disabled():
    """Valida comportamento quando Shadow EMA está desativada."""
    _, agent, trainer = _create_test_env_and_trainer(ema_enabled=False)
    assert trainer.ema_enabled is False
    assert trainer.ema_agent is None
    assert trainer.get_eval_agent() is agent


def test_shadow_ema_polyak_update():
    """Valida a atualização in-place de Polyak: theta_ema <- beta * theta_ema + (1-beta) * theta_online."""
    _, agent, trainer = _create_test_env_and_trainer(ema_enabled=True, ema_decay=0.90)

    # Snapshot inicial da EMA
    initial_ema_p0 = list(trainer.ema_agent.parameters())[0].clone()

    # Altera os parâmetros online manualmente (simulando passo do otimizador)
    with torch.no_grad():
        for p in agent.parameters():
            p.add_(10.0)

    # Executa update da EMA
    trainer.update_ema()

    updated_ema_p0 = list(trainer.ema_agent.parameters())[0]
    expected_p0 = 0.90 * initial_ema_p0 + 0.10 * (initial_ema_p0 + 10.0)
    torch.testing.assert_close(updated_ema_p0, expected_p0)


def test_shadow_ema_anti_collapse_tether():
    """Valida re-ancoragem suave da política online a partir da EMA sob risco de colapso."""
    _, agent, trainer = _create_test_env_and_trainer(ema_enabled=True, ema_decay=0.95)

    # Guarda cópia dos pesos saudáveis da EMA
    healthy_ema_p0 = list(trainer.ema_agent.parameters())[0].clone()

    # Simula colapso ou desvio extremo da política online
    with torch.no_grad():
        for p in agent.parameters():
            p.fill_(100.0)

    # Aplica sync_online_from_ema com alpha=0.20
    trainer.sync_online_from_ema(alpha=0.20)

    online_p0 = list(agent.parameters())[0]
    # Esperado: 0.80 * 100.0 + 0.20 * healthy_ema_p0
    expected_p0 = 0.80 * 100.0 + 0.20 * healthy_ema_p0
    torch.testing.assert_close(online_p0, expected_p0)


def test_shadow_ema_checkpoint_save_and_restore():
    """Valida salvamento e restauração completa de checkpoints contendo ema_state_dict."""
    _, agent, trainer = _create_test_env_and_trainer(ema_enabled=True, ema_decay=0.95)

    # Modifica a EMA para ter valores distintos da online
    with torch.no_grad():
        for p in trainer.ema_agent.parameters():
            p.add_(5.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ema_checkpoint.pt")

        # Salva checkpoint com ema_agent
        KnowledgeTransferManager.save_checkpoint(
            agent=agent,
            checkpoint_path=ckpt_path,
            extra_info={"test_run": True},
            auto_sync_web=False,
            ema_agent=trainer.ema_agent,
        )

        assert os.path.exists(ckpt_path)

        # Inspeciona metadados
        info = KnowledgeTransferManager.inspect_checkpoint(ckpt_path)
        assert info.get("has_ema") is True

        # Cria um segundo trainer e carrega os pesos da EMA
        _, agent2, trainer2 = _create_test_env_and_trainer(ema_enabled=True, ema_decay=0.95)
        success = trainer2.load_ema_weights(ckpt_path)
        assert success is True

        # Valida que os pesos da EMA no trainer2 batem exatamente com a EMA original
        for p_orig, p_loaded in zip(trainer.ema_agent.parameters(), trainer2.ema_agent.parameters()):
            torch.testing.assert_close(p_loaded, p_orig)
