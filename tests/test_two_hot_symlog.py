"""Testes Unitários para a Cabeça de Crítico Categórica Two-Hot Symlog.

Valida:
1. Funções matemáticas symlog e symexp (inversibilidade exata em ℝ).
2. Codificação Two-Hot (soma de probabilidades = 1.0, preservação da esperança matemática).
3. ValueHead (shapes, forward contínuo para GAE, logits para cross-entropy, entropia de incerteza).
4. Limitação estrita de gradientes (gradientes estáveis mesmo com retornos de 100.000).
5. Integração com UniversalS1Agent e RecurrentPPOTrainer.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.heads import ValueHead, symexp, symlog, two_hot_encode
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer


def test_symlog_symexp_invertibility():
    """Valida que symexp(symlog(x)) == x com alta precisão numérica em toda a reta real."""
    test_values = torch.tensor([
        -1e5, -1000.0, -100.0, -10.0, -1.0, -0.1, -1e-4,
        0.0,
        1e-4, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e5
    ], dtype=torch.float32)

    encoded = symlog(test_values)
    decoded = symexp(encoded)

    assert torch.allclose(test_values, decoded, rtol=1e-5, atol=1e-5), (
        f"Inversão falhou: original={test_values}, decodificado={decoded}"
    )


def test_two_hot_encode_distribution_properties():
    """Valida propriedades estocásticas da codificação Two-Hot: soma 1.0 e esperança linear."""
    bins = torch.linspace(-5.0, 5.0, 64)
    targets = torch.tensor([-4.2, -1.05, 0.0, 0.73, 2.85, 4.9], dtype=torch.float32)

    dist = two_hot_encode(targets, bins)

    assert dist.shape == (len(targets), 64)
    assert (dist >= 0.0).all(), "Todas as probabilidades devem ser não-negativas"

    # 1. Cada linha deve somar exatamente 1.0
    sums = dist.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # 2. Exatamente 1 ou 2 bins devem ter probabilidade não-nula por amostra
    non_zeros = (dist > 1e-6).sum(dim=-1)
    assert (non_zeros <= 2).all()

    # 3. A esperança matemática sobre os bins deve recuperar exatamente o alvo original
    expected = torch.sum(dist * bins, dim=-1)
    assert torch.allclose(targets, expected, atol=1e-4)


def test_value_head_forward_and_loss():
    """Valida o forward da ValueHead e o cálculo de perda de entropia cruzada."""
    vh = ValueHead(in_features=128, num_bins=127, v_min=-20.0, v_max=50.0)
    h = torch.randn(8, 16, 128)

    # 1. Forward padrão retorna shape (B, T, 1) contínuo
    val = vh(h)
    assert val.shape == (8, 16, 1)
    assert vh.last_logits is not None
    assert vh.last_logits.shape == (8, 16, 127)

    # 2. Forward com logits e entropia de incerteza
    val_out, logits, entropy = vh.forward_with_logits(h)
    assert torch.allclose(val, val_out)
    assert logits.shape == (8, 16, 127)
    assert entropy.shape == (8, 16)
    assert (entropy >= 0.0).all()

    # 3. Loss computada via Two-Hot Cross-Entropy
    targets = torch.randn(8, 16) * 5.0 + 10.0
    loss = vh.loss(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    loss.backward()
    for p in vh.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()


def test_gradient_boundedness_under_extreme_returns():
    """Comprova matematicamente que alvos colossais nunca explodem o gradiente da ValueHead."""
    vh = ValueHead(in_features=64, num_bins=63)
    h = torch.randn(4, 64)

    # Alvo normal
    val, logits, _ = vh.forward_with_logits(h)
    loss_normal = vh.loss(logits, torch.full((4,), 5.0))
    vh.zero_grad()
    loss_normal.backward()
    norm_normal = torch.norm(torch.stack([p.grad.norm() for p in vh.parameters()])).item()

    # Alvo extremo (100.000) que causaria NaN absoluto sob MSE
    val2, logits2, _ = vh.forward_with_logits(h)
    loss_extreme = vh.loss(logits2, torch.full((4,), 100000.0))
    vh.zero_grad()
    loss_extreme.backward()
    norm_extreme = torch.norm(torch.stack([p.grad.norm() for p in vh.parameters()])).item()

    assert norm_extreme < 10.0, f"Norma do gradiente explodiu ({norm_extreme:.2f})!"
    assert not torch.isnan(torch.tensor(norm_extreme))


def test_universal_agent_two_hot_integration():
    """Valida integração no UniversalS1Agent (inferência rápida, get_action e forward)."""
    obs_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    agent = UniversalS1Agent(obs_space, act_space, trunk_type="classic")
    assert isinstance(agent.value_head, ValueHead)

    obs_dict = {
        "obs": torch.randn(2, 4),
        "delta_obs": torch.zeros(2, 4),
        "prev_action": torch.zeros(2, dtype=torch.int64),
        "prev_reward": torch.zeros(2, dtype=torch.float32),
    }
    dist, val, next_hx = agent(obs_dict)

    assert val.shape == (2, 1, 1)
    assert agent.value_head.last_logits is not None
    assert agent.value_head.last_logits.shape[-1] == 255

    # Retorno com logits de valor explícito
    dist2, val2, next_hx2, val_logits = agent(obs_dict, return_value_logits=True)
    assert val_logits.shape == (2, 1, 255)
    assert torch.allclose(val, val2)

    # Act fast (produz escalares consistentes)
    fast_dict = {
        "obs": np.zeros(4, dtype=np.float32),
        "prev_action": 0,
        "prev_reward": 0.0,
    }
    dec = agent.act_fast(fast_dict, return_decision=True, return_value=True)
    assert dec.latent_value is not None
    assert isinstance(dec.latent_value, float)


def test_recurrent_ppo_trains_with_two_hot_value_head():
    """Valida que o loop completo de treinamento PPO opera com a Two-Hot Symlog Value Head."""
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
        trunk_type="classic",
    )

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        learning_rate=1e-3,
        rollout_steps=128,
        chunk_length=16,
        chunk_batch_size=8,
        n_epochs=2,
    )

    # 1. Coleta rollouts
    obs_dict, _ = env.reset(seed=42)
    obs_dict, hx, _, mean_ret = trainer.collect_rollouts(obs_dict, None, True)

    # 2. Executa época PPO com loss de valor Two-Hot Symlog
    metrics = trainer.train_epoch()

    assert "value_loss" in metrics
    assert metrics["value_loss"] > 0.0
    assert not np.isnan(metrics["value_loss"])
    assert not np.isnan(metrics["policy_loss"])
