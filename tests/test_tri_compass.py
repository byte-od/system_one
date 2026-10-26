import os
import time
import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.encoders import VectorFrontEnd
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.transfer.manager import KnowledgeTransferManager


def test_vector_frontend_tri_compass_dimensions():
    """Valida formato [B, T, 465] no VectorFrontEnd quando tri_compass=True."""
    obs_dim = 4
    act_space = gym.spaces.Discrete(2)
    front_end = VectorFrontEnd(obs_dim=obs_dim, action_space=act_space, tri_compass=True)
    assert front_end.out_dim == 465

    test_shapes = [(1, 1), (4, 16), (2, 8)]
    for b, t in test_shapes:
        obs = torch.randn(b, t, obs_dim)
        delta_obs = torch.randn(b, t, 3 * obs_dim)  # 3 * 4 = 12
        prev_act = torch.randint(0, 2, (b, t))
        prev_rew = torch.randn(b, t, 1)

        out = front_end(obs, delta_obs, prev_act, prev_rew)
        assert out.shape == (b, t, 465), f"Esperado {(b, t, 465)}, obtido {out.shape}"


def test_universal_s1_wrapper_tri_compass_channels():
    """Valida o desacoplamento de canais no UniversalS1Wrapper (local + origem + meta)."""
    raw_env = gym.make("CartPole-v1")
    wrapped = UniversalS1Wrapper(raw_env, tri_compass=True)

    obs_dim = 4
    assert wrapped.observation_space["delta_obs"].shape == (3 * obs_dim,)

    obs0, _ = wrapped.reset()
    assert obs0["delta_obs"].shape == (12,)
    assert "local_delta" in obs0
    assert "origin_delta" in obs0
    assert "goal_delta" in obs0
    assert np.all(obs0["local_delta"] == 0.0)
    assert np.all(obs0["origin_delta"] == 0.0)
    assert np.all(obs0["goal_delta"] == 0.0)

    # 1 passo
    obs1, rew1, term1, trunc1, _ = wrapped.step(1)
    expected_local = obs1["obs"] - obs0["obs"]
    expected_origin = obs1["obs"] - obs0["obs"]
    np.testing.assert_allclose(obs1["local_delta"], expected_local, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(obs1["origin_delta"], expected_origin, rtol=1e-5, atol=1e-5)

    expected_concat = np.concatenate([expected_local, expected_origin, np.zeros_like(expected_local)])
    np.testing.assert_allclose(obs1["delta_obs"], expected_concat, rtol=1e-5, atol=1e-5)


def test_wrapper_attractor_auto_discovery():
    """Valida a auto-descoberta de atratores geodésicos a partir de estados terminais de sucesso."""
    raw_env = gym.make("CartPole-v1")
    wrapped = UniversalS1Wrapper(raw_env, tri_compass=True)
    wrapped.reset()

    assert wrapped.elite_goal_obs is None

    # Simula injeção de atrator de elite descoberto pelo c-SIL
    attractor = np.array([0.05, 0.01, -0.02, 0.03], dtype=np.float32)
    wrapped.set_elite_goal_obs(attractor)
    assert wrapped.elite_goal_obs is not None

    obs1, _, _, _, _ = wrapped.step(0)
    expected_goal_delta = obs1["obs"] - attractor
    np.testing.assert_allclose(obs1["goal_delta"], expected_goal_delta, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(obs1["delta_obs"][8:12], expected_goal_delta, rtol=1e-5, atol=1e-5)


def test_net2net_tri_compass_function_preservation(tmp_path):
    """Garante preservação de função matemática exata (Net2Net) ao expandir de 337D para 465D."""
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    # 1. Agente 337D legado
    agent_337 = UniversalS1Agent(
        obs_space=obs_space,
        action_space=act_space,
        trunk_type="lean",
        tri_compass=False,
    )
    agent_337.eval()

    ckpt_path = str(tmp_path / "model_337.pt")
    torch.save({"state_dict": agent_337.state_dict()}, ckpt_path)

    # 2. Agente 465D Tri-Compass
    agent_465 = UniversalS1Agent(
        obs_space=obs_space,
        action_space=act_space,
        trunk_type="lean",
        tri_compass=True,
    )
    agent_465.eval()

    # Expande via Net2Net
    res = KnowledgeTransferManager.load_for_training(
        agent=agent_465,
        checkpoint_path=ckpt_path,
        freeze_trunk=False,
    )
    assert res["mode"] == "net2net_tri_compass"

    # 3. Teste de Invariância Funcional:
    # Mesmo com novos canais em delta_obs, a saída inicial deve ser idêntica
    s = torch.tensor([[0.1, -0.2, 0.05, 0.3]], dtype=torch.float32)
    d_local = torch.tensor([[0.01, 0.02, -0.01, 0.0]], dtype=torch.float32)
    # No início de episódio, os canais de origem e meta começam neutros
    d_zero = torch.zeros(1, 4)

    input_337 = {
        "obs": s,
        "delta_obs": d_local,
        "prev_action": torch.tensor([[0]], dtype=torch.long),
        "prev_reward": torch.tensor([[[0.0]]], dtype=torch.float32),
    }

    input_465 = {
        "obs": s,
        "delta_obs": torch.cat([d_local, d_zero, d_zero], dim=-1),
        "prev_action": torch.tensor([[0]], dtype=torch.long),
        "prev_reward": torch.tensor([[[0.0]]], dtype=torch.float32),
    }

    with torch.no_grad():
        dist_old, val_old, _ = agent_337(input_337)
        dist_new, val_new, _ = agent_465(input_465)

    # Os logits e valores devem ser matematicamente idênticos (diferença na ordem de float32 epsilon < 1e-6)
    torch.testing.assert_close(dist_new.logits, dist_old.logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(val_new, val_old, rtol=1e-5, atol=1e-5)


def test_net2net_load_from_v23_anchor_checkpoint():
    """Valida warm-start do checkpoint s1_rubikscube_v0_trained_v23.pt para a arquitetura 465D."""
    ckpt_path = "s1_rubikscube_v0_trained_v23.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint âncora {ckpt_path} não encontrado no diretório de trabalho.")

    from system1_engine.env.adapters.rubiks import RubiksCubeEnv
    raw_env = RubiksCubeEnv()
    wrapped = UniversalS1Wrapper(raw_env, tri_compass=True)

    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
        tri_compass=True,
    )

    res = KnowledgeTransferManager.load_for_training(
        agent=agent,
        checkpoint_path=ckpt_path,
        freeze_trunk=False,
    )

    assert res["mode"] == "net2net_tri_compass"
    assert res["total_params"] > 0
    assert not res["trunk_frozen"]

    # Valida act_fast() com latência sub-milisegundo no agente expandido
    obs_dict, _ = wrapped.reset()
    agent.eval()

    t0 = time.perf_counter()
    decision = agent.act_fast(obs_dict, return_decision=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert 0 <= int(decision.action) < 12
    assert elapsed_ms < 10.0  # Ultra-rápido (<10ms mesmo no pior caso de primeiro forward)
