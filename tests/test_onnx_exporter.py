"""Testes unitários para exportação automática ONNX e sincronização Web/Portfólio."""

from __future__ import annotations

import os
import tempfile
import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.onnx_exporter import auto_sync_web_model, resolve_web_model_filename, generate_web_manifest
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.transfer.manager import KnowledgeTransferManager


def test_resolve_web_model_filename():
    assert resolve_web_model_filename("CartPole-v1") == "s1_cartpole.onnx"
    assert resolve_web_model_filename("RubiksCubeMacro-v0") == "s1_rubiks_macro.onnx"
    assert resolve_web_model_filename("RubiksCube-v0") == "s1_rubiks_atomic.onnx"
    assert resolve_web_model_filename(checkpoint_path="s1_rubiks_macro_trained.pt") == "s1_rubiks_macro.onnx"
    assert resolve_web_model_filename(checkpoint_path="s1_cartpole.pt") == "s1_cartpole.onnx"
    assert resolve_web_model_filename(checkpoint_path="s1_cartpole_trained.pt") == "s1_cartpole.onnx"
    assert resolve_web_model_filename(checkpoint_path="s1_lunarlander_v3_trained.pt") == "s1_lunarlander_v3.onnx"
    assert resolve_web_model_filename(checkpoint_path="s1_ant_v5_trained_v3.pt") == "s1_ant_v5.onnx"
    assert resolve_web_model_filename(checkpoint_path="s1_mountaincar_v0_trained.pt") == "s1_mountaincar_v0.onnx"


def test_find_latest_trained_checkpoint(tmp_path):
    from pathlib import Path
    from system1_engine.core.onnx_exporter import find_latest_trained_checkpoint
    project_root = Path(__file__).resolve().parent.parent

    found_any = False
    for env in ("CartPole-v1", "Acrobot-v1", "MountainCar-v0", "LunarLander-v3", "Ant-v5", "RubiksCube-v0"):
        ckpt = find_latest_trained_checkpoint(env, project_root)
        if ckpt is not None:
            found_any = True
            assert "trained" in os.path.basename(ckpt).lower(), f"Checkpoint {ckpt} não contém 'trained' no nome"

    if not found_any:
        dummy_ckpt = tmp_path / "s1_cartpole_trained.pt"
        dummy_ckpt.write_text("dummy")
        ckpt = find_latest_trained_checkpoint("CartPole-v1", tmp_path)
        assert ckpt == str(dummy_ckpt)



def test_auto_sync_web_model_vector():
    env = gym.make("CartPole-v1")
    wrapped = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(obs_space=wrapped.observation_space, action_space=wrapped.action_space)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "s1_testvector_v0_test.pt")
        # Deve salvar o checkpoint e disparar a exportação web
        KnowledgeTransferManager.save_checkpoint(
            agent=agent,
            checkpoint_path=ckpt_path,
            extra_info={"env_id": "TestVector-v0"},
            auto_sync_web=True,
        )
        assert os.path.exists(ckpt_path)

        # Verifica se o arquivo ONNX foi gerado
        result = auto_sync_web_model(agent, checkpoint_path=ckpt_path, extra_info={"env_id": "TestVector-v0"})
        assert result is not None
        assert os.path.exists(result)
        assert os.path.getsize(result) > 100_000

        # Limpeza
        try:
            if os.path.exists(result):
                os.remove(result)
            generate_web_manifest()
        except OSError:
            pass


def test_auto_sync_web_model_lean_trunk():
    env = gym.make("CartPole-v1")
    wrapped = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
    )
    assert agent.trunk.hidden_dim == 512

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "s1_testlean_v0_test.pt")
        KnowledgeTransferManager.save_checkpoint(
            agent=agent,
            checkpoint_path=ckpt_path,
            extra_info={"env_id": "TestLean-v0"},
            auto_sync_web=True,
        )
        assert os.path.exists(ckpt_path)

        result = auto_sync_web_model(agent, checkpoint_path=ckpt_path, extra_info={"env_id": "TestLean-v0"})
        assert result is not None
        assert os.path.exists(result)
        assert os.path.getsize(result) > 1_000_000  # Lean trunk is ~6 MB in fp32

        # Verify execution in ONNX Runtime
        import onnxruntime as ort
        sess = ort.InferenceSession(result, providers=["CPUExecutionProvider"])
        obs_in = np.zeros((1, 1, 4), dtype=np.float32)
        delta_in = np.zeros((1, 1, 4), dtype=np.float32)
        act_in = np.zeros((1, 1), dtype=np.int64)
        rew_in = np.zeros((1, 1, 1), dtype=np.float32)
        hx_in = np.zeros((1, 1, 512), dtype=np.float32)

        outputs = sess.run(None, {
            "obs": obs_in,
            "delta_obs": delta_in,
            "prev_action": act_in,
            "prev_reward": rew_in,
            "hx": hx_in,
        })
        logits, val, next_hx = outputs
        assert logits.shape == (1, 2)
        assert val.shape == (1, 1)
        assert next_hx.shape == (1, 1, 512)

        try:
            if os.path.exists(result):
                os.remove(result)
            generate_web_manifest()
        except OSError:
            pass
