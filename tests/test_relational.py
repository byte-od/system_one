"""Testes de conformidade e integridade do Universal Relational Front-End (Pilar 4).

Valida:
  1. Atenção Relacional Linear O(N) com Kernelização Positiva phi(x) = ELU(x) + 1.0.
  2. UniversalRelationalStateEncoder em 2D e 3D com fusão residual.
  3. VectorFrontEnd com modo Relacional (Tri-Compass 465D e Clássico 337D).
  4. UniversalS1Agent com LRU + Relational Front-End.
  5. Exportação ONNX e paridade numérica estrita com ONNX Runtime.
  6. Detecção automática de Relational no KnowledgeTransferManager.
"""

import io
import gymnasium as gym
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.encoders import VectorFrontEnd
from system1_engine.core.onnx_exporter import S1OnnxExportWrapper
from system1_engine.core.relational import (
    LinearRelationalAttention,
    UniversalRelationalStateEncoder,
)
from system1_engine.transfer.manager import KnowledgeTransferManager


def test_linear_relational_attention_shapes_and_gradients():
    """Valida forward, shapes e retropropagação da Atenção Linear."""
    attn = LinearRelationalAttention(embed_dim=32, num_heads=4)
    x = torch.randn(16, 8, 32, requires_grad=True)
    out = attn(x)

    assert out.shape == (16, 8, 32)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert attn.q_proj.weight.grad is not None


def test_linear_relational_attention_4d_temporal():
    """Valida atenção linear com dimensão temporal [B, T, S, D]."""
    attn = LinearRelationalAttention(embed_dim=32, num_heads=4)
    x = torch.randn(4, 10, 8, 32)
    out = attn(x)

    assert out.shape == (4, 10, 8, 32)
    assert not torch.isnan(out).any()


def test_universal_relational_state_encoder_2d_and_3d():
    """Valida o codificador de estados relacional em 2D e 3D."""
    obs_dim = 324
    out_dim = 256
    encoder = UniversalRelationalStateEncoder(obs_dim=obs_dim, out_dim=out_dim, num_slots=8, slot_dim=32)

    # 3D: [B, T, obs_dim]
    x_3d = torch.randn(8, 5, obs_dim)
    y_3d = encoder(x_3d)
    assert y_3d.shape == (8, 5, out_dim)

    # 2D: [B, obs_dim]
    x_2d = torch.randn(8, obs_dim)
    y_2d = encoder(x_2d)
    assert y_2d.shape == (8, out_dim)


def test_vector_frontend_with_relational():
    """Valida VectorFrontEnd integrando UniversalRelationalStateEncoder com Tri-Compass."""
    action_space = gym.spaces.Discrete(12)
    obs_dim = 324

    # Tri-Compass Ativo (465D)
    front_tri = VectorFrontEnd(obs_dim=obs_dim, action_space=action_space, tri_compass=True, relational=True)
    assert front_tri.out_dim == 465
    assert isinstance(front_tri.state_encoder, UniversalRelationalStateEncoder)

    b, t = 4, 3
    obs = torch.randn(b, t, obs_dim)
    delta_obs = torch.randn(b, t, 3 * obs_dim)
    prev_a = torch.zeros(b, t, dtype=torch.long)
    prev_r = torch.zeros(b, t, 1)

    out_tri = front_tri(obs=obs, delta_obs=delta_obs, prev_action=prev_a, prev_reward=prev_r)
    assert out_tri.shape == (b, t, 465)

    # Modo Clássico (337D)
    front_classic = VectorFrontEnd(obs_dim=obs_dim, action_space=action_space, tri_compass=False, relational=True)
    assert front_classic.out_dim == 337
    out_classic = front_classic(obs=obs, delta_obs=obs, prev_action=prev_a, prev_reward=prev_r)
    assert out_classic.shape == (b, t, 337)


def test_agent_with_lru_and_relational():
    """Valida UniversalS1Agent com tronco LRU e FrontEnd Relacional."""
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(324,), dtype=np.float32)
    action_space = gym.spaces.Discrete(12)

    agent = UniversalS1Agent(
        obs_space=obs_space,
        action_space=action_space,
        trunk_type="lru",
        tri_compass=True,
        relational=True,
    )

    assert agent.relational is True
    assert isinstance(agent.front_end.state_encoder, UniversalRelationalStateEncoder)

    b, t = 2, 4
    input_dict = {
        "obs": torch.randn(b, t, 324),
        "delta_obs": torch.randn(b, t, 324 * 3),
        "prev_action": torch.zeros(b, t, dtype=torch.long),
        "prev_reward": torch.zeros(b, t, 1),
    }

    dist, val, next_hx = agent.forward(input_dict)
    assert dist.logits.shape == (b, t, 12)
    assert next_hx.shape == (1, b, 512)

    # Fast evaluation step (raw action & ReflexDecision)
    obs_fast = {
        "obs": np.zeros(324, dtype=np.float32),
        "delta_obs": np.zeros(324 * 3, dtype=np.float32),
        "prev_action": 0,
        "prev_reward": 0.0,
    }
    action = agent.act_fast(obs_fast, deterministic=True)
    assert 0 <= int(action) < 12

    decision = agent.act_fast(obs_fast, return_decision=True, deterministic=True)
    assert 0 <= int(decision.action) < 12
    assert decision.confidence >= 0.0


def test_relational_onnx_export_and_runtime_parity():
    """Valida a exportação simbólica ONNX e paridade numérica rigorosa com ONNX Runtime."""
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(324,), dtype=np.float32)
    action_space = gym.spaces.Discrete(12)

    agent = UniversalS1Agent(
        obs_space=obs_space,
        action_space=action_space,
        trunk_type="lru",
        tri_compass=True,
        relational=True,
    )
    agent.eval()

    export_wrapper = S1OnnxExportWrapper(agent)
    export_wrapper.eval()

    obs = torch.randn(1, 1, 324)
    delta_obs = torch.randn(1, 1, 324 * 3)
    prev_a = torch.zeros(1, 1, dtype=torch.long)
    prev_r = torch.zeros(1, 1, 1, dtype=torch.float32)
    hx = agent.trunk.get_initial_hx(batch_size=1)

    # PyTorch reference output
    with torch.no_grad():
        pt_logits, pt_val, pt_hx = export_wrapper(obs, delta_obs, prev_a, prev_r, hx)

    buf = io.BytesIO()
    torch.onnx.export(
        export_wrapper,
        (obs, delta_obs, prev_a, prev_r, hx),
        buf,
        opset_version=14,
        input_names=["obs", "delta_obs", "prev_action", "prev_reward", "hx"],
        output_names=["logits", "value", "next_hx"],
        dynamic_axes={
            "obs": {0: "batch"},
            "delta_obs": {0: "batch"},
            "prev_action": {0: "batch"},
            "prev_reward": {0: "batch"},
            "hx": {1: "batch"},
            "logits": {0: "batch"},
            "value": {0: "batch"},
            "next_hx": {1: "batch"},
        },
    )
    buf.seek(0)

    # ONNX model integrity check
    onnx_model = onnx.load(buf)
    onnx.checker.check_model(onnx_model)

    # ONNX Runtime evaluation
    buf.seek(0)
    session = ort.InferenceSession(buf.read(), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "obs": obs.numpy(),
        "delta_obs": delta_obs.numpy(),
        "prev_action": prev_a.numpy(),
        "prev_reward": prev_r.numpy(),
        "hx": hx.numpy(),
    }
    ort_outs = session.run(None, ort_inputs)

    # Paridade numérica com tolerância estrita (< 1e-4)
    np.testing.assert_allclose(pt_logits.numpy(), ort_outs[0], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(pt_hx.numpy(), ort_outs[2], atol=1e-4, rtol=1e-4)


def test_manager_detect_relational():
    """Valida a detecção estática de arquitetura relational pelo KnowledgeTransferManager."""
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(324,), dtype=np.float32)
    action_space = gym.spaces.Discrete(12)

    # Agente Relacional
    rel_agent = UniversalS1Agent(obs_space, action_space, relational=True)
    rel_dict = {"state_dict": rel_agent.state_dict(), "extra_info": {"relational": True}}
    assert KnowledgeTransferManager.detect_relational("dummy.pt", checkpoint_obj=rel_dict) is True

    # Agente Clássico (não relacional)
    classic_agent = UniversalS1Agent(obs_space, action_space, relational=False)
    classic_dict = {"state_dict": classic_agent.state_dict(), "extra_info": {"relational": False}}
    assert KnowledgeTransferManager.detect_relational("dummy.pt", checkpoint_obj=classic_dict) is False
