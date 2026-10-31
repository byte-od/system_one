"""Unit & Integration Tests for Linear Recurrent Units (LRU) and LRUResidualTrunk.

Pilar 3: Linear Recurrent Unit (Orvieto et al., ICML 2023 - DeepMind).
Verifica:
  1. Estabilidade espectral dos autovalores complexos (|lambda| < 1.0 e r in [r_min, r_max]).
  2. Equivalência numérica exata entre step() unitário e forward() sequencial.
  3. Mascaramento temporal de reset com dones sem vazamento entre trajetórias.
  4. Fluxo de gradientes em longos horizontes (T = 50 passos).
  5. Integração com UniversalS1Agent (trunk_type="lru").
  6. Exportação ONNX e paridade numérica com ONNX Runtime.
"""

import tempfile
import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn
import onnxruntime as ort

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.onnx_exporter import S1OnnxExportWrapper
from system1_engine.core.recurrent import LinearRecurrentUnit
from system1_engine.core.trunk import LRUResidualTrunk


def test_lru_polar_initialization_and_stability():
    """Valida a parametrização polar e garantia estrita de estabilidade (|lambda| < 1.0)."""
    d_in = 64
    d_out = 64
    d_hidden = 128
    r_min = 0.85
    r_max = 0.999

    lru = LinearRecurrentUnit(
        d_in=d_in,
        d_out=d_out,
        d_hidden=d_hidden,
        r_min=r_min,
        r_max=r_max,
    )

    lam_r, lam_i, gamma = lru.get_lambda_real_imag()
    modulus = torch.sqrt(lam_r ** 2 + lam_i ** 2)

    # Estabilidade matemática estrita: todos os autovalores devem estar dentro do círculo unitário
    assert (modulus < 1.0).all(), "Autovalores violam o critério de estabilidade (|lambda| >= 1.0)!"
    assert (modulus >= r_min - 1e-4).all(), f"Autovalores abaixo de r_min={r_min}!"
    assert (modulus <= r_max + 1e-4).all(), f"Autovalores acima de r_max={r_max}!"
    assert (gamma > 0.0).all(), "Fator de normalização de variância gamma deve ser estritamente positivo!"


def test_lru_step_vs_forward_equivalence():
    """Valida equivalência numérica exata entre step() unitário e forward() sequencial."""
    torch.manual_seed(42)
    B, T, D_in, D_out, D_hid = 4, 16, 32, 48, 64

    lru = LinearRecurrentUnit(d_in=D_in, d_out=D_out, d_hidden=D_hid).eval()
    x = torch.randn(B, T, D_in)
    hx = lru.get_initial_hx(B)

    # 1. Forward sequencial
    y_seq, next_hx_seq = lru(x, hx)

    # 2. Unroll passo a passo com step()
    curr_h = hx.squeeze(0)
    step_outs = []
    for t in range(T):
        yt, curr_h = lru.step(x[:, t], curr_h)
        step_outs.append(yt)

    y_step = torch.stack(step_outs, dim=1)
    next_hx_step = curr_h.unsqueeze(0)

    # Verifica paridade numérica estrita
    diff_y = torch.max(torch.abs(y_seq - y_step)).item()
    diff_h = torch.max(torch.abs(next_hx_seq - next_hx_step)).item()

    assert diff_y < 1e-6, f"Discrepância na saída y: {diff_y}"
    assert diff_h < 1e-6, f"Discrepância no estado oculto h: {diff_h}"


def test_lru_temporal_dones_masking():
    """Valida que dones=True reseta o estado oculto e impede contaminação entre episódios."""
    torch.manual_seed(123)
    B, T, D_in, D_out, D_hid = 2, 8, 16, 16, 16
    lru = LinearRecurrentUnit(d_in=D_in, d_out=D_out, d_hidden=D_hid).eval()

    x = torch.randn(B, T, D_in)
    dones = torch.zeros(B, T, dtype=torch.bool)
    # Define reset no passo 4 para o batch 0
    dones[0, 4] = True

    # 1. Roda com dones
    y_with_dones, _ = lru(x, dones=dones)

    # 2. Roda passo 4 do batch 0 com estado explicitamente zerado
    h_zero = lru.get_initial_hx(1).squeeze(0)
    y_step4_reset, _ = lru.step(x[0:1, 4], h_zero)

    diff = torch.max(torch.abs(y_with_dones[0, 4] - y_step4_reset[0])).item()
    assert diff < 1e-6, f"O reset temporal com dones não zerou o estado oculto corretamente: diff={diff}"


def test_lru_gradient_flow_long_horizon():
    """Valida propagação de gradientes através de T=50 passos temporais sem desvanecimento."""
    torch.manual_seed(999)
    B, T, D = 8, 50, 32

    x = torch.randn(B, T, D, requires_grad=True)
    lru = LinearRecurrentUnit(d_in=D, d_out=D, d_hidden=32, r_min=0.90, r_max=0.999)
    y, _ = lru(x)

    # Loss calculada exclusivamente no último passo temporal T-1
    loss = y[:, -1].sum()
    loss.backward()

    # O gradiente deve fluir até o passo t=0 com magnitude mensurável
    grad_t0 = x.grad[:, 0].norm().item()
    assert grad_t0 > 1e-9, f"Gradiente desvaneceu em T=50: norm={grad_t0}"
    assert not torch.isnan(x.grad).any(), "Gradiente contém NaNs!"


def test_lru_residual_trunk_dimensions_and_shapes():
    """Valida formatos e dimensões de entrada/saída do LRUResidualTrunk (2D e 3D)."""
    input_dim = 337
    fast_dim = 256
    slow_dim = 256
    trunk = LRUResidualTrunk(input_dim=input_dim, fast_dim=fast_dim, slow_dim=slow_dim)

    assert trunk.hidden_dim == 512
    hx0 = trunk.get_initial_hx(batch_size=4)
    assert hx0.shape == (1, 4, 512)

    # Entrada 3D [B, T, input_dim]
    x_3d = torch.randn(4, 10, input_dim)
    h_3d, next_hx_3d = trunk(x_3d, hx0)
    assert h_3d.shape == (4, 10, 512)
    assert next_hx_3d.shape == (1, 4, 512)

    # Entrada 2D [B, input_dim]
    x_2d = torch.randn(4, input_dim)
    h_2d, next_hx_2d = trunk(x_2d, hx0)
    assert h_2d.shape == (4, 512)
    assert next_hx_2d.shape == (1, 4, 512)


def test_universal_agent_with_lru_trunk():
    """Valida instanciação e passo de inferência do UniversalS1Agent com trunk_type='lru'."""
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(54, 6), dtype=float)
    act_space = gym.spaces.Discrete(12)

    agent = UniversalS1Agent(
        obs_space=obs_space,
        action_space=act_space,
        trunk_type="lru",
        tri_compass=True,
    )

    assert isinstance(agent.trunk, LRUResidualTrunk)
    assert agent.trunk.hidden_dim == 512

    obs_sample = {
        "obs": torch.zeros(1, 324, dtype=torch.float32),
        "delta_obs": torch.zeros(1, 324, dtype=torch.float32),
        "prev_action": torch.zeros(1, dtype=torch.long),
        "prev_reward": torch.zeros(1, 1, dtype=torch.float32),
    }
    act_out, logp_out, val_out, next_hx = agent.get_action(obs_sample)

    assert act_out is not None
    assert val_out is not None
    assert next_hx.shape == (1, 1, 512)

    # act_fast ultra-fast inference with numpy dict
    obs_np = {
        "obs": np.zeros(324, dtype=np.float32),
        "delta_obs": np.zeros(324, dtype=np.float32),
        "prev_action": 0,
        "prev_reward": 0.0,
    }
    decision = agent.act_fast(obs_np, return_decision=True)
    assert 0 <= decision.action < 12
    assert 0.0 <= decision.confidence <= 1.0


def test_lru_onnx_export_and_runtime_parity():
    """Valida exportação ONNX de agente LRU e paridade numérica exata com ONNX Runtime."""
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(54, 6), dtype=float)
    act_space = gym.spaces.Discrete(12)

    agent = UniversalS1Agent(
        obs_space=obs_space,
        action_space=act_space,
        trunk_type="lru",
        tri_compass=True,
    ).eval()

    export_model = S1OnnxExportWrapper(agent).eval()

    dummy_obs = torch.zeros(1, 1, 324, dtype=torch.float32)
    dummy_delta = torch.zeros(1, 1, 3 * 324, dtype=torch.float32)
    dummy_act = torch.zeros(1, 1, dtype=torch.long)
    dummy_rew = torch.zeros(1, 1, 1, dtype=torch.float32)
    dummy_hx = agent.get_initial_hx(1)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        torch.onnx.export(
            export_model,
            (dummy_obs, dummy_delta, dummy_act, dummy_rew, dummy_hx),
            f.name,
            input_names=["obs", "delta_obs", "prev_action", "prev_reward", "hx"],
            output_names=["logits", "value", "next_hx"],
            opset_version=17,
        )

        sess = ort.InferenceSession(f.name)
        ort_out = sess.run(
            None,
            {
                "obs": dummy_obs.numpy(),
                "delta_obs": dummy_delta.numpy(),
                "prev_action": dummy_act.numpy(),
                "prev_reward": dummy_rew.numpy(),
                "hx": dummy_hx.numpy(),
            },
        )

        logits_pt, val_pt, next_hx_pt = export_model(
            dummy_obs, dummy_delta, dummy_act, dummy_rew, dummy_hx
        )

        diff_logits = np.max(np.abs(ort_out[0] - logits_pt.detach().numpy()))
        diff_val = np.max(np.abs(ort_out[1] - val_pt.detach().numpy()))
        diff_hx = np.max(np.abs(ort_out[2] - next_hx_pt.detach().numpy()))

        assert diff_logits < 1e-5, f"Discrepância logits ONNX: {diff_logits}"
        assert diff_val < 1e-5, f"Discrepância value ONNX: {diff_val}"
        assert diff_hx < 1e-5, f"Discrepância hx ONNX: {diff_hx}"
