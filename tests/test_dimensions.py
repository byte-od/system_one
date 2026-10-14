import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.encoders import ImpalaVisualFrontEnd, VectorFrontEnd
from system1_engine.core.trunk import LeanResidualTrunk, System1Trunk


def test_vector_frontend_dimensions():
    """Verify VectorFrontEnd output strictly conforms to [B, T, 337]."""
    obs_dim = 4
    act_space = gym.spaces.Discrete(2)
    front_end = VectorFrontEnd(obs_dim=obs_dim, action_space=act_space)

    # Test arbitrary batch sizes and sequence lengths
    test_shapes = [(1, 1), (4, 16), (8, 32), (2, 5)]

    for b, t in test_shapes:
        obs = torch.randn(b, t, obs_dim)
        delta_obs = torch.randn(b, t, obs_dim)
        prev_act = torch.randint(0, 2, (b, t))
        prev_rew = torch.randn(b, t, 1)

        out = front_end(obs, delta_obs, prev_act, prev_rew)
        assert out.shape == (b, t, 337), f"Expected {(b, t, 337)}, got {out.shape}"
        assert not torch.isnan(out).any()


def test_visual_frontend_dimensions():
    """Verify ImpalaVisualFrontEnd output strictly conforms to [B, T, 337]."""
    in_channels = 6  # 2 * 3 (stacked RGB)
    act_space = gym.spaces.Discrete(4)
    front_end = ImpalaVisualFrontEnd(in_channels=in_channels, action_space=act_space)

    test_shapes = [(1, 1), (2, 4), (3, 8)]

    for b, t in test_shapes:
        visual_obs = torch.randn(b, t, in_channels, 84, 84)
        prev_act = torch.randint(0, 4, (b, t))
        prev_rew = torch.randn(b, t, 1)

        out = front_end(obs=visual_obs, prev_action=prev_act, prev_reward=prev_rew)
        assert out.shape == (b, t, 337), f"Expected {(b, t, 337)}, got {out.shape}"
        assert not torch.isnan(out).any()


def test_system1_trunk_dimensions_and_hx():
    """Verify System1Trunk accepts [B, T, 337] and returns [B, T, 256] and updated hx."""
    trunk = System1Trunk(input_dim=337, hidden_dim=256, num_res_blocks=2)

    test_shapes = [(1, 1), (4, 16), (8, 32)]

    for b, t in test_shapes:
        x = torch.randn(b, t, 337)
        hx = torch.randn(1, b, 256)

        h, next_hx = trunk(x, hx=hx, dones=None)
        assert h.shape == (b, t, 256), f"Expected {(b, t, 256)}, got {h.shape}"
        assert next_hx.shape == (1, b, 256), f"Expected {(1, b, 256)}, got {next_hx.shape}"

        # Test with dones mask
        dones = torch.zeros(b, t)
        dones[:, 0] = 1.0  # episode boundary at step 0
        h_masked, next_hx_masked = trunk(x, hx=hx, dones=dones)
        assert h_masked.shape == (b, t, 256)
        assert next_hx_masked.shape == (1, b, 256)


def test_lean_residual_trunk_dimensions_and_hx():
    """Verify LeanResidualTrunk accepts [B, T, 337] and returns [B, T, 512] and updated hx [1, B, 512]."""
    trunk = LeanResidualTrunk(input_dim=337, fast_dim=256, slow_dim=256)
    assert trunk.hidden_dim == 512

    test_shapes = [(1, 1), (4, 16), (8, 32)]

    for b, t in test_shapes:
        x = torch.randn(b, t, 337)
        hx = trunk.get_initial_hx(b)
        assert hx.shape == (1, b, 512)

        h, next_hx = trunk(x, hx=hx, dones=None)
        assert h.shape == (b, t, 512), f"Expected {(b, t, 512)}, got {h.shape}"
        assert next_hx.shape == (1, b, 512), f"Expected {(1, b, 512)}, got {next_hx.shape}"

        # Test 2D input
        x_2d = torch.randn(b, 337)
        h_2d, next_hx_2d = trunk(x_2d, hx=hx, dones=None)
        assert h_2d.shape == (b, 512)
        assert next_hx_2d.shape == (1, b, 512)

        # Test with dones mask
        dones = torch.zeros(b, t)
        dones[:, 0] = 1.0  # episode boundary at step 0
        h_masked, next_hx_masked = trunk(x, hx=hx, dones=dones)
        assert h_masked.shape == (b, t, 512)
        assert next_hx_masked.shape == (1, b, 512)


def test_lean_agent_parameter_count():
    """Verify Lean UniversalS1Agent parameter count is ~1.5M-1.7M."""
    agent = UniversalS1Agent(
        obs_space=gym.spaces.Box(low=-1, high=1, shape=(4,)),
        action_space=gym.spaces.Discrete(2),
        trunk_type="lean",
    )
    total_params = sum(p.numel() for p in agent.parameters())
    assert 1_400_000 <= total_params <= 1_800_000, f"Expected ~1.5M-1.7M params, got {total_params:,}"


def test_parameter_counts_and_memory():
    """Verify parameter count is ~650k-725k for vector and ~2.0M-2.5M for visual scenarios."""
    vec_agent = UniversalS1Agent(
        obs_space=gym.spaces.Box(low=-1, high=1, shape=(4,)),
        action_space=gym.spaces.Discrete(2),
    )
    vec_params = sum(p.numel() for p in vec_agent.parameters())
    # Target ~650k parameters (within 15% tolerance: 650k to 750k)
    assert 600_000 <= vec_params <= 800_000, f"Vector parameters {vec_params} out of target ~650k"

    vis_agent = UniversalS1Agent(
        obs_space=gym.spaces.Box(low=0, high=255, shape=(6, 84, 84)),
        action_space=gym.spaces.Discrete(4),
        is_visual=True,
    )
    vis_params = sum(p.numel() for p in vis_agent.parameters())
    # Target ~2.5M parameters (within range 2.0M to 2.8M)
    assert 1_800_000 <= vis_params <= 2_800_000, f"Visual parameters {vis_params} out of target ~2.5M"

    # Memory check: float32 weights < 35 MB
    vec_mem_mb = (vec_params * 4) / (1024 * 1024)
    vis_mem_mb = (vis_params * 4) / (1024 * 1024)
    assert vec_mem_mb < 35.0, f"Vector memory exceeds 35MB: {vec_mem_mb:.2f}MB"
    assert vis_mem_mb < 35.0, f"Visual memory exceeds 35MB: {vis_mem_mb:.2f}MB"
