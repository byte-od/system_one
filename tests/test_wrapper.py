import gymnasium as gym
import numpy as np
import pytest

from system1_engine.env.wrapper import UniversalS1Wrapper


def test_wrapper_reset_robustness():
    """Verify that on reset() after termination or truncation, all state is flushed."""
    env = UniversalS1Wrapper(gym.make("CartPole-v1"))

    # Initial reset
    obs_dict, info = env.reset()
    assert np.all(obs_dict["delta_obs"] == 0.0)
    assert obs_dict["prev_action"] == 0
    assert obs_dict["prev_reward"] == 0.0
    assert env.prev_obs is not None

    # Step until termination
    done = False
    step_count = 0
    while not done:
        action = env.action_space.sample()
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        step_count += 1
        done = terminated or truncated

    # At termination, buffers contain non-zero episode data
    assert env.prev_action is not None
    assert env.prev_obs is not None

    # Reset after termination
    obs_dict_reset, info = env.reset()

    # CRITICAL: Verify zero cross-episode residual data leakage
    assert np.all(obs_dict_reset["delta_obs"] == 0.0), "delta_obs must be strictly zero at episode start"
    assert obs_dict_reset["prev_action"] == 0, "prev_action must be reset to 0"
    assert obs_dict_reset["prev_reward"] == 0.0, "prev_reward must be reset to 0.0"
    assert env.prev_reward == 0.0
    assert env.prev_action == 0
    assert np.all(env.prev_obs == obs_dict_reset["obs"])


def test_wrapper_delta_computation():
    """Verify delta_obs decomposes into local, origin, and goal channels by default."""
    env = UniversalS1Wrapper(gym.make("CartPole-v1"))
    obs0_dict, _ = env.reset()
    s0 = obs0_dict["obs"]

    # Step 1
    obs1_dict, r0, _, _, _ = env.step(1)
    s1 = obs1_dict["obs"]
    expected_delta = s1 - s0

    np.testing.assert_allclose(obs1_dict["local_delta"], expected_delta, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(obs1_dict["origin_delta"], expected_delta, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(obs1_dict["delta_obs"][:4], expected_delta, rtol=1e-5, atol=1e-5)
    assert obs1_dict["prev_action"] == 1
    assert obs1_dict["prev_reward"] == float(r0)

    # Also verify classic 1-channel delta mode when tri_compass=False
    classic_env = UniversalS1Wrapper(gym.make("CartPole-v1"), tri_compass=False)
    c_obs0, _ = classic_env.reset()
    c_obs1, c_r0, _, _, _ = classic_env.step(1)
    np.testing.assert_allclose(c_obs1["delta_obs"], c_obs1["obs"] - c_obs0["obs"], rtol=1e-5, atol=1e-5)


def test_wrapper_continuous_action():
    """Verify wrapper works seamlessly for continuous action spaces (Box)."""
    # Create Pendulum-v1 (continuous Box action space)
    env = UniversalS1Wrapper(gym.make("Pendulum-v1"))
    obs_dict, _ = env.reset()

    assert isinstance(obs_dict["prev_action"], np.ndarray)
    assert obs_dict["prev_action"].shape == env.action_space.shape
    assert np.all(obs_dict["prev_action"] == 0.0)
    assert obs_dict["prev_reward"] == 0.0

    action = np.array([1.5], dtype=np.float32)
    next_dict, reward, _, _, _ = env.step(action)

    np.testing.assert_allclose(next_dict["prev_action"], action)
    assert next_dict["prev_reward"] == float(reward)


def test_wrapper_visual_standardization():
    """Verify visual observations in HWC, CHW, or arbitrary resolutions standardize to (2*C, 84, 84)."""
    # Create a mock visual gym environment returning (96, 96, 3) like CarRacing
    class MockHWCEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        def reset(self, **kwargs):
            return np.zeros((96, 96, 3), dtype=np.uint8), {}
        def step(self, action):
            return np.ones((96, 96, 3), dtype=np.uint8) * 128, 1.0, False, False, {}

    wrapped = UniversalS1Wrapper(MockHWCEnv())
    assert wrapped.is_visual
    assert wrapped.observation_space["obs"].shape == (6, 84, 84)

    obs0, _ = wrapped.reset()
    assert obs0["obs"].shape == (6, 84, 84)
    assert obs0["obs"].dtype == np.float32
    assert obs0["obs"].min() >= 0.0 and obs0["obs"].max() <= 1.0

    obs1, rew, _, _, _ = wrapped.step(np.array([0.5, 0.2, 0.0], dtype=np.float32))
    assert obs1["obs"].shape == (6, 84, 84)
    assert obs1["prev_action"].shape == (3,)
    assert rew == 1.0


def test_wrapper_discrete_and_structured_observation_spaces():
    """Verify that Discrete, MultiDiscrete, MultiBinary, and Tuple observation spaces vectorize accurately."""
    # 1. Discrete observation space (FrozenLake-v1)
    fl_env = UniversalS1Wrapper(gym.make("FrozenLake-v1"))
    assert not fl_env.is_visual
    assert fl_env.observation_space["obs"].shape == (16,)
    assert fl_env.observation_space["delta_obs"].shape == (48,)

    obs0, _ = fl_env.reset()
    assert obs0["obs"].shape == (16,)
    assert obs0["obs"][0] == 1.0
    assert np.all(obs0["delta_obs"] == 0.0)

    obs1, _, _, _, _ = fl_env.step(1)
    assert obs1["obs"].shape == (16,)
    # delta_obs is current_obs - prev_obs
    expected_delta = obs1["obs"] - obs0["obs"]
    np.testing.assert_allclose(obs1["local_delta"], expected_delta)
    np.testing.assert_allclose(obs1["delta_obs"][:16], expected_delta)
    fl_env.close()

    # Classic mode (tri_compass=False)
    fl_classic = UniversalS1Wrapper(gym.make("FrozenLake-v1"), tri_compass=False)
    assert fl_classic.observation_space["delta_obs"].shape == (16,)
    fl_classic.close()

    # 2. Tuple observation space (Blackjack-v1: 32 + 11 + 2 = 45)
    bj_env = UniversalS1Wrapper(gym.make("Blackjack-v1"))
    assert not bj_env.is_visual
    assert bj_env.observation_space["obs"].shape == (45,)
    assert bj_env.observation_space["delta_obs"].shape == (135,)

    bj_obs0, _ = bj_env.reset()
    assert bj_obs0["obs"].shape == (45,)
    assert bj_obs0["obs"].sum() == 3.0  # 1 one-hot active per subspace
    assert np.all(bj_obs0["delta_obs"] == 0.0)
    bj_env.close()


def test_wrapper_continuous_action_clipping():
    """Verify that continuous actions exceeding Box limits are defensively clipped."""
    raw_env = gym.make("Pendulum-v1")  # action_space: Box(-2.0, 2.0, (1,))
    env = UniversalS1Wrapper(raw_env)
    env.reset()

    # Pass an excessively large action
    excess_action = np.array([100.0], dtype=np.float32)
    obs_dict, _, _, _, _ = env.step(excess_action)

    # Action recorded and passed must be clipped to high (+2.0)
    assert obs_dict["prev_action"][0] == pytest.approx(2.0)

    # Pass an excessively negative action
    excess_neg = np.array([-999.0], dtype=np.float32)
    obs_dict, _, _, _, _ = env.step(excess_neg)
    assert obs_dict["prev_action"][0] == pytest.approx(-2.0)
    env.close()


def test_wrapper_vectorized_environment():
    """Verify UniversalS1Wrapper cleanly supports batched vectorized environments."""
    from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv

    raw_env = VectorizedRubiksEnv(num_envs=4, scramble_depth=2)
    env = UniversalS1Wrapper(raw_env)
    assert env.is_vectorized
    assert env.num_envs == 4

    obs0, info = env.reset()
    assert obs0["obs"].shape == (4, 324)
    assert obs0["delta_obs"].shape == (4, 972)
    assert obs0["prev_action"].shape == (4,)
    assert obs0["prev_reward"].shape == (4,)

    actions = np.array([0, 1, 2, 3], dtype=np.int64)
    obs1, rews, terms, truncs, info = env.step(actions)
    assert obs1["obs"].shape == (4, 324)
    assert obs1["delta_obs"].shape == (4, 972)
    assert rews.shape == (4,)
    assert terms.shape == (4,)
    assert truncs.shape == (4,)
    assert len(info) == 4


def test_wrapper_compass_mode():
    """Verify Bússola compass mode blends local velocity and global origin displacement."""
    env = UniversalS1Wrapper(gym.make("CartPole-v1"), tri_compass=False)
    env.compass_mode = True
    env.compass_alpha = 0.5

    obs0_dict, _ = env.reset()
    s0 = obs0_dict["obs"]
    assert env.init_obs is not None
    np.testing.assert_allclose(env.init_obs, s0)

    # Step 1: s1 - s0 is both local and global delta
    obs1_dict, _, _, _, _ = env.step(1)
    s1 = obs1_dict["obs"]
    expected_delta_1 = s1 - s0
    np.testing.assert_allclose(obs1_dict["delta_obs"], expected_delta_1, rtol=1e-5, atol=1e-5)

    # Step 2: local is s2 - s1, global is s2 - s0
    obs2_dict, _, _, _, _ = env.step(0)
    s2 = obs2_dict["obs"]
    local_delta = s2 - s1
    global_delta = s2 - s0
    expected_delta_2 = 0.5 * local_delta + 0.5 * global_delta
    np.testing.assert_allclose(obs2_dict["delta_obs"], expected_delta_2, rtol=1e-5, atol=1e-5)

    # Reset flushes init_obs to new start
    obs_reset, _ = env.reset()
    np.testing.assert_allclose(env.init_obs, obs_reset["obs"])
    assert np.all(obs_reset["delta_obs"] == 0.0)



