import time
import torch
import numpy as np
from system1_engine.env.adapters.rubiks import RubiksCubeEnv
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.recurrent import LinearRecurrentUnit


def test_lru_eval_fusion_equivalence_and_cache_invalidation():
    torch.manual_seed(42)
    lru = LinearRecurrentUnit(d_in=64, d_out=32, d_hidden=16)
    x = torch.randn(4, 1, 64)
    hx = torch.randn(1, 4, 32)

    lru.train()
    with torch.no_grad():
        y_train, h_train = lru(x, hx)

    lru.eval()
    with torch.no_grad():
        y_eval, h_eval = lru(x, hx)

    assert torch.allclose(y_train, y_eval, atol=1e-5)
    assert torch.allclose(h_train, h_eval, atol=1e-5)

    # Modify parameter in-place and ensure cache invalidates automatically via _version
    with torch.no_grad():
        lru.b_real.mul_(2.0)
        y_eval2, _ = lru(x, hx)
    lru.train()
    with torch.no_grad():
        y_train2, _ = lru(x, hx)
    assert torch.allclose(y_train2, y_eval2, atol=1e-5)


def test_simulate_2step_restores_exact_cube_state():
    raw_env = RubiksCubeEnv(curriculum=False, scramble_depth=5)
    raw_env.reset(seed=123)
    state_before = raw_env.core.state.copy()
    prev_score_before = raw_env._prev_score

    for a1 in range(12):
        for a2 in range(12):
            obs2, rew2, done2 = raw_env.simulate_2step(a1, a2)
            assert np.array_equal(raw_env.core.state, state_before)
            assert raw_env._prev_score == prev_score_before
            if not done2:
                assert obs2 is not None and obs2.shape == (324,)


def test_simulate_3step_restores_exact_cube_state_and_detects_win():
    raw_env = RubiksCubeEnv(curriculum=False, scramble_depth=3)
    raw_env.reset(seed=99)
    # Apply 3 known moves from solved state
    raw_env.core.reset()
    raw_env.core.apply_atomic("U")
    raw_env.core.apply_atomic("R")
    raw_env.core.apply_atomic("F")
    raw_env._prev_score = raw_env.core.get_score()
    state_before = raw_env.core.state.copy()

    # Inverse sequence F' (5), R' (9), U' (1) must solve in 3 steps
    obs3, rew3, done3 = raw_env.simulate_3step(5, 9, 1, return_obs=False)
    assert done3 is True
    assert obs3 is None
    assert rew3 > 10.0
    assert np.array_equal(raw_env.core.state, state_before)


def test_standardized_scar_memory_record_and_autoload():
    raw_env = RubiksCubeEnv(curriculum=False, scramble_depth=2)
    env = UniversalS1Wrapper(raw_env, tri_compass=True)
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        action_chunk_size=1,
        tri_compass=True,
    ).to("cpu")

    obs_dict, _ = env.reset(seed=11)
    traj_obs = [obs_dict["obs"], obs_dict["obs"]]
    traj_delta = [obs_dict["delta_obs"], obs_dict["delta_obs"]]
    ok = agent.record_victory_scar(
        traj_obs=traj_obs,
        traj_delta=traj_delta,
        traj_prev_act=[0, 1],
        traj_prev_rew=[0.0, 1.0],
        surprise=2.0,
    )
    assert ok is True
    assert agent.scar_memory is not None
    assert int(agent.scar_memory.scars_count.item()) == 1

    # Save state_dict and load into a fresh agent with scar_memory=None -> must auto-instantiate scar_memory
    sd = agent.state_dict()
    agent2 = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        action_chunk_size=1,
        tri_compass=True,
    ).to("cpu")
    assert agent2.scar_memory is None
    agent2.load_state_dict(sd, strict=True)
    assert agent2.scar_memory is not None
    assert int(agent2.scar_memory.scars_count.item()) == 1


def test_act_fast_ghost_jump_and_soft_scar():
    raw_env = RubiksCubeEnv(curriculum=False, scramble_depth=2)
    env = UniversalS1Wrapper(raw_env, tri_compass=True)
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        action_chunk_size=1,
        tri_compass=True,
    ).to("cpu")
    agent.eval()

    obs_dict, _ = env.reset(seed=7)
    agent.reset_memory()

    # Run with ghost_jump=True and scar_penalty=0.25
    act1 = agent.act_fast(
        obs_dict,
        deterministic=True,
        value_rerank=True,
        value_rerank_top_k=4,
        value_rerank_threshold=0.99,
        ghost_jump=True,
        scar_penalty=0.25,
        env=env,
    )
    assert isinstance(act1, int) and 0 <= act1 < 12
    assert len(agent._state_visit_counts) == 1

    # Run with ghost_jump=False (pure 1-step)
    act2 = agent.act_fast(
        obs_dict,
        deterministic=True,
        value_rerank=True,
        value_rerank_top_k=4,
        value_rerank_threshold=0.99,
        ghost_jump=False,
        scar_penalty=0.0,
        env=env,
    )
    assert isinstance(act2, int) and 0 <= act2 < 12


def test_cpu_reflex_latency_sub_08ms():
    raw_env = RubiksCubeEnv(curriculum=False, scramble_depth=4)
    env = UniversalS1Wrapper(raw_env, tri_compass=True)
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        action_chunk_size=1,
        tri_compass=True,
    ).to("cpu")
    agent.eval()

    obs_dict, _ = env.reset(seed=42)
    for _ in range(30):
        agent.act_fast(obs_dict, deterministic=True, value_rerank=False)

    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        agent.act_fast(obs_dict, deterministic=True, value_rerank=False)
        times.append((time.perf_counter() - t0) * 1000.0)

    assert np.median(times) < 0.80, f"Mediana de reflexo na CPU ({np.median(times):.3f}ms) excedeu 0.80ms"
