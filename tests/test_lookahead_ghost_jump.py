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
    assert abs(agent2.scar_memory.scale - 0.15) < 1e-6
    assert abs(agent2.scar_memory.resonance_beta - 0.20) < 1e-6


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


def test_temporal_cache_bit_exactness_and_victory_scar_after_inference_mode():
    torch.manual_seed(77)
    raw_env1 = RubiksCubeEnv(curriculum=False, scramble_depth=6)
    env1 = UniversalS1Wrapper(raw_env1, tri_compass=True)
    raw_env2 = RubiksCubeEnv(curriculum=False, scramble_depth=6)
    env2 = UniversalS1Wrapper(raw_env2, tri_compass=True)

    agent_cached = UniversalS1Agent(
        obs_space=env1.observation_space,
        action_space=env1.action_space,
        trunk_type="lru",
        action_chunk_size=1,
        tri_compass=True,
    ).to("cpu")
    agent_uncached = UniversalS1Agent(
        obs_space=env2.observation_space,
        action_space=env2.action_space,
        trunk_type="lru",
        action_chunk_size=1,
        tri_compass=True,
    ).to("cpu")
    agent_uncached.load_state_dict(agent_cached.state_dict())
    agent_cached.eval()
    agent_uncached.eval()

    obs1, _ = env1.reset(seed=999)
    obs2, _ = env2.reset(seed=999)
    agent_cached.reset_memory()
    agent_uncached.reset_memory()

    traj_obs, traj_delta, traj_act, traj_rew = [], [], [], []
    for _ in range(6):
        # Force uncached agent to recompute Pass 0 and Pass 1 from scratch
        agent_uncached._pass0_cache = None
        agent_uncached._pass1_child_cache = None

        a_c = agent_cached.act_fast(
            obs1,
            deterministic=True,
            value_rerank=True,
            value_rerank_top_k=4,
            value_rerank_threshold=0.99,
            ghost_jump=True,
            scar_penalty=0.25,
            env=env1,
        )
        a_u = agent_uncached.act_fast(
            obs2,
            deterministic=True,
            value_rerank=True,
            value_rerank_top_k=4,
            value_rerank_threshold=0.99,
            ghost_jump=True,
            scar_penalty=0.25,
            env=env2,
        )
        assert a_c == a_u
        assert torch.allclose(agent_cached.hx, agent_uncached.hx, atol=1e-6)
        assert agent_cached._pass0_cache is not None

        traj_obs.append(obs1["obs"])
        traj_delta.append(obs1["delta_obs"])
        traj_act.append(int(obs1.get("prev_action", 0)))
        traj_rew.append(float(obs1.get("prev_reward", 0.0)))

        obs1, _, done1, _, _ = env1.step(a_c)
        obs2, _, done2, _, _ = env2.step(a_u)
        if done1 or done2:
            break

    # Verify record_victory_scar works cleanly after @torch.inference_mode() in eval() mode
    ok = agent_cached.record_victory_scar(
        traj_obs=traj_obs,
        traj_delta=traj_delta,
        traj_prev_act=traj_act,
        traj_prev_rew=traj_rew,
        surprise=2.5,
    )
    assert ok is True


def test_scout_envs_excluded_from_curriculum_promotion():
    from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv
    from system1_engine.training.ppo import RecurrentPPOTrainer

    raw_env = VectorizedRubiksEnv(num_envs=8, curriculum=True, min_depth=1, curriculum_window=8)
    # 1. Direct adapter test: set_scout_envs(3) isolates Curriculum if used standalone
    raw_env.set_scout_envs(3)
    assert raw_env.curriculum_window == 5
    for idx in range(3):
        assert raw_env.envs[idx].record_curriculum is False
    for idx in range(3, 8):
        assert raw_env.envs[idx].record_curriculum is True

    # 2. Caminho A Universal Contract: RecurrentPPOTrainer forces 0 scouts (100% pure S1 fleet)
    env = UniversalS1Wrapper(raw_env, tri_compass=True)
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        action_chunk_size=1,
        tri_compass=True,
    ).to("cpu")

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=64,
        chunk_length=8,
        chunk_batch_size=4,
        scar_mode="hybrid",
        pure_scars=True,
        sil_enabled=False,
        teacher_sil_envs=3,  # Legacy arg must be clamped to 0 in Universal Caminho A
        device=torch.device("cpu"),
    )
    assert trainer.teacher_sil_envs == 0
    assert raw_env.curriculum_window == 8  # 100% da frota (8/8) conta na grade curricular
    assert raw_env.frontier_successes.maxlen == 8
    for idx in range(8):
        assert raw_env.envs[idx].record_curriculum is True

    env.reset(seed=123)
    assert len(raw_env.frontier_successes) == 0

    # Solving in env 0 and env 4 both count cleanly as pure System 1 curriculum wins
    inv0 = raw_env.inverse_action_map[raw_env.envs[0].core.ATOMIC_MOVES.index(raw_env.envs[0]._last_scramble_moves[0])]
    _, _, term0, _, _ = raw_env.envs[0].step(inv0)
    assert term0 is True
    assert len(raw_env.frontier_successes) == 1

    inv4 = raw_env.inverse_action_map[raw_env.envs[4].core.ATOMIC_MOVES.index(raw_env.envs[4]._last_scramble_moves[0])]
    _, _, term4, _, _ = raw_env.envs[4].step(inv4)
    assert term4 is True
    assert len(raw_env.frontier_successes) == 2
    assert raw_env.frontier_successes[-1] == 1.0


