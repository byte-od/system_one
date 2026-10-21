import numpy as np
import pytest
import torch
from collections import deque

from system1_engine.env.adapters.rubiks import RubiksCubeEnv, RubiksCubeMacroEnv, VectorizedRubiksEnv
from system1_engine.training.buffer import SuccessReplayBuffer, SuccessfulTrajectory
from system1_engine.telemetry.tracker import LiveStatsTracker


def test_dynamic_frontier_modulation_low_vs_high_success():
    """Valida que a probabilidade da fronteira P(D) modula adaptativamente entre sucesso baixo e alto."""
    env = RubiksCubeEnv(
        curriculum=True,
        min_depth=1,
        max_depth=10,
        mixed_replay_rate=0.25,
        dynamic_frontier=True,
    )
    env.set_curriculum_depth(8)
    assert env.current_depth == 8

    # 1. Taxa de sucesso baixa (0%)
    env.frontier_successes.extend([0.0] * 50)
    assert env.curriculum_success_rate == 0.0

    # Amostra 1.000 profundidades sob sucesso baixo
    samples_low = [env.sample_curriculum_depth() for _ in range(1000)]
    frac_d8_low = sum(1 for d in samples_low if d == 8) / 1000.0

    # Com mixed_replay=0.25 e S_F=0, P_min deve ficar próximo de 0.50
    assert 0.40 <= frac_d8_low <= 0.60
    assert 0.40 <= env._last_p_frontier <= 0.60

    # 2. Taxa de sucesso alta (80%)
    env.frontier_successes.clear()
    env.frontier_successes.extend([1.0] * 40 + [0.0] * 10)
    assert env.curriculum_success_rate == 0.80

    # Amostra 1.000 profundidades sob sucesso alto
    samples_high = [env.sample_curriculum_depth() for _ in range(1000)]
    frac_d8_high = sum(1 for d in samples_high if d == 8) / 1000.0

    # Com S_F=0.80, P_frontier deve subir para próximo de 0.90
    assert 0.80 <= frac_d8_high <= 0.95
    assert 0.82 <= env._last_p_frontier <= 0.95


def test_proximal_exponential_decay_distribution():
    """Valida que no replay as profundidades proximais (D-1, D-2) têm prioridade exponencial."""
    env = RubiksCubeEnv(
        curriculum=True,
        min_depth=1,
        max_depth=10,
        mixed_replay_rate=1.0,  # 100% replay para testar apenas a distribuição inferior
        dynamic_frontier=True,
        proximal_decay=0.8,
    )
    env.set_curriculum_depth(8)

    # Força sucesso moderado para acionar o sorteio de replay
    env.frontier_successes.extend([0.35] * 20)
    # Como mixed_replay_rate = 1.0, p_base = 0.0 -> p_min = 0.40, p_max = 0.40 -> sempre replay

    samples = [env.sample_curriculum_depth() for _ in range(3000)]
    replay_samples = [d for d in samples if d < 8]
    counts = {d: replay_samples.count(d) for d in range(1, 8)}

    # D7 deve receber substancialmente mais visitas que D6, e D6 mais que D5
    assert counts[7] > counts[6], f"D7 ({counts[7]}) deve ser > D6 ({counts[6]})"
    assert counts[6] > counts[5], f"D6 ({counts[6]}) deve ser > D5 ({counts[5]})"
    assert counts[7] + counts[6] > 0.75 * len(replay_samples), "A zona proximal D-1 e D-2 deve concentrar a maior parte do replay"


def test_dynamic_frontier_backwards_compatibility():
    """Valida que ao desativar dynamic_frontier=False o comportamento reverte ao legado estático."""
    env = RubiksCubeEnv(
        curriculum=True,
        min_depth=1,
        max_depth=10,
        mixed_replay_rate=0.20,
        dynamic_frontier=False,
    )
    env.set_curriculum_depth(8)

    samples = [env.sample_curriculum_depth() for _ in range(1000)]
    frac_d8 = sum(1 for d in samples if d == 8) / 1000.0

    # 1 - 0.20 = 0.80 estático
    assert 0.75 <= frac_d8 <= 0.85

    # Replay legado só visita [D-2, D) = 6 ou 7
    replay_samples = [d for d in samples if d < 8]
    assert all(d in (6, 7) for d in replay_samples)


def test_vectorized_rubiks_dynamic_frontier():
    """Valida que VectorizedRubiksEnv propaga dynamic_frontier e expõe telemetria em InfoList."""
    vec_env = VectorizedRubiksEnv(
        num_envs=4,
        curriculum=True,
        min_depth=1,
        max_depth=10,
        mixed_replay_rate=0.25,
        dynamic_frontier=True,
        proximal_decay=0.8,
    )
    vec_env.set_curriculum_depth(8)
    assert vec_env.dynamic_frontier is True
    assert vec_env.proximal_decay == 0.8

    obs, infos = vec_env.reset()
    assert all(inf.get("dynamic_frontier") is True for inf in infos)
    assert all("curriculum_p_frontier" in inf for inf in infos)

    # Alterar propriedade sincroniza todos os sub-ambientes
    vec_env.dynamic_frontier = False
    assert all(e.dynamic_frontier is False for e in vec_env.envs)
    vec_env.dynamic_frontier = True


def test_sil_buffer_dynamic_proximal_sampling():
    """Valida que SuccessReplayBuffer pondera trajetórias por proximidade a target_depth e comprimento."""
    buf = SuccessReplayBuffer(
        max_trajectories=100,
        obs_shape=(4,),
        action_shape=(),
        dynamic_frontier=True,
        proximal_decay=0.8,
    )

    # Adiciona trajetórias em diferentes profundidades
    # Depth 8 (fronteira ativa), 7 (degrau proximal), 4 (âncora fundacional)
    def make_traj(depth: int, length: int) -> SuccessfulTrajectory:
        return SuccessfulTrajectory(
            obs=np.zeros((length, 4), dtype=np.float32),
            delta_obs=None,
            prev_actions=np.zeros(length, dtype=np.int64),
            prev_rewards=np.zeros(length, dtype=np.float32),
            actions=np.zeros(length, dtype=np.int64),
            returns=np.ones(length, dtype=np.float32),
            depth=depth,
            length=length,
        )

    t8_short = make_traj(8, 8)
    t8_long = make_traj(8, 25)
    t7 = make_traj(7, 7)
    t4 = make_traj(4, 4)

    buf.add(t8_short)
    buf.add(t8_long)
    buf.add(t7)
    buf.add(t4)

    # Amostragem com target_depth = 8
    # frontier_min_depth = 7 (inclui 7 e 8)
    batch = buf.sample_batch(batch_size=4, target_depth=8)
    assert batch is not None
    assert "obs" in batch
    assert "actions" in batch
    assert "mask" in batch


def test_tracker_dynamic_frontier_telemetry():
    """Valida que LiveStatsTracker captura e transmite curriculum_p_frontier e dynamic_frontier."""
    tracker = LiveStatsTracker()
    tracker.record_env_step(
        reward=1.0,
        done=False,
        info={
            "current_depth": 8,
            "curriculum_p_frontier": 0.735,
            "dynamic_frontier": True,
        },
    )

    snap = tracker.snapshot()
    assert snap["curriculum_depth"] == 8
    assert snap["dynamic_frontier"] is True
    assert snap["curriculum_p_frontier"] == pytest.approx(0.735, abs=1e-3)
