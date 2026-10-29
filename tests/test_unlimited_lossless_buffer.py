import gymnasium as gym
import numpy as np
import torch

import system1_engine.env  # noqa: F401
from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.buffer import (
    FrontierTransition,
    SuccessReplayBuffer,
    SuccessfulTrajectory,
)
from system1_engine.training.ppo import RecurrentPPOTrainer


def test_unlimited_capacity_without_ceiling():
    """Garante que o buffer padrão não possui teto artificial (nem 2048 por nível nem 10240 global)."""
    buf = SuccessReplayBuffer(obs_shape=(16,), is_discrete=True)
    assert buf.max_trajectories is None
    assert buf.max_per_depth is None
    assert buf.get_capacity_for_depth(1) is None
    assert buf.get_capacity_for_depth(11) is None

    rng = np.random.default_rng(42)
    n_d11 = 2600
    for i in range(n_d11):
        obs = np.zeros((6, 16), dtype=np.float32)
        # Estado inicial único por embaralhamento
        obs[0] = rng.integers(0, 2, size=(16,)).astype(np.float32)
        obs[0, 0] = float(i)
        buf.add(
            FrontierTransition(
                obs=obs,
                delta_obs=None,
                prev_actions=np.zeros(6, dtype=np.int64),
                prev_rewards=np.zeros(6, dtype=np.float32),
                actions=np.ones(6, dtype=np.int64),
                returns=np.linspace(10.0, 5.0, 6, dtype=np.float32),
                depth=11,
                length=6,
                is_full_win=True,
            )
        )

    assert len(buf) == n_d11
    assert buf.get_depth_distribution()[11] == n_d11
    assert buf.count_full_wins() == n_d11
    assert buf.count_success_slices() == 0


def test_lossless_compaction_zero_precision_loss_rubiks_and_continuous():
    """Verifica compactação >20x no Cubo Mágico e 0.0 absoluto de erro numérico em estados contínuos."""
    # 1. Trajetória real do RubiksCube-v0 (obs 324 floats {0,1}, delta_obs 972 floats {-1,0,1})
    raw_cube = gym.make("RubiksCube-v0")
    cube_env = UniversalS1Wrapper(raw_cube, tri_compass=True)
    obs_dict, _ = cube_env.reset(seed=123)

    obs_seq = []
    delta_seq = []
    act_seq = []
    prev_act_seq = []
    prev_rew_seq = []
    for step_i in range(13):
        obs_seq.append(obs_dict["obs"].copy())
        delta_seq.append(obs_dict["delta_obs"].copy())
        prev_act_seq.append(int(obs_dict["prev_action"]))
        prev_rew_seq.append(float(obs_dict["prev_reward"]))
        act = step_i % 12
        act_seq.append(act)
        obs_dict, _, term, trunc, _ = cube_env.step(act)
        if term or trunc:
            obs_dict, _ = cube_env.reset(seed=200 + step_i)

    orig_cube_obs = np.array(obs_seq, dtype=np.float32)
    orig_cube_delta = np.array(delta_seq, dtype=np.float32)

    cube_buf = SuccessReplayBuffer(
        obs_shape=orig_cube_obs.shape[1:],
        is_discrete=True,
        lossless_compaction=True,
    )
    t_cube = FrontierTransition(
        obs=orig_cube_obs.copy(),
        delta_obs=orig_cube_delta.copy(),
        prev_actions=np.array(prev_act_seq, dtype=np.int64),
        prev_rewards=np.array(prev_rew_seq, dtype=np.float32),
        actions=np.array(act_seq, dtype=np.int64),
        returns=np.linspace(10.0, 1.0, len(obs_seq), dtype=np.float32),
        depth=11,
        length=len(obs_seq),
        is_full_win=True,
    )
    cube_buf.add(t_cube)

    # Compactação significativa (>20x) e igualdade bit-a-bit exata (erro = 0.0)
    assert cube_buf.compression_ratio > 20.0
    assert np.array_equal(t_cube.obs, orig_cube_obs)
    assert np.array_equal(t_cube.delta_obs, orig_cube_delta)
    assert t_cube.obs.dtype == np.float32
    assert t_cube.delta_obs.dtype == np.float32

    batch = cube_buf.sample_batch(batch_size=1, target_depth=11)
    assert batch is not None
    assert np.array_equal(batch["obs"][0].cpu().numpy(), orig_cube_obs)
    assert np.array_equal(batch["delta_obs"][0].cpu().numpy(), orig_cube_delta)

    # 2. Estados contínuos arbitrários IEEE-754 float32 (zero perda de mantissa)
    rng = np.random.default_rng(999)
    cont_obs = rng.standard_normal((14, 24)).astype(np.float32) * np.float32(3.1415927)
    cont_delta = rng.standard_normal((14, 72)).astype(np.float32) * np.float32(0.70710677)

    cont_buf = SuccessReplayBuffer(
        obs_shape=(24,),
        action_shape=(3,),
        is_discrete=False,
        lossless_compaction=True,
    )
    t_cont = SuccessfulTrajectory(
        obs=cont_obs.copy(),
        delta_obs=cont_delta.copy(),
        prev_actions=np.zeros((14, 3), dtype=np.float32),
        prev_rewards=np.zeros(14, dtype=np.float32),
        actions=rng.standard_normal((14, 3)).astype(np.float32),
        returns=np.linspace(5.0, 0.5, 14, dtype=np.float32),
        depth=1,
        length=14,
    )
    cont_buf.add(t_cont)

    # Reconstrução 100% bit-exata: nenhum truncamento para float16/int8 permitido
    assert np.array_equal(t_cont.obs, cont_obs)
    assert np.array_equal(t_cont.delta_obs, cont_delta)
    assert float(np.max(np.abs(t_cont.obs - cont_obs))) == 0.0
    assert float(np.max(np.abs(t_cont.delta_obs - cont_delta))) == 0.0


def test_scramble_deduplication_and_quality_upgrade_eliminates_easy_bias():
    """Garante que embaralhamentos fáceis nunca expulsam difíceis e que o mesmo estado faz upgrade in-place."""
    buf = SuccessReplayBuffer(obs_shape=(8,), is_discrete=True)

    obs_hard = np.zeros((13, 8), dtype=np.float32)
    obs_hard[0] = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.float32)

    obs_easy = np.zeros((9, 8), dtype=np.float32)
    obs_easy[0] = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.float32)

    # 1. Fatia parcial de sucesso no embaralhamento difícil
    slice_hard = FrontierTransition(
        obs=obs_hard[:6].copy(),
        delta_obs=None,
        prev_actions=np.zeros(6, dtype=np.int64),
        prev_rewards=np.zeros(6, dtype=np.float32),
        actions=np.ones(6, dtype=np.int64),
        returns=np.ones(6, dtype=np.float32) * 2.5,
        depth=11,
        length=6,
        is_full_win=False,
    )
    buf.add(slice_hard)
    assert len(buf) == 1
    assert buf.count_full_wins() == 0
    assert buf.count_success_slices() == 1

    # 2. Vitória rápida (9 movimentos) em OUTRO embaralhamento nunca expulsa o difícil
    win_easy = FrontierTransition(
        obs=obs_easy.copy(),
        delta_obs=None,
        prev_actions=np.zeros(9, dtype=np.int64),
        prev_rewards=np.zeros(9, dtype=np.float32),
        actions=np.ones(9, dtype=np.int64),
        returns=np.ones(9, dtype=np.float32) * 9.8,
        depth=11,
        length=9,
        is_full_win=True,
    )
    buf.add(win_easy)
    assert len(buf) == 2
    assert buf.count_full_wins() == 1
    assert buf.count_success_slices() == 1

    # 3. Vitória de 13 movimentos no MESMO embaralhamento difícil faz upgrade da fatia parcial -> vitória completa
    win_hard_13 = FrontierTransition(
        obs=obs_hard.copy(),
        delta_obs=None,
        prev_actions=np.zeros(13, dtype=np.int64),
        prev_rewards=np.zeros(13, dtype=np.float32),
        actions=np.ones(13, dtype=np.int64),
        returns=np.ones(13, dtype=np.float32) * 8.9,
        depth=11,
        length=13,
        is_full_win=True,
    )
    buf.add(win_hard_13)
    assert len(buf) == 2
    assert buf.count_full_wins() == 2
    assert buf.count_success_slices() == 0
    assert buf.total_upgraded == 1

    # 4. Solução mais enxuta (11 movimentos, maior retorno) para o MESMO embaralhamento difícil faz upgrade in-place
    win_hard_11 = FrontierTransition(
        obs=obs_hard[:11].copy(),
        delta_obs=None,
        prev_actions=np.zeros(11, dtype=np.int64),
        prev_rewards=np.zeros(11, dtype=np.float32),
        actions=np.ones(11, dtype=np.int64),
        returns=np.ones(11, dtype=np.float32) * 9.3,
        depth=11,
        length=11,
        is_full_win=True,
    )
    buf.add(win_hard_11)
    assert len(buf) == 2
    assert buf.total_upgraded == 2

    # 5. Solução pior (15 movimentos, retorno menor) para o MESMO embaralhamento é descartada
    obs_hard_15 = np.zeros((15, 8), dtype=np.float32)
    obs_hard_15[0] = obs_hard[0]
    win_hard_15 = FrontierTransition(
        obs=obs_hard_15,
        delta_obs=None,
        prev_actions=np.zeros(15, dtype=np.int64),
        prev_rewards=np.zeros(15, dtype=np.float32),
        actions=np.ones(15, dtype=np.int64),
        returns=np.ones(15, dtype=np.float32) * 8.1,
        depth=11,
        length=15,
        is_full_win=True,
    )
    buf.add(win_hard_15)
    assert len(buf) == 2
    assert buf.total_upgraded == 2
    lengths = sorted(t.length for t in buf._depth_buffers[11])
    assert lengths == [9, 11]


def test_strictly_wins_and_success_slices_with_on_policy_dpo_contrast():
    """Garante 0 falhas armazenadas no buffer e contraste DPO funcional via chunks negativos on-policy."""
    raw_env = gym.make("CartPole-v1")
    wrapped_env = UniversalS1Wrapper(raw_env, tri_compass=False)
    agent = UniversalS1Agent(
        obs_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        trunk_type="lean",
    )
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped_env,
        rollout_steps=64,
        chunk_length=8,
        chunk_batch_size=4,
        sil_enabled=True,
        sil_min_samples=2,
        sil_dpo_enabled=True,
        dpo_coef=0.05,
        dpo_beta=0.10,
        dpo_batch_size=4,
    )

    assert trainer.sil_max_trajectories is None
    assert trainer.sil_max_per_depth is None
    assert trainer.sil_buffer.max_trajectories is None
    assert trainer.sil_buffer.max_per_depth is None

    # 1. Episódio de falha pura -> NADA armazenado (nem em trajectories nem em failures)
    pure_fail_steps = [
        {"obs": np.full(4, float(i), dtype=np.float32), "delta_obs": np.zeros(4, dtype=np.float32), "prev_action": 0, "prev_reward": 0.0, "action": 0, "reward": -0.2, "log_prob": -0.7, "value": 0.1}
        for i in range(4)
    ]
    trainer._process_completed_trajectory(pure_fail_steps, is_win=False, cur_depth=5)
    assert len(trainer.sil_buffer) == 0
    assert len(trainer.sil_buffer.failures) == 0

    # 2. Episódio com avanço inicial + degradação -> armazena APENAS a fatia de sucesso, 0 falhas
    peak_steps = [
        {"obs": np.array([1.0, 0.2, 0.0, 0.0], dtype=np.float32), "delta_obs": np.zeros(4, dtype=np.float32), "prev_action": 0, "prev_reward": 0.0, "action": 0, "reward": 0.30, "log_prob": -0.6, "value": 1.0},
        {"obs": np.array([1.1, 0.3, 0.0, 0.0], dtype=np.float32), "delta_obs": np.zeros(4, dtype=np.float32), "prev_action": 0, "prev_reward": 0.3, "action": 1, "reward": 0.40, "log_prob": -0.5, "value": 1.4},
        {"obs": np.array([1.2, 0.4, 0.0, 0.0], dtype=np.float32), "delta_obs": np.zeros(4, dtype=np.float32), "prev_action": 1, "prev_reward": 0.4, "action": 0, "reward": -0.35, "log_prob": -0.8, "value": 0.8},
        {"obs": np.array([1.3, 0.5, 0.0, 0.0], dtype=np.float32), "delta_obs": np.zeros(4, dtype=np.float32), "prev_action": 0, "prev_reward": -0.35, "action": 1, "reward": -0.45, "log_prob": -0.9, "value": 0.3},
    ]
    trainer._process_completed_trajectory(peak_steps, is_win=False, cur_depth=5)
    assert len(trainer.sil_buffer) == 1
    assert trainer.sil_buffer.count_success_slices() == 1
    assert trainer.sil_buffer.count_full_wins() == 0
    assert len(trainer.sil_buffer.failures) == 0

    # 3. Duas vitórias completas distintas -> armazenadas integralmente, 0 falhas
    for win_id in range(2):
        win_steps = [
            {"obs": np.array([5.0 + win_id, float(t), 0.1, -0.1], dtype=np.float32), "delta_obs": np.zeros(4, dtype=np.float32), "prev_action": 0, "prev_reward": 0.0, "action": t % 2, "reward": 10.0 if t == 2 else 0.2, "log_prob": -0.5, "value": 3.0}
            for t in range(3)
        ]
        trainer._process_completed_trajectory(win_steps, is_win=True, cur_depth=5)

    assert len(trainer.sil_buffer) == 3
    assert trainer.sil_buffer.count_full_wins() == 2
    assert trainer.sil_buffer.count_success_slices() == 1
    assert len(trainer.sil_buffer.failures) == 0

    # 4. Coleta rollout on-policy e executa train_epoch: DPO contrasta vitórias com chunks negativos on-policy
    init_obs, _ = wrapped_env.reset(seed=77)
    trainer.collect_rollouts(current_obs_dict=init_obs, current_hx=None, episode_start=True)
    assert len(trainer.sil_buffer.failures) == 0

    neg_sample = trainer.buffer.sample_negative_chunks(batch_size=2)
    assert neg_sample is not None

    metrics = trainer.train_epoch()
    assert len(trainer.sil_buffer.failures) == 0
    assert metrics["sil_failures"] == 0
    assert np.isfinite(metrics["sil_dpo_loss"])
    assert metrics["sil_dpo_loss"] > 0.0
