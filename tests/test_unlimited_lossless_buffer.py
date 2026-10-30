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


def test_o1_running_telemetry_counters_accuracy_under_mutations():
    """Valida que todos os contadores O(1) de telemetria coincidem com a soma manual sob adições, upgrades e limpezas."""
    buf = SuccessReplayBuffer(obs_shape=(8,), is_discrete=True, lossless_compaction=True)
    rng = np.random.default_rng(123)

    # 1. Buffer vazio inicial
    assert len(buf) == 0
    assert buf.memory_bytes == 0
    assert buf.uncompressed_bytes == 0
    assert buf.count_full_wins() == 0
    assert buf.count_success_slices() == 0

    # 2. Inserção mista de 20 fatias e 30 vitórias completas em profundidades variadas
    for i in range(50):
        depth = (i % 5) + 1
        is_fw = (i >= 20)
        length = 4 + (i % 3)  # comprimentos 4, 5, 6 respeitam o filtro max_allowed
        obs = rng.integers(0, 2, size=(length, 8)).astype(np.float32)
        obs[0, 0] = float(i)
        t = FrontierTransition(
            obs=obs,
            delta_obs=None,
            prev_actions=np.zeros(length, dtype=np.int64),
            prev_rewards=np.zeros(length, dtype=np.float32),
            actions=rng.integers(0, 4, size=length, dtype=np.int64),
            returns=np.linspace(5.0, 1.0, length, dtype=np.float32),
            depth=depth,
            length=length,
            is_full_win=is_fw,
        )
        buf.add(t)

    # Verifica exatidão bit-a-bit dos contadores O(1) contra soma O(N)
    manual_trajs = [t for b in buf._depth_buffers.values() for t in b]
    assert len(buf) == len(manual_trajs) == 50
    assert buf.memory_bytes == sum(t.memory_bytes for t in manual_trajs)
    assert buf.uncompressed_bytes == sum(t.uncompressed_bytes for t in manual_trajs)
    assert buf.count_full_wins() == sum(1 for t in manual_trajs if t.is_full_win) == 30
    assert buf.count_success_slices() == sum(1 for t in manual_trajs if not t.is_full_win) == 20

    # 3. Upgrade in-place de uma fatia parcial para vitória completa com menor comprimento
    target_slice = buf._depth_buffers[1][0]
    upgraded_obs = target_slice.obs[:3].copy()
    better_t = FrontierTransition(
        obs=upgraded_obs,
        delta_obs=None,
        prev_actions=np.zeros(3, dtype=np.int64),
        prev_rewards=np.zeros(3, dtype=np.float32),
        actions=np.zeros(3, dtype=np.int64),
        returns=np.ones(3, dtype=np.float32) * 10.0,
        depth=1,
        length=3,
        is_full_win=True,
    )
    buf.add(better_t)

    manual_trajs = [t for b in buf._depth_buffers.values() for t in b]
    assert len(buf) == 50
    assert buf.memory_bytes == sum(t.memory_bytes for t in manual_trajs)
    assert buf.uncompressed_bytes == sum(t.uncompressed_bytes for t in manual_trajs)
    assert buf.count_full_wins() == 31
    assert buf.count_success_slices() == 19

    # 4. Limpeza total do buffer
    buf.clear()
    assert len(buf) == 0
    assert buf.memory_bytes == 0
    assert buf.uncompressed_bytes == 0
    assert buf.count_full_wins() == 0
    assert buf.count_success_slices() == 0


def test_o1_tournament_selection_prioritizes_full_wins_and_optimal_paths():
    """Valida que o Torneio O(1) enriquece a taxa de vitórias e caminhos eficientes sem tocar nas observações."""
    buf = SuccessReplayBuffer(obs_shape=(4,), is_discrete=True, lossless_compaction=True)
    rng = np.random.default_rng(42)

    # Cria um pool com 90% fatias parciais longas e 10% vitórias completas curtas e ótimas no nível 5
    for i in range(100):
        is_win = (i < 10)  # apenas 10% são vitórias completas
        length = 5 if is_win else 9  # vitória em 5 passos vs fatia em 9 passos (ambos <= max_allowed 10)
        ret_val = 10.0 if is_win else 2.0
        obs = rng.integers(0, 2, size=(length, 4)).astype(np.float32)
        obs[0, 0] = float(i)
        buf.add(
            FrontierTransition(
                obs=obs,
                delta_obs=None,
                prev_actions=np.zeros(length, dtype=np.int64),
                prev_rewards=np.zeros(length, dtype=np.float32),
                actions=np.zeros(length, dtype=np.int64),
                returns=np.ones(length, dtype=np.float32) * ret_val,
                depth=5,
                length=length,
                is_full_win=is_win,
            )
        )

    assert len(buf) == 100
    assert buf.count_full_wins() == 10
    assert buf.count_success_slices() == 90

    # 1. Amostragem uniforme pura (tournament_size=1) reproduz ~10% de vitórias
    np.random.seed(123)
    n_samples = 400
    sampled_uniform_wins = 0
    for _ in range(n_samples):
        b = buf.sample_batch(batch_size=1, target_depth=5, tournament_size=1)
        # Se for vitória completa, o retorno acumulado é 10.0
        if float(b["returns"][0, 0].item()) >= 9.0:
            sampled_uniform_wins += 1

    uniform_win_rate = sampled_uniform_wins / n_samples
    assert 0.05 <= uniform_win_rate <= 0.16, f"Uniform rate {uniform_win_rate} fora do esperado (~10%)"

    # 2. Torneio k=3 eleva a taxa de vitórias completas para ~25-35%
    sampled_tourn3_wins = 0
    for _ in range(n_samples):
        b = buf.sample_batch(batch_size=1, target_depth=5, tournament_size=3)
        if float(b["returns"][0, 0].item()) >= 9.0:
            sampled_tourn3_wins += 1

    tourn3_win_rate = sampled_tourn3_wins / n_samples
    assert tourn3_win_rate >= 0.20, f"Tournament-3 rate {tourn3_win_rate} deveria enriquecer vitórias (esperado >= 20%)"
    assert tourn3_win_rate > uniform_win_rate * 1.5, "Torneio deve enriquecer significativamente em relação ao uniforme"

    # 3. Torneio k=5 (Padrão de Produção) atinge enriquecimento ótimo (~40%) sem omitir fatias
    sampled_tourn5_wins = 0
    for _ in range(n_samples):
        # Omissão do parâmetro tournament_size valida o valor default padrão (k=5)
        b = buf.sample_batch(batch_size=1, target_depth=5)
        if float(b["returns"][0, 0].item()) >= 9.0:
            sampled_tourn5_wins += 1

    tourn5_win_rate = sampled_tourn5_wins / n_samples
    assert tourn5_win_rate >= 0.30, f"Tournament-5 default rate {tourn5_win_rate} deveria enriquecer vitórias para >= 30%"
    assert tourn5_win_rate > tourn3_win_rate, "k=5 deve prover maior pressão seletiva que k=3"

    # 4. Validação do modo only_full_wins=True em O(1) via _full_win_buffers
    b_only_wins = buf.sample_batch(batch_size=8, target_depth=5, only_full_wins=True)
    assert b_only_wins is not None
    # Todos os itens do lote devem ser vitórias completas (retorno 10.0 e comprimento 5)
    for i in range(8):
        assert float(b_only_wins["returns"][i, 0].item()) >= 9.0
        # A máscara deve ser exatamente 5
        assert int(b_only_wins["mask"][i].sum().item()) == 5
