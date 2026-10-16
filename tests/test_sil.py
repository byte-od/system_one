"""Testes formais para Self-Imitation Learning (Success Replay Buffer) e Prevenção de Comutadores."""

import pytest
import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.training.buffer import (
    FrontierTransition,
    FrontierTransitionBuffer,
    SuccessReplayBuffer,
    SuccessfulTrajectory,
)
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.env.adapters.rubiks import RubiksCubeEnv, VectorizedRubiksEnv


def test_success_replay_buffer_basic_ops():
    """Valida armazenamento, capacidade máxima circular e amostragem de lote com máscara."""
    buf = SuccessReplayBuffer(
        max_trajectories=4,
        is_visual=False,
        is_discrete=True,
        obs_shape=(324,),
        action_shape=(),
        device=torch.device("cpu"),
        hidden_dim=512,
    )
    assert len(buf) == 0

    # Cria trajetórias sintéticas com comprimentos distintos
    for d in [1, 2, 3, 4, 5]:
        traj = SuccessfulTrajectory(
            obs=np.ones((d + 2, 324), dtype=np.float32) * d,
            delta_obs=None,
            prev_actions=np.zeros(d + 2, dtype=np.int64),
            prev_rewards=np.zeros(d + 2, dtype=np.float32),
            actions=np.ones(d + 2, dtype=np.int64) * (d % 12),
            returns=np.linspace(10.0, 15.0, d + 2, dtype=np.float32),
            depth=d,
            length=d + 2,
        )
        buf.add(traj)

    # Buffer circular limitado a max_trajectories=4 (descartou d=1, contém d=2, 3, 4, 5)
    assert len(buf) == 4
    assert [t.depth for t in buf.trajectories] == [2, 3, 4, 5]

    # Amostra mini-lote
    batch = buf.sample_batch(batch_size=2)
    assert batch is not None
    assert "obs" in batch
    assert "actions" in batch
    assert "returns" in batch
    assert "mask" in batch
    assert "hx" in batch
    assert batch["obs"].shape[0] == 2
    assert batch["hx"].shape == (1, 2, 512)
    assert batch["mask"].shape == batch["returns"].shape


def test_success_replay_buffer_depth_targeting():
    """Garante que a amostragem prioriza trajetórias da profundidade alvo (fronteira curricular)."""
    buf = SuccessReplayBuffer(
        max_trajectories=16,
        obs_shape=(324,),
        action_shape=(),
        device=torch.device("cpu"),
        hidden_dim=512,
    )
    # Adiciona 5 trajetórias rasas (depth=2) e 3 profundas (depth=7)
    for _ in range(5):
        buf.add(
            SuccessfulTrajectory(
                obs=np.zeros((4, 324), dtype=np.float32),
                delta_obs=None,
                prev_actions=np.zeros(4, dtype=np.int64),
                prev_rewards=np.zeros(4, dtype=np.float32),
                actions=np.zeros(4, dtype=np.int64),
                returns=np.ones(4, dtype=np.float32) * 10.0,
                depth=2,
                length=4,
            )
        )
    for _ in range(3):
        buf.add(
            SuccessfulTrajectory(
                obs=np.ones((8, 324), dtype=np.float32),
                delta_obs=None,
                prev_actions=np.zeros(8, dtype=np.int64),
                prev_rewards=np.zeros(8, dtype=np.float32),
                actions=np.ones(8, dtype=np.int64),
                returns=np.ones(8, dtype=np.float32) * 15.0,
                depth=7,
                length=8,
            )
        )

    # Amostragem com target_depth=7 deve priorizar candidatos com depth >= 6
    batch = buf.sample_batch(batch_size=2, target_depth=7)
    assert batch is not None
    # Max length entre as trajetórias de depth=7 é 8
    assert batch["obs"].shape[1] == 8


def test_stratified_buffer_prevents_cross_depth_eviction():
    """Garante isolamento estrito entre profundidades: vitórias de D7 não podem despejar D5."""
    buf = SuccessReplayBuffer(
        max_trajectories=1024,
        max_per_depth=512,
        obs_shape=(324,),
        action_shape=(),
        device=torch.device("cpu"),
        hidden_dim=512,
    )

    # Adiciona 512 trajetórias de profundidade 5
    for i in range(512):
        buf.add(
            SuccessfulTrajectory(
                obs=np.ones((6, 324), dtype=np.float32) * 5.0,
                delta_obs=None,
                prev_actions=np.zeros(6, dtype=np.int64),
                prev_rewards=np.zeros(6, dtype=np.float32),
                actions=np.zeros(6, dtype=np.int64),
                returns=np.ones(6, dtype=np.float32) * 10.0,
                depth=5,
                length=6,
            )
        )

    assert len(buf) == 512
    assert buf.get_depth_distribution() == {5: 512}

    # Adiciona 512 trajetórias de profundidade 7
    for i in range(512):
        buf.add(
            SuccessfulTrajectory(
                obs=np.ones((8, 324), dtype=np.float32) * 7.0,
                delta_obs=None,
                prev_actions=np.zeros(8, dtype=np.int64),
                prev_rewards=np.zeros(8, dtype=np.float32),
                actions=np.ones(8, dtype=np.int64),
                returns=np.ones(8, dtype=np.float32) * 15.0,
                depth=7,
                length=8,
            )
        )

    assert len(buf) == 1024
    assert buf.get_depth_distribution() == {5: 512, 7: 512}

    # Adiciona mais 100 trajetórias de profundidade 7 (atinge 612 em D7)
    # Com buffer plano, isso despejaria 100 trajetórias de D5.
    # Com buffer estratificado, D5 deve permanecer INTACTO com 512 trajetórias!
    for i in range(100):
        buf.add(
            SuccessfulTrajectory(
                obs=np.ones((8, 324), dtype=np.float32) * 7.1,
                delta_obs=None,
                prev_actions=np.zeros(8, dtype=np.int64),
                prev_rewards=np.zeros(8, dtype=np.float32),
                actions=np.ones(8, dtype=np.int64),
                returns=np.ones(8, dtype=np.float32) * 15.0,
                depth=7,
                length=8,
            )
        )

    dist = buf.get_depth_distribution()
    assert dist[5] == 512, f"Erro: D5 sofreu despejo cruzado! Contagem: {dist[5]}"
    assert dist[7] == 512, f"Erro: D7 não respeitou max_per_depth=512! Contagem: {dist[7]}"
    assert len(buf) == 1024

    # Testa amostragem dual (50% fronteira D7, 50% âncora D5)
    batch = buf.sample_batch(batch_size=4, target_depth=7)
    assert batch is not None
    assert batch["obs"].shape[0] == 4
    # Comprimento máximo do lote deve ser 8 (max entre D5 de tam 6 e D7 de tam 8)
    assert batch["obs"].shape[1] == 8
    # Máscaras de atenção / padding: pelo menos uma com comprimento 6 e pelo menos uma com comprimento 8
    mask_sums = batch["mask"].sum(dim=1).cpu().numpy().tolist()
    assert 6.0 in mask_sums, "Lote deveria conter trajetórias âncoras (D5) de comprimento 6"
    assert 8.0 in mask_sums, "Lote deveria conter trajetórias da fronteira (D7) de comprimento 8"


def test_sil_integration_in_ppo_training():
    """Valida ciclo completo: coleta de episódio vitorioso, buffer SIL e gradientes SIL em train_epoch."""
    env = RubiksCubeEnv(scramble_depth=1, curriculum=False)
    wrapper_env = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(obs_space=wrapper_env.observation_space, action_space=wrapper_env.action_space)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapper_env,
        rollout_steps=32,
        chunk_length=8,
        chunk_batch_size=2,
        sil_enabled=True,
        sil_coef=0.10,
        sil_batch_size=2,
        sil_min_samples=1,
    )

    # Injeta artificialmente 2 trajetórias de sucesso para testar train_epoch
    for i in range(2):
        trainer.sil_buffer.add(
            SuccessfulTrajectory(
                obs=np.random.randn(5, 324).astype(np.float32),
                delta_obs=np.random.randn(5, 324).astype(np.float32),
                prev_actions=np.zeros(5, dtype=np.int64),
                prev_rewards=np.zeros(5, dtype=np.float32),
                actions=np.random.randint(0, 12, size=5),
                returns=np.linspace(5.0, 15.0, 5, dtype=np.float32),
                depth=1,
                length=5,
            )
        )

    # Coleta rollouts normais
    obs_dict, _ = wrapper_env.reset()
    trainer.collect_rollouts(obs_dict, current_hx=None, episode_start=True)

    # Executa train_epoch e verifica métricas de SIL
    metrics = trainer.train_epoch()
    assert "policy_loss" in metrics
    assert "sil_policy_loss" in metrics
    assert "sil_value_loss" in metrics
    assert metrics["sil_trajectories"] >= 2
    assert not np.isnan(metrics["policy_loss"])


def test_commutator_avoidance_in_act_fast():
    """Valida que act_fast detecta o comutador [a1, a2, inv(a1)] e evita inv(a2) para impedir loop."""
    env = RubiksCubeEnv(scramble_depth=1)
    wrapper_env = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(obs_space=wrapper_env.observation_space, action_space=wrapper_env.action_space)
    agent.eval()

    obs_dict, _ = wrapper_env.reset()

    # Movimentos atômicos: 4 = F, 5 = F', 6 = B, 7 = B'
    # Sequência: F (4), B' (7), F' (5)
    # Se executar B (6), fecha o comutador [F, B'] = I e volta ao mesmo estado
    agent._action_history = [4, 7, 5]  # a1=4, a2=7, a3=5 (inv de 4)

    # Chamada com qualidade ativada: deve incluir 6 (inv de 7) no avoid_set
    decision = agent.act_fast(obs_dict, return_decision=True, quality_filter=True)
    # Garante que a ação escolhida não é 6 (que fecharia o ciclo de identidade)
    assert decision.action != 6


def test_hierarchical_best_state_dict_prevents_depth_rollback():
    """Garante que o best_state_dict prioriza a profundidade mais alta conquistada (ex: D8 sobre D7)."""
    env = RubiksCubeEnv(scramble_depth=7, curriculum=True, min_depth=7, max_depth=10)
    wrapper_env = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(obs_space=wrapper_env.observation_space, action_space=wrapper_env.action_space)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapper_env,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
    )

    step_counter = [0]

    def fake_collect_rollouts(obs_dict, hx, episode_start, step_callback=None):
        step_counter[0] += 1
        trainer.total_steps += 16
        if step_counter[0] == 1:
            # Rollout 1 no D7: retorno altíssimo (12.89), marca pesos com assinatura 7.0
            env.current_depth = 7
            trainer.episode_returns.clear()
            for _ in range(10):
                trainer.episode_returns.append(12.89)
            with torch.no_grad():
                for p in agent.parameters():
                    p.fill_(0.07)
            return obs_dict, hx, episode_start, 12.89
        elif step_counter[0] == 2:
            # Rollout 2: promoveu para D8! Retorno menor em magnitude (7.37 < 12.89), assinatura 8.0
            env.current_depth = 8
            trainer.episode_returns.clear()
            for _ in range(10):
                trainer.episode_returns.append(7.37)
            with torch.no_grad():
                for p in agent.parameters():
                    p.fill_(0.08)
            return obs_dict, hx, episode_start, 7.37
        else:
            # Rollout 3 ainda no D8: oscilação menor (4.50 < 7.37), assinatura 4.0
            env.current_depth = 8
            trainer.episode_returns.clear()
            for _ in range(10):
                trainer.episode_returns.append(4.50)
            with torch.no_grad():
                for p in agent.parameters():
                    p.fill_(0.04)
            return obs_dict, hx, episode_start, 4.50

    trainer.collect_rollouts = fake_collect_rollouts  # type: ignore[method-assign]
    trainer.train_epoch = lambda: {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}  # type: ignore[method-assign]

    trainer.train(max_steps=48, verbose=False)

    # Deve ter restaurado os pesos do ápice do D8 (0.08, retorno 7.37), e NUNCA voltado para o D7 (0.07)!
    assert trainer.best_state_depth == 8
    assert trainer.best_depth_mean_return == pytest.approx(7.37, abs=1e-4)
    assert trainer.best_mean_return == pytest.approx(12.89, abs=1e-4)
    first_param_val = float(next(agent.parameters()).flatten()[0].item())
    assert first_param_val == pytest.approx(0.08, abs=1e-5)


def test_curriculum_warm_start_and_sil_prewarming():
    """Valida set_curriculum_depth e warmup_sil_buffer para Warm-Start Curricular sem regressão a D1."""
    vec_env = VectorizedRubiksEnv(num_envs=4, curriculum=True, min_depth=1, max_depth=10)
    assert vec_env.current_depth == 1
    assert vec_env.curriculum_promotions == 0

    # Warm-start direto para D7 preservando min_depth=1
    new_d = vec_env.set_curriculum_depth(7)
    assert new_d == 7
    assert vec_env.current_depth == 7
    assert vec_env.min_depth == 1
    assert vec_env.curriculum_promotions == 6
    assert all(e.current_depth == 7 and e.curriculum_promotions == 6 for e in vec_env.envs)

    wrapper_env = UniversalS1Wrapper(vec_env)
    agent = UniversalS1Agent(
        obs_space=wrapper_env.observation_space,
        action_space=wrapper_env.action_space,
        trunk_type="lean",
    )
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapper_env,
        rollout_steps=64,
        chunk_length=16,
        chunk_batch_size=4,
        sil_enabled=True,
    )

    # Executa warmup_sil_buffer em D1..D2
    n_added = trainer.warmup_sil_buffer(up_to_depth=2, episodes_per_depth=16, calibrate_curriculum=False)
    assert isinstance(n_added, int)
    assert 1 in trainer.last_warmup_solve_rates
    assert 2 in trainer.last_warmup_solve_rates
    assert len(trainer.sil_buffer) == n_added


def test_sil_trajectory_quality_pruning_and_priority_replacement():
    """Valida a Poda de Eficiência de Trajetórias e Substituição Prioritária por Qualidade (Opção 3)."""
    buf = SuccessReplayBuffer(
        max_trajectories=100,
        max_per_depth=2,
        obs_shape=(324,),
        action_shape=(),
        device=torch.device("cpu"),
        hidden_dim=512,
    )

    # 1. Adiciona trajetórias de D7 com 8 passos (ótima eficiência)
    for i in range(4):
        buf.add(
            SuccessfulTrajectory(
                obs=np.zeros((8, 324), dtype=np.float32),
                delta_obs=None,
                prev_actions=np.zeros(8, dtype=np.int64),
                prev_rewards=np.zeros(8, dtype=np.float32),
                actions=np.zeros(8, dtype=np.int64),
                returns=np.ones(8, dtype=np.float32) * (10.0 + i),
                depth=7,
                length=8,
            )
        )
    assert len(buf._depth_buffers[7]) == 2

    # 2. Trajetória excessivamente longa em D7 (25 passos) deve ser podada (max permitido é 14)
    pruned_before = buf.total_pruned
    buf.add(
        SuccessfulTrajectory(
            obs=np.zeros((25, 324), dtype=np.float32),
            delta_obs=None,
            prev_actions=np.zeros(25, dtype=np.int64),
            prev_rewards=np.zeros(25, dtype=np.float32),
            actions=np.zeros(25, dtype=np.int64),
            returns=np.ones(25, dtype=np.float32) * 5.0,
            depth=7,
            length=25,
        )
    )
    assert buf.total_pruned == pruned_before + 1
    assert all(t.length == 8 for t in buf._depth_buffers[7])

    # 3. Substituição prioritária:
    # Trajetória inferior (return 8.0) é rejeitada
    buf.add(
        SuccessfulTrajectory(
            obs=np.zeros((8, 324), dtype=np.float32),
            delta_obs=None,
            prev_actions=np.zeros(8, dtype=np.int64),
            prev_rewards=np.zeros(8, dtype=np.float32),
            actions=np.zeros(8, dtype=np.int64),
            returns=np.ones(8, dtype=np.float32) * 8.0,
            depth=7,
            length=8,
        )
    )
    assert min(t.returns[0] for t in buf._depth_buffers[7]) == 12.0

    # Trajetória superior (return 16.0, length 7) substitui a pior (12.0)
    buf.add(
        SuccessfulTrajectory(
            obs=np.zeros((7, 324), dtype=np.float32),
            delta_obs=None,
            prev_actions=np.zeros(7, dtype=np.int64),
            prev_rewards=np.zeros(7, dtype=np.float32),
            actions=np.zeros(7, dtype=np.int64),
            returns=np.ones(7, dtype=np.float32) * 16.0,
            depth=7,
            length=7,
        )
    )
    returns_in_buf = sorted([float(t.returns[0]) for t in buf._depth_buffers[7]])
    assert returns_in_buf == [13.0, 16.0]


def test_frontier_transition_from_step_and_alias():
    """Valida instanciação atômica via from_step e o alias FrontierTransitionBuffer."""
    assert FrontierTransitionBuffer is SuccessReplayBuffer

    trans = FrontierTransition.from_step(
        obs=np.ones(324, dtype=np.float32) * 2.0,
        action=5,
        reward=10.0,
        target_return=12.5,
        depth=3,
        delta_obs=np.ones(324, dtype=np.float32) * 0.1,
        prev_action=4,
        prev_reward=1.0,
        td_error=2.5,
    )
    assert isinstance(trans, SuccessfulTrajectory)
    assert isinstance(trans, FrontierTransition)
    assert trans.length == 1
    assert trans.depth == 3
    assert trans.obs.shape == (1, 324)
    assert trans.actions.shape == (1,)
    assert trans.actions[0] == 5
    assert trans.returns[0] == 12.5
    assert trans.td_error == 2.5

    # Testa adição direta ao buffer
    buf = FrontierTransitionBuffer(max_trajectories=10, obs_shape=(324,))
    buf.add(trans)
    assert len(buf) == 1
    batch = buf.sample_batch(batch_size=1)
    assert batch is not None
    assert batch["obs"].shape == (1, 1, 324)
    assert batch["actions"][0, 0] == 5
    assert batch["returns"][0, 0] == 12.5


def test_visual_observation_uint8_compression_and_decompression():
    """Valida compressão em uint8 para ambientes visuais (ex: ViZDoom/pixels) e descompressão float32."""
    buf = SuccessReplayBuffer(
        max_trajectories=10,
        is_visual=True,
        obs_shape=(3, 64, 64),
        action_shape=(),
        device=torch.device("cpu"),
    )

    # Imagem simulada em float32 normalizada [0.0, 1.0]
    raw_img = np.random.uniform(0.0, 1.0, size=(5, 3, 64, 64)).astype(np.float32)
    traj = SuccessfulTrajectory(
        obs=raw_img.copy(),
        delta_obs=None,
        prev_actions=np.zeros(5, dtype=np.int64),
        prev_rewards=np.zeros(5, dtype=np.float32),
        actions=np.zeros(5, dtype=np.int64),
        returns=np.ones(5, dtype=np.float32) * 5.0,
        depth=1,
        length=5,
    )

    buf.add(traj)
    # A observação guardada no buffer DEVE estar comprimida em uint8 (1 byte por pixel em vez de 4)
    stored_traj = buf._depth_buffers[1][0]
    assert stored_traj.obs.dtype == np.uint8
    assert stored_traj.obs.nbytes < raw_img.nbytes  # Redução de 75% em memória RAM!

    # Na amostragem, sample_batch deve descompactar de volta para float32 [0.0, 1.0]
    batch = buf.sample_batch(batch_size=1)
    assert batch is not None
    assert batch["obs"].dtype == torch.float32
    assert batch["obs"].shape == (1, 5, 3, 64, 64)
    # O valor descompactado deve ser aproximadamente igual ao original (erro de quantização < 1/255)
    decompressed = batch["obs"][0].cpu().numpy()
    np.testing.assert_allclose(decompressed, raw_img, atol=1.0 / 255.0)


def test_frontier_slicing_preserves_deep_solutions():
    """Valida Frontier Slicing: vitórias longas com desvios em D7 retêm o prefixo decisivo em vez de serem podadas."""
    # Buffer COM frontier_slicing ativado (como configurado no RecurrentPPOTrainer)
    buf = SuccessReplayBuffer(
        max_trajectories=10,
        max_per_depth=5,
        obs_shape=(324,),
        action_shape=(),
        device=torch.device("cpu"),
        frontier_slicing=True,
    )

    # Trajetória longa de 25 passos em D7 (max_allowed seria 14)
    traj = SuccessfulTrajectory(
        obs=np.tile(np.arange(25)[:, None], (1, 324)).astype(np.float32),
        delta_obs=None,
        prev_actions=np.arange(25, dtype=np.int64),
        prev_rewards=np.zeros(25, dtype=np.float32),
        actions=np.arange(25, dtype=np.int64),
        returns=np.linspace(10.0, 20.0, 25, dtype=np.float32),
        depth=7,
        length=25,
    )

    buf.add(traj)
    # Não deve ter sido podada! Deve ter sido cortada para o prefixo decisivo de max_allowed (17 quando len < 4)
    assert buf.total_pruned == 0
    assert len(buf) == 1
    stored = buf._depth_buffers[7][0]
    assert stored.length == 17  # Sliced para 17
    assert stored.obs.shape[0] == 17
    # Preserva o estado inicial de fronteira (passo 0) e o retorno alto acumulado
    assert stored.obs[0, 0] == 0.0
    assert stored.returns[0] == pytest.approx(10.0, abs=1e-4)


def test_frontier_priority_sampling_ratio():
    """Valida amostragem balanceada 75% fronteira e 25% âncoras quando target_depth está ativo."""
    buf = SuccessReplayBuffer(
        max_trajectories=20,
        obs_shape=(324,),
        action_shape=(),
        device=torch.device("cpu"),
        frontier_ratio=0.75,
    )

    # Adiciona 5 âncoras (D2..D3) e 5 fronteiras (D7)
    for _ in range(5):
        buf.add(
            SuccessfulTrajectory(
                obs=np.zeros((4, 324), dtype=np.float32),
                delta_obs=None,
                prev_actions=np.zeros(4, dtype=np.int64),
                prev_rewards=np.zeros(4, dtype=np.float32),
                actions=np.zeros(4, dtype=np.int64),
                returns=np.ones(4, dtype=np.float32) * 5.0,
                depth=2,
                length=4,
            )
        )
    for _ in range(5):
        buf.add(
            SuccessfulTrajectory(
                obs=np.ones((6, 324), dtype=np.float32),
                delta_obs=None,
                prev_actions=np.zeros(6, dtype=np.int64),
                prev_rewards=np.zeros(6, dtype=np.float32),
                actions=np.ones(6, dtype=np.int64),
                returns=np.ones(6, dtype=np.float32) * 15.0,
                depth=7,
                length=6,
            )
        )

    # Amostra lote de 4 com target_depth=7
    # 75% de 4 = 3 fronteiras (D7, length 6) e 25% de 4 = 1 âncora (D2, length 4)
    batch = buf.sample_batch(batch_size=4, target_depth=7)
    assert batch is not None
    mask_sums = batch["mask"].sum(dim=1).cpu().numpy().tolist()
    assert mask_sums.count(6.0) == 3, f"Esperado 3 amostras de fronteira (tam 6), obtido: {mask_sums}"
    assert mask_sums.count(4.0) == 1, f"Esperado 1 amostra de âncora (tam 4), obtido: {mask_sums}"




