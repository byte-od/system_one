import os
import tempfile
import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from system1_engine.core.agent import UniversalS1Agent, ReflexDecision
from system1_engine.core.heads import CategoricalPolicyHead, GaussianPolicyHead
from system1_engine.env.wrapper import UniversalS1Wrapper, ActionChunkWrapper
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def test_categorical_policy_head_chunking():
    hidden_dim = 256
    n_actions = 12
    k = 3
    head = CategoricalPolicyHead(in_features=hidden_dim, n_actions=n_actions, action_chunk_size=k)
    assert head.action_chunk_size == k

    B, T = 2, 4
    h = torch.randn(B, T, hidden_dim)
    dist = head(h)

    # 1. Forward output shape
    assert dist.logits.shape == (B, T, k, n_actions)

    # 2. Sample action shape
    actions = dist.sample()
    assert actions.shape == (B, T, k)
    assert actions.dtype == torch.long

    # 3. Log-prob joint sum over chunk dimension
    log_p = dist.log_prob(actions).sum(dim=-1)
    assert log_p.shape == (B, T)

    # Manual verification of joint log-prob sum
    raw_log_probs = torch.log_softmax(dist.logits, dim=-1)
    gathered = torch.gather(raw_log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)
    expected_log_p = gathered.sum(dim=-1)
    assert torch.allclose(log_p, expected_log_p, atol=1e-5)

    # 4. Mean entropy across chunk dimension
    ent = dist.entropy().mean(dim=-1)
    assert ent.shape == (B, T)
    probs = torch.softmax(dist.logits, dim=-1)
    expected_ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean(dim=-1)
    assert torch.allclose(ent, expected_ent, atol=1e-4)


def test_gaussian_policy_head_chunking():
    hidden_dim = 256
    act_dim = 3
    k = 4
    head = GaussianPolicyHead(in_features=hidden_dim, act_dim=act_dim, action_chunk_size=k)
    assert head.action_chunk_size == k

    B, T = 2, 5
    h = torch.randn(B, T, hidden_dim)
    dist = head(h)

    # 1. Sample action shape
    actions = dist.sample()
    assert actions.shape == (B, T, k, act_dim)

    # 2. Log-prob joint sum over chunk and act dimensions
    action_out, log_p, ent = head.get_action_and_log_prob(h)
    assert action_out.shape == (B, T, k, act_dim)
    assert log_p.shape == (B, T)
    assert ent.shape == (B, T)


def test_universal_s1_agent_chunking_discrete():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(12)
    k = 3

    agent = UniversalS1Agent(
        obs_space=obs_space,
        action_space=action_space,
        trunk_type="lean",
        action_chunk_size=k,
    )
    assert agent.action_chunk_size == k

    # Agent forward
    B, T = 2, 4
    obs = torch.randn(B, T, 4)
    prev_a = torch.zeros(B, T, dtype=torch.long)
    prev_r = torch.zeros(B, T, 1)

    obs_dict = {
        "obs": obs,
        "delta_obs": torch.zeros_like(obs),
        "prev_action": prev_a,
        "prev_reward": prev_r,
    }

    dist, val, hx = agent.forward(obs_dict)
    sampled_actions = dist.sample()
    assert sampled_actions.shape == (B, T, k)
    assert val.shape == (B, T, 1)

    # get_action single-step inference
    single_obs = torch.randn(B, 4)
    single_prev_a = torch.zeros(B, dtype=torch.long)
    single_prev_r = torch.zeros(B, 1)
    single_obs_dict = {
        "obs": single_obs,
        "delta_obs": torch.zeros_like(single_obs),
        "prev_action": single_prev_a,
        "prev_reward": single_prev_r,
    }
    act_out, logp_out, val_out, next_hx = agent.get_action(single_obs_dict)
    assert act_out.shape == (B, k)
    assert logp_out.shape == (B,)
    assert val_out.shape == (B, 1)

    # act_fast ultra-fast inference
    obs_dict = {
        "obs": np.random.randn(4).astype(np.float32),
        "delta_obs": np.zeros(4, dtype=np.float32),
        "prev_action": 0,
        "prev_reward": 0.0,
    }
    chunk_act = agent.act_fast(obs_dict, deterministic=True)
    assert isinstance(chunk_act, np.ndarray)
    assert chunk_act.shape == (k,)

    # act_fast with return_decision
    decision = agent.act_fast(obs_dict, return_decision=True)
    assert isinstance(decision, ReflexDecision)
    assert isinstance(decision.action, np.ndarray)
    assert decision.action.shape == (k,)
    assert 0.0 <= decision.confidence <= 1.0
    assert 0.0 <= decision.uncertainty <= 1.0
    assert decision.latency_ms > 0.0

    # act_with_confidence
    dec_conf = agent.act_with_confidence(obs_dict)
    assert isinstance(dec_conf, ReflexDecision)
    assert dec_conf.action.shape == (k,)


def test_universal_s1_agent_chunking_continuous():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
    k = 4

    agent = UniversalS1Agent(
        obs_space=obs_space,
        action_space=action_space,
        trunk_type="lean",
        action_chunk_size=k,
    )
    assert agent.action_chunk_size == k

    obs_dict = {
        "obs": np.random.randn(6).astype(np.float32),
        "delta_obs": np.zeros(6, dtype=np.float32),
        "prev_action": np.zeros(2, dtype=np.float32),
        "prev_reward": 0.0,
    }
    chunk_act = agent.act_fast(obs_dict, deterministic=True)
    assert isinstance(chunk_act, np.ndarray)
    assert chunk_act.shape == (k, 2)
    assert np.all(chunk_act >= -2.0) and np.all(chunk_act <= 2.0)


def test_wrapper_chunk_execution_discrete():
    raw_env = gym.make("CartPole-v1")
    wrapper = UniversalS1Wrapper(raw_env, gamma=0.99)
    obs_dict, _ = wrapper.reset(seed=42)

    # Step with an action chunk of 3 actions
    chunk = [1, 0, 1]
    next_obs, reward, terminated, truncated, info = wrapper.step(chunk)

    assert "obs" in next_obs
    assert "delta_obs" in next_obs
    assert "prev_action" in next_obs
    assert next_obs["prev_action"] in (0, 1)
    assert isinstance(reward, float)
    assert reward >= 1.0  # CartPole gives +1 per step alive
    wrapper.close()


def test_wrapper_chunk_early_termination():
    # Test early break on solved/terminated
    raw_env = gym.make("CartPole-v1")
    wrapper = UniversalS1Wrapper(raw_env, gamma=0.99)
    wrapper.reset(seed=42)

    # Force CartPole to fail quickly by pushing in same direction repeatedly
    done = False
    for _ in range(50):
        chunk = [1, 1, 1, 1]
        _, rew, term, trunc, _ = wrapper.step(chunk)
        if term or trunc:
            done = True
            break
    assert done
    wrapper.close()


def test_ppo_trainer_with_action_chunking():
    # Mini end-to-end PPO training run with k=2
    env = gym.make("CartPole-v1")
    wrapped_env = UniversalS1Wrapper(env)
    k = 2

    agent = UniversalS1Agent(
        obs_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        trunk_type="lean",
        action_chunk_size=k,
    )

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped_env,
        action_chunk_size=k,
        learning_rate=3e-4,
        rollout_steps=128,
        chunk_length=16,
        chunk_batch_size=4,
        sil_enabled=True,
        sil_dpo_enabled=True,
    )

    assert trainer.action_chunk_size == k
    assert trainer.buffer.action_shape == (k,)

    # Train for 2 iterations (256 steps)
    final_ret = trainer.train(max_steps=256, verbose=False)
    assert isinstance(final_ret, float)
    wrapped_env.close()


def test_checkpoint_save_and_autodetect():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(12)
    k = 3

    agent = UniversalS1Agent(
        obs_space=obs_space,
        action_space=action_space,
        trunk_type="lean",
        action_chunk_size=k,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_act_k3.pt")
        KnowledgeTransferManager.save_checkpoint(
            agent=agent,
            checkpoint_path=ckpt_path,
            extra_info={"env_id": "RubiksCube-v0", "steps": 1000},
            auto_sync_web=False,
        )

        assert os.path.exists(ckpt_path)

        # Inspect checkpoint
        info = KnowledgeTransferManager.inspect_checkpoint(ckpt_path)
        assert info.get("action_chunk_size") == 3
        assert info.get("env_id") == "RubiksCube-v0"

        # Auto-detect chunk size
        detected_k = KnowledgeTransferManager.detect_action_chunk_size(ckpt_path, action_space)
        assert detected_k == 3


def test_backward_compatibility_k1():
    # Verify that action_chunk_size=1 preserves exact standard behavior
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    agent_k1 = UniversalS1Agent(
        obs_space=obs_space,
        action_space=action_space,
        action_chunk_size=1,
    )
    assert agent_k1.action_chunk_size == 1

    obs_dict = {
        "obs": np.zeros(4, dtype=np.float32),
        "delta_obs": np.zeros(4, dtype=np.float32),
        "prev_action": 0,
        "prev_reward": 0.0,
    }
    action = agent_k1.act_fast(obs_dict, deterministic=True)
    # For k=1, returns an integer action
    assert isinstance(action, (int, np.integer))


def test_vectorized_rubiks_action_chunking():
    from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv
    raw_env = VectorizedRubiksEnv(num_envs=8, is_macro=False, scramble_depth=2, curriculum=False)
    wrapped = UniversalS1Wrapper(raw_env)
    k = 3

    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
        action_chunk_size=k,
    )

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped,
        action_chunk_size=k,
        learning_rate=3e-4,
        rollout_steps=128,
        chunk_length=16,
        chunk_batch_size=4,
        sil_enabled=True,
        sil_dpo_enabled=True,
    )

    final_ret = trainer.train(max_steps=128, verbose=False)
    assert isinstance(final_ret, float)
    wrapped.close()


def test_smdp_gae_discounting():
    """Valida o rigor matemático do desconto SMDP gamma^k no GAE e retornos."""
    from system1_engine.training.buffer import RecurrentRolloutBuffer
    buffer = RecurrentRolloutBuffer(
        buffer_size=4,
        chunk_length=4,
        obs_shape=(4,),
        action_shape=(),
    )
    obs_dict = {
        "obs": np.zeros(4, dtype=np.float32),
        "delta_obs": np.zeros(4, dtype=np.float32),
        "prev_action": 0,
        "prev_reward": 0.0,
    }
    # Adiciona 4 transições
    for i in range(4):
        buffer.add(
            obs_dict=obs_dict,
            action=0,
            reward=1.0,
            done=False,
            episode_start=(i == 0),
            value=0.5,
            log_prob=-0.2,
        )

    gamma = 0.90
    gae_lambda = 0.95
    k = 3
    eff_gamma = gamma ** k  # 0.729

    buffer.compute_gae(
        last_value=1.0,
        last_done=False,
        gamma=gamma,
        gae_lambda=gae_lambda,
        action_chunk_size=k,
    )

    # Verifica o delta no último passo: r_3 + eff_gamma * next_val - val_3
    # delta_3 = 1.0 + 0.729 * 1.0 - 0.5 = 1.229
    expected_delta_3 = 1.0 + eff_gamma * 1.0 - 0.5
    assert np.isclose(buffer.advantages[3], expected_delta_3, atol=1e-5)


def test_gaussian_policy_head_heteroscedastic():
    """Valida GaussianPolicyHead com log_std heterocedástico 2D (k, act_dim)."""
    hidden_dim = 64
    act_dim = 2
    k = 4
    head = GaussianPolicyHead(in_features=hidden_dim, act_dim=act_dim, action_chunk_size=k)
    # Atribui parâmetro 2D com desvios crescentes ao longo do horizonte temporal
    head.log_std = nn.Parameter(torch.zeros(k, act_dim))
    h = torch.randn(2, 3, hidden_dim)
    dist = head(h)
    assert dist.scale.shape == (2, 3, k, act_dim)
    act_sample = dist.sample()
    assert act_sample.shape == (2, 3, k, act_dim)


def test_onnx_export_with_action_chunking():
    """Valida que o exportador ONNX detecta e preserva action_chunk_size > 1."""
    from system1_engine.core.onnx_exporter import export_checkpoint_to_onnx
    from system1_engine.env.adapters.rubiks import RubiksCubeEnv
    raw_env = RubiksCubeEnv()
    wrapped = UniversalS1Wrapper(raw_env)
    k = 3

    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
        action_chunk_size=k,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_act_export.pt")
        KnowledgeTransferManager.save_checkpoint(
            agent=agent,
            checkpoint_path=ckpt_path,
            extra_info={"env_id": "RubiksCube-v0", "action_chunk_size": k},
            auto_sync_web=False,
        )

        onnx_path = export_checkpoint_to_onnx(ckpt_path, env_id="RubiksCube-v0", sync_portfolio=False)
        assert onnx_path is not None
        assert os.path.exists(onnx_path)
    wrapped.close()


def test_vectorized_act_training_and_telemetry():
    """Valida treino vetorizado com action chunking e coleta de métricas de telemetria sem erro de dimensões."""
    from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv
    from system1_engine.telemetry.tracker import LiveStatsTracker
    raw_env = VectorizedRubiksEnv(num_envs=4, curriculum=True, min_depth=1, max_depth=2)
    wrapped = UniversalS1Wrapper(raw_env)
    tracker = LiveStatsTracker()
    k = 4
    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
        action_chunk_size=k,
    )
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped,
        tracker=tracker,
        action_chunk_size=k,
        rollout_steps=16,
        chunk_length=8,
        sil_enabled=True,
        sil_dpo_enabled=True,
    )
    final_ret = trainer.train(max_steps=32, verbose=False)
    assert isinstance(final_ret, float)
    assert tracker.metrics.fps > 0
    assert tracker.metrics.total_steps >= 32
    wrapped.close()


def test_clean_state_preservation_on_mid_chunk_reset():
    """Valida que ambientes que resetam no meio de um chunk (ex: passo 1 de 4) NÃO recebem

    ações residuais dos passos restantes (passos 2 e 3), mantendo seu novo scramble puro com 0 passos.
    """
    from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv

    raw_env = VectorizedRubiksEnv(num_envs=2, curriculum=False, scramble_depth=2, max_steps=10)
    wrapped = UniversalS1Wrapper(raw_env)
    wrapped.reset()

    # Força o ambiente 0 a atingir término no primeiro passo simulando max_steps atingido (passo 9 -> 10)
    raw_env.envs[0]._steps = raw_env.envs[0].max_steps - 1
    raw_env.envs[1]._steps = 0

    # Executa chunk de 4 ações: [2, 4]
    chunk_actions = np.array([
        [0, 1, 2, 3],  # env 0: no passo 0 atinge term/trunc -> reseta imediatamente! Passos 1, 2, 3 devem ser ignorados
        [0, 1, 2, 3],  # env 1: prossegue normalmente pelos 4 passos
    ])

    obs_dict, rews, terms, truncs, info = wrapped.step(chunk_actions)

    # Env 0 deve ter resetado e os passos residuais (1, 2, 3) NÃO devem ter sido executados no novo cubo
    # Portanto, no novo episódio, _steps deve ser 0!
    assert raw_env.envs[0]._steps == 0, f"Env 0 executou passos residuais após reset: _steps={raw_env.envs[0]._steps}"
    # Env 1 não resetou, deve ter executado exatamente 4 passos
    assert raw_env.envs[1]._steps == 4, f"Env 1 deveria ter 4 passos, mas tem {raw_env.envs[1]._steps}"

    wrapped.close()


def test_adaptive_chunk_uncertainty_truncation():
    """Valida que o adaptive_chunk trunca a máscara de execução quando a incerteza sobe ou margem cai."""
    from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv

    raw_env = VectorizedRubiksEnv(num_envs=2, curriculum=False, scramble_depth=2)
    wrapped = UniversalS1Wrapper(raw_env)
    k = 4
    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
        action_chunk_size=k,
    )
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped,
        action_chunk_size=k,
        adaptive_chunk=True,
        adaptive_conf_threshold=0.65,
        adaptive_margin_threshold=0.15,
        rollout_steps=4,
    )

    # Simula cálculo de chunk_mask com logits sintéticos
    # Env 0: alta confiança no passo 1, baixa confiança no passo 2 (< 0.65)
    # Env 1: alta confiança em todos os passos
    E = 2
    num_acts = 12
    fake_logits = torch.zeros(E, k, num_acts)
    # Env 0:
    fake_logits[0, 0, 0] = 10.0  # conf ~ 1.0
    fake_logits[0, 1, 0] = 10.0  # conf ~ 1.0
    fake_logits[0, 2, :] = 0.0   # uniforme: conf = 1/12 = 0.083 < 0.65 -> trunca aqui!
    fake_logits[0, 3, 0] = 10.0  # deveria ser ignorado pois passo 2 truncou

    # Env 1: todos confiantes
    fake_logits[1, :, 0] = 10.0

    probs_t = torch.softmax(fake_logits, dim=-1)
    top2_t, _ = torch.topk(probs_t, k=2, dim=-1)
    confs_t = top2_t[..., 0]
    margins_t = top2_t[..., 0] - top2_t[..., 1]

    chunk_mask = np.ones((E, k), dtype=bool)
    for step_i in range(1, k):
        unc_mask = (confs_t[:, step_i] < trainer.adaptive_conf_threshold) | (margins_t[:, step_i] < trainer.adaptive_margin_threshold)
        unc_env_indices = torch.where(unc_mask)[0].cpu().numpy()
        chunk_mask[unc_env_indices, step_i:] = False

    # Env 0 deve ter executado passos 0 e 1, e abortado passos 2 e 3
    assert np.array_equal(chunk_mask[0], [True, True, False, False])
    # Env 1 deve ter executado todos os 4 passos
    assert np.array_equal(chunk_mask[1], [True, True, True, True])

    wrapped.close()


def test_anti_collapse_exploration_floor():
    """Valida que o mecanismo de anti-colapso preserva o piso de calibração >= 0.25 e temperatura >= 0.20."""
    from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv

    raw_env = VectorizedRubiksEnv(num_envs=4, curriculum=True, min_depth=1)
    wrapped = UniversalS1Wrapper(raw_env)
    k = 4
    agent = UniversalS1Agent(
        obs_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        trunk_type="lean",
        action_chunk_size=k,
    )
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped,
        action_chunk_size=k,
        exploration_scale="auto",
        entropy_coef=0.015,
    )

    # Verifica o piso mínimo configurado
    assert trainer._min_adaptive_scale >= 0.25
    assert trainer.entropy_coef == 0.015

    # Simula sucessos repetidos para forçar decaimento de escala
    dummy_infos = [{"curriculum_promotions": 0, "curriculum_promoted": False, "curriculum_success_rate": 0.95}]
    for _ in range(50):
        trainer._update_adaptive_exploration(mean_return=10.0, env_infos=dummy_infos)

    # Escala adaptativa NUNCA deve cair abaixo de 0.25
    assert trainer.exploration_scale >= 0.25, f"Exploration scale colapsou: {trainer.exploration_scale}"
    assert trainer._adaptive_scale >= 0.25

    wrapped.close()


def test_net2net_action_chunking_expansion():
    """Valida a expansão Net2Net de k=1 para k=2 preservando pesos e comportamento inicial."""
    import tempfile
    from system1_engine.transfer.manager import KnowledgeTransferManager

    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(324,), dtype=np.float32)
    act_space = gym.spaces.Discrete(12)

    # 1. Agente k=1
    agent_k1 = UniversalS1Agent(obs_space=obs_space, action_space=act_space, trunk_type="lean", action_chunk_size=1)
    # Define pesos arbitrários conhecidos na cabeça de política
    with torch.no_grad():
        agent_k1.policy_head.linear.weight.fill_(0.42)
        agent_k1.policy_head.linear.bias.fill_(0.15)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt_k1.pt")
        KnowledgeTransferManager.save_checkpoint(
            agent_k1,
            ckpt_path,
            extra_info={"env_id": "RubiksCube-v0", "steps": 50000000, "curriculum_depth": 8},
            auto_sync_web=False,
        )

        # 2. Agente k=2 (Action Chunking)
        agent_k2 = UniversalS1Agent(obs_space=obs_space, action_space=act_space, trunk_type="lean", action_chunk_size=2)
        assert agent_k2.policy_head.linear.weight.shape == (24, 512)

        # 3. Carrega via load_for_training (deve acionar Net2Net Action Chunking)
        res = KnowledgeTransferManager.load_for_training(agent_k2, ckpt_path)
        assert res["mode"] == "net2net_expansion"

        # Verifica preservação exata dos pesos nos 2 slots atômicos
        w = agent_k2.policy_head.linear.weight.data
        b = agent_k2.policy_head.linear.bias.data
        assert torch.allclose(w[:12], torch.full((12, 512), 0.42))
        assert torch.allclose(w[12:24], torch.full((12, 512), 0.42))
        assert torch.allclose(b[:12], torch.full((12,), 0.15))
        assert torch.allclose(b[12:24], torch.full((12,), 0.15))

        # Testa inferência rápida
        obs_dict = {
            "obs": np.zeros(324, dtype=np.float32),
            "delta_obs": np.zeros(324, dtype=np.float32),
            "prev_action": 0,
            "prev_reward": 0.0,
        }
        action = agent_k2.act_fast(obs_dict)
        assert isinstance(action, np.ndarray)
        assert len(action) == 2


def test_teacher_assisted_sil_rollouts():
    """Valida a coleta vetorizada com workers professores (1-Step Lookahead SIL) no ACT k=2."""
    from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv

    raw_env = VectorizedRubiksEnv(num_envs=4, curriculum=True, scramble_depth=6)
    wrapped = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(obs_space=wrapped.observation_space, action_space=wrapped.action_space, trunk_type="lean", action_chunk_size=2)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=wrapped,
        action_chunk_size=2,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
        sil_enabled=True,
        teacher_sil_envs=2,
        teacher_min_depth=2,
        teacher_confidence_threshold=0.99,
        teacher_top_k=3,
    )

    obs, _ = wrapped.reset()
    obs_dict, hx, ep_start, mean_ret = trainer.collect_rollouts(obs, None, True)
    assert isinstance(mean_ret, (float, np.floating))

    metrics = trainer.train_epoch()
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    wrapped.close()






