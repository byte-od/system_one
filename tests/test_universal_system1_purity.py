"""Testes de Contrato de Universalidade Pura do System 1 (Caminho A).

Garante que:
1. O treino PPO nunca invoca simuladores white-box (simulate_step / 2step / 3step) nem injeta
   escalas negativas artificiais (env_scales > 0 estritamente on-policy).
2. Ambientes genéricos Discrete(12) sem inverse_action_map nunca sofrem bloqueio falso de paridade
   (ex: 0 -> 1 não é tratado como inverso de Cubo Mágico).
3. Ambientes contínuos (gym.spaces.Box) funcionam ponta-a-ponta com UniversalS1Agent,
   GradientScarMemory.modulate_scale e gravação diferida de cicatrizes no PPO.
4. A GradientScarMemory mantém estabilidade de norma via EMA Convexo acoplado à surpresa e
   filtra vitórias já dominadas via TD-Surprise universal (R_0 - V(s_0)), sem depender de currículo.
5. O controlador homeostático de exploração possui ciclo universal de respiração/consolidação.
"""

import gymnasium as gym
import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.ensemble import ConfidenceGatedEnsemble, compute_purity_score
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer


class BlackBoxDiscrete12Env(gym.Env):
    """Ambiente caixa-preta Discrete(12) onde a sequência ótima exige 0 -> 1.

    Qualquer tentativa de chamar simulate_step levanta AssertionError.
    Qualquer heurística hardcoded `n_acts == 12` que bloqueie `0 -> 1` impede a vitória!
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(12)
        self._step_idx = 0
        self._stage = 0

    def simulate_step(self, action):
        raise AssertionError("Proibido chamar simulate_step white-box durante o treino Universal S1!")

    def simulate_2step(self, a1, a2):
        raise AssertionError("Proibido chamar simulate_2step white-box durante o treino Universal S1!")

    def simulate_3step(self, a1, a2, a3, return_obs=True):
        raise AssertionError("Proibido chamar simulate_3step white-box durante o treino Universal S1!")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        self._stage = 0
        obs = np.zeros(16, dtype=np.float32)
        obs[0] = 1.0
        return obs, {}

    def step(self, action: int):
        self._step_idx += 1
        a = int(action)
        reward = -0.05
        terminated = False
        if self._stage == 0 and a == 0:
            self._stage = 1
            reward = 1.0
        elif self._stage == 1 and a == 1:
            self._stage = 2
            reward = 10.0
            terminated = True
        truncated = self._step_idx >= 8 and not terminated
        obs = np.zeros(16, dtype=np.float32)
        obs[min(self._stage, 15)] = 1.0
        return obs, reward, terminated, truncated, {"is_success": terminated}


class ContinuousTargetBoxEnv(gym.Env):
    """Ambiente contínuo Box(3,) caixa-preta para validar universalidade contínua ponta-a-ponta."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._pos = np.zeros(3, dtype=np.float32)
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = np.array([0.2, -0.2, 0.1], dtype=np.float32)
        self._steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(8, dtype=np.float32)
        obs[:3] = self._pos
        return obs

    def step(self, action):
        self._steps += 1
        act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:3], -1.0, 1.0)
        self._pos = self._pos * 0.5 + 0.1 * act
        dist = float(np.linalg.norm(self._pos))
        solved = dist < 0.18
        reward = 10.0 if solved else float(1.0 - dist)
        truncated = self._steps >= 6 and not solved
        return self._get_obs(), reward, solved, truncated, {"is_success": solved}


def test_ppo_never_calls_simulate_step_and_keeps_pure_positive_scales():
    raw_env = BlackBoxDiscrete12Env()
    env = UniversalS1Wrapper(raw_env, tri_compass=True)
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        tri_compass=True,
    ).to("cpu")

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=32,
        chunk_length=8,
        chunk_batch_size=2,
        scar_mode="hybrid",
        pure_scars=True,
        sil_enabled=False,
        teacher_sil_envs=4,  # Deve ser forçado para 0 pelo Caminho A
        device=torch.device("cpu"),
    )
    assert trainer.teacher_sil_envs == 0

    obs, _ = env.reset(seed=42)
    obs, hx, ep_start, _ = trainer.collect_rollouts(obs, None, True)
    flat_scales = np.asarray(trainer.buffer.env_scales, dtype=np.float32).reshape(-1)
    assert (flat_scales >= 0.0).all(), "Nenhuma escala negativa artificial deve existir no buffer PPO"
    metrics = trainer.train_epoch()
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])


def test_discrete_12_actions_without_inverse_map_never_blocks_0_to_1():
    raw_env = BlackBoxDiscrete12Env()
    env = UniversalS1Wrapper(raw_env)
    assert env.inverse_action_map is None

    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
    ).to("cpu")
    agent.eval()

    # Força a política a preferir fortemente a ação 1 após ter executado a ação 0
    with torch.no_grad():
        agent.policy_head.linear.weight.zero_()
        agent.policy_head.linear.bias.fill_(-10.0)
        agent.policy_head.linear.bias[1] = 10.0

    obs, _ = env.reset(seed=7)
    obs, _, _, _, _ = env.step(0)  # Ação 0 executada
    agent._action_history.append(0)

    # Sem inverse_action_map, act_fast NÃO pode tratar 1 como inverso de 0!
    chosen = agent.act_fast(obs, deterministic=True, value_rerank=False, env=env)
    assert chosen == 1

    # E compute_purity_score sem inverse_action_map não penaliza [0, 1]
    assert compute_purity_score([0, 1], inverse_action_map=None) == 1.0
    assert compute_purity_score([0, 1], inverse_action_map={0: 1, 1: 0}) < 1.0

    # E ConfidenceGatedEnsemble também respeita ausência de inverse_action_map
    h_dummy = torch.zeros(1, 1, agent.policy_head.linear.in_features)
    ens_act, _ = ConfidenceGatedEnsemble.resolve_uncertain_action(agent, h_dummy, is_discrete=True)
    assert ens_act == 1


def test_continuous_box_action_space_ppo_and_scar_resonance():
    torch.manual_seed(123)
    raw_env = ContinuousTargetBoxEnv()
    env = UniversalS1Wrapper(raw_env, tri_compass=True)
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        tri_compass=True,
    ).to("cpu")

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=32,
        chunk_length=8,
        chunk_batch_size=2,
        scar_mode="hybrid",
        pure_scars=True,
        sil_enabled=False,
        device=torch.device("cpu"),
    )
    assert agent.scar_memory is not None

    obs, _ = env.reset(seed=99)
    obs, hx, ep_start, _ = trainer.collect_rollouts(obs, None, True)
    assert int(agent.scar_memory.scars_count.item()) >= 1, "Vitórias contínuas devem gravar cicatrizes universais"

    # Verifica simetria contínua de modulate_scale na GradientScarMemory após gravação de cicatriz
    base_scale = torch.ones(2, 4, 3) * 0.5
    pos_res = torch.ones(2, 4) * 2.0
    neg_res = torch.ones(2, 4) * -2.0
    s_pos = agent.scar_memory.modulate_scale(base_scale, pos_res)
    s_neg = agent.scar_memory.modulate_scale(base_scale, neg_res)
    assert (s_pos < base_scale).all(), "Ressonância positiva deve focar (reduzir sigma) políticas contínuas"
    assert (s_neg > base_scale).all(), "Dissonância negativa deve expandir sigma para escapar de estados ruins"

    metrics = trainer.train_epoch()
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])


def test_scar_convex_ema_norm_stability_and_td_surprise_gating():
    torch.manual_seed(2026)
    raw_env = BlackBoxDiscrete12Env()
    env = UniversalS1Wrapper(raw_env, tri_compass=True)
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        tri_compass=True,
    ).to("cpu")

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
        scar_mode="hybrid",
        pure_scars=True,
        sil_enabled=False,
        device=torch.device("cpu"),
    )
    scar_mem = agent.scar_memory
    assert scar_mem is not None

    obs_dict, _ = env.reset(seed=5)
    novel_win_traj = [
        {
            "obs": obs_dict["obs"].copy(),
            "delta_obs": obs_dict["delta_obs"].copy(),
            "prev_action": 0,
            "prev_reward": 0.0,
            "action": 0,
            "reward": 1.0,
            "value": 0.2,  # Crítico subestima fortemente R_0 (~10.9) -> alta TD-Surprise positiva!
            "log_prob": -1.2,
        },
        {
            "obs": obs_dict["obs"].copy(),
            "delta_obs": obs_dict["delta_obs"].copy(),
            "prev_action": 0,
            "prev_reward": 1.0,
            "action": 1,
            "reward": 10.0,
            "value": 0.5,
            "log_prob": -1.1,
        },
    ]

    # 1. Vitória inédita (alta TD-Surprise positiva) grava cicatriz
    trainer._process_completed_trajectory(novel_win_traj, is_win=True, cur_depth=1)
    assert int(scar_mem.scars_count.item()) == 1
    initial_norm = scar_mem.scar_norm
    assert initial_norm > 0.1

    # 2. Vitória já dominada (V(s_0) >= R_0 -> TD-Surprise <= 0) NÃO sobrescreve nem dilui W_scar!
    mastered_win_traj = [dict(step) for step in novel_win_traj]
    mastered_win_traj[0]["value"] = 12.0  # Crítico já prevê retorno total
    trainer._process_completed_trajectory(mastered_win_traj, is_win=True, cur_depth=1)
    assert int(scar_mem.scars_count.item()) == 1
    assert abs(scar_mem.scar_norm - initial_norm) < 1e-6

    # 3. Estabilidade do EMA Convexo sob 100 atualizações de baixa surpresa: W_scar não colapsa a zero!
    z = torch.randn(4, scar_mem.num_slots, scar_mem.slot_dim, requires_grad=True)
    h = torch.randn(4, scar_mem.num_slots, scar_mem.slot_dim, requires_grad=True)
    loss = (z * h).sum()
    loss.backward()
    for _ in range(100):
        scar_mem.record_scar(z.detach(), h.grad.detach(), surprise=0.05)
    assert scar_mem.scar_norm > 0.5 * initial_norm, "EMA Convexo não deve erodir a norma de W_scar para zero"


def test_universal_homeostatic_exploration_breathing_cooldown():
    raw_env = BlackBoxDiscrete12Env()
    env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
    ).to("cpu")

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
        exploration_scale="auto",
        device=torch.device("cpu"),
    )
    trainer._best_rollout_return = 10.0
    trainer._adaptive_scale = 1.0

    # Simula 16 rollouts estagnados no teto de exploração (1.0) sem bater recorde
    for _ in range(16):
        trainer._update_adaptive_exploration(2.0, [])

    # Deve acionar o ciclo universal de consolidação (~0.35) em vez de ficar preso em 1.0 eternamente
    assert abs(trainer._adaptive_scale - 0.35) < 1e-6


def test_sil_tournament_never_prefers_short_crashes_over_long_survival():
    """Garante universalidade em tarefas de sobrevivência/locomoção (ex: CartPole):
    trajetórias não-terminais-de-meta (is_full_win=False) devem ser ranqueadas por retorno R_0,
    nunca por 1/length (que selecionaria quedas precoces de 9 passos acima de sobrevivências de 500 passos).
    """
    from system1_engine.training.buffer import FrontierTransition, SuccessReplayBuffer

    buf = SuccessReplayBuffer(max_trajectories=32)
    obs_crash = np.zeros((9, 4), dtype=np.float32)
    obs_crash[0, 0] = -0.1
    short_crash = FrontierTransition(
        obs=obs_crash,
        delta_obs=np.zeros((9, 4), dtype=np.float32),
        prev_actions=np.zeros(9, dtype=np.int64),
        prev_rewards=np.ones(9, dtype=np.float32),
        actions=np.zeros(9, dtype=np.int64),
        returns=np.linspace(9.0, 1.0, 9, dtype=np.float32),
        depth=0,
        length=9,
        td_error=5.0,
        log_probs=np.zeros(9, dtype=np.float32),
        is_full_win=False,
    )
    obs_surv = np.zeros((400, 4), dtype=np.float32)
    obs_surv[0, 0] = 0.1
    long_survival = FrontierTransition(
        obs=obs_surv,
        delta_obs=np.zeros((400, 4), dtype=np.float32),
        prev_actions=np.zeros(400, dtype=np.int64),
        prev_rewards=np.ones(400, dtype=np.float32),
        actions=np.zeros(400, dtype=np.int64),
        returns=np.linspace(98.0, 1.0, 400, dtype=np.float32),
        depth=0,
        length=400,
        td_error=50.0,
        log_probs=np.zeros(400, dtype=np.float32),
        is_full_win=False,
    )
    buf.add(short_crash)
    buf.add(long_survival)
    assert len(buf._depth_buffers[0]) == 2, "Ambientes sem currículo (depth=0) não devem podar episódios longos!"

    np.random.seed(0)
    wins_picked = sum(
        1
        for _ in range(50)
        if buf._pick_trajectory(buf._depth_buffers[0], tournament_size=5).length == 400
    )
    assert wins_picked >= 35, f"Esperava preferência clara pela sobrevivência longa (R0=98 vs 9), obteve {wins_picked}/50"


def test_external_scout_batch_modulation_and_reset_scars():
    """Valida a precisão de máquina da modulação Fast-Weights por elemento do batch e a limpeza integral reset_scars()."""
    from system1_engine.core.scar_memory import GradientScarMemory

    sm = GradientScarMemory(slot_dim=16, num_slots=4, mode="hybrid", scale=0.15, resonance_beta=0.20)
    # 1. Simula gravação de cicatriz inicial
    z = torch.randn(1, 4, 16)
    g = torch.randn(1, 4, 16)
    sm.record_scar(z, g, surprise=1.5)
    assert int(sm.scars_count.item()) == 1
    assert sm.scar_norm > 0.0

    # 2. Testa invariância estrita de co-batching: fatias principais [0:4] devem ser bit-for-bit idênticas
    E_main = 4
    E_scout = 2
    slots_main = torch.randn(E_main, 1, 4, 16)
    slots_scout = torch.randn(E_scout, 1, 4, 16)
    slots_comb = torch.cat([slots_main, slots_scout], dim=0)

    scales_comb = torch.tensor([0.15, 0.15, 0.15, 0.15, 0.45, 0.50], dtype=torch.float32)
    betas_comb = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.60, 0.65], dtype=torch.float32)

    # Executa sem scope (escalares padrões)
    res_main_standalone, _ = sm.modulate_slots(slots_main)

    # Executa co-batched com scope vetorial
    with sm.batch_modulation_scope(scales_comb, betas_comb):
        res_comb, _ = sm.modulate_slots(slots_comb)

    diff_slots = torch.max(torch.abs(res_comb[:E_main] - res_main_standalone)).item()
    assert diff_slots < 1e-6, f"Fatia principal foi alterada pelo co-batching: diff={diff_slots}"

    # 3. Mesma invariância para logits discretos e scale contínuo
    logits_main = torch.randn(E_main, 1, 12)
    logits_scout = torch.randn(E_scout, 1, 12)
    logits_comb = torch.cat([logits_main, logits_scout], dim=0)
    res_vec_comb = torch.ones(E_main + E_scout, 1, 4) * 0.5
    res_vec_main = res_vec_comb[:E_main]

    mod_logits_standalone = sm.modulate_logits(logits_main, res_vec_main)
    with sm.batch_modulation_scope(scales_comb, betas_comb):
        mod_logits_comb = sm.modulate_logits(logits_comb, res_vec_comb)

    diff_logits = torch.max(torch.abs(mod_logits_comb[:E_main] - mod_logits_standalone)).item()
    assert diff_logits < 1e-6, f"Logits da fatia principal alterados pelo co-batching: diff={diff_logits}"

    # 4. Testa reset_scars() para garantia de Tabula Rasa 100% limpa
    sm.reset_scars()
    assert int(sm.scars_count.item()) == 0
    assert float(sm.total_surprise_accum.item()) == 0.0
    assert sm.scar_norm == 0.0
    assert torch.all(sm.W_scar == 0.0)
    assert torch.all(sm.g_target == 0.0)
    assert torch.all(sm.z_target == 0.0)


def test_external_scouts_universal_isolation_and_zero_buffer_pollution():
    """Garante que a frota de batedores roda fora dos N ambientes PPO, sem poluir RolloutBuffer ou grade X/N."""
    raw_env = BlackBoxDiscrete12Env()
    env = UniversalS1Wrapper(raw_env, tri_compass=True)
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        tri_compass=True,
    ).to("cpu")

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
        scar_mode="hybrid",
        pure_scars=True,
        sil_enabled=False,
        external_scout_envs=3,
        device=torch.device("cpu"),
    )
    try:
        assert trainer.external_scout_envs == 3
        assert trainer.scout_env is not None
        # O buffer PPO deve conter estritamente rollout_steps = 16 transições
        assert trainer.buffer.buffer_size == 16

        obs, _ = env.reset(seed=101)
        obs, hx, ep_start, _ = trainer.collect_rollouts(obs, None, True)

        # Buffer contém estritamente 16 passos PPO
        flat_scales = np.asarray(trainer.buffer.env_scales, dtype=np.float32).reshape(-1)
        assert len(flat_scales) == 16
        assert (flat_scales >= 0.0).all()

        metrics = trainer.train_epoch()
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
    finally:
        trainer.close_scout_env()


def test_external_scouts_box_continuous_environment():
    """Valida que batedores externos funcionam com ambientes contínuos Box sem erro de tipo."""
    raw_env = ContinuousTargetBoxEnv()
    env = UniversalS1Wrapper(raw_env, tri_compass=True)
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type="lru",
        tri_compass=True,
    ).to("cpu")

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
        scar_mode="hybrid",
        pure_scars=True,
        sil_enabled=False,
        external_scout_envs=2,
        device=torch.device("cpu"),
    )
    try:
        assert trainer.external_scout_envs == 2
        obs, _ = env.reset(seed=202)
        obs, hx, ep_start, _ = trainer.collect_rollouts(obs, None, True)
        assert trainer.buffer.buffer_size == 16
        metrics = trainer.train_epoch()
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
    finally:
        trainer.close_scout_env()

