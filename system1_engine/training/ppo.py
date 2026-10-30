from collections import deque
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry.tracker import LiveStatsTracker
from system1_engine.training.buffer import (
    FrontierTransition,
    FrontierTransitionBuffer,
    RecurrentRolloutBuffer,
    SuccessReplayBuffer,
    SuccessfulTrajectory,
)


class RecurrentPPOTrainer:
    """Pure PyTorch Recurrent PPO Trainer with GRU hidden state preservation and BPTT."""

    def __init__(
        self,
        agent: UniversalS1Agent,
        env: UniversalS1Wrapper,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.05,
        entropy_coef: float = 0.015,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        rollout_steps: int = 2048,
        chunk_length: int = 32,
        chunk_batch_size: int = 16,
        device: Union[str, torch.device] = "cpu",
        tracker: Optional[LiveStatsTracker] = None,
        exploration_scale: Union[float, str] = 1.0,
        hindsight_replay: bool = True,
        hindsight_bonus: float = 2.0,
        hindsight_min_gain: float = 0.02,
        sil_enabled: bool = True,
        sil_coef: float = 0.08,
        sil_batch_size: int = 8,
        sil_max_trajectories: Optional[int] = None,
        sil_max_per_depth: Optional[int] = None,
        min_capacity_per_depth: int = 32,
        sil_min_samples: int = 8,
        sil_tournament_size: int = 5,
        sil_dpo_enabled: bool = False,
        dpo_coef: float = 0.0,
        dpo_beta: float = 0.10,
        dpo_batch_size: int = 8,
        progressive_buffer_capacity: bool = True,
        max_failures_per_depth: int = 512,
        min_failures_per_depth: int = 16,
        sterility_memory_enabled: bool = False,
        sterility_capacity: int = 1024,
        sterility_sim_threshold: float = 0.85,
        sterility_penalty_scale: float = 2.0,
        sterility_step_increment: float = 1.0,
        action_chunk_size: Optional[int] = None,
        adaptive_chunk: bool = False,
        adaptive_conf_threshold: float = 0.65,
        adaptive_margin_threshold: float = 0.15,
        dynamic_frontier: bool = True,
        proximal_decay: float = 0.8,
        teacher_sil_envs: int = 0,
        teacher_min_depth: int = 5,
        teacher_confidence_threshold: float = 0.30,
        teacher_top_k: int = 4,
        entropy_floor_enabled: bool = True,
        entropy_floor: float = 0.18,
        entropy_floor_coef: float = 0.50,
        adaptive_entropy_coef: bool = True,
        min_entropy_coef: float = 0.003,
        max_entropy_coef: float = 0.080,
        pcgrad_enabled: bool = True,
        gradvac_max_alpha: float = 0.25,
        gradvac_ema_beta: float = 0.99,
    ) -> None:
        self.agent = agent.to(device)
        self.env = env
        self.dynamic_frontier = bool(dynamic_frontier)
        self.proximal_decay = float(proximal_decay)
        self.teacher_sil_envs = max(0, int(teacher_sil_envs))
        self.teacher_min_depth = max(1, int(teacher_min_depth))
        self.teacher_confidence_threshold = float(teacher_confidence_threshold)
        self.teacher_top_k = max(2, int(teacher_top_k))
        self.action_chunk_size = max(1, int(action_chunk_size or getattr(agent, "action_chunk_size", 1)))
        self.adaptive_chunk = bool(adaptive_chunk)
        self.adaptive_conf_threshold = float(adaptive_conf_threshold)
        self.adaptive_margin_threshold = float(adaptive_margin_threshold)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.rollout_steps = rollout_steps
        self.chunk_length = chunk_length
        self.chunk_batch_size = chunk_batch_size
        self.device = torch.device(device)
        self.tracker = tracker
        self.hindsight_replay = bool(hindsight_replay)
        self.hindsight_bonus = float(hindsight_bonus)
        self.hindsight_min_gain = float(hindsight_min_gain)

        # Self-Imitation Learning (SIL) & Contrastive Online DPO-SIL (c-SIL)
        self.sil_enabled = bool(sil_enabled)
        self.sil_coef = float(sil_coef)
        self.sil_batch_size = max(2, int(sil_batch_size))
        self.sil_min_samples = max(2, int(sil_min_samples))
        self.sil_max_trajectories: Optional[int] = (
            int(sil_max_trajectories)
            if (sil_max_trajectories is not None and int(sil_max_trajectories) > 0)
            else None
        )
        self.sil_max_per_depth: Optional[int] = (
            int(sil_max_per_depth)
            if (sil_max_per_depth is not None and int(sil_max_per_depth) > 0)
            else None
        )
        self.min_capacity_per_depth = max(1, int(min_capacity_per_depth))
        self.sil_dpo_enabled = bool(sil_dpo_enabled)
        self.dpo_coef = float(dpo_coef)
        self.dpo_beta = float(dpo_beta)
        self.dpo_batch_size = max(1, int(dpo_batch_size))
        self.sil_tournament_size = max(1, int(sil_tournament_size))
        self.progressive_buffer_capacity = bool(progressive_buffer_capacity)
        self.max_failures_per_depth = max(1, int(max_failures_per_depth))
        self.min_failures_per_depth = max(1, int(min_failures_per_depth))

        # Curvature-Gated Actor Surgery (Opção 3: Unificação Analítica M1 [PCGrad] + M3 [GradVac] + Crítico 100% PPO Puro)
        self.pcgrad_enabled = bool(pcgrad_enabled)
        self.gradvac_max_alpha = float(np.clip(gradvac_max_alpha, 0.0, 0.50))
        self.gradvac_ema_beta = float(np.clip(gradvac_ema_beta, 0.50, 0.9999))
        self._gradvac_ema: Dict[str, float] = {
            "FrontEnd": 0.05,
            "Trunk": 0.05,
            "PolicyHead": 0.05,
        }
        self._recent_frontier_wins: deque = deque(maxlen=128)

        # Dynamic Entropy Floor & Adaptive Entropy Controller
        self.entropy_floor_enabled = bool(entropy_floor_enabled)
        self.entropy_floor = float(entropy_floor)
        self.entropy_floor_coef = float(entropy_floor_coef)
        self.adaptive_entropy_coef = bool(adaptive_entropy_coef)
        self.base_entropy_coef = float(entropy_coef)
        self.min_entropy_coef = float(min_entropy_coef)
        self.max_entropy_coef = float(max_entropy_coef)
        if self.tracker is not None:
            self.tracker.metrics.entropy_floor = self.entropy_floor

        # Suporte a exploração automática / homeostática no treino PPO (com piso seguro anti-colapso)
        self._min_adaptive_scale = 0.25 if self.action_chunk_size > 1 else 0.08
        self._max_adaptive_scale = 1.00
        self._last_promotions: int = 0
        if isinstance(exploration_scale, str) and exploration_scale.lower() in ("auto", "adaptive", "homeostatic"):
            self.is_auto_exploration = True
            self.exploration_scale = 0.80
            self._adaptive_scale = 0.80
            self._reward_ema: Optional[float] = None
            self._stagnation_count = 0
            self._recent_actions: deque[int] = deque(maxlen=6)
        else:
            self.is_auto_exploration = False
            self.exploration_scale = float(np.clip(float(exploration_scale), self._min_adaptive_scale, 1.0))
            self._adaptive_scale = self.exploration_scale
            self._reward_ema = None
            self._stagnation_count = 0
            self._recent_actions = deque(maxlen=6)

        # Optimize only parameters that require grad (respects frozen trunk)
        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "No trainable parameters found!"
        opt_kwargs = {"lr": learning_rate, "eps": 1e-5}
        if self.device.type in ("mps", "cuda"):
            try:
                self.optimizer = optim.Adam(trainable_params, **opt_kwargs, fused=True)
            except Exception:
                self.optimizer = optim.Adam(trainable_params, **opt_kwargs)
        else:
            self.optimizer = optim.Adam(trainable_params, **opt_kwargs)
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.best_mean_return: float = -float("inf")

        # Rollout buffer
        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        obs_shape = env.observation_space["obs"].shape
        if self.action_chunk_size > 1:
            act_shape = (self.action_chunk_size,) if is_discrete else (self.action_chunk_size, *env.action_space.shape)
        else:
            act_shape = () if is_discrete else env.action_space.shape

        self.is_vectorized = getattr(env, "is_vectorized", False)
        self.num_envs = getattr(env, "num_envs", 1)

        if self.is_vectorized:
            self.steps_per_env = max(1, rollout_steps // (self.num_envs * chunk_length)) * chunk_length
            self.rollout_steps = self.steps_per_env * self.num_envs
        else:
            self.steps_per_env = self.rollout_steps

        self.trunk_hidden_dim = getattr(agent.trunk, "hidden_dim", 256)
        self.buffer = RecurrentRolloutBuffer(
            buffer_size=self.rollout_steps,
            chunk_length=chunk_length,
            is_visual=env.is_visual,
            is_discrete=is_discrete,
            obs_shape=obs_shape,
            action_shape=act_shape,
            device=self.device,
            hidden_dim=self.trunk_hidden_dim,
        )

        # Buffer estratificado por profundidade de trajetórias e transições de fronteira (Frontier Transition Buffer)
        self.sil_buffer = FrontierTransitionBuffer(
            max_trajectories=self.sil_max_trajectories,
            max_per_depth=self.sil_max_per_depth,
            min_capacity_per_depth=self.min_capacity_per_depth,
            is_visual=env.is_visual,
            is_discrete=is_discrete,
            obs_shape=obs_shape,
            action_shape=act_shape,
            device=self.device,
            hidden_dim=self.trunk_hidden_dim,
            frontier_slicing=True,
            frontier_ratio=0.75,
            progressive_capacity=self.progressive_buffer_capacity,
            max_failures_per_depth=self.max_failures_per_depth,
            min_failures_per_depth=self.min_failures_per_depth,
            dynamic_frontier=self.dynamic_frontier,
            proximal_decay=self.proximal_decay,
        )

        # Memória Latente Estéril Auto-Corretiva (Substituída nativamente por c-SIL)
        self.sterility_memory = None
        self.sterility_memory_enabled = False

        self._active_vec_trajectories: List[deque] = [deque(maxlen=200) for _ in range(self.num_envs)]
        self._active_single_trajectory: deque = deque(maxlen=200)

        # Buffers de tensores pré-alocados no dispositivo para eliminar overhead de heap e alocações repetidas
        self._buf_obs = torch.zeros((1, 1, *obs_shape), dtype=torch.float32, device=self.device)
        if is_discrete:
            self._buf_act = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        else:
            self._buf_act = torch.zeros((1, 1, *act_shape), dtype=torch.float32, device=self.device)
        self._buf_rew = torch.zeros((1, 1, 1), dtype=torch.float32, device=self.device)
        if not env.is_visual:
            if hasattr(env, "observation_space") and isinstance(env.observation_space, gym.spaces.Dict) and "delta_obs" in env.observation_space.spaces:
                delta_shape = env.observation_space.spaces["delta_obs"].shape
            else:
                delta_shape = obs_shape
            self._buf_delta: Optional[torch.Tensor] = torch.zeros((1, 1, *delta_shape), dtype=torch.float32, device=self.device)
        else:
            self._buf_delta = None

        self._rollout_input_dict: Dict[str, Any] = {
            "obs": self._buf_obs,
            "prev_action": self._buf_act,
            "prev_reward": self._buf_rew,
        }
        if self._buf_delta is not None:
            self._rollout_input_dict["delta_obs"] = self._buf_delta

        if self.is_vectorized:
            self._vec_buf_obs = torch.zeros((self.num_envs, 1, *obs_shape), dtype=torch.float32, device=self.device)
            if is_discrete:
                self._vec_buf_act = torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device)
            else:
                self._vec_buf_act = torch.zeros((self.num_envs, 1, *act_shape), dtype=torch.float32, device=self.device)
            self._vec_buf_rew = torch.zeros((self.num_envs, 1, 1), dtype=torch.float32, device=self.device)
            if not env.is_visual:
                if hasattr(env, "observation_space") and isinstance(env.observation_space, gym.spaces.Dict) and "delta_obs" in env.observation_space.spaces:
                    delta_shape = env.observation_space.spaces["delta_obs"].shape
                else:
                    delta_shape = obs_shape
                self._vec_buf_delta: Optional[torch.Tensor] = torch.zeros((self.num_envs, 1, *delta_shape), dtype=torch.float32, device=self.device)
            else:
                self._vec_buf_delta = None

            self._vec_input_dict: Dict[str, Any] = {
                "obs": self._vec_buf_obs,
                "prev_action": self._vec_buf_act,
                "prev_reward": self._vec_buf_rew,
            }
            if self._vec_buf_delta is not None:
                self._vec_input_dict["delta_obs"] = self._vec_buf_delta

            self._vec_recent_actions: List[deque[int]] = [deque(maxlen=6) for _ in range(self.num_envs)]
            if self.num_envs >= 4:
                # Cobertura Completa de Exploração de 0 à 1 em Ambientes Simultâneos (Quality-Diverse Spectrum):
                # - Estrato Elite (0 .. E//4 - 1): determinístico / ultra-greedy (0.02 a 0.20) para trajetórias ótimas
                # - Estrato Balanced (E//4 .. 3*E//4 - 1): faixa central adaptativa modulada por curr_scale
                # - Estrato Exploratory (3*E//4 .. E - 1): máxima dispersão térmica (0.75 a 1.00) para escapar de mínimos locais
                n_elite = max(1, self.num_envs // 4)
                n_exploratory = max(1, self.num_envs // 4)
                n_balanced = self.num_envs - n_elite - n_exploratory

                elite_mults = np.linspace(0.05, 0.40, n_elite, dtype=np.float32)
                balanced_mults = np.linspace(0.60, 1.20, n_balanced, dtype=np.float32)
                exploratory_mults = np.linspace(1.30, 2.00, n_exploratory, dtype=np.float32)
                multipliers = np.concatenate([elite_mults, balanced_mults, exploratory_mults])
                self._env_scale_multipliers = torch.from_numpy(multipliers).to(self.device).view(self.num_envs, 1, 1)

                # Âncoras fixas para garantir cobertura absoluta de 0 à 1 em qualquer regime térmico
                elite_fixed = np.linspace(0.02, 0.15, n_elite, dtype=np.float32)
                exploratory_fixed = np.linspace(0.80, 1.00, n_exploratory, dtype=np.float32)
                balanced_fixed = np.full(n_balanced, 0.0, dtype=np.float32)
                self._env_fixed_anchors = torch.from_numpy(
                    np.concatenate([elite_fixed, balanced_fixed, exploratory_fixed])
                ).to(self.device).view(self.num_envs, 1, 1)
            else:
                self._env_scale_multipliers = torch.ones((self.num_envs, 1, 1), dtype=torch.float32, device=self.device)
                self._env_fixed_anchors = None

        # Persistent episode statistics across rollout batches
        self.episode_returns: deque[float] = deque(maxlen=20)
        self.curr_ep_return: float = 0.0
        self.total_steps: int = 0
        self.best_state_depth: int = self._get_current_curriculum_depth()
        self.best_depth_mean_return: float = -float("inf")
        self._rollout_tracked_depth: int = self.best_state_depth

    @property
    def effective_gamma(self) -> float:
        """Effective discount factor across macro-actions in SMDP (gamma^k)."""
        if self.action_chunk_size > 1:
            return float(self.gamma ** self.action_chunk_size)
        return float(self.gamma)

    def _get_current_curriculum_depth(self) -> int:
        """Retorna a profundidade ou nível curricular atual do ambiente (0 se não aplicável)."""
        raw_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(raw_env, "current_depth") and raw_env.current_depth is not None:
            return int(raw_env.current_depth)
        if hasattr(self.env, "current_depth") and self.env.current_depth is not None:
            return int(self.env.current_depth)
        if hasattr(raw_env, "current_level") and raw_env.current_level is not None:
            return int(raw_env.current_level)
        return 0

    def warmup_sil_buffer(
        self,
        up_to_depth: int = 1,
        episodes_per_depth: int = 32,
        calibrate_curriculum: bool = True,
    ) -> int:
        """Pré-aquece o buffer SIL estratificado com trajetórias âncoras (níveis 1..up_to_depth) diretamente em self.env.

        Totalmente agnóstico de domínio (suporta ambientes vetorizados ou singulares, discretos ou contínuos).
        Se calibrate_curriculum=True e o agente demonstrar domínio consolidado (>=78% no nível 1),
        calibra automaticamente a fronteira curricular inicial para o primeiro nível com sucesso <78%.
        """
        if not self.sil_enabled or up_to_depth < 1:
            return 0

        raw_env = getattr(self.env, "unwrapped", self.env)
        has_curriculum_setter = hasattr(raw_env, "set_curriculum_depth")
        orig_depth = self._get_current_curriculum_depth()
        orig_mixed_rate = getattr(raw_env, "mixed_replay_rate", None)
        inv_map: Optional[Dict[int, int]] = getattr(raw_env, "inverse_action_map", None)

        max_d_target = max(1, int(up_to_depth)) if has_curriculum_setter else 1
        if getattr(raw_env, "max_depth", None) is not None:
            max_d_target = min(max_d_target, int(raw_env.max_depth))

        if orig_mixed_rate is not None:
            try:
                raw_env.mixed_replay_rate = 0.0
            except Exception:
                pass

        self.agent.eval()
        self.last_warmup_solve_rates: Dict[int, float] = {}
        len_before = len(self.sil_buffer)
        added_total = 0

        try:
            with torch.inference_mode():
                for d in range(1, max_d_target + 1):
                    if has_curriculum_setter:
                        raw_env.set_curriculum_depth(d)
                    max_ep_steps = min(getattr(raw_env, "max_steps", 40), max(6, d + 2))

                    if self.is_vectorized:
                        E = self.num_envs
                        active_quota = min(E, max(1, int(episodes_per_depth)))
                        obs_dict, _ = self.env.reset(seed=1000 + d * 100)
                        hx = torch.zeros((1, E, self.trunk_hidden_dim), dtype=torch.float32, device=self.device)

                        ep_active = np.zeros(E, dtype=bool)
                        ep_active[:active_quota] = True
                        ep_solved = np.zeros(active_quota, dtype=bool)
                        trajs: List[List[Dict[str, Any]]] = [[] for _ in range(active_quota)]
                        recent_acts: List[deque[int]] = [deque(maxlen=6) for _ in range(active_quota)]

                        for _step_idx in range(max_ep_steps):
                            if not np.any(ep_active):
                                break

                            input_dict = self._prepare_vec_rollout_input(obs_dict)
                            dist, _, next_hx = self.agent.forward(input_dict, hx=hx, dones=None)

                            if self.agent.is_discrete:
                                if self.action_chunk_size > 1:
                                    logits_sq = dist.logits.squeeze(1)
                                    probs = torch.softmax(logits_sq, dim=-1)
                                    greedy_acts = torch.argmax(probs, dim=-1)
                                    env_action = greedy_acts.cpu().numpy()
                                else:
                                    logits_sq = dist.logits[:, 0, :]
                                    probs = torch.softmax(logits_sq, dim=-1)
                                    sampled_probs = torch.softmax(logits_sq / 1.04, dim=-1)
                                    rand_draws = torch.rand(E, device=self.device)
                                    greedy_acts = torch.argmax(probs, dim=-1)
                                    stoch_acts = torch.multinomial(sampled_probs, num_samples=1).squeeze(-1)
                                    raw_acts = torch.where(rand_draws < 0.08, stoch_acts, greedy_acts)
                                    if E > 1:
                                        raw_acts[0] = greedy_acts[0]

                                    env_action = raw_acts.cpu().numpy()
                                    for e in range(active_quota):
                                        if not ep_active[e]:
                                            continue
                                        conf_e = float(probs[e, greedy_acts[e]].item())
                                        avoid_e = set()
                                        r_acts = recent_acts[e]
                                        if inv_map is not None and len(r_acts) >= 1 and conf_e < 0.96:
                                            inv_last = inv_map.get(r_acts[-1])
                                            if inv_last is not None:
                                                avoid_e.add(inv_last)
                                        if len(r_acts) >= 3:
                                            if r_acts[-1] == r_acts[-3]:
                                                avoid_e.add(r_acts[-2])
                                            elif r_acts[-1] == r_acts[-2] == r_acts[-3]:
                                                avoid_e.add(r_acts[-1])
                                            if inv_map is not None and conf_e < 0.95:
                                                inv_a1 = inv_map.get(r_acts[-3])
                                                inv_a2 = inv_map.get(r_acts[-2])
                                                if inv_a1 is not None and inv_a2 is not None and r_acts[-1] == inv_a1:
                                                    avoid_e.add(inv_a2)
                                        if int(env_action[e]) in avoid_e:
                                            top_idxs = torch.argsort(probs[e], descending=True)
                                            for alt_idx in top_idxs:
                                                alt_a = int(alt_idx.item())
                                                if alt_a not in avoid_e:
                                                    env_action[e] = alt_a
                                                    break
                            else:
                                env_action = dist.mean.squeeze(1).cpu().numpy()

                            next_obs_dict, rewards, terminateds, truncateds, infos = self.env.step(env_action)

                            for e in range(active_quota):
                                if not ep_active[e]:
                                    continue
                                rew_flt = float(rewards[e])
                                delta_v = (
                                    obs_dict["delta_obs"][e].copy()
                                    if not self.env.is_visual and obs_dict.get("delta_obs") is not None
                                    else None
                                )
                                if self.action_chunk_size > 1:
                                    prev_act_v = int(obs_dict["prev_action"][e]) if np.isscalar(obs_dict["prev_action"][e]) else obs_dict["prev_action"][e].copy()
                                    act_v = env_action[e].copy()
                                else:
                                    prev_act_v = int(obs_dict["prev_action"][e]) if self.agent.is_discrete else obs_dict["prev_action"][e].copy()
                                    act_v = int(env_action[e]) if self.agent.is_discrete else env_action[e].copy()
                                trajs[e].append(
                                    {
                                        "obs": obs_dict["obs"][e].copy(),
                                        "delta_obs": delta_v,
                                        "prev_action": prev_act_v,
                                        "prev_reward": float(obs_dict["prev_reward"][e]),
                                        "action": act_v,
                                        "reward": rew_flt,
                                    }
                                )
                                if self.agent.is_discrete:
                                    recent_acts[e].append(int(act_v[-1] if hasattr(act_v, '__len__') else act_v))
                                if terminateds[e] or truncateds[e]:
                                    ep_active[e] = False
                                    info_e = infos[e] if isinstance(infos, (list, tuple)) and e < len(infos) else {}
                                    is_win = (
                                        bool(info_e.get("is_solved", info_e.get("is_success", False)))
                                        or (bool(terminateds[e]) and rew_flt >= 5.0)
                                    )
                                    if is_win and len(trajs[e]) > 0:
                                        ep_solved[e] = True
                                        t_steps = trajs[e]
                                        T_traj = len(t_steps)
                                        traj_obs = np.array([s["obs"] for s in t_steps], dtype=np.float32)
                                        has_delta = any(s["delta_obs"] is not None for s in t_steps)
                                        traj_delta = np.array([s["delta_obs"] for s in t_steps], dtype=np.float32) if has_delta else None
                                        traj_prev_act = np.array([s["prev_action"] for s in t_steps])
                                        traj_prev_rew = np.array([s["prev_reward"] for s in t_steps], dtype=np.float32)
                                        traj_act = np.array([s["action"] for s in t_steps])
                                        traj_returns = np.zeros(T_traj, dtype=np.float32)
                                        running_r = 0.0
                                        for t_idx in reversed(range(T_traj)):
                                            running_r = t_steps[t_idx]["reward"] + self.gamma * running_r
                                            traj_returns[t_idx] = running_r

                                        self.sil_buffer.add(
                                            FrontierTransition(
                                                obs=traj_obs,
                                                delta_obs=traj_delta,
                                                prev_actions=traj_prev_act,
                                                prev_rewards=traj_prev_rew,
                                                actions=traj_act,
                                                returns=traj_returns,
                                                depth=d,
                                                length=T_traj,
                                                td_error=float(traj_returns[0]) if len(traj_returns) > 0 else 0.0,
                                            )
                                        )
                                        added_total += 1

                            obs_dict = next_obs_dict
                            hx = next_hx

                        self.last_warmup_solve_rates[d] = float(np.mean(ep_solved))
                    else:
                        n_eps = max(1, int(episodes_per_depth))
                        ep_solved_single = np.zeros(n_eps, dtype=bool)
                        for ep_i in range(n_eps):
                            obs_dict, _ = self.env.reset(seed=1000 + d * 100 + ep_i)
                            self.agent.reset_memory()
                            t_steps: List[Dict[str, Any]] = []
                            for _step_idx in range(max_ep_steps):
                                act = self.agent.act_fast(
                                    obs_dict,
                                    deterministic=(ep_i == 0),
                                    calibration=0.0 if ep_i == 0 else 0.08,
                                    quality_filter=True,
                                )
                                next_obs_dict, rew, term, trunc, info = self.env.step(act)
                                delta_v = (
                                    obs_dict["delta_obs"].copy()
                                    if not self.env.is_visual and obs_dict.get("delta_obs") is not None
                                    else None
                                )
                                t_steps.append(
                                    {
                                        "obs": obs_dict["obs"].copy(),
                                        "delta_obs": delta_v,
                                        "prev_action": int(obs_dict["prev_action"]) if self.agent.is_discrete else np.asarray(obs_dict["prev_action"]).copy(),
                                        "prev_reward": float(obs_dict["prev_reward"]),
                                        "action": int(act) if self.agent.is_discrete else np.asarray(act).copy(),
                                        "reward": float(rew),
                                    }
                                )
                                obs_dict = next_obs_dict
                                if term or trunc:
                                    is_win = (
                                        bool(info.get("is_solved", info.get("is_success", False)))
                                        or (bool(term) and float(rew) >= 5.0)
                                    )
                                    if is_win and len(t_steps) > 0:
                                        ep_solved_single[ep_i] = True
                                        T_traj = len(t_steps)
                                        traj_obs = np.array([s["obs"] for s in t_steps], dtype=np.float32)
                                        has_delta = any(s["delta_obs"] is not None for s in t_steps)
                                        traj_delta = np.array([s["delta_obs"] for s in t_steps], dtype=np.float32) if has_delta else None
                                        traj_prev_act = np.array([s["prev_action"] for s in t_steps])
                                        traj_prev_rew = np.array([s["prev_reward"] for s in t_steps], dtype=np.float32)
                                        traj_act = np.array([s["action"] for s in t_steps])
                                        traj_returns = np.zeros(T_traj, dtype=np.float32)
                                        running_r = 0.0
                                        for t_idx in reversed(range(T_traj)):
                                            running_r = t_steps[t_idx]["reward"] + self.effective_gamma * running_r
                                            traj_returns[t_idx] = running_r
                                        self.sil_buffer.add(
                                            FrontierTransition(
                                                obs=traj_obs,
                                                delta_obs=traj_delta,
                                                prev_actions=traj_prev_act,
                                                prev_rewards=traj_prev_rew,
                                                actions=traj_act,
                                                returns=traj_returns,
                                                depth=d,
                                                length=T_traj,
                                                td_error=float(traj_returns[0]) if len(traj_returns) > 0 else 0.0,
                                            )
                                        )
                                        added_total += 1
                                    break
                        self.last_warmup_solve_rates[d] = float(np.mean(ep_solved_single))
        finally:
            if orig_mixed_rate is not None:
                try:
                    raw_env.mixed_replay_rate = orig_mixed_rate
                except Exception:
                    pass

        if has_curriculum_setter:
            target_final_d = orig_depth if orig_depth >= 1 else max_d_target
            if calibrate_curriculum and getattr(raw_env, "curriculum", False):
                # Nunca rebaixa abaixo da profundidade explicitamente carregada do checkpoint (orig_depth)
                # e utiliza o limiar curricular adaptativo do próprio ambiente em vez de 78% fixo
                calib_d = 1
                for d in range(1, max_d_target + 1):
                    target_rate = (
                        raw_env.get_target_success_rate(d)
                        if hasattr(raw_env, "get_target_success_rate")
                        else 0.65
                    )
                    if self.last_warmup_solve_rates.get(d, 0.0) >= target_rate:
                        calib_d = max(calib_d, d + 1)
                    else:
                        break
                target_final_d = max(orig_depth, min(max_d_target, calib_d))
            raw_env.set_curriculum_depth(target_final_d)
            self.best_state_depth = raw_env.current_depth
            self._rollout_tracked_depth = raw_env.current_depth

        self.env.reset()

        if self.tracker is not None:
            self.tracker.record_sil_step(sil_trajectories=len(self.sil_buffer))
            if hasattr(raw_env, "current_depth") and getattr(raw_env, "curriculum", False):
                self.tracker.metrics.curriculum_depth = int(raw_env.current_depth)
                self.tracker.metrics.curriculum_promotions = int(getattr(raw_env, "curriculum_promotions", 0))

        return max(0, len(self.sil_buffer) - len_before)

    def _prepare_rollout_input(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Copia os dados do dicionário de observação diretamente para os tensores pré-alocados."""
        self._buf_obs[0, 0].copy_(torch.as_tensor(obs_dict["obs"], dtype=torch.float32))
        prev_act_raw = obs_dict["prev_action"]
        if self.agent.is_discrete:
            self._buf_act[0, 0] = int(prev_act_raw)
        else:
            self._buf_act[0, 0].copy_(torch.as_tensor(prev_act_raw, dtype=torch.float32))
        self._buf_rew[0, 0, 0] = float(obs_dict["prev_reward"])
        if self._buf_delta is not None and "delta_obs" in obs_dict and obs_dict["delta_obs"] is not None:
            self._buf_delta[0, 0].copy_(torch.as_tensor(obs_dict["delta_obs"], dtype=torch.float32))
        return self._rollout_input_dict

    def _prepare_vec_rollout_input(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Copia os dados vetorizados do dicionário diretamente para os tensores pré-alocados em lote."""
        self._vec_buf_obs[:, 0].copy_(torch.as_tensor(obs_dict["obs"], dtype=torch.float32))
        prev_act_raw = obs_dict["prev_action"]
        if self.agent.is_discrete:
            self._vec_buf_act[:, 0].copy_(torch.as_tensor(prev_act_raw, dtype=torch.long))
        else:
            self._vec_buf_act[:, 0].copy_(torch.as_tensor(prev_act_raw, dtype=torch.float32))
        self._vec_buf_rew[:, 0, 0].copy_(torch.as_tensor(obs_dict["prev_reward"], dtype=torch.float32))
        if self._vec_buf_delta is not None and "delta_obs" in obs_dict and obs_dict["delta_obs"] is not None:
            self._vec_buf_delta[:, 0].copy_(torch.as_tensor(obs_dict["delta_obs"], dtype=torch.float32))
        return self._vec_input_dict

    def _apply_teacher_lookahead(
        self,
        e: int,
        env_e: Any,
        dist_sample: Any,
        current_hx: Optional[torch.Tensor],
    ) -> Optional[Tuple[Union[int, np.ndarray], torch.Tensor]]:
        """Aplica busca proximal 1-step lookahead guiada pelo Critic para o worker professor e.
        Retorna (teacher_action, teacher_log_prob) se acionado, ou None se mantiver a política."""
        if not hasattr(env_e, "simulate_step"):
            return None

        d_e = int(getattr(env_e, "_episode_depth", getattr(env_e, "current_depth", 1)))
        if d_e < self.teacher_min_depth:
            return None

        if not self.agent.is_discrete or not hasattr(dist_sample, "logits"):
            return None

        raw_logits = dist_sample.logits[e]

        if self.action_chunk_size == 1:
            logits_1d = raw_logits.squeeze()
            probs = torch.softmax(logits_1d, dim=-1)
            top_p, top_indices = torch.topk(probs, k=min(self.teacher_top_k, len(probs)))
            conf = float(top_p[0].item())
            if conf >= self.teacher_confidence_threshold:
                return None

            cand_acts = [int(idx.item()) for idx in top_indices]
            cand_obs = []
            cand_rews = []
            valid_acts = []
            immediate_solve = None

            for a in cand_acts:
                try:
                    s_obs, s_rew, s_done, _ = env_e.simulate_step(a)
                except Exception:
                    continue
                if s_done:
                    immediate_solve = a
                    break
                cand_obs.append(s_obs["obs"] if isinstance(s_obs, dict) else s_obs)
                cand_rews.append(float(s_rew))
                valid_acts.append(a)

            if immediate_solve is not None:
                best_act = immediate_solve
            elif len(valid_acts) > 1:
                M = len(valid_acts)
                b_obs = torch.from_numpy(np.stack(cand_obs).astype(np.float32)).unsqueeze(1).to(self.device)
                b_delta = torch.zeros_like(b_obs)
                b_act = torch.tensor(valid_acts, dtype=torch.long, device=self.device).view(M, 1)
                b_rew = torch.tensor(cand_rews, dtype=torch.float32, device=self.device).view(M, 1, 1)

                hx_e = current_hx[:, e : e + 1, :].repeat(1, M, 1) if current_hx is not None else None
                z_cands = self.agent.front_end(obs=b_obs, delta_obs=b_delta, prev_action=b_act, prev_reward=b_rew)
                h_cands, _ = self.agent.trunk(z_cands, hx=hx_e)
                v_cands = self.agent.value_head(h_cands).squeeze(-1).squeeze(-1)
                q_scores = torch.tensor(cand_rews, device=self.device) + self.effective_gamma * v_cands
                best_idx = int(torch.argmax(q_scores).item())
                best_act = valid_acts[best_idx]
            elif len(valid_acts) == 1:
                best_act = valid_acts[0]
            else:
                return None

            best_t = torch.tensor(best_act, device=self.device)
            new_logp = dist_sample.log_prob(best_t[None, None])[e]
            new_logp = torch.clamp(new_logp, min=-6.0, max=0.0)
            return best_act, new_logp

        elif self.action_chunk_size == 2 and hasattr(env_e, "core"):
            logits_chunk = raw_logits.squeeze()  # [2, num_acts]
            p0 = torch.softmax(logits_chunk[0], dim=-1)
            top_p0, top_idx0 = torch.topk(p0, k=min(self.teacher_top_k, len(p0)))
            conf0 = float(top_p0[0].item())
            if conf0 >= self.teacher_confidence_threshold:
                return None

            # Step 1 lookahead
            cand_acts0 = [int(idx.item()) for idx in top_idx0]
            cand_obs0 = []
            cand_rews0 = []
            valid_acts0 = []
            imm0 = None
            for a in cand_acts0:
                try:
                    s_obs, s_rew, s_done, _ = env_e.simulate_step(a)
                except Exception:
                    continue
                if s_done:
                    imm0 = a
                    break
                cand_obs0.append(s_obs["obs"] if isinstance(s_obs, dict) else s_obs)
                cand_rews0.append(float(s_rew))
                valid_acts0.append(a)

            if imm0 is not None:
                a0_best = imm0
            elif len(valid_acts0) > 1:
                M = len(valid_acts0)
                b_obs = torch.from_numpy(np.stack(cand_obs0).astype(np.float32)).unsqueeze(1).to(self.device)
                b_delta = torch.zeros_like(b_obs)
                b_act = torch.tensor(valid_acts0, dtype=torch.long, device=self.device).view(M, 1)
                b_rew = torch.tensor(cand_rews0, dtype=torch.float32, device=self.device).view(M, 1, 1)
                hx_e = current_hx[:, e : e + 1, :].repeat(1, M, 1) if current_hx is not None else None
                z_cands = self.agent.front_end(obs=b_obs, delta_obs=b_delta, prev_action=b_act, prev_reward=b_rew)
                h_cands, _ = self.agent.trunk(z_cands, hx=hx_e)
                v_cands = self.agent.value_head(h_cands).squeeze(-1).squeeze(-1)
                q0 = torch.tensor(cand_rews0, device=self.device) + 0.99 * v_cands
                a0_best = valid_acts0[int(torch.argmax(q0).item())]
            elif len(valid_acts0) == 1:
                a0_best = valid_acts0[0]
            else:
                return None

            # Step 2 lookahead (aplica a0 temporariamente no simulador do core)
            m0_name = env_e.core.ATOMIC_MOVES[a0_best]
            inv0_name = env_e.core.INVERSE_ATOMIC[m0_name]
            env_e.core.apply_atomic(m0_name)

            p1 = torch.softmax(logits_chunk[1], dim=-1)
            top_p1, top_idx1 = torch.topk(p1, k=min(self.teacher_top_k + 2, len(p1)))
            cand_acts1 = [int(idx.item()) for idx in top_idx1 if env_e.core.ATOMIC_MOVES[int(idx.item())] != inv0_name][:self.teacher_top_k]
            if not cand_acts1:
                cand_acts1 = [int(top_idx1[0].item())]

            cand_obs1 = []
            cand_rews1 = []
            valid_acts1 = []
            imm1 = None
            for a in cand_acts1:
                try:
                    s_obs, s_rew, s_done, _ = env_e.simulate_step(a)
                except Exception:
                    continue
                if s_done:
                    imm1 = a
                    break
                cand_obs1.append(s_obs["obs"] if isinstance(s_obs, dict) else s_obs)
                cand_rews1.append(float(s_rew))
                valid_acts1.append(a)

            # Desfaz a0 no simulador imediatamente
            env_e.core.apply_atomic(inv0_name)

            if imm1 is not None:
                a1_best = imm1
            elif len(valid_acts1) > 1:
                M = len(valid_acts1)
                b_obs = torch.from_numpy(np.stack(cand_obs1).astype(np.float32)).unsqueeze(1).to(self.device)
                b_delta = torch.zeros_like(b_obs)
                b_act = torch.tensor(valid_acts1, dtype=torch.long, device=self.device).view(M, 1)
                b_rew = torch.tensor(cand_rews1, dtype=torch.float32, device=self.device).view(M, 1, 1)
                hx_e = current_hx[:, e : e + 1, :].repeat(1, M, 1) if current_hx is not None else None
                z_cands = self.agent.front_end(obs=b_obs, delta_obs=b_delta, prev_action=b_act, prev_reward=b_rew)
                h_cands, _ = self.agent.trunk(z_cands, hx=hx_e)
                v_cands = self.agent.value_head(h_cands).squeeze(-1).squeeze(-1)
                q1 = torch.tensor(cand_rews1, device=self.device) + 0.99 * v_cands
                a1_best = valid_acts1[int(torch.argmax(q1).item())]
            elif len(valid_acts1) == 1:
                a1_best = valid_acts1[0]
            else:
                a1_best = cand_acts1[0]

            chunk_act = np.array([a0_best, a1_best], dtype=np.int64)
            chunk_t = torch.tensor(chunk_act, device=self.device)
            new_logp = dist_sample.log_prob(chunk_t[None, None])[e].sum(dim=-1)
            new_logp = torch.clamp(new_logp, min=-8.0, max=0.0)
            return chunk_act, new_logp

        return None

    def _collect_rollouts_vectorized(
        self,
        current_obs_dict: Dict[str, Any],
        current_hx: Optional[torch.Tensor],
        episode_start: Union[bool, np.ndarray],
        step_callback: Optional[Callable[[], None]] = None,
    ) -> Tuple[Dict[str, Any], Optional[torch.Tensor], np.ndarray, float]:
        """Coleta interações vetorizadas em lote com num_envs simultâneos e inferência GPU batch."""
        self.buffer.reset()
        self.agent.eval()
        if self.tracker is not None:
            self.tracker.reset_step_timer()

        S = self.steps_per_env
        E = self.num_envs
        T = self.chunk_length
        K = S // T

        obs_shape = self.buffer.obs_shape
        delta_shape = self.env.observation_space.spaces["delta_obs"].shape if hasattr(self.env, "observation_space") and isinstance(self.env.observation_space, gym.spaces.Dict) and "delta_obs" in self.env.observation_space.spaces else obs_shape
        stored_obs = np.empty((S, E, *obs_shape), dtype=np.float32)
        stored_delta = np.empty((S, E, *delta_shape), dtype=np.float32) if not self.env.is_visual else None
        if self.agent.is_discrete:
            stored_prev_act = np.empty((S, E), dtype=np.int64)
            if self.action_chunk_size > 1:
                stored_act = np.empty((S, E, self.action_chunk_size), dtype=np.int64)
            else:
                stored_act = np.empty((S, E), dtype=np.int64)
        else:
            stored_prev_act = np.empty((S, E, *self.buffer.action_shape), dtype=np.float32)
            stored_act = np.empty((S, E, *self.buffer.action_shape), dtype=np.float32)
        stored_prev_rew = np.empty((S, E), dtype=np.float32)
        stored_rew = np.empty((S, E), dtype=np.float32)
        stored_done = np.empty((S, E), dtype=bool)
        stored_ep_start = np.empty((S, E), dtype=bool)
        stored_val = np.empty((S, E), dtype=np.float32)
        stored_logp = np.empty((S, E), dtype=np.float32)
        stored_scale = np.ones((S, E), dtype=np.float32)

        recorded_chunk_hx: List[List[torch.Tensor]] = [[None for _ in range(E)] for _ in range(K)]

        if current_hx is None:
            current_hx = torch.zeros((1, E, self.trunk_hidden_dim), dtype=torch.float32, device=self.device)
        if isinstance(episode_start, bool):
            ep_starts = np.full(E, episode_start, dtype=bool)
        else:
            ep_starts = np.asarray(episode_start, dtype=bool)

        curr_ep_returns = np.zeros(E, dtype=np.float32)
        stored_env_infos: Optional[List[Any]] = [] if self.hindsight_replay else None

        with torch.inference_mode():

            for step in range(S):
                # 1. Salva hx inicial dos chunks nas fronteiras temporais
                if step % T == 0:
                    chunk_k = step // T
                    hx_cpu = current_hx.detach().cpu()
                    for e in range(E):
                        recorded_chunk_hx[chunk_k][e] = hx_cpu[:, e : e + 1, :]

                input_dict = self._prepare_vec_rollout_input(current_obs_dict)
                t0 = time.perf_counter_ns()
                dist, value, next_hx = self.agent.forward(input_dict, hx=current_hx, dones=None)

                # Escala de exploração / homeostase dinâmica
                if self.is_auto_exploration:
                    curr_scale = float(self._adaptive_scale)
                else:
                    curr_scale = float(self.exploration_scale)

                if self.agent.is_discrete:
                    logits = dist.logits
                    num_acts = logits.shape[-1]
                    penalized_logits = logits.clone()

                    # Vectorized Action Quality Filter & Generic Reversible Move Pruning
                    raw_env = getattr(self.env, "unwrapped", self.env)
                    inv_map: Optional[Dict[int, int]] = getattr(raw_env, "inverse_action_map", None)
                    if inv_map is not None:
                        for e in range(E):
                            e_hist = self._vec_recent_actions[e]
                            if len(e_hist) >= 1:
                                inv_a = inv_map.get(e_hist[-1])
                                if inv_a is not None:
                                    if self.action_chunk_size > 1:
                                        penalized_logits[e, 0, 0, inv_a] -= 4.0
                                    else:
                                        penalized_logits[e, 0, inv_a] -= 4.0
                            if len(e_hist) >= 3:
                                if e_hist[-1] == e_hist[-3]:
                                    if self.action_chunk_size > 1:
                                        penalized_logits[e, 0, 0, e_hist[-2]] -= 3.0
                                    else:
                                        penalized_logits[e, 0, e_hist[-2]] -= 3.0
                                inv_a1 = inv_map.get(e_hist[-3])
                                inv_a2 = inv_map.get(e_hist[-2])
                                if inv_a1 is not None and inv_a2 is not None and e_hist[-1] == inv_a1:
                                    if self.action_chunk_size > 1:
                                        penalized_logits[e, 0, 0, inv_a2] -= 4.0
                                    else:
                                        penalized_logits[e, 0, inv_a2] -= 4.0
                    else:
                        for e in range(E):
                            e_hist = self._vec_recent_actions[e]
                            if len(e_hist) >= 3 and e_hist[-1] == e_hist[-3]:
                                if self.action_chunk_size > 1:
                                    penalized_logits[e, 0, 0, e_hist[-2]] -= 3.0
                                else:
                                    penalized_logits[e, 0, e_hist[-2]] -= 3.0

                    # Estratificação térmica por ambiente com cobertura garantida de 0 à 1 (Quality-Diverse Spectrum)
                    if hasattr(self, "_env_scale_multipliers") and self.num_envs >= 4:
                        dynamic_scales = curr_scale * self._env_scale_multipliers
                        if getattr(self, "_env_fixed_anchors", None) is not None:
                            env_scales = torch.where(
                                self._env_fixed_anchors > 0.0,
                                0.5 * dynamic_scales + 0.5 * self._env_fixed_anchors,
                                dynamic_scales,
                            ).clamp(min=0.02, max=1.00)
                        else:
                            env_scales = dynamic_scales.clamp(min=0.02, max=1.00)
                        if self.action_chunk_size > 1:
                            env_scales = env_scales.unsqueeze(2)
                        scaled_logits = torch.nan_to_num(penalized_logits / env_scales, nan=0.0, posinf=35.0, neginf=-35.0).clamp(min=-35.0, max=35.0)
                        raw_scaled_logits = torch.nan_to_num(logits / env_scales, nan=0.0, posinf=35.0, neginf=-35.0).clamp(min=-35.0, max=35.0)
                    elif curr_scale != 1.0:
                        eff_scale = max(0.02, curr_scale)
                        env_scales = torch.full((E, 1), eff_scale, device=self.device)
                        if self.action_chunk_size > 1:
                            env_scales = env_scales.unsqueeze(2)
                        scaled_logits = torch.nan_to_num(penalized_logits / eff_scale, nan=0.0, posinf=35.0, neginf=-35.0).clamp(min=-35.0, max=35.0)
                        raw_scaled_logits = torch.nan_to_num(logits / eff_scale, nan=0.0, posinf=35.0, neginf=-35.0).clamp(min=-35.0, max=35.0)
                    else:
                        env_scales = torch.ones((E, 1), device=self.device)
                        if self.action_chunk_size > 1:
                            env_scales = env_scales.unsqueeze(2)
                        scaled_logits = penalized_logits
                        raw_scaled_logits = logits

                    dist_sample = torch.distributions.Categorical(logits=scaled_logits)
                    action = dist_sample.sample()

                    # Avaliação de log_prob com a escala térmica exata sem penalidade de poda (elimina viés off-policy no PPO)
                    dist_eval_rollout = torch.distributions.Categorical(logits=raw_scaled_logits)
                    log_prob = dist_eval_rollout.log_prob(action)
                    if self.action_chunk_size > 1:
                        log_prob = log_prob.sum(dim=-1)
                    env_action = action.squeeze(1).cpu().numpy()

                    # -------------------------------------------------------------
                    # Teacher-Assisted SIL Rollouts (Lookahead Proximal Assimétrico)
                    # -------------------------------------------------------------
                    if self.teacher_sil_envs > 0 and self.sil_enabled:
                        raw_env_obj = getattr(self.env, "unwrapped", self.env)
                        if hasattr(raw_env_obj, "envs"):
                            for e in range(min(self.teacher_sil_envs, E)):
                                env_e = raw_env_obj.envs[e]
                                res = self._apply_teacher_lookahead(
                                    e=e,
                                    env_e=env_e,
                                    dist_sample=dist_eval_rollout,
                                    current_hx=current_hx,
                                )
                                if res is not None:
                                    t_act, t_logp = res
                                    env_action[e] = t_act
                                    log_prob[e] = t_logp

                    # Chunking adaptativo por incerteza: aborta a execução aberta caso a confiança caia
                    chunk_mask = None
                    if self.adaptive_chunk and self.action_chunk_size > 1:
                        probs_t = torch.softmax(scaled_logits.squeeze(1), dim=-1)
                        top2_t, _ = torch.topk(probs_t, k=min(2, num_acts), dim=-1)
                        confs_t = top2_t[..., 0]
                        margins_t = top2_t[..., 0] - top2_t[..., 1]
                        chunk_mask = np.ones((E, self.action_chunk_size), dtype=bool)
                        for step_i in range(1, self.action_chunk_size):
                            unc_mask = (confs_t[:, step_i] < self.adaptive_conf_threshold) | (margins_t[:, step_i] < self.adaptive_margin_threshold)
                            unc_env_indices = torch.where(unc_mask)[0].cpu().numpy()
                            chunk_mask[unc_env_indices, step_i:] = False

                    # Atualiza histórico de ações por ambiente (registra a última ação executada)
                    for e in range(E):
                        last_act_e = int(env_action[e, -1]) if self.action_chunk_size > 1 else int(env_action[e])
                        self._vec_recent_actions[e].append(last_act_e)
                else:
                    if hasattr(self, "_env_scale_multipliers") and self.num_envs >= 4:
                        dynamic_scales = curr_scale * self._env_scale_multipliers
                        if getattr(self, "_env_fixed_anchors", None) is not None:
                            env_scales = torch.where(
                                self._env_fixed_anchors > 0.0,
                                0.5 * dynamic_scales + 0.5 * self._env_fixed_anchors,
                                dynamic_scales,
                            ).clamp(min=0.02, max=1.00)
                        else:
                            env_scales = dynamic_scales.clamp(min=0.02, max=1.00)
                        if self.action_chunk_size > 1:
                            env_scales = env_scales.unsqueeze(2)
                        scaled_std = dist.scale * env_scales
                    elif curr_scale != 1.0:
                        eff_scale = max(0.20, curr_scale)
                        env_scales = torch.full((E, 1), eff_scale, device=self.device)
                        if self.action_chunk_size > 1:
                            env_scales = env_scales.unsqueeze(2)
                        scaled_std = dist.scale * eff_scale
                    else:
                        env_scales = torch.ones((E, 1), device=self.device)
                        if self.action_chunk_size > 1:
                            env_scales = env_scales.unsqueeze(2)
                        scaled_std = dist.scale

                    dist_sample = torch.distributions.Normal(loc=dist.mean, scale=scaled_std)
                    action = dist_sample.sample()
                    if self.action_chunk_size > 1:
                        log_prob = dist_sample.log_prob(action).sum(dim=(-2, -1))
                    else:
                        log_prob = dist_sample.log_prob(action).sum(dim=-1)
                    env_action = action.squeeze(1).cpu().numpy()
                    if isinstance(self.env.action_space, gym.spaces.Box):
                        env_action = np.clip(env_action, self.env.action_space.low, self.env.action_space.high)
                    chunk_mask = None

                lat_us = (time.perf_counter_ns() - t0) / 1000.0

                if self.tracker is not None:
                    if step % 4 == 0 or step == S - 1:
                        if self.agent.is_discrete:
                            probs = dist.probs[0, 0]
                            n_acts = probs.shape[-1]
                            top_probs, _ = torch.topk(probs, k=min(2, n_acts), dim=-1)
                            conf = float(top_probs[..., 0].mean().item())
                            ent = float(-torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item())
                            max_ent = float(np.log(n_acts)) if n_acts > 1 else 1.0
                            unc = float(np.clip(ent / max_ent, 0.0, 1.0))
                        else:
                            std = dist.scale[0, 0].cpu().numpy()
                            std = np.clip(std, a_min=1e-6, a_max=100.0)
                            ent = float(0.5 * np.mean(np.sum(1.0 + np.log(2.0 * np.pi * (std ** 2)), axis=-1)))
                            var = float(np.mean(std ** 2))
                            unc = float(np.clip(2.0 / (1.0 + np.exp(-var / 0.5)) - 1.0, 0.0, 1.0))
                            conf = float(np.clip(1.0 - unc, 0.0, 1.0))

                        self.tracker.record_inference(
                            latency_us=lat_us,
                            uncertainty=unc,
                            confidence=conf,
                            entropy=ent,
                            calibration=curr_scale,
                        )

                # Environment step vetorizado (com chunk_mask adaptativo)
                try:
                    next_obs_dict, rewards, terminateds, truncateds, env_infos = self.env.step(env_action, chunk_mask=chunk_mask)
                except TypeError:
                    next_obs_dict, rewards, terminateds, truncateds, env_infos = self.env.step(env_action)
                dones = terminateds | truncateds
                if stored_env_infos is not None:
                    stored_env_infos.append(env_infos)

                # Sincroniza tensores para numpy uma única vez por passo de rollout
                step_logp_np = log_prob.squeeze(1).detach().cpu().numpy()
                step_vals_np = value.squeeze(1).squeeze(1).detach().cpu().numpy()

                stored_val[step] = step_vals_np
                stored_logp[step] = step_logp_np
                stored_scale[step] = env_scales.view(E, -1)[:, 0].detach().cpu().numpy()

                if self.sil_enabled:
                    has_delta = "delta_obs" in current_obs_dict and current_obs_dict["delta_obs"] is not None
                    for e in range(E):
                        delta_v = (
                            current_obs_dict["delta_obs"][e].copy()
                            if has_delta
                            else None
                        )
                        prev_act_v = current_obs_dict["prev_action"][e]
                        if hasattr(prev_act_v, "copy"):
                            prev_act_v = prev_act_v.copy()
                        act_v = env_action[e]
                        if hasattr(act_v, "copy"):
                            act_v = act_v.copy()
                        entry = {
                            "obs": current_obs_dict["obs"][e].copy(),
                            "delta_obs": delta_v,
                            "prev_action": prev_act_v,
                            "prev_reward": float(current_obs_dict["prev_reward"][e]),
                            "action": act_v,
                            "reward": float(rewards[e]),
                            "log_prob": float(step_logp_np[e]),
                            "value": float(step_vals_np[e]),
                        }
                        self._active_vec_trajectories[e].append(entry)

                if self.tracker is not None:
                    primary_info = env_infos[0] if isinstance(env_infos, list) and len(env_infos) > 0 else {}
                    self.tracker.record_env_step(
                        reward=float(rewards[0]),
                        done=bool(dones[0]),
                        info=primary_info,
                        num_steps=self.num_envs * self.action_chunk_size,
                    )

                stored_obs[step] = current_obs_dict["obs"]
                if stored_delta is not None and "delta_obs" in current_obs_dict:
                    stored_delta[step] = current_obs_dict["delta_obs"]
                stored_prev_act[step] = current_obs_dict["prev_action"]
                stored_prev_rew[step] = current_obs_dict["prev_reward"]
                stored_act[step] = env_action
                stored_rew[step] = rewards
                stored_done[step] = terminateds
                stored_ep_start[step] = ep_starts

                curr_ep_returns += rewards
                self.total_steps += E * self.action_chunk_size

                done_indices = np.where(dones)[0]
                if len(done_indices) > 0:
                    for d_i in done_indices:
                        self.episode_returns.append(float(curr_ep_returns[d_i]))
                        curr_ep_returns[d_i] = 0.0
                        self._vec_recent_actions[d_i].clear()
                        d_info = env_infos[d_i] if isinstance(env_infos, (list, tuple)) and d_i < len(env_infos) else {}
                        new_d = self._get_current_curriculum_depth()
                        if (isinstance(d_info, dict) and d_info.get("curriculum_promoted", False)) or new_d > self._rollout_tracked_depth:
                            self._rollout_tracked_depth = max(self._rollout_tracked_depth, new_d)
                            self.episode_returns.clear()

                        traj_steps = self._active_vec_trajectories[d_i]
                        is_solved = bool(d_info.get("is_solved", False)) if isinstance(d_info, dict) else False
                        is_win = is_solved or (bool(terminateds[d_i]) and float(rewards[d_i]) >= 5.0)

                        if len(traj_steps) > 0:
                            cur_d = int(d_info.get("scramble_depth", d_info.get("current_depth", self._get_current_curriculum_depth()))) if isinstance(d_info, dict) else self._get_current_curriculum_depth()
                            self._process_completed_trajectory(traj_steps, is_win, cur_d)
                        self._active_vec_trajectories[d_i].clear()

                    next_hx[:, done_indices, :] = 0.0
                    ep_starts = dones.copy()
                else:
                    ep_starts.fill(False)

                current_obs_dict = next_obs_dict
                current_hx = next_hx

                if step_callback is not None:
                    step_callback()

            # Bootstrap values para o último estado
            input_dict = self._prepare_vec_rollout_input(current_obs_dict)
            _, last_value, _ = self.agent.forward(input_dict, hx=current_hx, dones=None)
            last_values = last_value.squeeze(1).squeeze(1).cpu().numpy()
            last_dones = terminateds

        # GAE computado em paralelo ao longo das trajetórias de cada ambiente
        rew_es = stored_rew.T.copy()
        done_es = stored_done.T
        val_es = stored_val.T

        if self.hindsight_replay:
            self._apply_hindsight_relabelling(rew_es, done_es, val_es, step_infos=stored_env_infos)

        adv_es = np.zeros((E, S), dtype=np.float32)

        last_gae = np.zeros(E, dtype=np.float32)

        for t in reversed(range(S)):
            if t == S - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_val = last_values.astype(np.float32)
            else:
                next_non_terminal = 1.0 - done_es[:, t].astype(np.float32)
                next_val = val_es[:, t + 1]

            delta = rew_es[:, t] + self.effective_gamma * next_val * next_non_terminal - val_es[:, t]
            last_gae = delta + self.effective_gamma * self.gae_lambda * next_non_terminal * last_gae
            adv_es[:, t] = last_gae

        ret_es = adv_es + val_es

        # Transposição [S, E, ...] -> [E, S, ...] e linearização ordenada por ambiente
        flat_obs = stored_obs.transpose(1, 0, *range(2, stored_obs.ndim)).reshape(E * S, *obs_shape)
        flat_delta = (
            stored_delta.transpose(1, 0, *range(2, stored_delta.ndim)).reshape(E * S, *delta_shape)
            if stored_delta is not None
            else None
        )
        if self.agent.is_discrete:
            flat_prev_act = stored_prev_act.T.reshape(E * S)
            if self.action_chunk_size > 1:
                flat_act = stored_act.transpose(1, 0, *range(2, stored_act.ndim)).reshape(E * S, self.action_chunk_size)
            else:
                flat_act = stored_act.T.reshape(E * S)
        else:
            flat_prev_act = stored_prev_act.transpose(1, 0, *range(2, stored_prev_act.ndim)).reshape(E * S, *self.buffer.action_shape)
            flat_act = stored_act.transpose(1, 0, *range(2, stored_act.ndim)).reshape(E * S, *self.buffer.action_shape)

        flat_prev_rew = stored_prev_rew.T.reshape(E * S)
        flat_rew = rew_es.reshape(E * S)
        flat_done = done_es.reshape(E * S)
        flat_ep_start = stored_ep_start.T.reshape(E * S)
        flat_val = val_es.reshape(E * S)
        flat_logp = stored_logp.T.reshape(E * S)
        flat_adv = adv_es.reshape(E * S)
        flat_ret = ret_es.reshape(E * S)
        flat_scale = stored_scale.T.reshape(E * S)

        # Montagem dos estados ocultos iniciais dos chunks em ordem de ambiente
        all_chunk_hx: List[torch.Tensor] = []
        for e in range(E):
            for chunk_k in range(K):
                all_chunk_hx.append(recorded_chunk_hx[chunk_k][e])

        # Normalização global de vantagens
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        # Atualiza buffer
        self.buffer.set_vectorized_data(
            obs=flat_obs,
            delta_obs=flat_delta,
            prev_actions=flat_prev_act,
            prev_rewards=flat_prev_rew,
            actions=flat_act,
            rewards=flat_rew,
            dones=flat_done,
            episode_starts=flat_ep_start,
            values=flat_val,
            log_probs=flat_logp,
            chunk_hx=all_chunk_hx,
            advantages=flat_adv,
            returns=flat_ret,
            env_scales=flat_scale,
        )

        mean_return = (
            float(np.mean(self.episode_returns)) if len(self.episode_returns) > 0 else 0.0
        )

        self._update_adaptive_exploration(mean_return, env_infos)

        return current_obs_dict, current_hx, ep_starts, mean_return

    def _apply_hindsight_relabelling(
        self,
        rew_es: np.ndarray,
        done_es: np.ndarray,
        val_es: np.ndarray,
        step_infos: Optional[List[Any]] = None,
    ) -> None:
        """Aplica Hindsight Experience Replay (HER) / Reversão Retrospectiva On-Policy (B.2).

        Para episódios incompletos/não-resolvidos no rollout, localiza o ápice de progresso
        intermediário t* e injeta bônus retrospectivo para garantir que a vantagem GAE
        reforce os passos de avanço, quebrando platôs de recompensa esparsa.
        """
        E, S = rew_es.shape

        for e in range(E):
            ep_start = 0
            for t in range(S):
                is_last_step = (t == S - 1)
                is_terminal = bool(done_es[e, t])

                if is_terminal or is_last_step:
                    ep_end = t
                    ep_len = ep_end - ep_start + 1

                    if ep_len >= 2:
                        was_solved = False
                        if step_infos is not None and len(step_infos) > ep_end:
                            info_end = step_infos[ep_end]
                            if isinstance(info_end, list) and len(info_end) > e:
                                info_e = info_end[e]
                                was_solved = bool(info_e.get("is_solved", False) or info_e.get("success", False))

                        if not was_solved:
                            progress = np.empty(ep_len, dtype=np.float32)
                            for step_offset, step_idx in enumerate(range(ep_start, ep_end + 1)):
                                p_val = None
                                if step_infos is not None and len(step_infos) > step_idx:
                                    info_step = step_infos[step_idx]
                                    if isinstance(info_step, list) and len(info_step) > e:
                                        info_dict = info_step[e]
                                        if "score" in info_dict:
                                            p_val = float(info_dict["score"])
                                        elif "progress" in info_dict:
                                            p_val = float(info_dict["progress"])

                                if p_val is None:
                                    p_val = float(val_es[e, step_idx])

                                progress[step_offset] = p_val

                            best_rel_idx = int(np.argmax(progress))
                            peak_progress = float(progress[best_rel_idx])
                            init_progress = float(progress[0])
                            delta_p = peak_progress - init_progress

                            if delta_p >= self.hindsight_min_gain:
                                peak_t = ep_start + best_rel_idx
                                bonus = float(self.hindsight_bonus * np.clip(delta_p, 0.0, 1.0))
                                rew_es[e, peak_t] += bonus

                                if peak_t < ep_end:
                                    decay_steps = ep_end - peak_t
                                    step_penalty = float(0.10 * bonus / max(1, decay_steps))
                                    rew_es[e, peak_t + 1 : ep_end + 1] -= step_penalty

                    ep_start = t + 1

    def _update_adaptive_exploration(
        self,
        mean_return: float,
        env_infos: Union[List[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Atualiza a escala de auto-calibração com base no momentum real de recompensa e platôs curriculares."""
        if self.is_auto_exploration:
            primary_info = (
                env_infos[0]
                if isinstance(env_infos, list) and len(env_infos) > 0
                else (env_infos if isinstance(env_infos, dict) else {})
            )

            cur_promotions = primary_info.get("curriculum_promotions", 0)
            cur_promoted = primary_info.get("curriculum_promoted", False)
            cur_success_rate = primary_info.get("curriculum_success_rate", None)

            if self._reward_ema is None:
                self._reward_ema = float(mean_return)
                delta_ret = 0.0
            else:
                delta_ret = mean_return - self._reward_ema
                self._reward_ema = 0.85 * self._reward_ema + 0.15 * mean_return

            # Escala de referência relativa independente da magnitude absoluta de recompensa
            scale_ref = max(1.0, abs(self._reward_ema))
            # Progresso real: ganho de pelo menos 2.5% em relação à escala de retorno típica
            is_progress = delta_ret > 0.025 * scale_ref

            if cur_promoted or cur_promotions > getattr(self, "_last_promotions", 0):
                # Promoção curricular: novo patamar requer exploração revigorada
                self._last_promotions = cur_promotions
                self._stagnation_count = 0
                self._adaptive_scale = max(self._adaptive_scale, 0.75)
            elif is_progress or (cur_success_rate is not None and cur_success_rate >= 0.80):
                # Progresso consistente ou quase domínio do nível: consolida e refina política
                self._stagnation_count = 0
                self._adaptive_scale = max(self._min_adaptive_scale, self._adaptive_scale - 0.03)
            else:
                # Platô ou declínio de retorno: incrementa contagem e aquece exploração para escapar de mínimos locais
                self._stagnation_count += 1
                if self._stagnation_count >= 2:
                    self._adaptive_scale = min(self._max_adaptive_scale, self._adaptive_scale + 0.04)

        # Dynamic Entropy Floor: Re-aquecimento térmico imediato de rollout em caso de inanição de entropia
        if self.entropy_floor_enabled and self.tracker is not None and getattr(self.tracker.metrics, "entropies", None):
            recent_entropies = list(self.tracker.metrics.entropies)[-50:]
            if recent_entropies:
                recent_ent = float(np.mean(recent_entropies))
                if recent_ent < self.entropy_floor:
                    # Injeta impulso térmico no próximo rollout para diversificar ramificações
                    self._adaptive_scale = min(self._max_adaptive_scale, max(self._adaptive_scale, 0.85))
                    self.exploration_scale = float(self._adaptive_scale)
        elif self.is_auto_exploration:
            self.exploration_scale = float(self._adaptive_scale)

    def _process_completed_trajectory(
        self,
        traj_steps: List[Dict[str, Any]],
        is_win: bool,
        cur_depth: int,
    ) -> None:
        """Processa e armazena estritamente vitórias e fatias de sucesso no buffer SIL com Peak Slicing e Bellman Isolado.

        Aplica a síntese universal híbrida sem teto e com compactação 100% lossless:
        1. Em vitórias completas (is_win=True), preserva o episódio integral com is_full_win=True.
        2. Em episódios incompletos/truncados (is_win=False), aplica Peak Slicing no passo de maior
           progresso cumulativo (t*), descartando a cauda de degradação e armazenando apenas a fatia
           de ganho positivo líquido (is_full_win=False).
        3. Calcula o retorno de Bellman isolado na fatia [0...t*], ancorando o valor terminal via
           bootstrapping de V(s_{t*+1}) para blindar a vantagem SIL positiva (max(0, R - V) > 0).
        4. Nenhuma trajetória de falha ou cauda de degradação é persistida no SuccessReplayBuffer;
           o contraste DPO opera sob demanda contra chunks negativos on-policy frescos do rollout atual.
        """
        if not self.sil_enabled or len(traj_steps) == 0:
            return

        traj_steps = list(traj_steps)
        cur_d = max(1, int(cur_depth))
        T_traj = len(traj_steps)

        raw_env_ref = getattr(self.env, "unwrapped", self.env)
        frontier_d = int(getattr(raw_env_ref, "current_depth", getattr(self.env, "current_depth", cur_d)) or cur_d)
        if cur_d >= frontier_d:
            self._recent_frontier_wins.append(1.0 if is_win else 0.0)

        if is_win:
            # Vitória Completa: preserva o episódio inteiro com retorno terminal canônico
            traj_rews = [s["reward"] for s in traj_steps]
            traj_returns = np.zeros(T_traj, dtype=np.float32)
            running_r = 0.0
            for t_idx in reversed(range(T_traj)):
                running_r = traj_rews[t_idx] + self.effective_gamma * running_r
                traj_returns[t_idx] = running_r

            traj_obs = np.array([s["obs"] for s in traj_steps], dtype=np.float32)
            has_delta = any(s["delta_obs"] is not None for s in traj_steps)
            traj_delta = (
                np.array([s["delta_obs"] for s in traj_steps], dtype=np.float32)
                if has_delta
                else None
            )
            traj_prev_act = np.array([s["prev_action"] for s in traj_steps])
            traj_prev_rew = np.array([s["prev_reward"] for s in traj_steps], dtype=np.float32)
            traj_act = np.array([s["action"] for s in traj_steps])
            traj_logp = np.array([s.get("log_prob", 0.0) for s in traj_steps], dtype=np.float32)

            self.sil_buffer.add(
                FrontierTransition(
                    obs=traj_obs,
                    delta_obs=traj_delta,
                    prev_actions=traj_prev_act,
                    prev_rewards=traj_prev_rew,
                    actions=traj_act,
                    returns=traj_returns,
                    depth=cur_d,
                    length=T_traj,
                    td_error=float(traj_returns[0]) if len(traj_returns) > 0 else 0.0,
                    log_probs=traj_logp,
                    is_full_win=True,
                )
            )
        else:
            # Episódio não resolvido: busca o pico de progresso cumulativo (Peak Slicing)
            cum_rews = np.cumsum([s["reward"] for s in traj_steps])
            t_star = int(np.argmax(cum_rews))
            peak_gain = float(cum_rews[t_star])

            # Critérios de elegibilidade:
            # Pelo menos 2 passos de manobra coordenada e ganho líquido mensurável (> 0.05)
            min_peak_steps = 2
            min_cum_gain = 0.05
            is_eligible_peak = (t_star >= min_peak_steps - 1) and (peak_gain >= min_cum_gain)

            if is_eligible_peak:
                L_slice = t_star + 1
                peak_slice = traj_steps[:L_slice]

                # Bootstrapping com V(s_{t*+1}) para blindar a vantagem de Bellman contra underestimation
                if t_star + 1 < T_traj:
                    next_val = float(traj_steps[t_star + 1].get("value", 0.0))
                else:
                    next_val = float(traj_steps[t_star].get("value", 0.0))

                slice_returns = np.zeros(L_slice, dtype=np.float32)
                running_r = next_val
                for t_idx in reversed(range(L_slice)):
                    running_r = float(peak_slice[t_idx]["reward"]) + self.effective_gamma * running_r
                    slice_returns[t_idx] = running_r

                peak_obs = np.array([s["obs"] for s in peak_slice], dtype=np.float32)
                has_delta = any(s["delta_obs"] is not None for s in peak_slice)
                peak_delta = (
                    np.array([s["delta_obs"] for s in peak_slice], dtype=np.float32)
                    if has_delta
                    else None
                )
                peak_prev_act = np.array([s["prev_action"] for s in peak_slice])
                peak_prev_rew = np.array([s["prev_reward"] for s in peak_slice], dtype=np.float32)
                peak_act = np.array([s["action"] for s in peak_slice])
                peak_logp = np.array([s.get("log_prob", 0.0) for s in peak_slice], dtype=np.float32)

                self.sil_buffer.add(
                    FrontierTransition(
                        obs=peak_obs,
                        delta_obs=peak_delta,
                        prev_actions=peak_prev_act,
                        prev_rewards=peak_prev_rew,
                        actions=peak_act,
                        returns=slice_returns,
                        depth=cur_d,
                        length=L_slice,
                        td_error=float(slice_returns[0]) if len(slice_returns) > 0 else 0.0,
                        log_probs=peak_logp,
                        is_full_win=False,
                    )
                )

    def collect_rollouts(
        self,
        current_obs_dict: Dict[str, Any],
        current_hx: Optional[torch.Tensor],
        episode_start: Union[bool, np.ndarray],
        step_callback: Optional[Callable[[], None]] = None,
    ) -> Tuple[Dict[str, Any], Optional[torch.Tensor], Union[bool, np.ndarray], float]:
        """Collects rollout_steps interactions in the environment."""
        if self.is_vectorized:
            return self._collect_rollouts_vectorized(
                current_obs_dict, current_hx, episode_start, step_callback=step_callback
            )

        self.buffer.reset()
        self.agent.eval()
        if self.tracker is not None:
            self.tracker.reset_step_timer()
        last_env_info: Dict[str, Any] = {}
        stored_step_infos: Optional[List[Any]] = [] if self.hindsight_replay else None


        with torch.no_grad():
            for _ in range(self.rollout_steps):
                input_dict = self._prepare_rollout_input(current_obs_dict)

                # Step agent with latency tracking
                t0 = time.perf_counter_ns()
                dist, value, next_hx = self.agent.forward(input_dict, hx=current_hx, dones=None)

                # Escala de exploração / homeostase dinâmica
                if self.is_auto_exploration:
                    curr_scale = float(self._adaptive_scale)
                else:
                    curr_scale = float(self.exploration_scale)

                if self.agent.is_discrete:
                    logits = dist.logits
                    num_acts = logits.shape[-1]
                    if num_acts == 12:
                        penalized_logits = logits.clone()
                        if len(self._recent_actions) >= 1:
                            last_a = self._recent_actions[-1]
                            inv_a = (last_a + 1) if (last_a % 2 == 0) else (last_a - 1)
                            penalized_logits[..., inv_a] -= 4.0
                        if len(self._recent_actions) >= 3:
                            if self._recent_actions[-1] == self._recent_actions[-3]:
                                penalized_logits[..., self._recent_actions[-2]] -= 3.0
                            a1, a2, a3 = self._recent_actions[-3], self._recent_actions[-2], self._recent_actions[-1]
                            inv_a1 = (a1 + 1) if (a1 % 2 == 0) else (a1 - 1)
                            inv_a2 = (a2 + 1) if (a2 % 2 == 0) else (a2 - 1)
                            if a3 == inv_a1:
                                penalized_logits[..., inv_a2] -= 4.0
                        logits = penalized_logits
                    elif self.is_auto_exploration and self._stagnation_count >= 3 and len(self._recent_actions) >= 3:
                        if self._recent_actions[-1] == self._recent_actions[-3]:
                            avoid_act = self._recent_actions[-2]
                            penalized_logits = logits.clone()
                            penalized_logits[..., avoid_act] -= 3.0
                            logits = penalized_logits

                    if curr_scale != 1.0:
                        eff_scale = max(0.20, curr_scale)
                        scaled_logits = torch.nan_to_num(logits / eff_scale, nan=0.0, posinf=35.0, neginf=-35.0).clamp(min=-35.0, max=35.0)
                        dist_sample = torch.distributions.Categorical(logits=scaled_logits)
                    elif self.is_auto_exploration and self._stagnation_count >= 3:
                        dist_sample = torch.distributions.Categorical(logits=logits)
                    else:
                        dist_sample = dist
                    action = dist_sample.sample()
                    log_prob = dist_sample.log_prob(action)
                    if self.action_chunk_size > 1:
                        log_prob = log_prob.sum(dim=-1)
                        env_action = action.squeeze(0).squeeze(0).cpu().numpy()
                    else:
                        env_action = int(action.item())

                    # Chunking adaptativo por incerteza
                    chunk_mask = None
                    if self.adaptive_chunk and self.action_chunk_size > 1:
                        probs_t = torch.softmax(scaled_logits.squeeze(0).squeeze(0), dim=-1)
                        top2_t, _ = torch.topk(probs_t, k=min(2, num_acts), dim=-1)
                        confs_t = top2_t[..., 0]
                        margins_t = top2_t[..., 0] - top2_t[..., 1]
                        chunk_mask = np.ones(self.action_chunk_size, dtype=bool)
                        for step_i in range(1, self.action_chunk_size):
                            if confs_t[step_i] < self.adaptive_conf_threshold or margins_t[step_i] < self.adaptive_margin_threshold:
                                chunk_mask[step_i:] = False
                                break

                    if self.is_auto_exploration:
                        last_a = int(env_action[-1]) if self.action_chunk_size > 1 else int(env_action)
                        self._recent_actions.append(last_a)
                else:
                    if curr_scale != 1.0:
                        eff_scale = max(0.20, curr_scale)
                        scaled_std = dist.scale * eff_scale
                        dist_sample = torch.distributions.Normal(loc=dist.mean, scale=scaled_std)
                    else:
                        dist_sample = dist
                    action = dist_sample.sample()
                    if self.action_chunk_size > 1:
                        log_prob = dist_sample.log_prob(action).sum(dim=(-2, -1))
                    else:
                        log_prob = dist_sample.log_prob(action).sum(dim=-1)
                    env_action = action.squeeze(0).squeeze(0).cpu().numpy()
                    if isinstance(self.env.action_space, gym.spaces.Box):
                        env_action = np.clip(env_action, self.env.action_space.low, self.env.action_space.high)
                    chunk_mask = None
                lat_us = (time.perf_counter_ns() - t0) / 1000.0

                if self.tracker is not None:
                    if self.agent.is_discrete:
                        probs = dist.probs.squeeze(0).squeeze(0)
                        n_acts = probs.shape[-1]
                        top_probs, _ = torch.topk(probs, k=min(2, n_acts), dim=-1)
                        conf = float(top_probs[..., 0].mean().item())
                        ent = float(-torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item())
                        max_ent = float(np.log(n_acts)) if n_acts > 1 else 1.0
                        unc = float(np.clip(ent / max_ent, 0.0, 1.0))
                    else:
                        std = dist.scale.squeeze(0).squeeze(0).cpu().numpy()
                        std = np.clip(std, a_min=1e-6, a_max=100.0)
                        ent = float(0.5 * np.mean(np.sum(1.0 + np.log(2.0 * np.pi * (std ** 2)), axis=-1)))
                        var = float(np.mean(std ** 2))
                        unc = float(np.clip(2.0 / (1.0 + np.exp(-var / 0.5)) - 1.0, 0.0, 1.0))
                        conf = float(np.clip(1.0 - unc, 0.0, 1.0))

                    self.tracker.record_inference(
                        latency_us=lat_us,
                        uncertainty=unc,
                        confidence=conf,
                        entropy=ent,
                        calibration=curr_scale,
                    )

                # Environment step (com chunk_mask adaptativo)
                try:
                    next_obs_dict, reward, terminated, truncated, env_info = self.env.step(env_action, chunk_mask=chunk_mask)
                except TypeError:
                    next_obs_dict, reward, terminated, truncated, env_info = self.env.step(env_action)
                done = terminated or truncated
                last_env_info = env_info
                if stored_step_infos is not None:
                    stored_step_infos.append(env_info)

                if self.sil_enabled:
                    delta_v = (
                        current_obs_dict["delta_obs"].copy()
                        if "delta_obs" in current_obs_dict and current_obs_dict["delta_obs"] is not None
                        else None
                    )
                    prev_act_v = current_obs_dict["prev_action"]
                    if hasattr(prev_act_v, "copy"):
                        prev_act_v = prev_act_v.copy()
                    act_v = env_action
                    if hasattr(act_v, "copy"):
                        act_v = act_v.copy()
                    entry_s = {
                        "obs": current_obs_dict["obs"].copy(),
                        "delta_obs": delta_v,
                        "prev_action": prev_act_v,
                        "prev_reward": float(current_obs_dict["prev_reward"]),
                        "action": act_v,
                        "reward": float(reward),
                        "log_prob": float(log_prob.squeeze(0).squeeze(0).item()),
                        "value": float(value.squeeze(0).squeeze(0).item()),
                    }
                    self._active_single_trajectory.append(entry_s)

                if self.tracker is not None:
                    self.tracker.record_env_step(
                        reward=reward,
                        done=done,
                        info=env_info,
                        num_steps=self.action_chunk_size,
                    )

                self.curr_ep_return += reward
                self.total_steps += self.action_chunk_size

                if step_callback is not None:
                    step_callback()

                # Save transition into buffer: 'terminated' flags true terminal failure for GAE
                self.buffer.add(
                    obs_dict=current_obs_dict,
                    action=action.squeeze(0).squeeze(0),
                    reward=reward,
                    done=terminated,
                    episode_start=episode_start,
                    value=value.squeeze(0).squeeze(0),
                    log_prob=log_prob.squeeze(0).squeeze(0),
                    hx=current_hx,
                    env_scale=curr_scale,
                )

                if done:
                    self.episode_returns.append(self.curr_ep_return)
                    self.curr_ep_return = 0.0

                    traj_steps = self._active_single_trajectory
                    is_solved = bool(env_info.get("is_solved", False)) if isinstance(env_info, dict) else False
                    is_win = is_solved or (bool(terminated) and float(reward) >= 5.0)

                    if len(traj_steps) > 0:
                        cur_d = int(env_info.get("scramble_depth", env_info.get("current_depth", self._get_current_curriculum_depth()))) if isinstance(env_info, dict) else self._get_current_curriculum_depth()
                        self._process_completed_trajectory(traj_steps, is_win, cur_d)
                    self._active_single_trajectory.clear()

                    current_obs_dict, _ = self.env.reset()
                    new_d = self._get_current_curriculum_depth()
                    if new_d > self._rollout_tracked_depth:
                        self._rollout_tracked_depth = new_d
                        self.episode_returns.clear()
                    current_hx = None
                    episode_start = True
                else:
                    current_obs_dict = next_obs_dict
                    current_hx = next_hx
                    episode_start = False

            # Bootstrap value for last state
            input_dict = self._prepare_rollout_input(current_obs_dict)
            _, last_value, _ = self.agent.forward(input_dict, hx=current_hx, dones=None)

            if self.hindsight_replay:
                rew_arr = np.asarray(self.buffer.rewards, dtype=np.float32).reshape(1, -1)
                done_arr = np.asarray(self.buffer.dones, dtype=bool).reshape(1, -1)
                val_arr = np.asarray(self.buffer.values, dtype=np.float32).reshape(1, -1)
                step_infos_wrapped = [[inf] for inf in stored_step_infos] if stored_step_infos is not None else None
                self._apply_hindsight_relabelling(rew_arr, done_arr, val_arr, step_infos=step_infos_wrapped)
                self.buffer.rewards = [float(r) for r in rew_arr.flatten()]

            self.buffer.compute_gae(
                last_value=last_value.item(),
                last_done=terminated,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                action_chunk_size=self.action_chunk_size,
            )


            # Global advantage normalization across entire rollout buffer
            adv = self.buffer.advantages
            self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        mean_return = (
            float(np.mean(self.episode_returns)) if len(self.episode_returns) > 0 else 0.0
        )

        self._update_adaptive_exploration(mean_return, last_env_info)

        return current_obs_dict, current_hx, episode_start, mean_return

    def _get_actor_and_value_blocks(self) -> Tuple[Dict[str, List[nn.Parameter]], List[nn.Parameter]]:
        """Particiona os parâmetros treináveis nos 3 blocos do Caminho do Ator e na ValueHead isolada."""
        actor_blocks: Dict[str, List[nn.Parameter]] = {
            "FrontEnd": [p for p in self.agent.front_end.parameters() if p.requires_grad],
            "Trunk": [p for p in self.agent.trunk.parameters() if p.requires_grad],
            "PolicyHead": [p for p in self.agent.policy_head.parameters() if p.requires_grad],
        }
        value_params = [p for p in self.agent.value_head.parameters() if p.requires_grad]
        return actor_blocks, value_params

    def _get_on_policy_win_density(self) -> float:
        """Retorna a densidade recente de vitórias on-policy na fronteira (w_on in [0.0, 1.0])."""
        if len(self._recent_frontier_wins) > 0:
            return float(np.clip(np.mean(self._recent_frontier_wins), 0.0, 1.0))
        raw_env = getattr(self.env, "unwrapped", self.env)
        cur_sr = getattr(raw_env, "last_curriculum_success_rate", None)
        if cur_sr is not None:
            return float(np.clip(float(cur_sr), 0.0, 1.0))
        return 0.0

    def _apply_curvature_gated_projection_inplace(
        self,
        params: List[nn.Parameter],
        ref_list: List[torch.Tensor],
        alpha_target: float,
    ) -> Tuple[bool, float]:
        """Unificação Analítica Contínua M1 (PCGrad 90°) + M3 (GradVac Lei dos Senos) preservadora de norma.

        Quando alpha_target == 0.0 (fronteira extrema esparsa), reduz-se identicamente ao PCGrad Simétrico 90°:
            coef = -<g, r> / ||r||^2.
        Quando alpha_target > 0.0 (sinal on-policy denso), rotaciona suavemente pela Lei dos Senos até cos == alpha_target,
        alinhando os subespaços recorrentes sem encolher a norma original ||g||_2.
        """
        dot_val = 0.0
        norm_p_sq = 0.0
        norm_r_sq = 0.0
        for p, r in zip(params, ref_list):
            if p.grad is not None:
                g = p.grad.detach()
                dot_val += float(torch.sum(g * r).item())
                norm_p_sq += float(torch.sum(g * g).item())
            norm_r_sq += float(torch.sum(r * r).item())

        norm_p = math.sqrt(max(0.0, norm_p_sq))
        norm_r = math.sqrt(max(0.0, norm_r_sq))
        if norm_p < 1e-12 or norm_r < 1e-12:
            return False, 0.0

        cos_val = dot_val / (norm_p * norm_r + 1e-12)
        cos_c = max(-1.0, min(1.0, cos_val))
        alpha_t = max(0.0, min(self.gradvac_max_alpha, float(alpha_target)))

        if cos_c < alpha_t:
            sin_c = math.sqrt(max(0.0, 1.0 - cos_c * cos_c))
            sin_t = math.sqrt(max(1e-12, 1.0 - alpha_t * alpha_t))
            coef = norm_p * (alpha_t * sin_c - cos_c * sin_t) / (norm_r * sin_t + 1e-12)
            new_norm_sq = 0.0
            for p, r in zip(params, ref_list):
                if p.grad is None:
                    p.grad = (coef * r).clone()
                else:
                    p.grad.add_(r, alpha=coef)
                new_norm_sq += float(torch.sum(p.grad.detach() * p.grad.detach()).item())
            new_norm = math.sqrt(max(0.0, new_norm_sq))
            if new_norm > 1e-6 * norm_p and sin_c > 1e-4:
                mult = min(norm_p / new_norm, 5.0)
                if abs(mult - 1.0) > 1e-7:
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(mult)
            return True, cos_val
        return False, cos_val

    def _compute_unified_memory_loss(
        self,
        target_depth: Optional[int],
        only_full_wins: bool = False,
    ) -> Tuple[Optional[torch.Tensor], float, float]:
        """Calcula a perda unificada de memória off-policy (SIL-Policy + DPO contrastivo).

        A ValueHead permanece 100% isolada (sil_val_sq.detach(), sem sil_val_loss), garantindo
        gradiente EXATAMENTE None no Crítico e zero viés de otimismo em horizontes profundos.
        Quando only_full_wins=True (usado para extrair o escudo g_mem_ref nos mini-batches PPO),
        considera apenas vitórias terminais completas verificadas (is_full_win=True), impedindo que
        fatias parciais de episódios falhos/mínimos locais bloqueiem o desaprender corretivo do PPO.
        """
        if not self.sil_enabled:
            return None, 0.0, 0.0

        total_mem_loss: Optional[torch.Tensor] = None
        sil_p_loss = 0.0
        dpo_p_loss = 0.0

        avail_count = (
            self.sil_buffer.count_full_wins()
            if (only_full_wins and hasattr(self.sil_buffer, "count_full_wins"))
            else len(self.sil_buffer)
        )
        if avail_count >= self.sil_min_samples and self.sil_coef > 0.0:
            sil_batch = self.sil_buffer.sample_batch(
                batch_size=self.sil_batch_size,
                target_depth=target_depth,
                only_full_wins=only_full_wins,
                tournament_size=self.sil_tournament_size,
            )
            if sil_batch is not None:
                sil_in = {
                    "obs": sil_batch["obs"],
                    "prev_action": sil_batch["prev_actions"],
                    "prev_reward": sil_batch["prev_rewards"],
                }
                if not self.agent.is_visual and sil_batch["delta_obs"] is not None:
                    sil_in["delta_obs"] = sil_batch["delta_obs"]

                sil_dist, sil_val, _ = self.agent(sil_in, hx=sil_batch["hx"], dones=sil_batch["dones"])
                sil_val_sq = sil_val.squeeze(-1)
                sil_ret = sil_batch["returns"]
                sil_mask = sil_batch["mask"]

                # Surpresa positiva estritamente com crítico desconectado do grafo (100% PPO Pure Critic)
                sil_adv = torch.clamp(sil_ret - sil_val_sq.detach(), min=0.0)

                if self.agent.is_discrete:
                    if self.action_chunk_size > 1:
                        sil_logp = sil_dist.log_prob(sil_batch["actions"]).sum(dim=-1)
                    else:
                        sil_logp = sil_dist.log_prob(sil_batch["actions"])
                else:
                    if self.action_chunk_size > 1:
                        sil_logp = sil_dist.log_prob(sil_batch["actions"]).sum(dim=(-2, -1))
                    else:
                        sil_logp = sil_dist.log_prob(sil_batch["actions"]).sum(dim=-1)

                denom = sil_mask.sum().clamp(min=1.0)
                sil_policy_loss = -(sil_logp * sil_adv * sil_mask).sum() / denom
                sil_p_loss = float(sil_policy_loss.item())
                total_mem_loss = self.sil_coef * sil_policy_loss

        if self.sil_dpo_enabled and self.dpo_coef > 0.0:
            pairs = self.sil_buffer.sample_contrastive_pairs(
                batch_size=self.dpo_batch_size,
                target_depth=target_depth,
                only_full_wins=only_full_wins,
                tournament_size=self.sil_tournament_size,
            )
            if pairs is None:
                # Modo sem armazenamento de falhas: contrasta vitórias/fatias de sucesso do buffer
                # contra chunks negativos frescos (vantagem GAE < 0) do rollout on-policy atual
                win_cand = self.sil_buffer.sample_batch(
                    batch_size=self.dpo_batch_size,
                    target_depth=target_depth,
                    only_full_wins=only_full_wins,
                    tournament_size=self.sil_tournament_size,
                )
                if win_cand is not None:
                    k_win = int(win_cand["obs"].shape[0])
                    loss_cand = self.buffer.sample_negative_chunks(batch_size=k_win)
                    if loss_cand is not None:
                        pairs = (win_cand, loss_cand)

            if pairs is not None:
                win_batch, loss_batch = pairs
                win_in = {
                    "obs": win_batch["obs"],
                    "prev_action": win_batch["prev_actions"],
                    "prev_reward": win_batch["prev_rewards"],
                }
                if not self.agent.is_visual and win_batch["delta_obs"] is not None:
                    win_in["delta_obs"] = win_batch["delta_obs"]

                loss_in = {
                    "obs": loss_batch["obs"],
                    "prev_action": loss_batch["prev_actions"],
                    "prev_reward": loss_batch["prev_rewards"],
                }
                if not self.agent.is_visual and loss_batch["delta_obs"] is not None:
                    loss_in["delta_obs"] = loss_batch["delta_obs"]

                win_dist, _, _ = self.agent(win_in, hx=win_batch["hx"], dones=win_batch["dones"])
                loss_dist, _, _ = self.agent(loss_in, hx=loss_batch["hx"], dones=loss_batch["dones"])

                if self.agent.is_discrete:
                    if self.action_chunk_size > 1:
                        win_logp = win_dist.log_prob(win_batch["actions"]).sum(dim=-1)
                        loss_logp = loss_dist.log_prob(loss_batch["actions"]).sum(dim=-1)
                    else:
                        win_logp = win_dist.log_prob(win_batch["actions"])
                        loss_logp = loss_dist.log_prob(loss_batch["actions"])
                else:
                    if self.action_chunk_size > 1:
                        win_logp = win_dist.log_prob(win_batch["actions"]).sum(dim=(-2, -1))
                        loss_logp = loss_dist.log_prob(loss_batch["actions"]).sum(dim=(-2, -1))
                    else:
                        win_logp = win_dist.log_prob(win_batch["actions"]).sum(dim=-1)
                        loss_logp = loss_dist.log_prob(loss_batch["actions"]).sum(dim=-1)

                win_ref_logp = win_batch["old_log_probs"]
                loss_ref_logp = loss_batch["old_log_probs"]
                win_mask = win_batch["mask"]
                loss_mask = loss_batch["mask"]

                win_len = win_mask.sum(dim=-1).clamp(min=1.0)
                loss_len = loss_mask.sum(dim=-1).clamp(min=1.0)

                h_w = ((win_logp - win_ref_logp) * win_mask).sum(dim=-1) / win_len
                h_l = ((loss_logp - loss_ref_logp) * loss_mask).sum(dim=-1) / loss_len

                b_eff = min(int(h_w.shape[0]), int(h_l.shape[0]))
                if b_eff > 0:
                    dpo_logits = self.dpo_beta * (h_w[:b_eff] - h_l[:b_eff])
                    dpo_loss = -torch.nn.functional.logsigmoid(dpo_logits).mean()
                    dpo_p_loss = float(dpo_loss.item())
                    scaled_dpo = self.dpo_coef * dpo_loss
                    total_mem_loss = scaled_dpo if total_mem_loss is None else (total_mem_loss + scaled_dpo)

        return total_mem_loss, sil_p_loss, dpo_p_loss

    def train_epoch(self) -> Dict[str, float]:
        """Executa época PPO acoplada à Cirurgia de Gradiente Simétrica por Curvatura no Caminho do Ator.

        - ValueHead permanece 100% exclusiva do PPO on-policy (zero gradiente de SIL/DPO).
        - FrontEnd, Trunk e PolicyHead recebem projeção bilateral preservadora de norma com
          alpha_t = clip(EMA_cos_b, 0.0, gradvac_max_alpha * w_on), unificando continuamente
          GradVac (em regime denso) e PCGrad 90° (em fronteira extrema esparsa).
        """
        self.agent.train()
        total_policy_loss = 0.0
        total_val_loss = 0.0
        total_ent_loss = 0.0
        total_floor_loss = 0.0
        n_updates = 0
        conflict_hits = 0
        conflict_checks = 0

        raw_env = getattr(self.env, "unwrapped", self.env)
        target_depth = getattr(raw_env, "current_depth", getattr(self.env, "current_depth", None))

        actor_blocks, value_params = self._get_actor_and_value_blocks()
        w_on = self._get_on_policy_win_density()
        alpha_cap = self.gradvac_max_alpha * w_on

        ppo_actor_accum: Dict[str, Optional[List[torch.Tensor]]] = {
            b_name: None for b_name in actor_blocks
        }

        # Extrai referência inicial do gradiente unificado de memória (SIL + DPO) APENAS sobre vitórias
        # completas verificadas (only_full_wins=True) para proteger todos os mini-batches PPO contra
        # esquecimento catastrófico sem jamais bloquear o PPO de escapar de mínimos locais / falhas parciais.
        g_mem_ref: Optional[Dict[str, Optional[List[torch.Tensor]]]] = None
        if self.sil_enabled and self.pcgrad_enabled:
            self.optimizer.zero_grad()
            mem_ref_loss, _, _ = self._compute_unified_memory_loss(target_depth, only_full_wins=True)
            if mem_ref_loss is not None:
                mem_ref_loss.backward()
                g_mem_ref = {}
                for b_name, params in actor_blocks.items():
                    if params and any(p.grad is not None for p in params):
                        g_mem_ref[b_name] = [
                            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                            for p in params
                        ]
                    else:
                        g_mem_ref[b_name] = None
            self.optimizer.zero_grad()

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_chunks_generator(self.chunk_batch_size):
                input_dict = {
                    "obs": batch.obs,
                    "prev_action": batch.prev_actions,
                    "prev_reward": batch.prev_rewards,
                }
                if not self.agent.is_visual:
                    input_dict["delta_obs"] = batch.delta_obs

                dist, values, _, val_logits = self.agent(
                    input_dict, hx=batch.hx, dones=batch.dones, return_value_logits=True
                )
                values_sq = values.squeeze(-1)  # [B, T]

                if self.agent.is_discrete:
                    if getattr(batch, "env_scales", None) is not None:
                        eff_scales = batch.env_scales.clamp(min=0.02, max=1.00)
                        if self.action_chunk_size > 1:
                            eff_scales = eff_scales.unsqueeze(-1).unsqueeze(-1)
                        else:
                            eff_scales = eff_scales.unsqueeze(-1)
                        scaled_logits = torch.nan_to_num(dist.logits / eff_scales, nan=0.0, posinf=35.0, neginf=-35.0).clamp(min=-35.0, max=35.0)
                        dist_eval = torch.distributions.Categorical(logits=scaled_logits)
                    elif self.exploration_scale != 1.0:
                        eff_scale = max(0.08, float(self.exploration_scale))
                        scaled_logits = torch.nan_to_num(dist.logits / eff_scale, nan=0.0, posinf=35.0, neginf=-35.0).clamp(min=-35.0, max=35.0)
                        dist_eval = torch.distributions.Categorical(logits=scaled_logits)
                    else:
                        dist_eval = dist
                    if self.action_chunk_size > 1:
                        new_log_probs = dist_eval.log_prob(batch.actions).sum(dim=-1)  # [B, T]
                        entropy = dist.entropy().mean(dim=-1)                          # [B, T]
                    else:
                        new_log_probs = dist_eval.log_prob(batch.actions)              # [B, T]
                        entropy = dist.entropy()                                       # [B, T]
                else:
                    if getattr(batch, "env_scales", None) is not None:
                        eff_scales = batch.env_scales.clamp(min=0.02, max=1.00)
                        if self.action_chunk_size > 1:
                            eff_scales = eff_scales.unsqueeze(-1).unsqueeze(-1)
                        else:
                            eff_scales = eff_scales.unsqueeze(-1)
                        scaled_std = dist.scale * eff_scales
                        dist_eval = torch.distributions.Normal(loc=dist.mean, scale=scaled_std)
                    elif self.exploration_scale != 1.0:
                        eff_scale = max(0.08, float(self.exploration_scale))
                        scaled_std = dist.scale * eff_scale
                        dist_eval = torch.distributions.Normal(loc=dist.mean, scale=scaled_std)
                    else:
                        dist_eval = dist
                    if self.action_chunk_size > 1:
                        new_log_probs = dist_eval.log_prob(batch.actions).sum(dim=(-2, -1))
                        entropy = dist.entropy().sum(dim=(-2, -1)) / self.action_chunk_size
                    else:
                        new_log_probs = dist_eval.log_prob(batch.actions).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1)

                log_ratio = new_log_probs - batch.old_log_probs
                log_ratio = torch.clamp(log_ratio, min=-10.0, max=2.5)
                ratio = torch.exp(log_ratio)

                norm_adv = batch.advantages

                surr1 = ratio * norm_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * norm_adv
                )
                surr = torch.min(surr1, surr2)
                policy_loss = -torch.clamp(surr, min=-15.0, max=15.0).mean()

                val_loss = self.agent.value_head.loss(val_logits, batch.returns)

                batch_entropy = entropy.mean()
                entropy_loss = -batch_entropy

                floor_loss = torch.tensor(0.0, device=self.device)
                floor_active = False
                if self.entropy_floor_enabled and batch_entropy < self.entropy_floor:
                    deficit = (self.entropy_floor - batch_entropy) / max(1e-4, self.entropy_floor)
                    floor_loss = self.entropy_floor_coef * (deficit ** 2)
                    floor_active = True

                loss = (
                    policy_loss
                    + self.value_coef * val_loss
                    + self.entropy_coef * entropy_loss
                    + floor_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                # Acumula direção média do PPO nos blocos do Ator antes da projeção
                if self.sil_enabled and self.pcgrad_enabled:
                    for b_name, params in actor_blocks.items():
                        if not params or not any(p.grad is not None for p in params):
                            continue
                        cur_acc = ppo_actor_accum[b_name]
                        if cur_acc is None:
                            ppo_actor_accum[b_name] = [
                                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                                for p in params
                            ]
                        else:
                            for acc_t, p in zip(cur_acc, params):
                                if p.grad is not None:
                                    acc_t.add_(p.grad.detach())

                # Proteção Simétrica em todos os mini-batches do Ator (ValueHead permanece 100% intocada!)
                if g_mem_ref is not None and self.pcgrad_enabled:
                    for b_name, params in actor_blocks.items():
                        gm_list = g_mem_ref.get(b_name, None)
                        if not params or gm_list is None:
                            continue
                        alpha_t = min(alpha_cap, max(0.0, self._gradvac_ema.get(b_name, 0.0)))
                        conflict_checks += 1
                        is_c, cos_obs = self._apply_curvature_gated_projection_inplace(
                            params, gm_list, alpha_t
                        )
                        if is_c:
                            conflict_hits += 1
                        ema_prev = self._gradvac_ema.get(b_name, 0.0)
                        self._gradvac_ema[b_name] = (
                            self.gradvac_ema_beta * ema_prev + (1.0 - self.gradvac_ema_beta) * cos_obs
                        )

                # Telemetry: calculate per-block gradient norms
                fe_grads = [p.grad for p in self.agent.front_end.parameters() if p.grad is not None]
                trunk_grads = [p.grad for p in self.agent.trunk.parameters() if p.grad is not None]
                head_grads = [p.grad for p in self.agent.policy_head.parameters() if p.grad is not None]

                grad_norms = {
                    "FrontEnd": float(torch.norm(torch.stack([torch.norm(g) for g in fe_grads])).item()) if fe_grads else 0.0,
                    "Trunk": float(torch.norm(torch.stack([torch.norm(g) for g in trunk_grads])).item()) if trunk_grads else 0.0,
                    "PolicyHead": float(torch.norm(torch.stack([torch.norm(g) for g in head_grads])).item()) if head_grads else 0.0,
                }

                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                clip_frac = float(((ratio - 1.0).abs() > self.clip_range).float().mean().item())
                if self.tracker is not None:
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    self.tracker.record_training_epoch(
                        policy_loss=policy_loss.item(),
                        value_loss=val_loss.item(),
                        clip_fraction=clip_frac,
                        grad_norms=grad_norms,
                        lr=current_lr,
                        entropy_floor_loss=float(floor_loss.item()),
                        entropy_floor_active=floor_active,
                        effective_entropy_coef=float(self.entropy_coef),
                    )

                total_policy_loss += policy_loss.item()
                total_val_loss += val_loss.item()
                total_ent_loss += entropy_loss.item()
                total_floor_loss += float(floor_loss.item())
                n_updates += 1

        # Passo Unificado de Consolidação de Memória Off-Policy (SIL-Policy + c-SIL/DPO)
        # Substitui integralmente os antigos passos isolados de SIL e DPO, mantendo ValueHead com gradiente None.
        sil_p_loss = 0.0
        sil_v_loss = 0.0
        dpo_p_loss = 0.0
        if self.sil_enabled:
            self.optimizer.zero_grad()
            mem_loss, sil_p_loss, dpo_p_loss = self._compute_unified_memory_loss(target_depth)
            if mem_loss is not None:
                mem_loss.backward()
                # Garante isolamento absoluto bit-a-bit da ValueHead (zero vazamento de momentum no Adam)
                for vp in value_params:
                    vp.grad = None

                if self.pcgrad_enabled and n_updates > 0:
                    inv_n = 1.0 / float(n_updates)
                    for b_name, params in actor_blocks.items():
                        gp_acc = ppo_actor_accum.get(b_name, None)
                        if not params or gp_acc is None:
                            continue
                        gp_mean_list = [t * inv_n for t in gp_acc]
                        alpha_t = min(alpha_cap, max(0.0, self._gradvac_ema.get(b_name, 0.0)))
                        conflict_checks += 1
                        is_c, cos_obs = self._apply_curvature_gated_projection_inplace(
                            params, gp_mean_list, alpha_t
                        )
                        if is_c:
                            conflict_hits += 1
                        ema_prev = self._gradvac_ema.get(b_name, 0.0)
                        self._gradvac_ema[b_name] = (
                            self.gradvac_ema_beta * ema_prev + (1.0 - self.gradvac_ema_beta) * cos_obs
                        )

                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

        if self.tracker is not None:
            self.tracker.record_sil_step(
                sil_trajectories=len(self.sil_buffer),
                sil_policy_loss=sil_p_loss,
                sil_value_loss=sil_v_loss,
                sil_dpo_loss=dpo_p_loss,
                sil_failures=len(self.sil_buffer.failures),
            )

        # Cálculo da entropia média observada nos mini-batches
        mean_ent_val = float(-(total_ent_loss / max(1, n_updates))) if n_updates > 0 else 0.0

        # Dynamic Entropy Floor: Adaptação e Re-aquecimento do Coeficiente de Entropia
        if self.adaptive_entropy_coef and n_updates > 0:
            if mean_ent_val < self.entropy_floor:
                deficit_ratio = (self.entropy_floor - mean_ent_val) / max(1e-4, self.entropy_floor)
                # Re-aquece proporcionalmente ao déficit de exploração
                new_coef = self.entropy_coef * (1.0 + 0.5 * deficit_ratio)
                self.entropy_coef = float(np.clip(new_coef, self.min_entropy_coef, self.max_entropy_coef))
            else:
                # Decaimento suave de volta para o base_entropy_coef
                if self.entropy_coef > self.base_entropy_coef:
                    self.entropy_coef = max(self.base_entropy_coef, self.entropy_coef * 0.95)
                elif self.entropy_coef < self.base_entropy_coef:
                    self.entropy_coef = min(self.base_entropy_coef, self.entropy_coef * 1.05)

        if self.scheduler is not None:
            self.scheduler.step()

        ster_prototypes = 0
        ster_rehab = 0

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_val_loss / max(1, n_updates),
            "entropy_loss": total_ent_loss / max(1, n_updates),
            "entropy_floor_loss": total_floor_loss / max(1, n_updates),
            "effective_entropy_coef": float(self.entropy_coef),
            "sil_policy_loss": sil_p_loss,
            "sil_value_loss": sil_v_loss,
            "sil_dpo_loss": dpo_p_loss,
            "sil_trajectories": len(self.sil_buffer),
            "sil_failures": len(self.sil_buffer.failures),
            "sterility_prototypes": ster_prototypes,
            "sterility_rehabilitated": ster_rehab,
            "pcgrad_conflict_rate": float(conflict_hits / max(1, conflict_checks)),
            "pcgrad_alpha_cap": float(alpha_cap),
            "pcgrad_win_density": float(w_on),
        }

    def train(
        self,
        max_steps: int = 40000,
        target_return: Optional[float] = 475.0,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, float], None]] = None,
        step_callback: Optional[Callable[[], None]] = None,
        verbose: bool = True,
    ) -> float:
        """Trains the agent until max_steps or target_return is reached."""
        obs_dict, _ = self.env.reset(seed=seed)
        hx: Optional[torch.Tensor] = None
        episode_start = True

        remaining_steps = max(1, max_steps - self.total_steps)
        total_updates = max(1, remaining_steps // self.rollout_steps)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_updates,
        )

        best_mean_return = -float("inf")
        best_depth_mean_return = -float("inf")
        best_state_depth = self._get_current_curriculum_depth()
        self.best_state_depth = best_state_depth
        self.best_depth_mean_return = best_depth_mean_return
        self._rollout_tracked_depth = best_state_depth
        iteration = 0

        while self.total_steps < max_steps:
            iteration += 1
            obs_dict, hx, episode_start, mean_return = self.collect_rollouts(
                obs_dict, hx, episode_start, step_callback=step_callback
            )
            metrics = self.train_epoch()

            cur_depth = self._get_current_curriculum_depth()
            if cur_depth > best_state_depth:
                # Promoção curricular: o novo nível mais profundo tem precedência hierárquica absoluta!
                # Ancora imediatamente os pesos recém-promovidos para evitar rollback para níveis anteriores.
                best_state_depth = cur_depth
                self.best_state_depth = best_state_depth
                best_depth_mean_return = max(0.0, mean_return) if len(self.episode_returns) >= 5 else 0.0
                self.best_depth_mean_return = best_depth_mean_return
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in self.agent.state_dict().items()
                }

            if len(self.episode_returns) > 0:
                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    self.best_mean_return = best_mean_return
                if cur_depth >= best_state_depth and mean_return > best_depth_mean_return:
                    best_depth_mean_return = mean_return
                    self.best_depth_mean_return = best_depth_mean_return
                    self.best_state_depth = cur_depth
                    self.best_state_dict = {
                        k: v.cpu().clone() for k, v in self.agent.state_dict().items()
                    }
                if self.tracker is not None:
                    self.tracker.metrics.best_mean_return = best_mean_return

            if verbose and iteration % 2 == 0:
                log_parts = [
                    f"[Step {self.total_steps:6d}/{max_steps}] Mean Return (last 20 ep): {mean_return:.2f}"
                ]
                if self.tracker is not None and self.tracker.metrics.fps > 0:
                    log_parts.append(f"FPS: {self.tracker.metrics.fps:.1f}")
                if self.tracker is not None and len(self.tracker.metrics.confidences) > 0:
                    mean_conf = float(np.mean(self.tracker.metrics.confidences))
                    log_parts.append(f"Confiança: {mean_conf*100:.1f}%")

                calib = float(self._adaptive_scale if self.is_auto_exploration else self.exploration_scale)
                if self.tracker is not None and len(self.tracker.metrics.calibrations) > 0:
                    calib = float(self.tracker.metrics.calibrations[-1])
                log_parts.append(f"Calibração: {calib:.2f}")

                if self.tracker is not None and len(self.tracker.metrics.step_latencies_us) > 0:
                    lat_p50 = float(np.percentile(self.tracker.metrics.step_latencies_us, 50))
                    if lat_p50 >= 1000.0:
                        log_parts.append(f"Latência: {lat_p50/1000.0:.1f}ms")
                    else:
                        log_parts.append(f"Latência: {lat_p50:.0f}µs")

                raw_env = getattr(self.env, "unwrapped", self.env)
                if hasattr(raw_env, "current_depth"):
                    max_d = getattr(raw_env, "max_depth", 5)
                    log_parts.append(f"Profundidade: {raw_env.current_depth}/{max_d}")
                elif hasattr(raw_env, "current_level"):
                    max_l = getattr(raw_env, "max_level", 4)
                    log_parts.append(f"Nível: {raw_env.current_level}/{max_l}")

                if self.sil_enabled and getattr(self, "sil_dpo_enabled", False):
                    dpo_l = self.tracker.metrics.sil_dpo_losses[-1] if (self.tracker and self.tracker.metrics.sil_dpo_losses) else 0.0
                    n_fail = getattr(self.tracker.metrics, "sil_failures", len(self.sil_buffer.failures)) if self.tracker else len(self.sil_buffer.failures)
                    log_parts.append(f"DPO Loss: {dpo_l:.3f} ({n_fail} falhas)")

                print(" | ".join(log_parts))

            if callback is not None:
                callback(self.total_steps, mean_return)

            if (
                target_return is not None
                and len(self.episode_returns) >= 20
                and mean_return >= target_return
            ):
                if verbose:
                    print(
                        f"Target return {target_return} achieved! "
                        f"({mean_return:.2f} at {self.total_steps} steps)"
                    )
                break

        if getattr(self, "best_state_dict", None) is not None:
            self.agent.load_state_dict(self.best_state_dict)
            if verbose:
                restored_ret = best_depth_mean_return if best_depth_mean_return > -float("inf") else best_mean_return
                if best_state_depth > 0:
                    print(
                        f"\n🏆 [Best State Dict Restored] Modelo restaurado no ápice da Profundidade D{best_state_depth} "
                        f"(Melhor Retorno no Nível: {restored_ret:.2f} | Pico Global: {best_mean_return:.2f})"
                    )
                else:
                    print(f"\n🏆 [Best State Dict Restored] Modelo restaurado no ápice de performance (Melhor Retorno: {best_mean_return:.2f})")

        self.best_mean_return = best_mean_return
        self.final_return = float(np.mean(self.episode_returns)) if len(self.episode_returns) > 0 else 0.0
        return self.final_return
