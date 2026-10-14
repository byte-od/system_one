from collections import deque
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
from system1_engine.training.buffer import RecurrentRolloutBuffer


class RecurrentPPOTrainer:
    """Pure PyTorch Recurrent PPO Trainer with GRU hidden state preservation and BPTT."""

    def __init__(
        self,
        agent: UniversalS1Agent,
        env: UniversalS1Wrapper,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.05,
        entropy_coef: float = 0.001,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        rollout_steps: int = 1024,
        chunk_length: int = 16,
        chunk_batch_size: int = 16,
        device: Union[str, torch.device] = "cpu",
        tracker: Optional[LiveStatsTracker] = None,
        exploration_scale: Union[float, str] = 1.0,
        hindsight_replay: bool = False,
        hindsight_bonus: float = 2.0,
        hindsight_min_gain: float = 0.02,
    ) -> None:
        self.agent = agent.to(device)
        self.env = env
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

        # Suporte a exploração automática / homeostática no treino PPO (sem limite artificial, 0 à 1)
        self._min_adaptive_scale = 0.08
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
            self.exploration_scale = float(np.clip(float(exploration_scale), 0.08, 1.0))
            self._adaptive_scale = self.exploration_scale
            self._reward_ema = None
            self._stagnation_count = 0
            self._recent_actions = deque(maxlen=6)

        # Optimize only parameters that require grad (respects frozen trunk)
        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "No trainable parameters found!"
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate, eps=1e-5)
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None

        # Rollout buffer
        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        obs_shape = env.observation_space["obs"].shape
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

        # Buffers de tensores pré-alocados no dispositivo para eliminar overhead de heap e alocações repetidas
        self._buf_obs = torch.zeros((1, 1, *obs_shape), dtype=torch.float32, device=self.device)
        if is_discrete:
            self._buf_act = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        else:
            self._buf_act = torch.zeros((1, 1, *act_shape), dtype=torch.float32, device=self.device)
        self._buf_rew = torch.zeros((1, 1, 1), dtype=torch.float32, device=self.device)
        if not env.is_visual:
            self._buf_delta: Optional[torch.Tensor] = torch.zeros((1, 1, *obs_shape), dtype=torch.float32, device=self.device)
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
                self._vec_buf_delta: Optional[torch.Tensor] = torch.zeros((self.num_envs, 1, *obs_shape), dtype=torch.float32, device=self.device)
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

        S = self.steps_per_env
        E = self.num_envs
        T = self.chunk_length
        K = S // T

        obs_shape = self.buffer.obs_shape
        stored_obs = np.empty((S, E, *obs_shape), dtype=np.float32)
        stored_delta = np.empty((S, E, *obs_shape), dtype=np.float32) if not self.env.is_visual else None
        if self.agent.is_discrete:
            stored_prev_act = np.empty((S, E), dtype=np.int64)
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

                    # Vectorized Action Quality Filter & Inverse Move Pruning (Melhoria 3)
                    if num_acts == 12:
                        for e in range(E):
                            e_hist = self._vec_recent_actions[e]
                            if len(e_hist) >= 1:
                                last_a = e_hist[-1]
                                inv_a = (last_a + 1) if (last_a % 2 == 0) else (last_a - 1)
                                penalized_logits[e, 0, inv_a] -= 4.0
                            if len(e_hist) >= 3 and e_hist[-1] == e_hist[-3]:
                                penalized_logits[e, 0, e_hist[-2]] -= 3.0
                    else:
                        for e in range(E):
                            e_hist = self._vec_recent_actions[e]
                            if len(e_hist) >= 3 and e_hist[-1] == e_hist[-3]:
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
                        scaled_logits = torch.nan_to_num(penalized_logits / env_scales, nan=0.0, posinf=35.0, neginf=-35.0).clamp(min=-35.0, max=35.0)
                    elif curr_scale != 1.0:
                        eff_scale = max(0.02, curr_scale)
                        scaled_logits = torch.nan_to_num(penalized_logits / eff_scale, nan=0.0, posinf=35.0, neginf=-35.0).clamp(min=-35.0, max=35.0)
                    else:
                        scaled_logits = penalized_logits

                    dist_sample = torch.distributions.Categorical(logits=scaled_logits)
                    action = dist_sample.sample()
                    log_prob = dist_sample.log_prob(action)
                    env_action = action.squeeze(1).cpu().numpy()

                    # Atualiza histórico de ações por ambiente
                    for e in range(E):
                        self._vec_recent_actions[e].append(int(env_action[e]))
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
                        scaled_std = dist.scale * env_scales
                    elif curr_scale != 1.0:
                        eff_scale = max(0.02, curr_scale)
                        scaled_std = dist.scale * eff_scale
                    else:
                        scaled_std = dist.scale

                    dist_sample = torch.distributions.Normal(loc=dist.mean, scale=scaled_std)
                    action = dist_sample.sample()
                    log_prob = dist_sample.log_prob(action).sum(dim=-1)
                    env_action = action.squeeze(1).cpu().numpy()
                    if isinstance(self.env.action_space, gym.spaces.Box):
                        env_action = np.clip(env_action, self.env.action_space.low, self.env.action_space.high)

                lat_us = (time.perf_counter_ns() - t0) / 1000.0

                if self.tracker is not None:
                    if self.agent.is_discrete:
                        probs = dist.probs[0, 0]
                        n_acts = probs.shape[-1]
                        top_probs, _ = torch.topk(probs, k=min(2, n_acts))
                        conf = float(top_probs[0].item())
                        ent = float(-torch.sum(probs * torch.log(probs + 1e-8)).item())
                        max_ent = float(np.log(n_acts)) if n_acts > 1 else 1.0
                        unc = float(np.clip(ent / max_ent, 0.0, 1.0))
                    else:
                        std = dist.scale[0, 0].cpu().numpy()
                        std = np.clip(std, a_min=1e-6, a_max=100.0)
                        ent = float(0.5 * np.sum(1.0 + np.log(2.0 * np.pi * (std ** 2))))
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

                # Environment step vetorizado
                next_obs_dict, rewards, terminateds, truncateds, env_infos = self.env.step(env_action)
                dones = terminateds | truncateds
                if stored_env_infos is not None:
                    stored_env_infos.append(env_infos)


                if self.tracker is not None:
                    primary_info = env_infos[0] if isinstance(env_infos, list) and len(env_infos) > 0 else {}
                    self.tracker.record_env_step(
                        reward=float(rewards[0]),
                        done=bool(dones[0]),
                        info=primary_info,
                        num_steps=self.num_envs,
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
                stored_val[step] = value.squeeze(1).squeeze(1).cpu().numpy()
                stored_logp[step] = log_prob.squeeze(1).cpu().numpy()

                curr_ep_returns += rewards
                self.total_steps += E

                done_indices = np.where(dones)[0]
                if len(done_indices) > 0:
                    for d_i in done_indices:
                        self.episode_returns.append(float(curr_ep_returns[d_i]))
                        curr_ep_returns[d_i] = 0.0
                        self._vec_recent_actions[d_i].clear()
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

            delta = rew_es[:, t] + self.gamma * next_val * next_non_terminal - val_es[:, t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            adv_es[:, t] = last_gae

        ret_es = adv_es + val_es

        # Transposição [S, E, ...] -> [E, S, ...] e linearização ordenada por ambiente
        flat_obs = stored_obs.transpose(1, 0, *range(2, stored_obs.ndim)).reshape(E * S, *obs_shape)
        flat_delta = (
            stored_delta.transpose(1, 0, *range(2, stored_delta.ndim)).reshape(E * S, *obs_shape)
            if stored_delta is not None
            else None
        )
        if self.agent.is_discrete:
            flat_prev_act = stored_prev_act.T.reshape(E * S)
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
        if not self.is_auto_exploration:
            return

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

        self.exploration_scale = float(self._adaptive_scale)

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
                    # Quebra de ciclos (Cycle Breaking) sob estagnação no modo automático
                    if self.is_auto_exploration and self._stagnation_count >= 3 and len(self._recent_actions) >= 3:
                        if self._recent_actions[-1] == self._recent_actions[-3]:
                            avoid_act = self._recent_actions[-2]
                            penalized_logits = logits.clone()
                            penalized_logits[..., avoid_act] -= 3.0
                            logits = penalized_logits

                    if curr_scale != 1.0:
                        eff_scale = max(0.02, curr_scale)
                        scaled_logits = torch.nan_to_num(logits / eff_scale, nan=0.0, posinf=35.0, neginf=-35.0).clamp(min=-35.0, max=35.0)
                        dist_sample = torch.distributions.Categorical(logits=scaled_logits)
                    elif self.is_auto_exploration and self._stagnation_count >= 3:
                        dist_sample = torch.distributions.Categorical(logits=logits)
                    else:
                        dist_sample = dist
                    action = dist_sample.sample()
                    log_prob = dist_sample.log_prob(action)
                    env_action = int(action.item())
                    if self.is_auto_exploration:
                        self._recent_actions.append(env_action)
                else:
                    if curr_scale != 1.0:
                        eff_scale = max(0.02, curr_scale)
                        scaled_std = dist.scale * eff_scale
                        dist_sample = torch.distributions.Normal(loc=dist.mean, scale=scaled_std)
                    else:
                        dist_sample = dist
                    action = dist_sample.sample()
                    log_prob = dist_sample.log_prob(action).sum(dim=-1)
                    env_action = action.squeeze(0).squeeze(0).cpu().numpy()
                    if isinstance(self.env.action_space, gym.spaces.Box):
                        env_action = np.clip(env_action, self.env.action_space.low, self.env.action_space.high)
                lat_us = (time.perf_counter_ns() - t0) / 1000.0

                if self.tracker is not None:
                    if self.agent.is_discrete:
                        probs = dist.probs.squeeze(0).squeeze(0)
                        n_acts = probs.shape[-1]
                        top_probs, _ = torch.topk(probs, k=min(2, n_acts))
                        conf = float(top_probs[0].item())
                        ent = float(-torch.sum(probs * torch.log(probs + 1e-8)).item())
                        max_ent = float(np.log(n_acts)) if n_acts > 1 else 1.0
                        unc = float(np.clip(ent / max_ent, 0.0, 1.0))
                    else:
                        std = dist.scale.squeeze(0).squeeze(0).cpu().numpy()
                        std = np.clip(std, a_min=1e-6, a_max=100.0)
                        ent = float(0.5 * np.sum(1.0 + np.log(2.0 * np.pi * (std ** 2))))
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

                # Environment step
                next_obs_dict, reward, terminated, truncated, env_info = self.env.step(env_action)
                done = terminated or truncated
                last_env_info = env_info
                if stored_step_infos is not None:
                    stored_step_infos.append(env_info)


                if self.tracker is not None:
                    self.tracker.record_env_step(reward=reward, done=done, info=env_info)

                self.curr_ep_return += reward
                self.total_steps += 1

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
                )

                if done:
                    self.episode_returns.append(self.curr_ep_return)
                    self.curr_ep_return = 0.0
                    current_obs_dict, _ = self.env.reset()
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
            )


            # Global advantage normalization across entire rollout buffer
            adv = self.buffer.advantages
            self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        mean_return = (
            float(np.mean(self.episode_returns)) if len(self.episode_returns) > 0 else 0.0
        )

        self._update_adaptive_exploration(mean_return, last_env_info)

        return current_obs_dict, current_hx, episode_start, mean_return

    def train_epoch(self) -> Dict[str, float]:
        """Performs PPO mini-batch updates over stored chunks."""
        self.agent.train()
        total_policy_loss = 0.0
        total_val_loss = 0.0
        total_ent_loss = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_chunks_generator(self.chunk_batch_size):
                input_dict = {
                    "obs": batch.obs,
                    "prev_action": batch.prev_actions,
                    "prev_reward": batch.prev_rewards,
                }
                if not self.agent.is_visual:
                    input_dict["delta_obs"] = batch.delta_obs

                dist, values, _ = self.agent(input_dict, hx=batch.hx, dones=batch.dones)
                values_sq = values.squeeze(-1)  # [B, T]

                if self.agent.is_discrete:
                    if self.exploration_scale != 1.0:
                        eff_scale = max(0.08, float(self.exploration_scale))
                        scaled_logits = torch.nan_to_num(dist.logits / eff_scale, nan=0.0, posinf=30.0, neginf=-30.0).clamp(min=-30.0, max=30.0)
                        dist_eval = torch.distributions.Categorical(logits=scaled_logits)
                    else:
                        dist_eval = dist
                    new_log_probs = dist_eval.log_prob(batch.actions)  # [B, T]
                    entropy = dist_eval.entropy()                       # [B, T]
                else:
                    if self.exploration_scale != 1.0:
                        eff_scale = max(0.08, float(self.exploration_scale))
                        scaled_std = dist.scale * eff_scale
                        dist_eval = torch.distributions.Normal(loc=dist.mean, scale=scaled_std)
                    else:
                        dist_eval = dist
                    new_log_probs = dist_eval.log_prob(batch.actions).sum(dim=-1)
                    entropy = dist_eval.entropy().sum(dim=-1)

                # Ratio
                ratio = torch.exp(new_log_probs - batch.old_log_probs)

                # Advantages are already globally normalized across rollout
                norm_adv = batch.advantages

                # Clipped surrogate policy loss
                surr1 = ratio * norm_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * norm_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Robust MSE value loss
                val_loss = 0.5 * ((values_sq - batch.returns) ** 2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * val_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

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
                    )

                total_policy_loss += policy_loss.item()
                total_val_loss += val_loss.item()
                total_ent_loss += entropy_loss.item()
                n_updates += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_val_loss / max(1, n_updates),
            "entropy_loss": total_ent_loss / max(1, n_updates),
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

        total_updates = max(1, max_steps // self.rollout_steps)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_updates,
        )

        best_mean_return = -float("inf")
        iteration = 0

        while self.total_steps < max_steps:
            iteration += 1
            obs_dict, hx, episode_start, mean_return = self.collect_rollouts(
                obs_dict, hx, episode_start, step_callback=step_callback
            )
            metrics = self.train_epoch()

            if len(self.episode_returns) > 0:
                best_mean_return = max(best_mean_return, mean_return)
                self.best_mean_return = best_mean_return
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

        self.best_mean_return = best_mean_return
        self.final_return = float(np.mean(self.episode_returns)) if len(self.episode_returns) > 0 else 0.0
        return self.final_return
