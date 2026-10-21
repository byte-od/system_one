from typing import Any, Optional, TypedDict, Union
import copy
import gymnasium as gym
import numpy as np
from PIL import Image


class S1Observation(TypedDict):
    obs: np.ndarray
    delta_obs: Optional[np.ndarray]
    prev_action: Union[int, np.ndarray]
    prev_reward: float


class UniversalS1Wrapper(gym.Wrapper):
    """Universal System 1 Environment Wrapper.

    Augments standard observations with:
    - delta_obs: s_t - s_{t-1} for vector observations, or channel stacking for visual.
    - prev_action: a_{t-1}.
    - prev_reward: r_{t-1}.

    Prevents causal blindness and cyclic attractors in reflexive policies.
    """

    def __init__(self, env: gym.Env, is_visual: Optional[bool] = None, gamma: float = 0.99) -> None:
        super().__init__(env)
        self.gamma = float(gamma)
        if is_visual is None:
            # Auto-detect visual observations:
            # 1. Box with 3 dims where min dim in (1, 3, 4) and max spatial dims >= 10
            # 2. Box with 2 dims where both dims >= 10 (grayscale)
            is_visual = False
            if isinstance(env.observation_space, gym.spaces.Box):
                shp = env.observation_space.shape
                if len(shp) == 3 and (shp[0] in (1, 3, 4) or shp[2] in (1, 3, 4)) and max(shp) >= 10:
                    is_visual = True
                elif len(shp) == 2 and min(shp) >= 10:
                    is_visual = True
        self.is_visual = is_visual

        self.visual_channels = 3
        if self.is_visual and isinstance(env.observation_space, gym.spaces.Box):
            shp = env.observation_space.shape
            if len(shp) == 3:
                if shp[0] in (1, 3, 4) and shp[2] not in (1, 3, 4):
                    self.visual_channels = 1 if shp[0] == 1 else 3
                elif shp[2] in (1, 3, 4):
                    self.visual_channels = 1 if shp[2] == 1 else 3
            elif len(shp) == 2:
                self.visual_channels = 1

        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vectorized = self.num_envs > 1

        self.prev_obs: Optional[np.ndarray] = None
        self.init_obs: Optional[np.ndarray] = None
        self.compass_mode: bool = True
        self.compass_alpha: float = 0.5
        self.prev_action: Optional[Union[int, np.ndarray]] = None
        self.prev_reward: Union[float, np.ndarray] = 0.0 if not self.is_vectorized else np.zeros(self.num_envs, dtype=np.float32)
        self.latest_obs_dict: Optional[S1Observation] = None

        # Build dictionary observation space
        self._setup_spaces()

    def _get_initial_action(self) -> Union[int, np.ndarray]:
        if self.is_vectorized:
            if isinstance(self.action_space, gym.spaces.Discrete):
                return np.zeros(self.num_envs, dtype=np.int64)
            elif isinstance(self.action_space, gym.spaces.Box):
                return np.zeros((self.num_envs, *self.action_space.shape), dtype=np.float32)
            else:
                raise NotImplementedError(f"Unsupported action space: {type(self.action_space)}")

        if isinstance(self.action_space, gym.spaces.Discrete):
            return 0
        elif isinstance(self.action_space, gym.spaces.Box):
            return np.zeros(self.action_space.shape, dtype=np.float32)
        else:
            raise NotImplementedError(f"Unsupported action space: {type(self.action_space)}")

    def _format_visual_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Padroniza frames visuais para (C, 84, 84) float32 normalizado em [0.0, 1.0]."""
        frame = raw_frame

        # Fast-path para frames já no padrão universal (C, 84, 84) float32 normalizado [0, 1]
        if (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape == (self.visual_channels, 84, 84)
            and frame.dtype == np.float32
            and (frame.size == 0 or (frame.max() <= 1.01 and frame.min() >= -0.01))
        ):
            return np.clip(frame, 0.0, 1.0)

        # Trata dimensões e canais (CHW vs HWC vs HW)
        if frame.ndim == 3:
            if frame.shape[0] in (1, 3, 4) and frame.shape[2] not in (1, 3, 4):
                if frame.shape[0] == 1:
                    frame = frame.squeeze(0)  # [H, W]
                elif frame.shape[0] in (3, 4):
                    frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
            elif frame.shape[2] == 1:
                frame = frame.squeeze(-1)  # [H, W]

        # Normaliza tipo para uint8 para redimensionamento PIL bilinear de alta fidelidade
        if frame.dtype != np.uint8:
            if frame.max() <= 1.01 and frame.min() >= -0.01:
                frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame_uint8 = frame

        if self.visual_channels == 1:
            pil_img = Image.fromarray(frame_uint8)
            if pil_img.mode != "L":
                pil_img = pil_img.convert("L")
            pil_img = pil_img.resize((84, 84), Image.Resampling.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            return np.expand_dims(arr, axis=0)  # [1, 84, 84]
        else:
            if frame_uint8.ndim == 3 and frame_uint8.shape[2] == 4:
                frame_uint8 = frame_uint8[:, :, :3]
            pil_img = Image.fromarray(frame_uint8)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img = pil_img.resize((84, 84), Image.Resampling.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            return np.transpose(arr, (2, 0, 1))  # [3, 84, 84]

    @staticmethod
    def _compute_vector_dim(space: gym.spaces.Space) -> int:
        """Calcula a dimensão total equivalente para espaços de observação vetoriais e discretos."""
        if isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            if hasattr(space, "shape") and space.shape:
                return int(np.prod(space.shape))
            return int(space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return int(np.sum(space.nvec))
        elif isinstance(space, gym.spaces.Tuple):
            return sum(UniversalS1Wrapper._compute_vector_dim(s) for s in space.spaces)
        else:
            raise NotImplementedError(
                f"Espaço de observação não suportado: {type(space)}. "
                "Espaços suportados: Box, Discrete, MultiDiscrete, MultiBinary, Tuple."
            )

    @classmethod
    def _vectorize_obs(cls, raw_obs: Any, space: gym.spaces.Space) -> np.ndarray:
        """Converte qualquer observação não-visual em um vetor float32 1D padronizado."""
        if isinstance(space, gym.spaces.Box):
            return np.asarray(raw_obs, dtype=np.float32).flatten()
        elif isinstance(space, gym.spaces.Discrete):
            n = int(space.n)
            vec = np.zeros(n, dtype=np.float32)
            idx = int(raw_obs)
            if 0 <= idx < n:
                vec[idx] = 1.0
            return vec
        elif isinstance(space, gym.spaces.MultiBinary):
            return np.asarray(raw_obs, dtype=np.float32).flatten()
        elif isinstance(space, gym.spaces.MultiDiscrete):
            total_dim = int(np.sum(space.nvec))
            vec = np.zeros(total_dim, dtype=np.float32)
            offset = 0
            for val, n in zip(raw_obs, space.nvec):
                idx = int(val)
                if 0 <= idx < n:
                    vec[offset + idx] = 1.0
                offset += int(n)
            return vec
        elif isinstance(space, gym.spaces.Tuple):
            sub_vecs = [cls._vectorize_obs(sub_val, sub_sp) for sub_val, sub_sp in zip(raw_obs, space.spaces)]
            return np.concatenate(sub_vecs, axis=0).astype(np.float32)
        else:
            return np.asarray(raw_obs, dtype=np.float32).flatten()

    def _setup_spaces(self) -> None:
        obs_space = self.env.observation_space
        act_space = self.action_space

        spaces: dict[str, gym.spaces.Space] = {
            "prev_action": act_space,
            "prev_reward": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
        }

        if self.is_visual:
            assert isinstance(obs_space, gym.spaces.Box), "Visual envs must have Box observation space"
            # Visual observations are standardized to (C, 84, 84) and stacked to (2 * C, 84, 84)
            spaces["obs"] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2 * self.visual_channels, 84, 84),
                dtype=np.float32,
            )
        else:
            obs_dim = self._compute_vector_dim(obs_space)
            if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1:
                low = obs_space.low.astype(np.float32)
                high = obs_space.high.astype(np.float32)
            elif isinstance(obs_space, gym.spaces.Box):
                low = obs_space.low.flatten().astype(np.float32)
                high = obs_space.high.flatten().astype(np.float32)
            elif isinstance(obs_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
                low = np.zeros(obs_dim, dtype=np.float32)
                high = np.ones(obs_dim, dtype=np.float32)
            else:
                low = np.full((obs_dim,), -np.inf, dtype=np.float32)
                high = np.full((obs_dim,), np.inf, dtype=np.float32)

            spaces["obs"] = gym.spaces.Box(
                low=low,
                high=high,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            spaces["delta_obs"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(spaces)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[S1Observation, Any]:
        """Resets the environment and flushes all temporal and causal memory buffers."""
        raw_obs, info = self.env.reset(seed=seed, options=options)

        if self.is_vectorized:
            current_obs = np.asarray(raw_obs, dtype=np.float32)
            self.prev_obs = np.array(current_obs, copy=True)
            self.init_obs = np.array(current_obs, copy=True)
            self.prev_action = self._get_initial_action()
            self.prev_reward = np.zeros(self.num_envs, dtype=np.float32)

            delta_obs = np.zeros_like(current_obs, dtype=np.float32)
            obs_dict: S1Observation = {
                "obs": current_obs,
                "delta_obs": delta_obs,
                "prev_action": copy.deepcopy(self.prev_action),
                "prev_reward": np.array(self.prev_reward, copy=True),
            }
            self.latest_obs_dict = obs_dict
            return obs_dict, info

        if self.is_visual:
            current_obs = self._format_visual_frame(np.asarray(raw_obs))
            self.prev_obs = np.array(current_obs, copy=True)
            self.prev_action = self._get_initial_action()
            self.prev_reward = 0.0

            stacked_obs = np.concatenate([current_obs, current_obs], axis=0)
            obs_dict: S1Observation = {
                "obs": stacked_obs,
                "delta_obs": None,
                "prev_action": copy.deepcopy(self.prev_action),
                "prev_reward": float(self.prev_reward),
            }
        else:
            current_obs = self._vectorize_obs(raw_obs, self.env.observation_space)
            self.prev_obs = np.array(current_obs, copy=True)
            self.init_obs = np.array(current_obs, copy=True)
            self.prev_action = self._get_initial_action()
            self.prev_reward = 0.0

            delta_obs = np.zeros_like(current_obs, dtype=np.float32)
            obs_dict = {
                "obs": current_obs,
                "delta_obs": delta_obs,
                "prev_action": copy.deepcopy(self.prev_action),
                "prev_reward": float(self.prev_reward),
            }

        self.latest_obs_dict = obs_dict
        return obs_dict, info

    def step(
        self, action: Any, chunk_mask: Optional[np.ndarray] = None
    ) -> tuple[S1Observation, Any, Any, Any, Any]:
        """Takes an environment step and computes temporal differential state.
        
        Supports both single-step execution and Temporal Action Chunking (ACT) sequences
        with automatic open-loop unrolling, reward discounting, early termination on done,
        and uncertainty-based adaptive chunk termination.
        """
        assert self.prev_obs is not None, "step() called before reset()"

        # =====================================================================
        # 1. AMBIENTE VETORIZADO (num_envs > 1)
        # =====================================================================
        if self.is_vectorized:
            act_arr = np.asarray(action)
            # Verifica se é um chunk temporal vetorizado [num_envs, k] onde k > 1
            if act_arr.ndim == 2 and act_arr.shape[0] == self.num_envs and act_arr.shape[1] > 1:
                E, k = act_arr.shape
                total_rewards = np.zeros(E, dtype=np.float32)
                discounts = np.ones(E, dtype=np.float32)
                all_terminateds = np.zeros(E, dtype=bool)
                all_truncateds = np.zeros(E, dtype=bool)
                active_mask = np.ones(E, dtype=bool)
                last_obs = None
                last_info = None
                last_executed_actions = act_arr[:, 0].copy()

                for step_i in range(k):
                    if chunk_mask is not None:
                        active_mask &= chunk_mask[:, step_i]
                        if not np.any(active_mask):
                            break
                    sub_acts = act_arr[:, step_i]
                    try:
                        next_obs, rews, terms, truncs, info = self.env.step(sub_acts, active_mask=active_mask)
                    except TypeError:
                        next_obs, rews, terms, truncs, info = self.env.step(sub_acts)
                    last_obs = next_obs
                    last_info = info
                    dones = np.asarray(terms, dtype=bool) | np.asarray(truncs, dtype=bool)
                    
                    # Atualiza recompensas acumuladas e ações apenas nos ambientes ativos
                    rews_arr = np.asarray(rews, dtype=np.float32)
                    total_rewards += np.where(active_mask, discounts * rews_arr, 0.0)
                    last_executed_actions[active_mask] = sub_acts[active_mask]
                    discounts[active_mask] *= self.gamma

                    # Trata encerramentos
                    newly_done = dones & active_mask
                    if np.any(newly_done):
                        all_terminateds[newly_done] = np.asarray(terms, dtype=bool)[newly_done]
                        all_truncateds[newly_done] = np.asarray(truncs, dtype=bool)[newly_done]
                        active_mask[newly_done] = False

                    if not np.any(active_mask):
                        break

                current_obs = np.asarray(last_obs, dtype=np.float32)
                local_delta = current_obs - self.prev_obs
                if self.compass_mode and self.init_obs is not None:
                    global_delta = current_obs - self.init_obs
                    delta_obs = (1.0 - self.compass_alpha) * local_delta + self.compass_alpha * global_delta
                else:
                    delta_obs = local_delta
                done_mask = np.asarray(all_terminateds | all_truncateds, dtype=bool)
                if np.any(done_mask):
                    delta_obs[done_mask] = 0.0
                    if self.compass_mode and self.init_obs is not None:
                        self.init_obs[done_mask] = current_obs[done_mask]

                obs_dict: S1Observation = {
                    "obs": current_obs,
                    "delta_obs": delta_obs,
                    "prev_action": copy.deepcopy(last_executed_actions),
                    "prev_reward": total_rewards,
                }
                self.prev_obs = np.array(current_obs, copy=True)
                self.prev_action = copy.deepcopy(last_executed_actions)
                self.prev_reward = total_rewards
                self.latest_obs_dict = obs_dict
                return obs_dict, total_rewards, all_terminateds, all_truncateds, last_info

            # Passo unitário vetorizado clássico
            next_obs, rewards, terminateds, truncateds, info = self.env.step(action)
            current_obs = np.asarray(next_obs, dtype=np.float32)
            local_delta = current_obs - self.prev_obs
            if self.compass_mode and self.init_obs is not None:
                global_delta = current_obs - self.init_obs
                delta_obs = (1.0 - self.compass_alpha) * local_delta + self.compass_alpha * global_delta
            else:
                delta_obs = local_delta
            done_mask = np.asarray(terminateds | truncateds, dtype=bool)
            if np.any(done_mask):
                delta_obs[done_mask] = 0.0
                if self.compass_mode and self.init_obs is not None:
                    self.init_obs[done_mask] = current_obs[done_mask]

            rew_arr = np.asarray(rewards, dtype=np.float32)
            obs_dict = {
                "obs": current_obs,
                "delta_obs": delta_obs,
                "prev_action": copy.deepcopy(action),
                "prev_reward": rew_arr,
            }
            self.prev_obs = np.array(current_obs, copy=True)
            self.prev_action = copy.deepcopy(action)
            self.prev_reward = rew_arr
            self.latest_obs_dict = obs_dict
            return obs_dict, rew_arr, terminateds, truncateds, info

        # =====================================================================
        # 2. AMBIENTE SINGULAR (num_envs == 1)
        # =====================================================================
        is_chunk = False
        if isinstance(self.action_space, gym.spaces.Discrete):
            if isinstance(action, (list, tuple, np.ndarray)):
                act_arr = np.asarray(action)
                if act_arr.ndim == 1 and len(act_arr) > 1:
                    is_chunk = True
        elif isinstance(self.action_space, gym.spaces.Box):
            act_arr = np.asarray(action, dtype=np.float32)
            if act_arr.ndim == len(self.action_space.shape) + 1 and act_arr.shape[0] > 1:
                is_chunk = True

        if is_chunk:
            total_reward = 0.0
            discount = 1.0
            terminated = False
            truncated = False
            last_obs = None
            last_info = None
            last_executed_act = act_arr[0]

            for step_i in range(len(act_arr)):
                if chunk_mask is not None and step_i < len(chunk_mask) and not chunk_mask[step_i]:
                    break
                sub_act = act_arr[step_i]
                if isinstance(self.action_space, gym.spaces.Discrete):
                    sub_act = int(sub_act)
                elif isinstance(self.action_space, gym.spaces.Box):
                    sub_act = np.clip(sub_act, self.action_space.low, self.action_space.high)
                last_executed_act = sub_act
                next_obs, rew, term, trunc, info = self.env.step(sub_act)
                total_reward += discount * float(rew)
                discount *= self.gamma
                last_obs = next_obs
                last_info = info
                if term or trunc:
                    terminated = term
                    truncated = trunc
                    break

            if self.is_visual:
                current_obs = self._format_visual_frame(np.asarray(last_obs))
                stacked_obs = np.concatenate([current_obs, self.prev_obs], axis=0)
                obs_dict = {
                    "obs": stacked_obs,
                    "delta_obs": None,
                    "prev_action": copy.deepcopy(last_executed_act),
                    "prev_reward": total_reward,
                }
            else:
                current_obs = self._vectorize_obs(last_obs, self.env.observation_space)
                local_delta = current_obs - self.prev_obs
                if self.compass_mode and self.init_obs is not None:
                    global_delta = current_obs - self.init_obs
                    delta_obs = (1.0 - self.compass_alpha) * local_delta + self.compass_alpha * global_delta
                else:
                    delta_obs = local_delta
                if terminated or truncated:
                    delta_obs = np.zeros_like(current_obs)
                    if self.compass_mode:
                        self.init_obs = np.array(current_obs, copy=True)
                obs_dict = {
                    "obs": current_obs,
                    "delta_obs": delta_obs,
                    "prev_action": copy.deepcopy(last_executed_act),
                    "prev_reward": total_reward,
                }

            self.prev_obs = np.array(current_obs, copy=True)
            self.prev_action = copy.deepcopy(last_executed_act)
            self.prev_reward = total_reward
            self.latest_obs_dict = obs_dict
            return obs_dict, total_reward, terminated, truncated, last_info

        # Passo unitário singular clássico
        if isinstance(self.action_space, gym.spaces.Box):
            action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        reward_float = float(reward)

        if self.is_visual:
            current_obs = self._format_visual_frame(np.asarray(next_obs))
            stacked_obs = np.concatenate([current_obs, self.prev_obs], axis=0)
            obs_dict: S1Observation = {
                "obs": stacked_obs,
                "delta_obs": None,
                "prev_action": copy.deepcopy(action),
                "prev_reward": reward_float,
            }
        else:
            current_obs = self._vectorize_obs(next_obs, self.env.observation_space)
            local_delta = current_obs - self.prev_obs
            if self.compass_mode and self.init_obs is not None:
                global_delta = current_obs - self.init_obs
                delta_obs = (1.0 - self.compass_alpha) * local_delta + self.compass_alpha * global_delta
            else:
                delta_obs = local_delta
            obs_dict = {
                "obs": current_obs,
                "delta_obs": delta_obs,
                "prev_action": copy.deepcopy(action),
                "prev_reward": reward_float,
            }

        # Update buffers for next step
        self.prev_obs = np.array(current_obs, copy=True)
        self.prev_action = copy.deepcopy(action)
        self.prev_reward = reward_float
        if (terminated or truncated) and self.compass_mode:
            self.init_obs = np.array(current_obs, copy=True)
        self.latest_obs_dict = obs_dict

        return obs_dict, reward_float, terminated, truncated, info


class ActionChunkWrapper(gym.Wrapper):
    """Wrapper explícito de alto nível para Action Chunking Temporal (ACT).

    Permite envelopar qualquer ambiente Gymnasium (vetorizado ou singular)
    para executar fatias temporais de tamanho k com término antecipado em caso de vitória/derrota.
    """

    def __init__(self, env: gym.Env, chunk_size: int = 4, gamma: float = 0.99) -> None:
        super().__init__(env)
        self.chunk_size = max(1, int(chunk_size))
        self.gamma = float(gamma)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # Se action já for um chunk, delega para a lógica de unrolling
        act_arr = np.asarray(action)
        if (act_arr.ndim == 1 and len(act_arr) > 1) or (act_arr.ndim == 2 and act_arr.shape[1] > 1):
            return self.env.step(action)
        # Se for unitário, executa normalmente
        return self.env.step(action)

