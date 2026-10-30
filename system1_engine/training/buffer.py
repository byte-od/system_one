from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import heapq
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import zlib
import numpy as np
import torch


@dataclass
class RecurrentBatch:
    obs: torch.Tensor
    delta_obs: Optional[torch.Tensor]
    prev_actions: torch.Tensor
    prev_rewards: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor
    dones: torch.Tensor
    hx: torch.Tensor
    env_scales: Optional[torch.Tensor] = None


class RecurrentRolloutBuffer:
    """Rollout Buffer for Recurrent PPO with temporal chunking for BPTT.

    Stores sequential transitions, computes GAE, and generates mini-batches
    of short sequence chunks with preserved initial hidden states and done masks.
    """

    def __init__(
        self,
        buffer_size: int = 1024,
        chunk_length: int = 16,
        is_visual: bool = False,
        is_discrete: bool = True,
        obs_shape: Tuple[int, ...] = (4,),
        action_shape: Tuple[int, ...] = (),
        device: torch.device = torch.device("cpu"),
        hidden_dim: int = 256,
    ) -> None:
        assert buffer_size % chunk_length == 0, (
            f"buffer_size ({buffer_size}) must be divisible by chunk_length ({chunk_length})"
        )
        self.buffer_size = buffer_size
        self.chunk_length = chunk_length
        self.num_chunks = buffer_size // chunk_length
        self.is_visual = is_visual
        self.is_discrete = is_discrete
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.hidden_dim = hidden_dim

        self.reset()

    def reset(self) -> None:
        """Clears buffer contents."""
        self.obs: List[np.ndarray] = []
        self.delta_obs: List[Optional[np.ndarray]] = []
        self.prev_actions: List[Union[int, np.ndarray]] = []
        self.prev_rewards: List[float] = []

        self.actions: List[Union[int, np.ndarray]] = []
        self.rewards: List[float] = []
        self.episode_starts: List[bool] = []  # True if step is start of episode
        self.dones: List[bool] = []           # True if step ended episode
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.env_scales: List[float] = []

        # Hidden states recorded at chunk boundaries [num_chunks, 1, 1, 256]
        self.chunk_hx: List[torch.Tensor] = []
        self.step_idx = 0

    def add(
        self,
        obs_dict: Dict[str, Any],
        action: Union[int, np.ndarray, torch.Tensor],
        reward: float,
        done: bool,
        episode_start: bool,
        value: Union[float, torch.Tensor],
        log_prob: Union[float, torch.Tensor],
        hx: Optional[torch.Tensor] = None,
        env_scale: float = 1.0,
    ) -> None:
        """Adds a transition to the rollout buffer."""
        # Record initial hidden state at chunk boundaries
        if self.step_idx % self.chunk_length == 0:
            if hx is not None:
                # Store detached CPU copy: shape [1, 1, hidden_dim]
                self.chunk_hx.append(hx.detach().cpu())
            else:
                self.chunk_hx.append(torch.zeros(1, 1, self.hidden_dim))

        self.obs.append(np.asarray(obs_dict["obs"], dtype=np.float32))
        delta = obs_dict.get("delta_obs")
        if delta is not None:
            self.delta_obs.append(np.asarray(delta, dtype=np.float32))
        else:
            self.delta_obs.append(None)

        self.prev_actions.append(obs_dict["prev_action"])
        self.prev_rewards.append(float(obs_dict["prev_reward"]))

        if isinstance(action, torch.Tensor):
            if self.is_discrete:
                if action.numel() > 1:
                    self.actions.append(action.cpu().numpy().astype(np.int64))
                else:
                    self.actions.append(int(action.item()))
            else:
                self.actions.append(action.cpu().numpy())
        else:
            self.actions.append(action)

        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.episode_starts.append(bool(episode_start))

        val = value.item() if isinstance(value, torch.Tensor) else float(value)
        lp = log_prob.item() if isinstance(log_prob, torch.Tensor) else float(log_prob)
        self.values.append(val)
        self.log_probs.append(lp)
        self.env_scales.append(float(env_scale))

        self.step_idx += 1

    def compute_gae(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        action_chunk_size: int = 1,
    ) -> None:
        """Computes Generalized Advantage Estimation (GAE) and target returns under SMDP."""
        assert self.step_idx == self.buffer_size, (
            f"Buffer not full: {self.step_idx}/{self.buffer_size}"
        )
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        last_gae = 0.0
        eff_gamma = float(gamma ** max(1, int(action_chunk_size)))

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = self.values[t + 1]

            delta = self.rewards[t] + eff_gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + eff_gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + np.asarray(self.values, dtype=np.float32)

    def set_vectorized_data(
        self,
        obs: np.ndarray,
        delta_obs: Optional[np.ndarray],
        prev_actions: np.ndarray,
        prev_rewards: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        episode_starts: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        chunk_hx: List[torch.Tensor],
        advantages: np.ndarray,
        returns: np.ndarray,
        env_scales: Optional[np.ndarray] = None,
    ) -> None:
        """Armazena transições coletadas diretamente em batch de ambientes vetorizados."""
        self.obs = obs
        self.delta_obs = delta_obs
        self.prev_actions = prev_actions
        self.prev_rewards = prev_rewards
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.episode_starts = episode_starts
        self.values = values
        self.log_probs = log_probs
        self.chunk_hx = chunk_hx
        self.advantages = advantages
        self.returns = returns
        self.env_scales = env_scales if env_scales is not None else np.ones(self.buffer_size, dtype=np.float32)
        self.step_idx = self.buffer_size

    def get_chunks_generator(
        self,
        chunk_batch_size: int = 4,
    ) -> Generator[RecurrentBatch, None, None]:
        """Generates randomized mini-batches of sequential chunks for BPTT."""
        chunk_indices = np.random.permutation(self.num_chunks)
        t_len = self.chunk_length

        for start_idx in range(0, self.num_chunks, chunk_batch_size):
            batch_chunk_idxs = chunk_indices[start_idx : start_idx + chunk_batch_size]
            b_size = len(batch_chunk_idxs)

            # Assemble batch tensors: [B, T, ...]
            b_obs = []
            b_delta = []
            b_prev_act = []
            b_prev_rew = []
            b_actions = []
            b_log_probs = []
            b_adv = []
            b_ret = []
            b_vals = []
            b_dones = []
            b_hx = []
            b_scales = []

            for c_idx in batch_chunk_idxs:
                step_start = c_idx * t_len
                step_end = step_start + t_len

                b_obs.append(self.obs[step_start:step_end])
                if not self.is_visual:
                    b_delta.append(self.delta_obs[step_start:step_end])
                b_prev_act.append(self.prev_actions[step_start:step_end])
                b_prev_rew.append(self.prev_rewards[step_start:step_end])
                b_actions.append(self.actions[step_start:step_end])
                b_log_probs.append(self.log_probs[step_start:step_end])
                b_adv.append(self.advantages[step_start:step_end])
                b_ret.append(self.returns[step_start:step_end])
                b_vals.append(self.values[step_start:step_end])
                # Episode start mask: 1.0 if new episode starts at step t, 0.0 otherwise
                b_dones.append(self.episode_starts[step_start:step_end])

                # Initial hidden state for this chunk: shape [1, 1, 256]
                b_hx.append(self.chunk_hx[c_idx])
                if hasattr(self, "env_scales") and len(self.env_scales) > 0:
                    b_scales.append(self.env_scales[step_start:step_end])

            # Convert to PyTorch tensors on device
            tensor_obs = torch.tensor(np.array(b_obs), dtype=torch.float32, device=self.device)
            tensor_delta = (
                torch.tensor(np.array(b_delta), dtype=torch.float32, device=self.device)
                if not self.is_visual
                else None
            )

            if self.is_discrete:
                tensor_prev_act = torch.tensor(
                    np.array(b_prev_act), dtype=torch.long, device=self.device
                )
                tensor_actions = torch.tensor(
                    np.array(b_actions), dtype=torch.long, device=self.device
                )
            else:
                tensor_prev_act = torch.tensor(
                    np.array(b_prev_act), dtype=torch.float32, device=self.device
                )
                tensor_actions = torch.tensor(
                    np.array(b_actions), dtype=torch.float32, device=self.device
                )

            tensor_prev_rew = torch.tensor(
                np.array(b_prev_rew), dtype=torch.float32, device=self.device
            ).unsqueeze(-1)
            tensor_log_probs = torch.tensor(
                np.array(b_log_probs), dtype=torch.float32, device=self.device
            )
            tensor_adv = torch.tensor(
                np.array(b_adv), dtype=torch.float32, device=self.device
            )
            tensor_ret = torch.tensor(
                np.array(b_ret), dtype=torch.float32, device=self.device
            )
            tensor_vals = torch.tensor(
                np.array(b_vals), dtype=torch.float32, device=self.device
            )
            tensor_dones = torch.tensor(
                np.array(b_dones), dtype=torch.float32, device=self.device
            )

            # Stack hx along batch dimension: [1, B, 256]
            tensor_hx = torch.cat(b_hx, dim=1).to(self.device)
            tensor_scales = (
                torch.tensor(np.array(b_scales), dtype=torch.float32, device=self.device)
                if len(b_scales) == b_size
                else None
            )

            yield RecurrentBatch(
                obs=tensor_obs,
                delta_obs=tensor_delta,
                prev_actions=tensor_prev_act,
                prev_rewards=tensor_prev_rew,
                actions=tensor_actions,
                old_log_probs=tensor_log_probs,
                advantages=tensor_adv,
                returns=tensor_ret,
                old_values=tensor_vals,
                dones=tensor_dones,
                hx=tensor_hx,
                env_scales=tensor_scales,
            )

    def sample_negative_chunks(
        self,
        batch_size: int = 8,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Extrai sob demanda chunks negativos (vantagem GAE média < 0) do rollout on-policy atual.

        Permite contraste DPO sem armazenar nenhuma trajetória de falha no SuccessReplayBuffer.
        """
        if (
            self.step_idx < self.buffer_size
            or not hasattr(self, "advantages")
            or len(self.advantages) < self.buffer_size
            or self.num_chunks <= 0
        ):
            return None

        adv_arr = np.asarray(self.advantages, dtype=np.float32).reshape(self.num_chunks, self.chunk_length)
        chunk_mean_adv = adv_arr.mean(axis=1)
        neg_idxs = np.where(chunk_mean_adv < 0.0)[0]
        if len(neg_idxs) == 0:
            return None

        k = max(1, int(batch_size))
        chosen = np.random.choice(neg_idxs, size=k, replace=(len(neg_idxs) < k))
        t_len = self.chunk_length

        has_valid_delta = (
            not self.is_visual
            and self.delta_obs is not None
            and len(self.delta_obs) >= self.buffer_size
            and self.delta_obs[0] is not None
        )

        b_obs = []
        b_delta = []
        b_prev_act = []
        b_prev_rew = []
        b_actions = []
        b_log_probs = []
        b_dones = []
        b_hx = []

        for c_idx in chosen:
            step_start = int(c_idx) * t_len
            step_end = step_start + t_len
            obs_slice = np.asarray(self.obs[step_start:step_end])
            if self.is_visual and obs_slice.dtype == np.uint8:
                obs_slice = obs_slice.astype(np.float32) / 255.0
            b_obs.append(obs_slice)
            if has_valid_delta:
                b_delta.append(self.delta_obs[step_start:step_end])
            b_prev_act.append(self.prev_actions[step_start:step_end])
            b_prev_rew.append(self.prev_rewards[step_start:step_end])
            b_actions.append(self.actions[step_start:step_end])
            b_log_probs.append(self.log_probs[step_start:step_end])
            b_dones.append(self.episode_starts[step_start:step_end])
            b_hx.append(self.chunk_hx[int(c_idx)])

        tensor_obs = torch.tensor(np.array(b_obs), dtype=torch.float32, device=self.device)
        tensor_delta = (
            torch.tensor(np.array(b_delta), dtype=torch.float32, device=self.device)
            if has_valid_delta
            else None
        )
        if self.is_discrete:
            tensor_prev_act = torch.tensor(np.array(b_prev_act), dtype=torch.long, device=self.device)
            if tensor_prev_act.dim() == 2:
                tensor_prev_act = tensor_prev_act.unsqueeze(-1)
            tensor_actions = torch.tensor(np.array(b_actions), dtype=torch.long, device=self.device)
        else:
            tensor_prev_act = torch.tensor(np.array(b_prev_act), dtype=torch.float32, device=self.device)
            tensor_actions = torch.tensor(np.array(b_actions), dtype=torch.float32, device=self.device)

        tensor_prev_rew = torch.tensor(np.array(b_prev_rew), dtype=torch.float32, device=self.device).unsqueeze(-1)
        tensor_log_probs = torch.tensor(np.array(b_log_probs), dtype=torch.float32, device=self.device)
        tensor_dones = torch.tensor(np.array(b_dones), dtype=torch.float32, device=self.device)
        tensor_hx = torch.cat(b_hx, dim=1).to(self.device)
        tensor_mask = torch.ones_like(tensor_log_probs, dtype=torch.float32, device=self.device)

        return {
            "obs": tensor_obs,
            "delta_obs": tensor_delta,
            "prev_actions": tensor_prev_act,
            "prev_rewards": tensor_prev_rew,
            "actions": tensor_actions,
            "old_log_probs": tensor_log_probs,
            "mask": tensor_mask,
            "hx": tensor_hx,
            "dones": tensor_dones,
        }


PackedArraySpec = Tuple[bytes, bool, Tuple[int, ...], np.dtype, np.dtype]


def _pack_array_lossless(
    arr: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[PackedArraySpec]]:
    """Compacta um array NumPy estritamente sem perdas (100% bit-a-bit idêntico após reconstrução).

    - Nunca degrada precisão numérica: só reduz o tipo intermediário para int8 ou float16 quando
      `np.array_equal(cand.astype(orig_dtype), arr)` é estritamente verdadeiro (erro = 0.0 exato).
    - Caso contrário (ex: estados contínuos float32 com mantissa arbitrária), preserva todos os 32 bits
      IEEE-754 intactos e aplica zlib nível 1 apenas se houver ganho real de bytes.
    """
    arr_c = np.ascontiguousarray(arr)
    orig_dtype = arr_c.dtype
    shape = arr_c.shape
    if arr_c.size == 0:
        return arr_c, None

    wire_arr = arr_c
    if orig_dtype == np.float32:
        mn = float(np.min(arr_c))
        mx = float(np.max(arr_c))
        if np.isfinite(mn) and np.isfinite(mx):
            if -128.0 <= mn and mx <= 127.0:
                cand_i8 = arr_c.astype(np.int8)
                if np.array_equal(cand_i8.astype(np.float32), arr_c):
                    wire_arr = cand_i8
            if wire_arr is arr_c and -65504.0 <= mn and mx <= 65504.0:
                cand_f16 = arr_c.astype(np.float16)
                if np.array_equal(cand_f16.astype(np.float32), arr_c):
                    wire_arr = cand_f16

    raw_bytes = wire_arr.tobytes()
    z_bytes = zlib.compress(raw_bytes, level=1)
    if len(z_bytes) < len(raw_bytes):
        best_bytes, is_zlib = z_bytes, True
    else:
        best_bytes, is_zlib = raw_bytes, False

    if len(best_bytes) < arr_c.nbytes:
        return None, (best_bytes, is_zlib, shape, orig_dtype, wire_arr.dtype)
    return arr_c, None


def _unpack_array_lossless(packed: PackedArraySpec) -> np.ndarray:
    """Reconstrói bit-a-bit o array NumPy original a partir de sua representação sem perdas."""
    payload, is_zlib, shape, orig_dtype, wire_dtype = packed
    raw_buf = zlib.decompress(payload) if is_zlib else payload
    view = np.frombuffer(raw_buf, dtype=wire_dtype).reshape(shape)
    if wire_dtype != orig_dtype:
        return view.astype(orig_dtype)
    return view.copy()


class SuccessfulTrajectory:
    """Trajetória completa de vitória verificada ou fatia positiva de sucesso, com compactação lossless."""

    def __init__(
        self,
        obs: np.ndarray,
        delta_obs: Optional[np.ndarray],
        prev_actions: np.ndarray,
        prev_rewards: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        depth: int,
        length: int,
        log_probs: Optional[np.ndarray] = None,
        is_full_win: bool = True,
        _entry_id: int = 0,
    ) -> None:
        self._obs_raw: Optional[np.ndarray] = None
        self._obs_packed: Optional[PackedArraySpec] = None
        self._delta_raw: Optional[np.ndarray] = None
        self._delta_packed: Optional[PackedArraySpec] = None
        self._initial_state_key: Optional[bytes] = None

        self.obs = obs
        self.delta_obs = delta_obs
        self.prev_actions = prev_actions
        self.prev_rewards = prev_rewards
        self.actions = actions
        self.returns = returns
        self.depth = int(depth)
        self.length = int(length)
        self.log_probs = log_probs
        self.is_full_win = bool(is_full_win)
        self._entry_id = int(_entry_id)

    @property
    def obs(self) -> np.ndarray:
        if self._obs_packed is not None:
            return _unpack_array_lossless(self._obs_packed)
        assert self._obs_raw is not None
        return self._obs_raw

    @obs.setter
    def obs(self, value: np.ndarray) -> None:
        arr = np.asarray(value)
        self._obs_raw = arr
        self._obs_packed = None
        self._initial_state_key = None

    @property
    def delta_obs(self) -> Optional[np.ndarray]:
        if self._delta_packed is not None:
            return _unpack_array_lossless(self._delta_packed)
        return self._delta_raw

    @delta_obs.setter
    def delta_obs(self, value: Optional[np.ndarray]) -> None:
        self._delta_raw = np.asarray(value) if value is not None else None
        self._delta_packed = None

    @property
    def initial_state_key(self) -> bytes:
        """Assinatura canônica de 128 bits do estado inicial obs[0] (calculada sem descompactar repetidamente)."""
        if self._initial_state_key is None:
            cur_obs = self.obs
            first = np.ascontiguousarray(cur_obs[0]) if cur_obs.shape[0] > 0 else np.ascontiguousarray(cur_obs)
            self._initial_state_key = hashlib.blake2b(first.tobytes(), digest_size=16).digest()
        return self._initial_state_key

    @property
    def is_compacted(self) -> bool:
        return (self._obs_packed is not None) or (self._delta_packed is not None)

    @property
    def memory_bytes(self) -> int:
        """Bytes efetivamente ocupados pelos tensores da trajetória na memória RAM."""
        total = 0
        if self._obs_packed is not None:
            total += len(self._obs_packed[0])
        elif self._obs_raw is not None:
            total += int(self._obs_raw.nbytes)
        if self._delta_packed is not None:
            total += len(self._delta_packed[0])
        elif self._delta_raw is not None:
            total += int(self._delta_raw.nbytes)
        for arr in (self.prev_actions, self.prev_rewards, self.actions, self.returns, self.log_probs):
            if isinstance(arr, np.ndarray):
                total += int(arr.nbytes)
        return total

    @property
    def uncompressed_bytes(self) -> int:
        """Bytes equivalentes caso obs e delta_obs estivessem descomprimidos em seus tipos originais."""
        total = 0
        if self._obs_packed is not None:
            _, _, shape, orig_dt, _ = self._obs_packed
            total += int(np.prod(shape, dtype=np.int64)) * np.dtype(orig_dt).itemsize
        elif self._obs_raw is not None:
            total += int(self._obs_raw.nbytes)
        if self._delta_packed is not None:
            _, _, shape, orig_dt, _ = self._delta_packed
            total += int(np.prod(shape, dtype=np.int64)) * np.dtype(orig_dt).itemsize
        elif self._delta_raw is not None:
            total += int(self._delta_raw.nbytes)
        for arr in (self.prev_actions, self.prev_rewards, self.actions, self.returns, self.log_probs):
            if isinstance(arr, np.ndarray):
                total += int(arr.nbytes)
        return total

    def compact_lossless(self) -> None:
        """Compacta obs e delta_obs in-place com 0.0 de perda de precisão (100% reversível bit-a-bit)."""
        _ = self.initial_state_key
        if self._obs_raw is not None and self._obs_packed is None:
            self._obs_raw, self._obs_packed = _pack_array_lossless(self._obs_raw)
        if self._delta_raw is not None and self._delta_packed is None:
            self._delta_raw, self._delta_packed = _pack_array_lossless(self._delta_raw)

    def __len__(self) -> int:
        return self.length

    @property
    def quality_key(self) -> Tuple[bool, float, int]:
        ret = float(self.returns[0]) if (self.returns is not None and len(self.returns) > 0) else 0.0
        return (bool(getattr(self, "is_full_win", True)), ret, -int(self.length))

    def _cmp_tuple(self) -> Tuple[bool, float, int, int]:
        ret = float(self.returns[0]) if (self.returns is not None and len(self.returns) > 0) else 0.0
        return (bool(getattr(self, "is_full_win", True)), ret, -int(self.length), int(self._entry_id))

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, SuccessfulTrajectory):
            return NotImplemented
        return self._cmp_tuple() < other._cmp_tuple()

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, SuccessfulTrajectory):
            return NotImplemented
        return self._cmp_tuple() <= other._cmp_tuple()

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, SuccessfulTrajectory):
            return NotImplemented
        return self._cmp_tuple() > other._cmp_tuple()

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, SuccessfulTrajectory):
            return NotImplemented
        return self._cmp_tuple() >= other._cmp_tuple()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SuccessfulTrajectory):
            return NotImplemented
        return self is other or (self._cmp_tuple() == other._cmp_tuple())


class FrontierTransition(SuccessfulTrajectory):
    """Transição atômica ou micro-manobra decisiva de fronteira (Bellman 1-step / n-step breakthrough)."""

    def __init__(
        self,
        obs: np.ndarray,
        delta_obs: Optional[np.ndarray],
        prev_actions: np.ndarray,
        prev_rewards: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        depth: int,
        length: int,
        log_probs: Optional[np.ndarray] = None,
        is_full_win: bool = True,
        _entry_id: int = 0,
        td_error: float = 0.0,
    ) -> None:
        super().__init__(
            obs=obs,
            delta_obs=delta_obs,
            prev_actions=prev_actions,
            prev_rewards=prev_rewards,
            actions=actions,
            returns=returns,
            depth=depth,
            length=length,
            log_probs=log_probs,
            is_full_win=is_full_win,
            _entry_id=_entry_id,
        )
        self.td_error = float(td_error)

    @classmethod
    def from_step(
        cls,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        target_return: float,
        depth: int,
        delta_obs: Optional[np.ndarray] = None,
        prev_action: Union[int, np.ndarray] = 0,
        prev_reward: float = 0.0,
        td_error: float = 0.0,
        log_probs: Optional[np.ndarray] = None,
    ) -> "FrontierTransition":
        """Cria uma transição atômica 1-step formatada como micro-trajetória de comprimento 1."""
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 1 or obs_arr.ndim == 3:
            obs_arr = np.expand_dims(obs_arr, axis=0)

        if np.isscalar(action):
            act_arr = np.asarray([action], dtype=np.int64)
        else:
            act_arr = np.expand_dims(np.asarray(action), axis=0)

        if np.isscalar(prev_action):
            prev_act_arr = np.asarray([prev_action], dtype=np.int64)
        else:
            prev_act_arr = np.expand_dims(np.asarray(prev_action), axis=0)

        delta_arr = None
        if delta_obs is not None:
            delta_arr = np.asarray(delta_obs)
            if delta_arr.ndim == 1 or delta_arr.ndim == 3:
                delta_arr = np.expand_dims(delta_arr, axis=0)

        return cls(
            obs=obs_arr,
            delta_obs=delta_arr,
            prev_actions=prev_act_arr,
            prev_rewards=np.array([prev_reward], dtype=np.float32),
            actions=act_arr,
            returns=np.array([target_return], dtype=np.float32),
            depth=int(depth),
            length=1,
            log_probs=log_probs,
            td_error=float(td_error),
            is_full_win=True,
        )


class SuccessReplayBuffer:
    """Buffer estratificado sem teto artificial e com compactação 100% lossless para vitórias e fatias de sucesso.

    No modo padrão (max_trajectories=None, max_per_depth=None):
    - Capacidade ilimitada por profundidade: jamais expulsa scrambles difíceis por causa de scrambles curtos
      (elimina completamente o Easy-Scramble Bias e a memorização precoce de pool fixo).
    - Deduplicação por estado inicial (hash(obs[0])): se o mesmo scramble for resolvido novamente, atualiza
      in-place apenas se a nova trajetória for superior (ex: vitória completa substituindo fatia parcial,
      ou solução com menos movimentos para o mesmo scramble).
    - Compactação estritamente sem perdas (lossless_compaction=True): reduz ~30x o consumo de RAM em
      domínios simbólicos/vetoriais sem alterar um único bit dos tensores float32 entregues à rede.
    """

    def __init__(
        self,
        max_trajectories: Optional[int] = None,
        max_per_depth: Optional[int] = None,
        min_capacity_per_depth: int = 32,
        is_visual: bool = False,
        is_discrete: bool = True,
        obs_shape: Tuple[int, ...] = (4,),
        action_shape: Tuple[int, ...] = (),
        device: torch.device = torch.device("cpu"),
        hidden_dim: int = 512,
        frontier_slicing: bool = False,
        frontier_ratio: float = 0.75,
        progressive_capacity: bool = True,
        max_failures_per_depth: int = 512,
        min_failures_per_depth: int = 16,
        dynamic_frontier: bool = True,
        proximal_decay: float = 0.8,
        lossless_compaction: bool = True,
        deduplicate_scrambles: Optional[bool] = None,
    ) -> None:
        self.max_trajectories: Optional[int] = (
            max(2, int(max_trajectories))
            if (max_trajectories is not None and int(max_trajectories) > 0)
            else None
        )
        if max_per_depth is not None and int(max_per_depth) > 0:
            raw_mpd = max(1, int(max_per_depth))
            self.max_per_depth: Optional[int] = (
                min(raw_mpd, self.max_trajectories) if self.max_trajectories is not None else raw_mpd
            )
        else:
            self.max_per_depth = None

        if self.max_per_depth is not None:
            self.min_capacity_per_depth = max(1, min(int(min_capacity_per_depth), self.max_per_depth))
        elif self.max_trajectories is not None:
            self.min_capacity_per_depth = max(1, min(int(min_capacity_per_depth), self.max_trajectories))
        else:
            self.min_capacity_per_depth = max(1, int(min_capacity_per_depth))

        self.is_visual = is_visual
        self.is_discrete = is_discrete
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.hidden_dim = hidden_dim
        self.frontier_slicing = frontier_slicing
        self.frontier_ratio = float(frontier_ratio)
        self.progressive_capacity = bool(progressive_capacity)
        self.max_failures_per_depth = max(1, int(max_failures_per_depth))
        self.min_failures_per_depth = max(1, min(int(min_failures_per_depth), self.max_failures_per_depth))
        self.dynamic_frontier = bool(dynamic_frontier)
        self.proximal_decay = float(proximal_decay)
        self.lossless_compaction = bool(lossless_compaction)
        self.deduplicate_scrambles = (
            (self.max_trajectories is None and self.max_per_depth is None)
            if deduplicate_scrambles is None
            else bool(deduplicate_scrambles)
        )

        self._depth_buffers: Dict[int, List[SuccessfulTrajectory]] = defaultdict(list)
        self._scramble_index: Dict[int, Dict[bytes, int]] = defaultdict(dict)
        self._full_win_buffers: Dict[int, List[SuccessfulTrajectory]] = defaultdict(list)
        self._full_win_scramble_index: Dict[int, Dict[bytes, int]] = defaultdict(dict)
        self._failure_buffers: Dict[int, deque[SuccessfulTrajectory]] = defaultdict(
            lambda: deque(maxlen=self.max_failures_per_depth)
        )
        self.total_added: int = 0
        self.total_upgraded: int = 0
        self.total_pruned: int = 0
        self.total_failures_added: int = 0
        self._entry_counter: int = 0

        # Contadores incrementais O(1) de telemetria e integridade
        self._running_total_trajectories: int = 0
        self._running_memory_bytes: int = 0
        self._running_uncompressed_bytes: int = 0
        self._running_full_wins: int = 0
        self._running_success_slices: int = 0

    def _track_add(self, trajectory: SuccessfulTrajectory) -> None:
        """Atualiza contadores incrementais O(1) na inserção de trajetória."""
        self._running_total_trajectories += 1
        self._running_memory_bytes += trajectory.memory_bytes
        self._running_uncompressed_bytes += trajectory.uncompressed_bytes
        if getattr(trajectory, "is_full_win", True):
            self._running_full_wins += 1
        else:
            self._running_success_slices += 1

    def _track_remove(self, trajectory: SuccessfulTrajectory) -> None:
        """Atualiza contadores incrementais O(1) na remoção/substituição de trajetória."""
        self._running_total_trajectories = max(0, self._running_total_trajectories - 1)
        self._running_memory_bytes = max(0, self._running_memory_bytes - trajectory.memory_bytes)
        self._running_uncompressed_bytes = max(0, self._running_uncompressed_bytes - trajectory.uncompressed_bytes)
        if getattr(trajectory, "is_full_win", True):
            self._running_full_wins = max(0, self._running_full_wins - 1)
        else:
            self._running_success_slices = max(0, self._running_success_slices - 1)

    @property
    def is_unlimited(self) -> bool:
        return self.max_trajectories is None and self.max_per_depth is None

    @property
    def memory_bytes(self) -> int:
        """Total de bytes ocupados em RAM pelas trajetórias armazenadas em O(1)."""
        if hasattr(self, "_running_memory_bytes"):
            return self._running_memory_bytes
        return sum(t.memory_bytes for b in self._depth_buffers.values() for t in b)

    @property
    def uncompressed_bytes(self) -> int:
        """Total de bytes equivalentes sem compactação lossless em O(1)."""
        if hasattr(self, "_running_uncompressed_bytes"):
            return self._running_uncompressed_bytes
        return sum(t.uncompressed_bytes for b in self._depth_buffers.values() for t in b)

    @property
    def compression_ratio(self) -> float:
        """Fator real de redução de memória RAM obtido pela compactação 100% lossless."""
        mem = self.memory_bytes
        if mem <= 0:
            return 1.0
        return float(self.uncompressed_bytes) / float(mem)

    @property
    def trajectories(self) -> List[SuccessfulTrajectory]:
        """Retorna uma lista plana de todas as trajetórias armazenadas em todas as profundidades ordenadas."""
        all_trajs: List[SuccessfulTrajectory] = []
        for depth in sorted(self._depth_buffers.keys()):
            all_trajs.extend(self._depth_buffers[depth])
        return all_trajs

    @trajectories.setter
    def trajectories(self, trajs: List[SuccessfulTrajectory]) -> None:
        self.clear()
        for t in trajs:
            self.add(t)

    @property
    def failures(self) -> List[SuccessfulTrajectory]:
        """Retorna uma lista plana de todas as trajetórias de falha em todas as profundidades."""
        all_fails: List[SuccessfulTrajectory] = []
        for depth in sorted(self._failure_buffers.keys()):
            all_fails.extend(list(self._failure_buffers[depth]))
        return all_fails

    def clear(self) -> None:
        """Limpa todos os estratos do buffer."""
        self._depth_buffers.clear()
        self._scramble_index.clear()
        self._full_win_buffers.clear()
        self._full_win_scramble_index.clear()
        self._failure_buffers.clear()
        self.total_added = 0
        self.total_upgraded = 0
        self.total_pruned = 0
        self.total_failures_added = 0
        self._entry_counter = 0
        self._running_total_trajectories = 0
        self._running_memory_bytes = 0
        self._running_uncompressed_bytes = 0
        self._running_full_wins = 0
        self._running_success_slices = 0

    def get_depth_distribution(self) -> Dict[int, int]:
        """Retorna a contagem de trajetórias armazenadas por profundidade."""
        return {d: len(trajs) for d, trajs in sorted(self._depth_buffers.items()) if len(trajs) > 0}

    def get_failure_depth_distribution(self) -> Dict[int, int]:
        """Retorna a contagem de falhas armazenadas por profundidade."""
        return {d: len(fails) for d, fails in sorted(self._failure_buffers.items()) if len(fails) > 0}

    def get_capacity_for_depth(self, depth: int) -> Optional[int]:
        """Calcula a cota máxima de trajetórias para um estrato específico (None se ilimitado)."""
        eff_max = self.max_per_depth if self.max_per_depth is not None else self.max_trajectories
        if eff_max is None:
            return None
        if not self.progressive_capacity or depth <= 0 or eff_max <= 4:
            return eff_max
        cap = int(self.min_capacity_per_depth * (2 ** max(0, depth - 1)))
        return max(1, min(eff_max, cap))

    def get_failure_capacity_for_depth(self, depth: int) -> int:
        """Calcula a cota máxima de falhas para DPO em um estrato específico."""
        if not self.progressive_capacity or depth <= 0 or self.max_failures_per_depth <= 4:
            return self.max_failures_per_depth
        f_cap = int(self.min_failures_per_depth * (2 ** max(0, depth - 1)))
        return max(1, min(self.max_failures_per_depth, f_cap))

    def get_depth_capacities(self) -> Dict[int, Optional[int]]:
        """Retorna as cotas de capacidade ativas para todas as profundidades presentes (None = ilimitado)."""
        return {d: self.get_capacity_for_depth(d) for d in sorted(self._depth_buffers.keys())}

    def __len__(self) -> int:
        """Retorna o total de trajetórias armazenadas em O(1)."""
        if hasattr(self, "_running_total_trajectories"):
            return self._running_total_trajectories
        return sum(len(b) for b in self._depth_buffers.values())

    def add_failure(self, trajectory: SuccessfulTrajectory) -> None:
        """Método legado para testes unitários isolados de contraste explícito."""
        depth = int(getattr(trajectory, "depth", 1))

        if self.is_visual and trajectory.obs.dtype != np.uint8:
            if np.issubdtype(trajectory.obs.dtype, np.floating):
                max_val = float(np.max(trajectory.obs)) if trajectory.obs.size > 0 else 1.0
                if max_val <= 1.0 + 1e-4:
                    comp_obs = np.clip(trajectory.obs * 255.0, 0, 255).astype(np.uint8)
                else:
                    comp_obs = np.clip(trajectory.obs, 0, 255).astype(np.uint8)
            else:
                comp_obs = np.clip(trajectory.obs, 0, 255).astype(np.uint8)
            trajectory.obs = comp_obs

        if self.lossless_compaction:
            trajectory.compact_lossless()

        f_cap = self.get_failure_capacity_for_depth(depth)
        if depth not in self._failure_buffers or self._failure_buffers[depth].maxlen != f_cap:
            existing = list(self._failure_buffers[depth]) if depth in self._failure_buffers else []
            self._failure_buffers[depth] = deque(existing[-f_cap:], maxlen=f_cap)

        self._failure_buffers[depth].append(trajectory)
        self.total_failures_added += 1

    def add(self, trajectory: SuccessfulTrajectory, slice_frontier: Optional[bool] = None) -> None:
        """Adiciona uma vitória completa ou fatia de sucesso ao estrato correspondente.

        No modo padrão sem teto (quota is None):
        - Compacta sem perdas (lossless_compaction=True) com 0.0 erro numérico.
        - Deduplica por estado inicial (hash(obs[0])): nunca expulsa outros scrambles, apenas atualiza
          o próprio scramble caso uma solução estritamente melhor/mais curta seja descoberta.
        """
        depth = int(getattr(trajectory, "depth", 1))
        d_buf = self._depth_buffers[depth]
        do_slice = self.frontier_slicing if slice_frontier is None else bool(slice_frontier)

        # 1. Filtro de Eficiência / Poda de Loops Erráticos ou Frontier Slicing
        if depth > 0:
            if len(d_buf) < 4:
                max_allowed = max(10, int(depth * 2.5))
            else:
                max_allowed = max(8, int(depth * 2.0))
            if trajectory.length > max_allowed:
                if do_slice:
                    L = max_allowed
                    cur_obs = trajectory.obs
                    cur_delta = trajectory.delta_obs
                    sliced_obs = cur_obs[:L]
                    sliced_delta = cur_delta[:L] if cur_delta is not None else None
                    sliced_prev_actions = trajectory.prev_actions[:L]
                    sliced_prev_rewards = trajectory.prev_rewards[:L]
                    sliced_actions = trajectory.actions[:L]
                    sliced_returns = trajectory.returns[:L]
                    sliced_logp = (
                        trajectory.log_probs[:L]
                        if getattr(trajectory, "log_probs", None) is not None
                        else None
                    )
                    td_err = getattr(trajectory, "td_error", 0.0)
                    orig_full_win = bool(getattr(trajectory, "is_full_win", True))
                    if isinstance(trajectory, FrontierTransition):
                        trajectory = FrontierTransition(
                            obs=sliced_obs,
                            delta_obs=sliced_delta,
                            prev_actions=sliced_prev_actions,
                            prev_rewards=sliced_prev_rewards,
                            actions=sliced_actions,
                            returns=sliced_returns,
                            depth=depth,
                            length=L,
                            log_probs=sliced_logp,
                            td_error=td_err,
                            is_full_win=orig_full_win,
                        )
                    else:
                        trajectory = SuccessfulTrajectory(
                            obs=sliced_obs,
                            delta_obs=sliced_delta,
                            prev_actions=sliced_prev_actions,
                            prev_rewards=sliced_prev_rewards,
                            actions=sliced_actions,
                            returns=sliced_returns,
                            depth=depth,
                            length=L,
                            log_probs=sliced_logp,
                            is_full_win=orig_full_win,
                        )
                else:
                    self.total_pruned += 1
                    return

        # 2. Compressão de Observações Visuais para uint8 [0, 255] em jogos pixel
        if self.is_visual and trajectory.obs.dtype != np.uint8:
            cur_obs = trajectory.obs
            if np.issubdtype(cur_obs.dtype, np.floating):
                max_val = float(np.max(cur_obs)) if cur_obs.size > 0 else 1.0
                if max_val <= 1.0 + 1e-4:
                    comp_obs = np.clip(cur_obs * 255.0, 0, 255).astype(np.uint8)
                else:
                    comp_obs = np.clip(cur_obs, 0, 255).astype(np.uint8)
            else:
                comp_obs = np.clip(cur_obs, 0, 255).astype(np.uint8)
            trajectory.obs = comp_obs

        # 3. Compactação 100% Lossless (0.0 perda de qualidade)
        init_key = trajectory.initial_state_key
        if self.lossless_compaction:
            trajectory.compact_lossless()

        self._entry_counter += 1
        trajectory._entry_id = self._entry_counter

        quota = self.get_capacity_for_depth(depth)

        # Modo Ilimitado (sem teto): crescimento livre + deduplicação/upgrade por estado inicial O(1)
        if quota is None and self.max_trajectories is None:
            if self.deduplicate_scrambles:
                s_map = self._scramble_index[depth]
                existing_idx = s_map.get(init_key, None)
                if existing_idx is not None and existing_idx < len(d_buf):
                    if trajectory.quality_key > d_buf[existing_idx].quality_key:
                        old_t = d_buf[existing_idx]
                        self._track_remove(old_t)
                        d_buf[existing_idx] = trajectory
                        self._track_add(trajectory)

                        if self.is_unlimited:
                            fw_map = self._full_win_scramble_index[depth]
                            fw_buf = self._full_win_buffers[depth]
                            fw_idx = fw_map.get(init_key, None)
                            if fw_idx is not None and fw_idx < len(fw_buf):
                                if trajectory.is_full_win:
                                    fw_buf[fw_idx] = trajectory
                            elif trajectory.is_full_win:
                                fw_map[init_key] = len(fw_buf)
                                fw_buf.append(trajectory)

                        self.total_added += 1
                        self.total_upgraded += 1
                    else:
                        self.total_pruned += 1
                    return
                s_map[init_key] = len(d_buf)
            d_buf.append(trajectory)
            self._track_add(trajectory)
            if self.is_unlimited and trajectory.is_full_win:
                if self.deduplicate_scrambles:
                    self._full_win_scramble_index[depth][init_key] = len(self._full_win_buffers[depth])
                self._full_win_buffers[depth].append(trajectory)
            self.total_added += 1
            return

        # Modo com cota explícita finita (compatibilidade com testes legados de cota fixa)
        if quota is not None and len(d_buf) >= quota:
            worst_traj = d_buf[0]
            if trajectory > worst_traj:
                removed_traj = heapq.heapreplace(d_buf, trajectory)
                self._track_remove(removed_traj)
                self._track_add(trajectory)
                self.total_added += 1
            else:
                self.total_pruned += 1
                return
        else:
            heapq.heappush(d_buf, trajectory)
            self._track_add(trajectory)
            self.total_added += 1

        if self.max_trajectories is not None:
            while len(self) > self.max_trajectories:
                non_empty_depths = [d for d in sorted(self._depth_buffers.keys()) if len(self._depth_buffers[d]) > 0]
                if not non_empty_depths:
                    break

                anchor_floor = 8
                eligible_depths = [
                    d for d in non_empty_depths
                    if len(self._depth_buffers[d]) > min(
                        anchor_floor,
                        self.get_capacity_for_depth(d) or anchor_floor,
                    )
                ]
                if not eligible_depths:
                    eligible_depths = non_empty_depths

                target_d = eligible_depths[0]
                popped_traj = heapq.heappop(self._depth_buffers[target_d])
                self._track_remove(popped_traj)
                if len(self._depth_buffers[target_d]) == 0:
                    del self._depth_buffers[target_d]

    def _collate_trajectories(
        self,
        sampled_trajs: List[SuccessfulTrajectory],
    ) -> Dict[str, torch.Tensor]:
        """Empacota uma lista de trajetórias heterogêneas em lote pré-alocado no dispositivo."""
        k = len(sampled_trajs)
        max_len = max(t.length for t in sampled_trajs)

        # Descompacta cada trajetória amostrada exatamente uma vez para eliminar descompressões redundantes
        obs_list = [t.obs for t in sampled_trajs]
        delta_list = [t.delta_obs for t in sampled_trajs]

        # Prealocação de tensores padronizados com zeros
        b_obs = np.zeros((k, max_len, *self.obs_shape), dtype=np.float32)
        has_delta = any(d_arr is not None for d_arr in delta_list)
        if has_delta:
            max_delta_dim = max(
                (d_arr.shape[-1] if d_arr.ndim > 1 else d_arr.shape[0])
                for d_arr in delta_list
                if d_arr is not None
            )
            b_delta = np.zeros((k, max_len, max_delta_dim), dtype=np.float32)
        else:
            b_delta = None

        act_dtype = np.int64 if self.is_discrete else np.float32
        if self.is_discrete:
            b_prev_act = np.zeros((k, max_len), dtype=np.int64)
        elif len(self.action_shape) > 1 and self.action_shape[0] > 1:
            b_prev_act = np.zeros((k, max_len, *self.action_shape[1:]), dtype=np.float32)
        else:
            b_prev_act = np.zeros((k, max_len, *self.action_shape), dtype=act_dtype)
        b_prev_rew = np.zeros((k, max_len, 1), dtype=np.float32)
        b_act = np.zeros((k, max_len, *self.action_shape), dtype=act_dtype)
        b_ret = np.zeros((k, max_len), dtype=np.float32)
        b_logp = np.zeros((k, max_len), dtype=np.float32)
        b_mask = np.zeros((k, max_len), dtype=np.float32)

        for i, t in enumerate(sampled_trajs):
            L = t.length
            obs_arr = obs_list[i]
            delta_arr = delta_list[i]
            if self.is_visual and obs_arr.dtype == np.uint8:
                b_obs[i, :L] = obs_arr.astype(np.float32) / 255.0
            else:
                b_obs[i, :L] = obs_arr
            if has_delta and delta_arr is not None:
                d_dim = min(b_delta.shape[-1], delta_arr.shape[-1])
                b_delta[i, :L, :d_dim] = delta_arr[:, :d_dim]
            b_prev_act[i, :L] = t.prev_actions
            b_prev_rew[i, :L, 0] = t.prev_rewards
            b_act[i, :L] = t.actions
            b_ret[i, :L] = t.returns
            if getattr(t, "log_probs", None) is not None:
                b_logp[i, :min(L, len(t.log_probs))] = t.log_probs[:min(L, len(t.log_probs))]
            b_mask[i, :L] = 1.0

        tensor_obs = torch.tensor(b_obs, dtype=torch.float32, device=self.device)
        tensor_delta = torch.tensor(b_delta, dtype=torch.float32, device=self.device) if has_delta else None
        tensor_prev_act = torch.tensor(
            b_prev_act,
            dtype=torch.long if self.is_discrete else torch.float32,
            device=self.device,
        )
        if self.is_discrete and tensor_prev_act.dim() == 2:
            tensor_prev_act = tensor_prev_act.unsqueeze(-1)

        tensor_prev_rew = torch.tensor(b_prev_rew, dtype=torch.float32, device=self.device)
        tensor_act = torch.tensor(
            b_act,
            dtype=torch.long if self.is_discrete else torch.float32,
            device=self.device,
        )
        tensor_ret = torch.tensor(b_ret, dtype=torch.float32, device=self.device)
        tensor_old_logp = torch.tensor(b_logp, dtype=torch.float32, device=self.device)
        tensor_mask = torch.tensor(b_mask, dtype=torch.float32, device=self.device)
        tensor_hx = torch.zeros(1, k, self.hidden_dim, device=self.device, dtype=torch.float32)

        return {
            "obs": tensor_obs,
            "delta_obs": tensor_delta,
            "prev_actions": tensor_prev_act,
            "prev_rewards": tensor_prev_rew,
            "actions": tensor_act,
            "returns": tensor_ret,
            "old_log_probs": tensor_old_logp,
            "mask": tensor_mask,
            "hx": tensor_hx,
            "dones": None,
        }

    def count_full_wins(self) -> int:
        """Retorna o total de trajetórias de vitória terminal completa (is_full_win == True) em O(1)."""
        if hasattr(self, "_running_full_wins"):
            return self._running_full_wins
        return sum(
            1
            for b in self._depth_buffers.values()
            for t in b
            if getattr(t, "is_full_win", True)
        )

    def count_success_slices(self) -> int:
        """Retorna o total de fatias de sucesso parcial (Peak Slicing, is_full_win == False) em O(1)."""
        if hasattr(self, "_running_success_slices"):
            return self._running_success_slices
        return sum(
            1
            for b in self._depth_buffers.values()
            for t in b
            if not getattr(t, "is_full_win", True)
        )

    def _pick_trajectory(
        self,
        pool_list: List[SuccessfulTrajectory],
        tournament_size: int = 3,
    ) -> SuccessfulTrajectory:
        """Seleciona uma trajetória do pool via Torneio O(1) avaliando metadados leves de cabeçalho.

        Agnóstico e universal:
        1. Prioriza vitória completa verificada (is_full_win == True).
        2. Prioriza eficiência de caminho (maior razão profundidade/passos: d / L).
        3. Prioriza maior retorno acumulado (returns[0]).

        Nunca descompacta arrays comprimidos de observações durante a seleção, garantindo
        latência estritamente O(1) da ordem de microssegundos mesmo com 1M+ trajetórias.
        """
        n = len(pool_list)
        if n == 0:
            raise ValueError("Cannot pick from empty trajectory pool")
        if tournament_size <= 1 or n == 1:
            return pool_list[np.random.randint(0, n)]

        k = min(tournament_size, n)
        idxs = np.random.randint(0, n, size=k)
        cands = [pool_list[int(i)] for i in idxs]

        def _traj_score(t: SuccessfulTrajectory) -> Tuple[int, float, float]:
            is_fw = 1 if getattr(t, "is_full_win", True) else 0
            d = float(getattr(t, "depth", 1))
            length = max(1, int(getattr(t, "length", 1)))
            eff = (d / length) if d > 0 else (1.0 / length)
            ret = float(t.returns[0]) if (t.returns is not None and len(t.returns) > 0) else 0.0
            return (is_fw, eff, ret)

        return max(cands, key=_traj_score)

    def sample_batch(
        self,
        batch_size: int = 8,
        target_depth: Optional[int] = None,
        only_full_wins: bool = False,
        tournament_size: int = 3,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Amostra um mini-lote padronizado de trajetórias/transições com balanceamento fronteira/âncora,
        seleção por torneio O(1) e máscara temporal.
        """
        if len(self) == 0:
            return None

        if only_full_wins:
            if self.is_unlimited and hasattr(self, "_full_win_buffers"):
                pools: Dict[int, List[SuccessfulTrajectory]] = {
                    d: trajs for d, trajs in self._full_win_buffers.items() if len(trajs) > 0
                }
            else:
                pools = {
                    d: [t for t in trajs if getattr(t, "is_full_win", True)]
                    for d, trajs in self._depth_buffers.items()
                }
                pools = {d: trajs for d, trajs in pools.items() if len(trajs) > 0}
        else:
            pools = {d: trajs for d, trajs in self._depth_buffers.items() if len(trajs) > 0}

        total_available = sum(len(trajs) for trajs in pools.values())
        k = min(batch_size, total_available)
        if k <= 0:
            return None

        # Saturação de peso por estrato em 256.0 no modo ilimitado: impede que estratos maduros
        # com 100.000+ vitórias afoguem novas fronteiras recém-desbloqueadas (ex: D12/D13),
        # mantendo amostragem uniforme sobre todas as vitórias dentro do estrato escolhido.
        depth_weight_cap = 256.0

        if target_depth is None:
            eligible_depths = list(pools.keys())
            depth_counts = np.minimum(
                np.array([len(pools[d]) for d in eligible_depths], dtype=np.float64),
                depth_weight_cap,
            )
            depth_probs = depth_counts / depth_counts.sum()
            chosen_depths = np.random.choice(eligible_depths, size=k, replace=True, p=depth_probs)
            sampled_trajs = [
                self._pick_trajectory(pools[d], tournament_size=tournament_size)
                for d in chosen_depths
            ]
        else:
            frontier_min_depth = max(1, target_depth - 1)
            frontier_depths = [d for d in pools.keys() if d >= frontier_min_depth]
            anchor_depths = [d for d in pools.keys() if d < frontier_min_depth]

            sampled_trajs: List[SuccessfulTrajectory] = []

            if len(frontier_depths) > 0 and len(anchor_depths) > 0:
                # Prioritização de Fronteira: 75% fronteira ativa / 25% âncoras fundacionais
                n_frontier = max(1, int(round(k * self.frontier_ratio)))
                if n_frontier >= k and k > 1:
                    n_frontier = k - 1
                n_anchor = k - n_frontier

                # Amostragem de fronteira
                f_counts = np.minimum(
                    np.array([len(pools[d]) for d in frontier_depths], dtype=np.float64),
                    depth_weight_cap,
                )
                if getattr(self, "dynamic_frontier", True):
                    f_depth_arr = np.array(frontier_depths, dtype=np.float64)
                    f_weights = f_counts * np.exp(-self.proximal_decay * np.abs(target_depth - f_depth_arr))
                else:
                    f_weights = f_counts
                f_weights = f_weights / f_weights.sum()
                f_chosen_depths = np.random.choice(frontier_depths, size=n_frontier, replace=True, p=f_weights)

                sampled_trajs.extend(
                    self._pick_trajectory(pools[d], tournament_size=tournament_size)
                    for d in f_chosen_depths
                )

                # Amostragem de âncoras
                if n_anchor > 0:
                    a_counts = np.minimum(
                        np.array([len(pools[d]) for d in anchor_depths], dtype=np.float64),
                        depth_weight_cap,
                    )
                    a_weights = a_counts / a_counts.sum()
                    a_chosen_depths = np.random.choice(anchor_depths, size=n_anchor, replace=True, p=a_weights)
                    sampled_trajs.extend(
                        self._pick_trajectory(pools[d], tournament_size=tournament_size)
                        for d in a_chosen_depths
                    )

            elif len(frontier_depths) > 0:
                f_counts = np.minimum(
                    np.array([len(pools[d]) for d in frontier_depths], dtype=np.float64),
                    depth_weight_cap,
                )
                if getattr(self, "dynamic_frontier", True):
                    f_depth_arr = np.array(frontier_depths, dtype=np.float64)
                    f_weights = f_counts * np.exp(-self.proximal_decay * np.abs(target_depth - f_depth_arr))
                else:
                    f_weights = f_counts
                f_weights = f_weights / f_weights.sum()
                chosen_depths = np.random.choice(frontier_depths, size=k, replace=True, p=f_weights)

                sampled_trajs.extend(
                    self._pick_trajectory(pools[d], tournament_size=tournament_size)
                    for d in chosen_depths
                )
            else:
                a_counts = np.minimum(
                    np.array([len(pools[d]) for d in anchor_depths], dtype=np.float64),
                    depth_weight_cap,
                )
                a_weights = a_counts / a_counts.sum()
                chosen_depths = np.random.choice(anchor_depths, size=k, replace=True, p=a_weights)
                sampled_trajs.extend(
                    self._pick_trajectory(pools[d], tournament_size=tournament_size)
                    for d in chosen_depths
                )

        return self._collate_trajectories(sampled_trajs)

    def sample_contrastive_pairs(
        self,
        batch_size: int = 8,
        target_depth: Optional[int] = None,
        only_full_wins: bool = False,
        tournament_size: int = 3,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Amostra pares contrastivos (vitória, falha) emparelhados pelo MESMO estrato (d_w == d_l).
        
        Garante simetria de contexto ecológico: nunca contrasta uma vitória trivial (D1)
        com uma falha complexa (D7), evitando viés espúrio de horizonte.
        
        Retorna (batch_win, batch_loss) ou None se não houver pares elegíveis suficientes.
        """
        if only_full_wins:
            if self.is_unlimited and hasattr(self, "_full_win_buffers"):
                win_pools: Dict[int, List[SuccessfulTrajectory]] = {
                    d: trajs for d, trajs in self._full_win_buffers.items() if len(trajs) > 0
                }
            else:
                win_pools = {
                    d: [t for t in trajs if getattr(t, "is_full_win", True)]
                    for d, trajs in self._depth_buffers.items()
                }
        else:
            win_pools = self._depth_buffers

        eligible_depths = [
            d for d, w_list in win_pools.items()
            if len(w_list) > 0 and len(self._failure_buffers.get(d, [])) > 0
        ]
        if not eligible_depths:
            return None

        total_pairs = sum(min(len(win_pools[d]), len(self._failure_buffers[d])) for d in eligible_depths)
        k = max(1, min(batch_size, total_pairs))

        # Separação entre fronteira e âncoras se target_depth fornecido
        if target_depth is not None:
            frontier_min = max(1, target_depth - 1)
            frontier_depths = [d for d in eligible_depths if d >= frontier_min]
            anchor_depths = [d for d in eligible_depths if d < frontier_min]
        else:
            frontier_depths = eligible_depths
            anchor_depths = []

        chosen_depths: List[int] = []
        if frontier_depths and anchor_depths:
            n_front = max(1, int(round(k * self.frontier_ratio)))
            if n_front >= k and k > 1:
                n_front = k - 1
            n_anc = k - n_front
            if getattr(self, "dynamic_frontier", True):
                f_w = np.array([np.exp(-self.proximal_decay * abs(target_depth - d)) for d in frontier_depths], dtype=np.float64)
                f_w /= f_w.sum()
                chosen_depths.extend(np.random.choice(frontier_depths, size=n_front, replace=True, p=f_w))
            else:
                chosen_depths.extend(np.random.choice(frontier_depths, size=n_front, replace=True))
            chosen_depths.extend(np.random.choice(anchor_depths, size=n_anc, replace=True))
        elif frontier_depths:
            if getattr(self, "dynamic_frontier", True) and target_depth is not None:
                f_w = np.array([np.exp(-self.proximal_decay * abs(target_depth - d)) for d in frontier_depths], dtype=np.float64)
                f_w /= f_w.sum()
                chosen_depths.extend(np.random.choice(frontier_depths, size=k, replace=True, p=f_w))
            else:
                chosen_depths.extend(np.random.choice(frontier_depths, size=k, replace=True))
        else:
            chosen_depths.extend(np.random.choice(anchor_depths, size=k, replace=True))

        sampled_wins: List[SuccessfulTrajectory] = []
        sampled_losses: List[SuccessfulTrajectory] = []

        for d in chosen_depths:
            w_candidates = win_pools[d]
            l_candidates = list(self._failure_buffers[d])
            w_traj = self._pick_trajectory(w_candidates, tournament_size=tournament_size)
            l_idx = np.random.randint(0, len(l_candidates))
            sampled_wins.append(w_traj)
            sampled_losses.append(l_candidates[l_idx])

        batch_win = self._collate_trajectories(sampled_wins)
        batch_loss = self._collate_trajectories(sampled_losses)

        return batch_win, batch_loss


# Alias universal para a Arquitetura de Transições de Fronteira
FrontierTransitionBuffer = SuccessReplayBuffer

__all__ = [
    "RecurrentBatch",
    "RecurrentRolloutBuffer",
    "SuccessfulTrajectory",
    "FrontierTransition",
    "SuccessReplayBuffer",
    "FrontierTransitionBuffer",
]

