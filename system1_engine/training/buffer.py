from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
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

        self.step_idx += 1

    def compute_gae(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Computes Generalized Advantage Estimation (GAE) and target returns."""
        assert self.step_idx == self.buffer_size, (
            f"Buffer not full: {self.step_idx}/{self.buffer_size}"
        )
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
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
            )
