from collections import defaultdict, deque
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


@dataclass
class SuccessfulTrajectory:
    """Trajetória completa de um episódio vitorioso/resolvido ou negativo (falha)."""
    obs: np.ndarray                  # [T, obs_dim]
    delta_obs: Optional[np.ndarray]  # [T, obs_dim] ou None
    prev_actions: np.ndarray         # [T, ...]
    prev_rewards: np.ndarray         # [T]
    actions: np.ndarray              # [T, ...]
    returns: np.ndarray              # [T] retornos descontados reais
    depth: int                       # Profundidade curricular ou do scramble
    length: int                      # Duração T em passos
    log_probs: Optional[np.ndarray] = None  # [T] log_probs da política de rollout (pi_old / pi_ref)


@dataclass
class FrontierTransition(SuccessfulTrajectory):
    """Transição atômica ou micro-manobra decisiva de fronteira (Bellman 1-step / n-step breakthrough)."""
    td_error: float = 0.0

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
        )


class SuccessReplayBuffer:
    """Buffer estratificado por profundidade de trajetórias completas e micro-transições para SIL e Frontier Replay.

    Preserva trajetórias vitoriosas e transições de avanço onde o agente atingiu o objetivo (ex: cubo resolvido),
    organizadas em estratos por profundidade curricular (D1..D10+) para eliminar esquecimento
    catastrófico e despejo cruzado (cross-depth eviction).
    
    A amostragem equilibra 75% de transições da fronteira ativa (reforço imediato) com 25% de
    âncoras fundacionais de níveis anteriores (regularização contra regressão).
    """

    def __init__(
        self,
        max_trajectories: int = 10240,
        max_per_depth: int = 2048,
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
    ) -> None:
        self.max_trajectories = max(2, int(max_trajectories))
        self.max_per_depth = max(1, min(int(max_per_depth), self.max_trajectories))
        self.min_capacity_per_depth = max(1, min(int(min_capacity_per_depth), self.max_per_depth))
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

        self._depth_buffers: Dict[int, List[SuccessfulTrajectory]] = defaultdict(list)
        self._failure_buffers: Dict[int, deque[SuccessfulTrajectory]] = defaultdict(
            lambda: deque(maxlen=self.max_failures_per_depth)
        )
        self.total_added: int = 0
        self.total_pruned: int = 0
        self.total_failures_added: int = 0

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
        """Limpa todos os estratos vitoriosos e negativos do buffer."""
        self._depth_buffers.clear()
        self._failure_buffers.clear()
        self.total_added = 0
        self.total_pruned = 0
        self.total_failures_added = 0

    def get_depth_distribution(self) -> Dict[int, int]:
        """Retorna a contagem de trajetórias armazenadas por profundidade."""
        return {d: len(trajs) for d, trajs in sorted(self._depth_buffers.items()) if len(trajs) > 0}

    def get_failure_depth_distribution(self) -> Dict[int, int]:
        """Retorna a contagem de falhas armazenadas por profundidade."""
        return {d: len(fails) for d, fails in sorted(self._failure_buffers.items()) if len(fails) > 0}

    def get_capacity_for_depth(self, depth: int) -> int:
        """Calcula a cota máxima de trajetórias para um estrato específico.

        No modo progressivo, aloca capacidade proporcional à entropia combinatorial do nível:
        cap(d) = clamp(C_0 * 2^(d - 1), C_0, C_max)
        Onde C_0 = min_capacity_per_depth (ex: 32) e C_max = max_per_depth (ex: 2048).
        """
        if not self.progressive_capacity or depth <= 0 or self.max_per_depth <= 4:
            return self.max_per_depth
        cap = int(self.min_capacity_per_depth * (2 ** max(0, depth - 1)))
        return max(1, min(self.max_per_depth, cap))

    def get_failure_capacity_for_depth(self, depth: int) -> int:
        """Calcula a cota máxima de falhas para DPO em um estrato específico.

        No modo progressivo, aloca cota reduzida para erros triviais rasos (F_0 = 16)
        e cota expandida para erros sutis de alta complexidade (F_max = max_failures_per_depth, ex: 512).
        """
        if not self.progressive_capacity or depth <= 0 or self.max_failures_per_depth <= 4:
            return self.max_failures_per_depth
        f_cap = int(self.min_failures_per_depth * (2 ** max(0, depth - 1)))
        return max(1, min(self.max_failures_per_depth, f_cap))

    def get_depth_capacities(self) -> Dict[int, int]:
        """Retorna as cotas de capacidade ativas para todas as profundidades presentes."""
        return {d: self.get_capacity_for_depth(d) for d in sorted(self._depth_buffers.keys())}

    def __len__(self) -> int:
        return sum(len(b) for b in self._depth_buffers.values())

    def add_failure(self, trajectory: SuccessfulTrajectory) -> None:
        """Adiciona uma trajetória de falha/erro ao buffer negativo estratificado para DPO contrastivo."""
        depth = int(getattr(trajectory, "depth", 1))

        # Compressão de Observações Visuais para uint8 [0, 255]
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

        # Alocação dinâmica de capacidade progressiva para falhas
        f_cap = self.get_failure_capacity_for_depth(depth)
        if depth not in self._failure_buffers or self._failure_buffers[depth].maxlen != f_cap:
            existing = list(self._failure_buffers[depth]) if depth in self._failure_buffers else []
            self._failure_buffers[depth] = deque(existing[-f_cap:], maxlen=f_cap)

        self._failure_buffers[depth].append(trajectory)
        self.total_failures_added += 1

    def add(self, trajectory: SuccessfulTrajectory, slice_frontier: Optional[bool] = None) -> None:
        """Adiciona uma trajetória ou micro-manobra de vitória ao estrato correspondente.

        Aplica poda de loops erráticos ou Frontier Slicing (para resgatar o avanço de fronteira),
        compressão de observações visuais para uint8, e substituição prioritária por qualidade/retorno.
        """
        depth = int(getattr(trajectory, "depth", 1))
        d_buf = self._depth_buffers[depth]
        do_slice = self.frontier_slicing if slice_frontier is None else bool(slice_frontier)

        # 1. Filtro de Eficiência / Poda de Loops Erráticos ou Frontier Slicing:
        # Rejeita trajetórias excessivamente longas/erráticas ou extrai o prefixo de fronteira decisivo.
        if depth > 0:
            if len(d_buf) < 4:
                max_allowed = max(10, int(depth * 2.5))
            else:
                max_allowed = max(8, int(depth * 2.0))
            if trajectory.length > max_allowed:
                if do_slice:
                    # Frontier Slicing: preserva a descoberta no nível de fronteira cortando para o prefixo decisivo
                    L = max_allowed
                    sliced_obs = trajectory.obs[:L]
                    sliced_delta = (
                        trajectory.delta_obs[:L]
                        if trajectory.delta_obs is not None
                        else None
                    )
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
                        )
                else:
                    self.total_pruned += 1
                    return

        # 2. Compressão de Observações Visuais para uint8 [0, 255] (Redução de 75% a 90% de RAM em jogos pixel)
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

        t_ret = float(trajectory.returns[0]) if len(trajectory.returns) > 0 else 0.0
        t_len = int(trajectory.length)
        new_quality = (t_ret, -t_len)

        # 3. Quota local por estrato com substituição prioritária por qualidade:
        quota = self.get_capacity_for_depth(depth)
        if len(d_buf) >= quota:
            worst_idx = min(
                range(len(d_buf)),
                key=lambda i: (
                    float(d_buf[i].returns[0]) if len(d_buf[i].returns) > 0 else 0.0,
                    -int(d_buf[i].length),
                ),
            )
            worst_traj = d_buf[worst_idx]
            worst_quality = (
                float(worst_traj.returns[0]) if len(worst_traj.returns) > 0 else 0.0,
                -int(worst_traj.length),
            )
            if new_quality >= worst_quality:
                d_buf[worst_idx] = trajectory
                self.total_added += 1
            else:
                self.total_pruned += 1
                return
        else:
            d_buf.append(trajectory)
            self.total_added += 1

        # 4. Limite global com Proteção de Âncoras Fundacionais:
        # Se o total exceder max_trajectories, descarte a pior trajetória de profundidades que possuem
        # excedente além do piso fundacional (anchor_floor = 8). Nunca extingue um estrato completo prematuramente.
        while len(self) > self.max_trajectories:
            non_empty_depths = [d for d in sorted(self._depth_buffers.keys()) if len(self._depth_buffers[d]) > 0]
            if not non_empty_depths:
                break

            anchor_floor = 8
            eligible_depths = [
                d for d in non_empty_depths
                if len(self._depth_buffers[d]) > min(anchor_floor, self.get_capacity_for_depth(d))
            ]
            if not eligible_depths:
                eligible_depths = non_empty_depths

            target_d = eligible_depths[0]
            worst_idx = min(
                range(len(self._depth_buffers[target_d])),
                key=lambda i: (
                    float(self._depth_buffers[target_d][i].returns[0]) if len(self._depth_buffers[target_d][i].returns) > 0 else 0.0,
                    -int(self._depth_buffers[target_d][i].length),
                ),
            )
            self._depth_buffers[target_d].pop(worst_idx)
            if len(self._depth_buffers[target_d]) == 0:
                del self._depth_buffers[target_d]

    def _collate_trajectories(
        self,
        sampled_trajs: List[SuccessfulTrajectory],
    ) -> Dict[str, torch.Tensor]:
        """Empacota uma lista de trajetórias heterogêneas em lote pré-alocado no dispositivo."""
        k = len(sampled_trajs)
        max_len = max(t.length for t in sampled_trajs)

        # Prealocação de tensores padronizados com zeros
        b_obs = np.zeros((k, max_len, *self.obs_shape), dtype=np.float32)
        has_delta = any(t.delta_obs is not None for t in sampled_trajs)
        b_delta = np.zeros((k, max_len, *self.obs_shape), dtype=np.float32) if has_delta else None

        act_dtype = np.int64 if self.is_discrete else np.float32
        b_prev_act = np.zeros((k, max_len, *self.action_shape), dtype=act_dtype)
        b_prev_rew = np.zeros((k, max_len, 1), dtype=np.float32)
        b_act = np.zeros((k, max_len, *self.action_shape), dtype=act_dtype)
        b_ret = np.zeros((k, max_len), dtype=np.float32)
        b_logp = np.zeros((k, max_len), dtype=np.float32)
        b_mask = np.zeros((k, max_len), dtype=np.float32)

        for i, t in enumerate(sampled_trajs):
            L = t.length
            if self.is_visual and t.obs.dtype == np.uint8:
                b_obs[i, :L] = t.obs.astype(np.float32) / 255.0
            else:
                b_obs[i, :L] = t.obs
            if has_delta and t.delta_obs is not None:
                b_delta[i, :L] = t.delta_obs
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

    def sample_batch(
        self,
        batch_size: int = 8,
        target_depth: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Amostra um mini-lote padronizado de trajetórias/transições com balanceamento fronteira/âncora e máscara temporal."""
        if len(self) == 0:
            return None

        total_available = len(self)
        k = min(batch_size, total_available)
        if k <= 0:
            return None

        if target_depth is None:
            all_trajs = self.trajectories
            sampled_indices = np.random.choice(len(all_trajs), size=k, replace=(k > len(all_trajs)))
            sampled_trajs = [all_trajs[i] for i in sampled_indices]
        else:
            frontier_min_depth = max(1, target_depth - 1)
            frontier = [t for t in self.trajectories if t.depth >= frontier_min_depth]
            anchors = [t for t in self.trajectories if t.depth < frontier_min_depth]

            if len(frontier) > 0 and len(anchors) > 0:
                # Prioritização de Fronteira: 75% fronteira ativa / 25% âncoras fundacionais
                n_frontier = max(1, int(round(k * self.frontier_ratio)))
                if n_frontier >= k and k > 1:
                    n_frontier = k - 1
                n_anchor = k - n_frontier
                f_indices = np.random.choice(len(frontier), size=n_frontier, replace=(n_frontier > len(frontier)))
                a_indices = np.random.choice(len(anchors), size=n_anchor, replace=(n_anchor > len(anchors)))
                sampled_trajs = [frontier[i] for i in f_indices] + [anchors[j] for j in a_indices]
            elif len(frontier) > 0:
                f_indices = np.random.choice(len(frontier), size=k, replace=(k > len(frontier)))
                sampled_trajs = [frontier[i] for i in f_indices]
            else:
                a_indices = np.random.choice(len(anchors), size=k, replace=(k > len(anchors)))
                sampled_trajs = [anchors[j] for j in a_indices]

        return self._collate_trajectories(sampled_trajs)

    def sample_contrastive_pairs(
        self,
        batch_size: int = 8,
        target_depth: Optional[int] = None,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Amostra pares contrastivos (vitória, falha) emparelhados pelo MESMO estrato (d_w == d_l).
        
        Garante simetria de contexto ecológico: nunca contrasta uma vitória trivial (D1)
        com uma falha complexa (D7), evitando viés espúrio de horizonte.
        
        Retorna (batch_win, batch_loss) ou None se não houver pares elegíveis suficientes.
        """
        eligible_depths = [
            d for d in self._depth_buffers.keys()
            if len(self._depth_buffers[d]) > 0 and len(self._failure_buffers.get(d, [])) > 0
        ]
        if not eligible_depths:
            return None

        total_pairs = sum(min(len(self._depth_buffers[d]), len(self._failure_buffers[d])) for d in eligible_depths)
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
            chosen_depths.extend(np.random.choice(frontier_depths, size=n_front, replace=True))
            chosen_depths.extend(np.random.choice(anchor_depths, size=n_anc, replace=True))
        elif frontier_depths:
            chosen_depths.extend(np.random.choice(frontier_depths, size=k, replace=True))
        else:
            chosen_depths.extend(np.random.choice(anchor_depths, size=k, replace=True))

        sampled_wins: List[SuccessfulTrajectory] = []
        sampled_losses: List[SuccessfulTrajectory] = []

        for d in chosen_depths:
            w_candidates = self._depth_buffers[d]
            l_candidates = list(self._failure_buffers[d])
            w_idx = np.random.randint(0, len(w_candidates))
            l_idx = np.random.randint(0, len(l_candidates))
            sampled_wins.append(w_candidates[w_idx])
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

