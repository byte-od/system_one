import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution

from system1_engine.core.encoders import ImpalaVisualFrontEnd, VectorFrontEnd
from system1_engine.core.ensemble import ConfidenceGatedEnsemble
from system1_engine.core.heads import CategoricalPolicyHead, GaussianPolicyHead, ValueHead
from system1_engine.core.trunk import LeanResidualTrunk, LRUResidualTrunk, System1Trunk


@dataclass(frozen=True)
class ReflexDecision:
    """Resultado estruturado e amortizado da decisão do System 1 com telemetria de incerteza."""

    action: Union[int, np.ndarray]
    confidence: float          # [0.0, 1.0] - Probabilidade da ação escolhida
    uncertainty: float         # [0.0, 1.0] - Entropia normalizada
    margin: float              # [0.0, 1.0] - Diferença entre as duas maiores probabilidades
    is_uncertain: bool         # Indicador de incerteza do reflexo
    entropy: float             # Entropia bruta de Shannon
    latent_value: Optional[float] = None  # Valor estimado V(s) se solicitado
    calibration: float = 0.0   # [0.0, 1.0] - Nível de calibração / temperatura aplicada
    latency_ms: float = 0.0    # Latência de execução do reflexo em ms
    ensemble_fallback_triggered: bool = False  # Flag indicando se o fallback de ensemble foi ativado sob incerteza
    value_rerank_triggered: bool = False       # Flag indicando se o re-ranqueamento local por valor foi ativado


class UniversalS1Agent(nn.Module):
    """Universal System 1 RL Agent.

    Strict decoupling between:
      1. Front-end (Perception): VectorFrontEnd or ImpalaVisualFrontEnd
      2. Trunk (Recurrent Reflexive Core): System1Trunk or LeanResidualTrunk
      3. Heads (Decision & Value): PolicyHead and ValueHead

    Guarantees:
      - Memory footprint: ~650k-1.6M parameters (vector) or ~2.5M (visual).
      - Inference latency: <= 1.0 ms on CPU for act_fast().
      - Zero catastrophic forgetting via weight freezing on trunk.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        is_visual: Optional[bool] = None,
        trunk: Optional[nn.Module] = None,
        reward_fixed: bool = False,
        trunk_type: str = "classic",
        action_chunk_size: int = 1,
        tri_compass: Optional[bool] = None,
        relational: bool = True,
    ) -> None:
        super().__init__()
        raw_obs_space = obs_space
        if isinstance(obs_space, gym.spaces.Dict) and "obs" in obs_space.spaces:
            obs_space = obs_space.spaces["obs"]
        self.obs_space = obs_space
        self.action_space = action_space
        self.action_chunk_size = max(1, int(action_chunk_size))
        self.relational = bool(relational)
        self.scar_memory = None

        if is_visual is None:
            is_visual = (
                isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3
            )
        self.is_visual = is_visual

        if self.is_visual:
            self.tri_compass = False
        elif tri_compass is None:
            if isinstance(raw_obs_space, gym.spaces.Dict) and "delta_obs" in raw_obs_space.spaces:
                obs_dim = int(np.prod(obs_space.shape))
                delta_dim = int(np.prod(raw_obs_space.spaces["delta_obs"].shape))
                tri_compass = (delta_dim >= 3 * obs_dim)
            else:
                tri_compass = True
            self.tri_compass = bool(tri_compass)
        else:
            self.tri_compass = bool(tri_compass)

        # 1. Perception Front-End (Scenario-dependent)
        if self.is_visual:
            assert isinstance(obs_space, gym.spaces.Box), "Visual space must be gym.spaces.Box"
            in_channels = obs_space.shape[0]
            self.front_end: Union[VectorFrontEnd, ImpalaVisualFrontEnd] = (
                ImpalaVisualFrontEnd(
                    in_channels=in_channels,
                    action_space=action_space,
                    reward_fixed=reward_fixed,
                )
            )
        else:
            assert isinstance(obs_space, gym.spaces.Box), "Vector space must be gym.spaces.Box"
            obs_dim = int(np.prod(obs_space.shape))
            self.front_end = VectorFrontEnd(
                obs_dim=obs_dim,
                action_space=action_space,
                reward_fixed=reward_fixed,
                tri_compass=self.tri_compass,
                relational=self.relational,
            )

        # 2. Universal Recurrent Trunk (Can be shared or transferred)
        front_end_out_dim = getattr(self.front_end, "out_dim", 465 if self.tri_compass else 337)
        if trunk is not None:
            self.trunk = trunk
        elif str(trunk_type).lower() in ("lru", "lru_residual", "v3"):
            self.trunk = LRUResidualTrunk(input_dim=front_end_out_dim, fast_dim=256, slow_dim=256)
        elif str(trunk_type).lower() in ("lean", "lean_residual", "v2"):
            self.trunk = LeanResidualTrunk(input_dim=front_end_out_dim, fast_dim=256, slow_dim=256)
        else:
            self.trunk = System1Trunk(input_dim=front_end_out_dim, hidden_dim=256, num_res_blocks=2)

        trunk_hidden_dim = getattr(self.trunk, "hidden_dim", 256)

        # 3. Policy & Value Heads (Scenario-dependent)
        if isinstance(action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.policy_head: Union[CategoricalPolicyHead, GaussianPolicyHead] = (
                CategoricalPolicyHead(
                    in_features=trunk_hidden_dim,
                    n_actions=int(action_space.n),
                    action_chunk_size=self.action_chunk_size,
                )
            )
        elif isinstance(action_space, gym.spaces.Box):
            self.is_discrete = False
            act_dim = int(np.prod(action_space.shape))
            self.policy_head = GaussianPolicyHead(
                in_features=trunk_hidden_dim,
                act_dim=act_dim,
                action_chunk_size=self.action_chunk_size,
            )
        else:
            raise NotImplementedError(f"Unsupported action space: {type(action_space)}")

        self.value_head = ValueHead(in_features=trunk_hidden_dim)

        # Runtime hidden state for act_fast inference and homeostatic auto-calibration
        self.hx: Optional[torch.Tensor] = None
        self._adaptive_calibration: float = 0.0
        self._reward_ema: float = 0.0
        self._stagnation_count: int = 0
        self._action_history: List[int] = []
        self._last_latent: Optional[torch.Tensor] = None
        self.default_env: Optional[Any] = None

    @property
    def last_latent(self) -> Optional[torch.Tensor]:
        """Returns the most recent latent decision state tensor h from the trunk."""
        return self._last_latent

    def set_sterility_memory(self, memory: Optional[Any]) -> None:
        """[DEPRECATED]: No-op. O aprendizado contrastivo c-SIL substitui a memória de esterilidade."""
        if memory is not None:
            warnings.warn(
                "UniversalS1Agent.set_sterility_memory is deprecated and has no effect; c-SIL handles failure aversion natively.",
                DeprecationWarning,
                stacklevel=2,
            )

    def set_env(self, env: Optional[Any]) -> None:
        """Associa um ambiente ao agente para habilitar re-ranqueamento local de valor (Value Re-ranking)."""
        self.default_env = env

    def set_scar_memory(self, scar_memory: Optional[Any]) -> None:
        """Associa um módulo de Memória de Cicatriz de Gradiente (GradientScarMemory) ao agente."""
        self.scar_memory = scar_memory
        if hasattr(self.front_end, "state_encoder") and hasattr(self.front_end.state_encoder, "scar_memory"):
            self.front_end.state_encoder.scar_memory = scar_memory

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        """Auto-detecta e instancia GradientScarMemory se presente no state_dict antes do carregamento."""
        if (
            self.scar_memory is None
            and isinstance(state_dict, dict)
            and any(k.startswith("scar_memory.") or ".scar_memory." in k for k in state_dict.keys())
        ):
            from system1_engine.core.scar_memory import GradientScarMemory
            enc = getattr(self.front_end, "state_encoder", None)
            slot_dim = getattr(enc, "slot_dim", 32) if enc is not None else 32
            num_slots = getattr(enc, "num_slots", 8) if enc is not None else 8
            sm = GradientScarMemory(
                slot_dim=slot_dim,
                num_slots=num_slots,
                mode="hybrid",
                scale=0.0,
                resonance_beta=0.0,
            ).to(self.device)
            self.set_scar_memory(sm)
        if self.scar_memory is not None and isinstance(state_dict, dict):
            state_dict = dict(state_dict)
            for prefix in ("scar_memory.", "front_end.state_encoder.scar_memory."):
                if f"{prefix}W_scar" in state_dict and f"{prefix}z_target" not in state_dict:
                    state_dict[f"{prefix}z_target"] = self.scar_memory.z_target.clone()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def record_victory_scar(
        self,
        traj_obs: Any,
        traj_delta: Optional[Any] = None,
        traj_prev_act: Optional[Any] = None,
        traj_prev_rew: Optional[Any] = None,
        surprise: float = 1.5,
    ) -> bool:
        """Grava permanentemente toda a trajetória de uma vitória [0..T-1] na Memória de Cicatrizes O(1) e Intuição."""
        if self.is_visual:
            return False
        enc = getattr(self.front_end, "state_encoder", None)
        if enc is None or not hasattr(enc, "slot_proj"):
            return False
        if self.scar_memory is None:
            from system1_engine.core.scar_memory import GradientScarMemory
            slot_dim = getattr(enc, "slot_dim", 32)
            num_slots = getattr(enc, "num_slots", 8)
            self.set_scar_memory(
                GradientScarMemory(slot_dim=slot_dim, num_slots=num_slots, mode="hybrid").to(self.device)
            )
        if getattr(self.scar_memory, "mode", "none") == "none":
            return False

        dev = self.device
        obs_arr = np.asarray(traj_obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]
        T_len, obs_dim = obs_arr.shape[0], obs_arr.shape[-1]
        delta_dim = (3 * obs_dim) if self.tri_compass else obs_dim

        with torch.inference_mode(False), torch.enable_grad():
            t_obs = torch.tensor(obs_arr, dtype=torch.float32, device=dev).unsqueeze(0).requires_grad_(True)
            if traj_delta is not None:
                d_arr = np.asarray(traj_delta, dtype=np.float32)
                if d_arr.ndim == 1:
                    d_arr = d_arr[None, :]
                t_delta = torch.tensor(d_arr, dtype=torch.float32, device=dev).unsqueeze(0)
            else:
                t_delta = torch.zeros((1, T_len, delta_dim), dtype=torch.float32, device=dev)

            if traj_prev_act is not None:
                t_act = torch.as_tensor(traj_prev_act, dtype=torch.long, device=dev).view(1, T_len)
            else:
                t_act = torch.zeros((1, T_len), dtype=torch.long, device=dev)

            if traj_prev_rew is not None:
                t_rew = torch.as_tensor(traj_prev_rew, dtype=torch.float32, device=dev).view(1, T_len, 1)
            else:
                t_rew = torch.zeros((1, T_len, 1), dtype=torch.float32, device=dev)

            obs_t = {
                "obs": t_obs,
                "delta_obs": t_delta,
                "prev_action": t_act,
                "prev_reward": t_rew,
            }
            _, val, _ = self.forward(obs_t)
            slots = getattr(enc, "_last_slots", None)
            if slots is not None and slots.requires_grad:
                g_slots = torch.autograd.grad(val.sum(), slots)[0].detach()
                self.scar_memory.record_scar(slots.detach(), g_slots, surprise=float(surprise))
                return True
        return False

    @property
    def device(self) -> torch.device:
        """Returns the device on which model parameters currently reside."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def reset_memory(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """Resets the persistent internal hidden state and homeostatic adaptive calibration."""
        self.hx = None
        self._adaptive_calibration = 0.0
        self._reward_ema = 0.0
        self._stagnation_count = 0
        self._action_history = []
        self._state_visit_counts: Dict[int, int] = {}

    def get_initial_hx(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns the canonical zero-initialized hidden state tensor for this agent."""
        if hasattr(self.trunk, "get_initial_hx"):
            return self.trunk.get_initial_hx(batch_size=batch_size, device=device or self.device)
        trunk_hdim = getattr(self.trunk, "hidden_dim", 256)
        return torch.zeros(1, batch_size, trunk_hdim, device=device or self.device)

    def extract_features(
        self,
        obs_dict: Dict[str, Any],
    ) -> torch.Tensor:
        """Passes observation dictionary through the perceptual front-end."""
        obs = obs_dict["obs"]
        prev_action = obs_dict["prev_action"]
        prev_reward = obs_dict["prev_reward"]
        delta_obs = obs_dict.get("delta_obs")

        if self.is_visual:
            return self.front_end(
                obs=obs,
                prev_action=prev_action,
                prev_reward=prev_reward,
                delta_obs=delta_obs,
            )
        else:
            assert delta_obs is not None, "delta_obs required for vector front-end"
            return self.front_end(
                obs=obs,
                delta_obs=delta_obs,
                prev_action=prev_action,
                prev_reward=prev_reward,
                local_delta=obs_dict.get("local_delta"),
                origin_delta=obs_dict.get("origin_delta"),
                goal_delta=obs_dict.get("goal_delta"),
            )

    def forward(
        self,
        obs_dict: Dict[str, Any],
        hx: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        return_latent: bool = False,
        return_value_logits: bool = False,
    ) -> Any:
        """Unified forward pass returning action distribution, state value, and next hidden state."""
        z_in = self.extract_features(obs_dict)
        h, next_hx = self.trunk(z_in, hx=hx, dones=dones)
        dist = self.policy_head(h)
        val = self.value_head(h)
        self._last_latent = h

        # Modulação de Logits por Ressonância (Mecanismo 2 e Híbrido)
        if getattr(self, "scar_memory", None) is not None and getattr(self.scar_memory, "has_resonance", False):
            res = None
            if self.scar_memory.mode == "hybrid" and hasattr(self.front_end, "state_encoder"):
                res = getattr(self.front_end.state_encoder, "_last_resonance", None)
            elif self.scar_memory.mode == "dual_gradient":
                norm_t = self.scar_memory.g_target / (torch.norm(self.scar_memory.g_target) + 1e-6)
                if torch.is_grad_enabled() and z_in.requires_grad:
                    g_loc = torch.autograd.grad(val.sum(), z_in, retain_graph=True)[0]
                    res = torch.matmul(g_loc[..., :32], norm_t)

            if res is not None and hasattr(dist, "logits") and self.is_discrete:
                new_logits = self.scar_memory.modulate_logits(dist.logits, res)
                dist = torch.distributions.Categorical(logits=new_logits)
        if return_latent:
            return dist, val, next_hx, h
        if return_value_logits:
            return dist, val, next_hx, self.value_head.last_logits
        return dist, val, next_hx

    def get_action(
        self,
        obs_dict: Dict[str, Any],
        hx: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        calibration: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout step returning (action, log_prob, value, next_hx)."""
        dist, value, next_hx = self.forward(obs_dict, hx=hx, dones=None)
        if calibration is not None:
            calib = float(np.clip(calibration, 0.0, 1.0))
            is_det = (calib <= 0.0)
            eff_temp = max(0.05, calib)
            eff_scale = calib
        else:
            is_det = deterministic
            eff_temp = 1.0
            eff_scale = 1.0

        if is_det:
            if self.is_discrete:
                action = torch.argmax(dist.logits, dim=-1)
            else:
                action = dist.mean
        else:
            if self.is_discrete:
                if eff_temp != 1.0:
                    scaled_dist = torch.distributions.Categorical(logits=dist.logits / eff_temp)
                    action = scaled_dist.sample()
                else:
                    action = dist.sample()
            else:
                if eff_scale != 1.0:
                    scaled_dist = torch.distributions.Normal(loc=dist.mean, scale=dist.scale * eff_scale)
                    action = scaled_dist.sample()
                else:
                    action = dist.sample()

        if self.is_discrete:
            log_prob = dist.log_prob(action)
            if self.action_chunk_size > 1:
                log_prob = log_prob.sum(dim=-1)
        else:
            log_prob = dist.log_prob(action).sum(dim=-1)
            if self.action_chunk_size > 1:
                log_prob = log_prob.sum(dim=-1)

        if action.dim() >= 2 and action.shape[1] == 1:
            action = action.squeeze(1)
        if log_prob.dim() >= 2 and log_prob.shape[1] == 1:
            log_prob = log_prob.squeeze(1)
        if value.dim() >= 2 and value.shape[1] == 1:
            value = value.squeeze(1)

        return action, log_prob, value, next_hx

    @torch.no_grad()
    def act_fast(
        self,
        obs_dict: Dict[str, Any],
        return_decision: bool = False,
        uncertainty_threshold: float = 0.70,
        confidence_threshold: float = 0.50,
        return_value: bool = False,
        deterministic: bool = True,
        noise_scale: float = 0.0,
        calibration: Optional[Union[float, str]] = None,
        auto_calibrate: bool = False,
        avoid_action: Optional[Union[int, List[int], Set[int]]] = None,
        quality_filter: bool = True,
        ensemble_fallback: bool = False,
        level: Optional[int] = None,
        adaptive_chunk: bool = False,
        value_rerank: bool = False,
        value_rerank_top_k: int = 2,
        value_rerank_threshold: float = 0.85,
        ghost_jump: bool = True,
        scar_penalty: float = 0.25,
        env: Optional[Any] = None,
        simulator: Optional[Callable[[int], Tuple[Any, float, bool, Dict[str, Any]]]] = None,
    ) -> Union[int, np.ndarray, ReflexDecision]:
        """Ultra-fast single-forward reflex inference.

        Optimized for zero autograd overhead, minimal allocation, and <= 0.8ms latency on CPU.
        Supports deterministic (mean/argmax), calibrated sampling (0.0 to 1.0), and homeostatic
        auto-calibration (auto_calibrate=True or calibration='auto') based on reward momentum.
        If return_decision=True, returns ReflexDecision with uncertainty gating signals and active calibration.
        If ensemble_fallback=True and is_uncertain=True, triggers ConfidenceGatedEnsemble sub-millisecond fallback.
        """
        t0 = time.perf_counter()
        resolved_level = level or (
            obs_dict.get("current_depth")
            or obs_dict.get("scramble_depth")
            or obs_dict.get("level")
            if isinstance(obs_dict, dict)
            else None
        )
        raw_obs = obs_dict["obs"]
        raw_action = obs_dict["prev_action"]
        raw_reward = obs_dict["prev_reward"]

        # Fast tensor conversions
        if self.is_visual:
            if isinstance(raw_obs, torch.Tensor):
                t_obs = raw_obs
                if t_obs.dim() == 3:
                    t_obs = t_obs.unsqueeze(0).unsqueeze(0)
                elif t_obs.dim() == 4:
                    t_obs = t_obs.unsqueeze(1)
            else:
                arr = np.asarray(raw_obs, dtype=np.float32)
                t_obs = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        else:
            if isinstance(raw_obs, torch.Tensor):
                t_obs = raw_obs
                if t_obs.dim() == 1:
                    t_obs = t_obs.unsqueeze(0).unsqueeze(0)
                elif t_obs.dim() == 2:
                    t_obs = t_obs.unsqueeze(1)
            else:
                arr = np.asarray(raw_obs, dtype=np.float32)
                t_obs = torch.from_numpy(arr).view(1, 1, -1)
                if value_rerank:
                    if not hasattr(self, "_state_visit_counts"):
                        self._state_visit_counts = {}
                    if len(self._state_visit_counts) < 512:
                        h_key = hash(arr.tobytes())
                        self._state_visit_counts[h_key] = self._state_visit_counts.get(h_key, 0) + 1

        dev = self.device
        to_dev = dev.type != "cpu"

        if self.is_visual:
            if isinstance(raw_action, torch.Tensor):
                t_act = raw_action
            elif self.is_discrete:
                t_act = torch.tensor([[int(raw_action)]], dtype=torch.long)
            else:
                t_act = torch.from_numpy(np.asarray(raw_action, dtype=np.float32)).view(1, 1, -1)

            if isinstance(raw_reward, torch.Tensor):
                t_rew = raw_reward.view(1, 1, 1)
            else:
                t_rew = torch.as_tensor(raw_reward, dtype=torch.float32).view(1, 1, 1)

            if to_dev:
                t_obs = t_obs.to(dev)
                t_act = t_act.to(dev)
                t_rew = t_rew.to(dev)

            z_in = self.front_end(
                obs=t_obs,
                prev_action=t_act,
                prev_reward=t_rew,
            )
        else:
            raw_delta = obs_dict.get("delta_obs")
            if raw_delta is None:
                t_delta = torch.zeros_like(t_obs)
            elif isinstance(raw_delta, torch.Tensor):
                t_delta = raw_delta
                if t_delta.dim() == 1:
                    t_delta = t_delta.unsqueeze(0).unsqueeze(0)
                elif t_delta.dim() == 2:
                    t_delta = t_delta.unsqueeze(1)
            else:
                arr_d = np.asarray(raw_delta, dtype=np.float32)
                t_delta = torch.from_numpy(arr_d).view(1, 1, -1)

            if isinstance(raw_action, torch.Tensor):
                t_act = raw_action
            elif self.is_discrete:
                t_act = torch.tensor([[int(raw_action)]], dtype=torch.long)
            else:
                t_act = torch.from_numpy(np.asarray(raw_action, dtype=np.float32)).view(1, 1, -1)

            if isinstance(raw_reward, torch.Tensor):
                t_rew = raw_reward.view(1, 1, 1)
            else:
                t_rew = torch.as_tensor(raw_reward, dtype=torch.float32).view(1, 1, 1)

            if to_dev:
                t_obs = t_obs.to(dev)
                t_delta = t_delta.to(dev)
                t_act = t_act.to(dev)
                t_rew = t_rew.to(dev)

            z_in = self.front_end(
                obs=t_obs,
                delta_obs=t_delta,
                prev_action=t_act,
                prev_reward=t_rew,
            )

        # Recurrent pass through trunk with internal persistent memory
        h, self.hx = self.trunk(z_in, hx=self.hx, dones=None)
        self._last_latent = h

        # Resolve sampling parameters based on homeostatic auto-calibration, fixed calibration, or legacy flags
        is_auto_calib = auto_calibrate or (
            isinstance(calibration, str) and calibration.lower() in ("auto", "adaptive", "homeostatic")
        )

        if is_auto_calib:
            rew_val = float(np.asarray(raw_reward).flat[0])
            delta_rew = rew_val - self._reward_ema
            self._reward_ema = 0.9 * self._reward_ema + 0.1 * rew_val

            # Se houve recompensa positiva ou delta positivo perceptível: progresso / alívio
            if rew_val > 0.01 or delta_rew > 0.01:
                self._stagnation_count = 0
                self._adaptive_calibration = max(0.0, self._adaptive_calibration - 0.20)
            else:
                # Estagnação ou recompensa nula/negativa: frustração acumulada eleva temperatura
                self._stagnation_count += 1
                if self._stagnation_count >= 2:
                    self._adaptive_calibration = min(0.80, self._adaptive_calibration + 0.10)

        # Resolução do conjunto de ações a evitar / quebra de ciclos
        avoid_set = set()
        if avoid_action is not None:
            if isinstance(avoid_action, (list, tuple, set)):
                avoid_set.update(int(a) for a in avoid_action if int(a) >= 0)
            elif int(avoid_action) >= 0:
                avoid_set.add(int(avoid_action))

        hist = getattr(self, "_action_history", [])
        if (is_auto_calib and self._stagnation_count >= 2) or (quality_filter and len(hist) >= 3):
            if len(hist) >= 3 and hist[-1] == hist[-3]:
                # Oscilação A -> B -> A sob estagnação: evita B para romper o ciclo vicioso
                avoid_set.add(hist[-2])
            elif len(hist) >= 3 and hist[-1] == hist[-2] and hist[-2] == hist[-3]:
                # Repetição tripla consecutiva
                avoid_set.add(hist[-1])

        # Policy decision
        if self.is_discrete:
            if self.action_chunk_size > 1:
                k = self.action_chunk_size
                n_acts = self.policy_head.n_actions
                raw_linear = self.policy_head.linear(h).squeeze(0).squeeze(0)
                logits_chunk = raw_linear.view(k, n_acts)

                chunk_actions: List[int] = []
                chunk_confs: List[float] = []
                chunk_uncs: List[float] = []
                chunk_margins: List[float] = []
                chunk_entropies: List[float] = []

                if is_auto_calib:
                    calib = float(np.clip(self._adaptive_calibration, 0.0, 0.80))
                    is_deterministic_mode = (calib <= 0.01)
                    eff_temp = 1.0 + 0.5 * calib
                elif calibration is not None:
                    calib = float(np.clip(float(calibration), 0.0, 1.0))
                    is_deterministic_mode = (calib <= 0.0)
                    eff_temp = 1.0 + 0.5 * calib
                else:
                    is_deterministic_mode = deterministic
                    eff_temp = 1.0
                    calib = 0.0 if deterministic else (noise_scale if noise_scale > 0.0 else 1.0)

                for step_i in range(k):
                    l_sq = logits_chunk[step_i]
                    probs_i = torch.softmax(l_sq, dim=-1)
                    best_act_i = int(torch.argmax(probs_i, dim=-1).item())
                    if n_acts > 1:
                        top_p, _ = torch.topk(probs_i, k=min(2, n_acts))
                        conf_i = float(top_p[0].item())
                        margin_i = float((top_p[0] - top_p[1]).item())
                        ent_i = float(-torch.sum(probs_i * torch.log(probs_i + 1e-8)).item())
                        max_ent_i = float(np.log(n_acts))
                        unc_i = float(np.clip(ent_i / max_ent_i, 0.0, 1.0))
                    else:
                        conf_i = 1.0
                        margin_i = 1.0
                        ent_i = 0.0
                        unc_i = 0.0

                    if is_deterministic_mode:
                        act_i = best_act_i
                    else:
                        if float(np.random.rand()) < calib:
                            p_sampled = torch.softmax(l_sq / eff_temp, dim=-1)
                            act_i = int(torch.multinomial(p_sampled, num_samples=1).item())
                        else:
                            act_i = best_act_i

                    # Anti-reversal dentro do chunk para cubo atômico (12 ações)
                    if n_acts == 12 and len(chunk_actions) >= 1:
                        last_a = chunk_actions[-1]
                        inv_a = (last_a + 1) if (last_a % 2 == 0) else (last_a - 1)
                        if act_i == inv_a and conf_i < 0.95 and n_acts > 1:
                            top_indices = torch.argsort(probs_i, descending=True)
                            for alt_idx in top_indices:
                                alt_a = int(alt_idx.item())
                                if alt_a != inv_a:
                                    act_i = alt_a
                                    break

                    chunk_actions.append(act_i)
                    chunk_confs.append(conf_i)
                    chunk_uncs.append(unc_i)
                    chunk_margins.append(margin_i)
                    chunk_entropies.append(ent_i)

                    if adaptive_chunk and step_i > 0 and (unc_i >= uncertainty_threshold or conf_i < confidence_threshold):
                        break

                action_arr = np.array(chunk_actions, dtype=np.int64)
                if not hasattr(self, "_action_history"):
                    self._action_history = []
                for a in chunk_actions:
                    self._action_history.append(a)
                while len(self._action_history) > 8:
                    self._action_history.pop(0)

                if not return_decision:
                    return action_arr

                latent_val = float(self.value_head(h).squeeze().item()) if return_value else None
                latency_ms = (time.perf_counter() - t0) * 1000.0
                mean_unc = float(np.mean(chunk_uncs)) if len(chunk_uncs) > 0 else 0.0
                mean_conf = float(np.mean(chunk_confs)) if len(chunk_confs) > 0 else 1.0
                is_unc = any(u >= uncertainty_threshold for u in chunk_uncs) or (mean_conf < confidence_threshold)

                return ReflexDecision(
                    action=action_arr,
                    confidence=mean_conf,
                    uncertainty=mean_unc,
                    margin=float(chunk_margins[0]) if len(chunk_margins) > 0 else 1.0,
                    is_uncertain=is_unc,
                    entropy=float(np.mean(chunk_entropies)) if len(chunk_entropies) > 0 else 0.0,
                    latent_value=latent_val,
                    calibration=calib,
                    latency_ms=latency_ms,
                    ensemble_fallback_triggered=False,
                )

            logits_sq = self.policy_head.linear(h).squeeze(0).squeeze(0)
            probs = torch.softmax(logits_sq, dim=-1)
            best_action = int(torch.argmax(probs, dim=-1).item())
            n_acts = probs.shape[0]

            if n_acts > 1:
                top_probs, _ = torch.topk(probs, k=min(2, n_acts))
                confidence = float(top_probs[0].item())
                margin = float((top_probs[0] - top_probs[1]).item())
                entropy = float(-torch.sum(probs * torch.log(probs + 1e-8)).item())
                max_entropy = float(np.log(n_acts))
                uncertainty = float(np.clip(entropy / max_entropy, 0.0, 1.0))
            else:
                confidence = 1.0
                margin = 1.0
                entropy = 0.0
                uncertainty = 0.0

            # Calibração efetiva modulada pela incerteza intrínseca da rede
            if is_auto_calib:
                if self._adaptive_calibration <= 0.0:
                    calib = 0.0
                else:
                    calib = float(np.clip(
                        self._adaptive_calibration * (0.6 + 0.4 * uncertainty) + 0.10 * max(0.0, uncertainty - 0.50),
                        0.0,
                        0.80,
                    ))
                is_deterministic_mode = (calib <= 0.01)
                eff_temp = 1.0 + 0.5 * calib
            elif calibration is not None:
                calib = float(np.clip(float(calibration), 0.0, 1.0))
                is_deterministic_mode = (calib <= 0.0)
                eff_temp = 1.0 + 0.5 * calib
            else:
                is_deterministic_mode = deterministic
                eff_temp = 1.0
                calib = 0.0 if deterministic else (noise_scale if noise_scale > 0.0 else 1.0)

            # Filtro de Qualidade de Ação (Action Quality Filter)
            # Em cubo atômico (12 ações), o movimento imediatamente inverso (ex: U -> U')
            # anula o passo anterior e desperdiça tempo. Evita reversão direta se não houver convicção quase total.
            # Também detecta e quebra o comutador de 4 passos [a1, a2, inv(a1)] -> inv(a2) que preserva a identidade [F, B'] = I.
            if (quality_filter or (is_auto_calib and self._stagnation_count >= 2)) and probs.shape[0] == 12:
                if len(hist) >= 1:
                    last_act = hist[-1]
                    inv_act = (last_act + 1) if (last_act % 2 == 0) else (last_act - 1)
                    if confidence < 0.96:
                        avoid_set.add(inv_act)
                if len(hist) >= 3:
                    a1, a2, a3 = hist[-3], hist[-2], hist[-1]
                    inv_a1 = (a1 + 1) if (a1 % 2 == 0) else (a1 - 1)
                    inv_a2 = (a2 + 1) if (a2 % 2 == 0) else (a2 - 1)
                    if a3 == inv_a1 and confidence < 0.95:
                        avoid_set.add(inv_a2)

            if is_deterministic_mode:
                action = best_action
            else:
                # Inovação proporcional: sorteia inovação estocástica com probabilidade = calib
                if float(np.random.rand()) < calib:
                    probs_sampled = torch.softmax(logits_sq / eff_temp, dim=-1)
                    action = int(torch.multinomial(probs_sampled, num_samples=1).item())
                else:
                    action = best_action

            # Quebra estrita de ciclos (Cycle Breaker) e filtro de qualidade:
            if len(avoid_set) > 0 and action in avoid_set and probs.shape[0] > 1:
                top_indices = torch.argsort(probs, descending=True)
                for alt_idx in top_indices:
                    alt_action = int(alt_idx.item())
                    if alt_action not in avoid_set:
                        action = alt_action
                        break

            # -----------------------------------------------------------------
            # Value Re-ranking (Lookahead via Intuition + Critic + Soft Scar + 2-Step Ghost Jump)
            # -----------------------------------------------------------------
            value_rerank_triggered = False
            effective_env = env or getattr(self, "default_env", None)
            sim_fn = simulator or (getattr(effective_env, "simulate_step", None) if effective_env is not None else None)

            if value_rerank and self.action_chunk_size == 1 and n_acts > 1 and sim_fn is not None:
                visit_counts = getattr(self, "_state_visit_counts", {})
                raw_inner_env = getattr(effective_env, "env", None) if (simulator is None and not self.is_visual) else None
                use_fast_raw_sim = (
                    raw_inner_env is not None
                    and hasattr(raw_inner_env, "simulate_step")
                    and hasattr(effective_env, "prev_obs")
                    and effective_env.prev_obs is not None
                )

                # Scar Wake-Up Gate: se o reflexo puro de alta confiança vai pisar num estado já visitado, acorda o Lookahead
                force_rerank_by_scar = False
                if confidence >= value_rerank_threshold and use_fast_raw_sim and scar_penalty > 0.0 and visit_counts:
                    try:
                        r_obs_chk, _, s_done_chk, _ = raw_inner_env.simulate_step(action)
                        if not s_done_chk:
                            chk_hash = hash(np.asarray(r_obs_chk, dtype=np.float32).tobytes())
                            if visit_counts.get(chk_hash, 0) >= 1:
                                force_rerank_by_scar = True
                    except Exception:
                        pass

                if confidence < value_rerank_threshold or force_rerank_by_scar:
                    cand_k = max(2, min(int(value_rerank_top_k), n_acts))
                    top_p, top_indices = torch.topk(probs, k=min(cand_k + 4, n_acts))
                    cand_actions = [int(idx.item()) for idx in top_indices if int(idx.item()) not in avoid_set][:cand_k]
                    if not cand_actions:
                        cand_actions = [int(top_indices[0].item())]

                    if len(cand_actions) > 1:
                        immediate_solve = None
                        cand_obs_list = []
                        cand_delta_list = []
                        cand_act_list = []
                        cand_rew_list = []
                        cand_pen_list = []
                        valid_cands = []

                        for c_act in cand_actions:
                            if use_fast_raw_sim:
                                try:
                                    r_obs, s_rew, s_done, _ = raw_inner_env.simulate_step(c_act)
                                except Exception:
                                    continue
                                if s_done:
                                    immediate_solve = c_act
                                    break
                                obs_arr = np.asarray(r_obs, dtype=np.float32)
                                v_cnt = visit_counts.get(hash(obs_arr.tobytes()), 0) if (scar_penalty > 0.0 and visit_counts) else 0
                                cand_obs_list.append(obs_arr)
                                cand_act_list.append(c_act)
                                cand_rew_list.append(float(s_rew))
                                cand_pen_list.append(float(scar_penalty * v_cnt))
                                valid_cands.append(c_act)
                            else:
                                try:
                                    sim_res = sim_fn(c_act)
                                except Exception:
                                    continue

                                if isinstance(sim_res, tuple) and len(sim_res) >= 3:
                                    s_obs, s_rew, s_done = sim_res[0], sim_res[1], sim_res[2]
                                else:
                                    continue

                                if s_done:
                                    immediate_solve = c_act
                                    break

                                if isinstance(s_obs, dict):
                                    obs_arr = np.asarray(s_obs["obs"], dtype=np.float32)
                                    delta_arr = s_obs.get("delta_obs", np.zeros_like(obs_arr))
                                else:
                                    obs_arr = np.asarray(s_obs, dtype=np.float32)
                                    delta_arr = np.zeros_like(obs_arr)

                                v_cnt = (
                                    visit_counts.get(hash(obs_arr.tobytes()), 0)
                                    if (not self.is_visual and scar_penalty > 0.0 and visit_counts)
                                    else 0
                                )
                                cand_obs_list.append(obs_arr)
                                cand_delta_list.append(delta_arr)
                                cand_act_list.append(c_act)
                                cand_rew_list.append(float(s_rew))
                                cand_pen_list.append(float(scar_penalty * v_cnt))
                                valid_cands.append(c_act)

                        if immediate_solve is not None:
                            action = immediate_solve
                            value_rerank_triggered = True
                        elif len(valid_cands) > 1:
                            M = len(valid_cands)
                            stacked_obs = np.stack(cand_obs_list)
                            if self.is_visual:
                                b_obs = torch.from_numpy(stacked_obs).unsqueeze(1).to(dev)
                                b_delta = None
                            elif use_fast_raw_sim:
                                b_obs = torch.from_numpy(stacked_obs).unsqueeze(1)
                                if getattr(effective_env, "tri_compass", False):
                                    d_loc = stacked_obs - effective_env.prev_obs
                                    d_orig = (
                                        stacked_obs - effective_env.init_obs
                                        if effective_env.init_obs is not None
                                        else np.zeros_like(stacked_obs)
                                    )
                                    d_goal = (
                                        stacked_obs - effective_env.elite_goal_obs
                                        if effective_env.elite_goal_obs is not None
                                        else np.zeros_like(stacked_obs)
                                    )
                                    b_delta = torch.from_numpy(np.concatenate([d_loc, d_orig, d_goal], axis=-1)).unsqueeze(1)
                                else:
                                    if getattr(effective_env, "compass_mode", False) and effective_env.init_obs is not None:
                                        b_delta = torch.from_numpy(stacked_obs - effective_env.init_obs).unsqueeze(1)
                                    else:
                                        b_delta = torch.zeros_like(b_obs)
                                if to_dev:
                                    b_obs = b_obs.to(dev)
                                    b_delta = b_delta.to(dev)
                            else:
                                b_obs = torch.from_numpy(stacked_obs).unsqueeze(1).to(dev)
                                b_delta = torch.from_numpy(np.stack(cand_delta_list)).unsqueeze(1).to(dev)

                            b_act = torch.tensor(cand_act_list, dtype=torch.long, device=dev).view(M, 1)
                            b_rew = torch.tensor(cand_rew_list, dtype=torch.float32, device=dev).view(M, 1, 1)

                            z_cands = self.front_end(obs=b_obs, delta_obs=b_delta, prev_action=b_act, prev_reward=b_rew)
                            if self.hx is not None:
                                if isinstance(self.hx, tuple):
                                    hx_cands = tuple(
                                        h_t.repeat(1, M, 1) if h_t.dim() == 3 else h_t.repeat(M, 1)
                                        for h_t in self.hx
                                    )
                                else:
                                    hx_cands = self.hx.repeat(1, M, 1) if self.hx.dim() == 3 else self.hx.repeat(M, 1)
                            else:
                                hx_cands = None

                            h_cands, nhx_cands = self.trunk(z_cands, hx=hx_cands, dones=None)
                            v_cands = self.value_head(h_cands).squeeze(-1).squeeze(-1)
                            q_scores = (
                                torch.tensor(cand_rew_list, device=dev)
                                + 0.99 * v_cands
                                - torch.tensor(cand_pen_list, device=dev)
                            )

                            # Salto Fantasma de 2º e 3º Passo guiado pela Intuição + Cicatrizes (Opções 1, 2 e 3)
                            sim_2step_fn = (
                                getattr(raw_inner_env, "simulate_2step", None)
                                if (ghost_jump and use_fast_raw_sim)
                                else None
                            )
                            sim_3step_fn = (
                                getattr(raw_inner_env, "simulate_3step", None)
                                if (ghost_jump and use_fast_raw_sim)
                                else None
                            )
                            if sim_2step_fn is not None:
                                next_logits = self.policy_head.linear(h_cands).squeeze(1)
                                next_probs = torch.softmax(next_logits, dim=-1)
                                next_top_p, next_top_idx = torch.topk(next_probs, k=min(3, n_acts), dim=-1)

                                q1_order = torch.argsort(q_scores, descending=True)
                                best_m1 = int(q1_order[0].item())
                                second_m1 = int(q1_order[1].item()) if M > 1 else best_m1
                                worst_m1 = int(q1_order[-1].item())
                                best_q1_val = float(q_scores[best_m1].item())
                                q1_margin = float((q_scores[best_m1] - q_scores[second_m1]).item())

                                # Opção 1 (Poda de Clone Cicatrizado): se o pior clone pisou em cicatriz e está atrás, libera slot SIMD
                                prune_worst = (
                                    (M >= 4)
                                    and (cand_pen_list[worst_m1] > 0.0)
                                    and ((best_q1_val - float(q_scores[worst_m1].item())) > 0.25)
                                )
                                alive_m = [m_i for m_i in range(M) if not (prune_worst and m_i == worst_m1)]

                                solve_2step_act = None
                                p_idx_list: List[int] = []
                                obs2_list = []
                                a1_list: List[int] = []
                                act2_list: List[int] = []
                                rew2_list: List[float] = []
                                pen2_list: List[float] = []
                                r1_ref_list: List[float] = []
                                best_m1_r2 = -1e9

                                for idx_m in alive_m:
                                    a1 = valid_cands[idx_m]
                                    if n_acts == 12:
                                        inv_a1 = (a1 + 1) if (a1 % 2 == 0) else (a1 - 1)
                                        valid_a2s = [
                                            int(next_top_idx[idx_m, k_i].item())
                                            for k_i in range(next_top_idx.shape[1])
                                            if int(next_top_idx[idx_m, k_i].item()) != inv_a1
                                        ]
                                    else:
                                        valid_a2s = [
                                            int(next_top_idx[idx_m, k_i].item())
                                            for k_i in range(next_top_idx.shape[1])
                                        ]
                                    if not valid_a2s:
                                        valid_a2s = [int(next_top_idx[idx_m, 0].item())]

                                    # Opção 2 (Bifurcação Assimétrica no 2º Passo para o líder sob dúvida intuitiva)
                                    margin2 = (
                                        float((next_top_p[idx_m, 0] - next_top_p[idx_m, 1]).item())
                                        if next_top_p.shape[1] > 1
                                        else 1.0
                                    )
                                    branch_k = (
                                        2
                                        if (
                                            idx_m == best_m1
                                            and (prune_worst or margin2 < 0.15)
                                            and margin2 < 0.25
                                            and len(valid_a2s) >= 2
                                        )
                                        else 1
                                    )

                                    for a2 in valid_a2s[:branch_k]:
                                        o2, rew2, done2 = sim_2step_fn(a1, a2)
                                        if done2:
                                            solve_2step_act = a1
                                            break
                                        if idx_m == best_m1 and float(rew2) > best_m1_r2:
                                            best_m1_r2 = float(rew2)
                                        if o2 is not None:
                                            o2_arr = np.asarray(o2, dtype=np.float32)
                                            v_cnt2 = (
                                                visit_counts.get(hash(o2_arr.tobytes()), 0)
                                                if (scar_penalty > 0.0 and visit_counts)
                                                else 0
                                            )
                                            p_idx_list.append(idx_m)
                                            obs2_list.append(o2_arr)
                                            a1_list.append(a1)
                                            act2_list.append(a2)
                                            rew2_list.append(float(rew2))
                                            pen2_list.append(float(scar_penalty * v_cnt2))
                                            r1_ref_list.append(cand_rew_list[idx_m])
                                    if solve_2step_act is not None:
                                        break

                                if solve_2step_act is not None:
                                    action = solve_2step_act
                                    value_rerank_triggered = True
                                elif (
                                    not force_rerank_by_scar
                                    and cand_pen_list[best_m1] == 0.0
                                    and len(rew2_list) > 0
                                    and (
                                        (q1_margin >= 0.25 and best_m1_r2 >= max(rew2_list))
                                        or (
                                            getattr(self, "_super_margin_threshold", None) is not None
                                            and q1_margin >= float(getattr(self, "_super_margin_threshold"))
                                        )
                                    )
                                ):
                                    # Opção 1 (Gate Condicional): líder do Passo 1 já domina em valor e em alinhamento físico de 2º passo
                                    action = valid_cands[best_m1]
                                    value_rerank_triggered = True
                                elif len(obs2_list) > 0:
                                    M2 = len(obs2_list)
                                    st_obs2 = np.stack(obs2_list)
                                    b_obs2 = torch.from_numpy(st_obs2).unsqueeze(1)
                                    st_parent_obs = stacked_obs[p_idx_list]
                                    if getattr(effective_env, "tri_compass", False):
                                        d_loc2 = st_obs2 - st_parent_obs
                                        d_orig2 = (
                                            st_obs2 - effective_env.init_obs
                                            if effective_env.init_obs is not None
                                            else np.zeros_like(st_obs2)
                                        )
                                        d_goal2 = (
                                            st_obs2 - effective_env.elite_goal_obs
                                            if effective_env.elite_goal_obs is not None
                                            else np.zeros_like(st_obs2)
                                        )
                                        b_delta2 = torch.from_numpy(np.concatenate([d_loc2, d_orig2, d_goal2], axis=-1)).unsqueeze(1)
                                    else:
                                        if getattr(effective_env, "compass_mode", False) and effective_env.init_obs is not None:
                                            b_delta2 = torch.from_numpy(st_obs2 - effective_env.init_obs).unsqueeze(1)
                                        else:
                                            b_delta2 = torch.zeros_like(b_obs2)
                                    if to_dev:
                                        b_obs2 = b_obs2.to(dev)
                                        b_delta2 = b_delta2.to(dev)

                                    b_act2 = torch.tensor(act2_list, dtype=torch.long, device=dev).view(M2, 1)
                                    b_rew2 = (
                                        torch.tensor(rew2_list, dtype=torch.float32, device=dev)
                                        - torch.tensor(r1_ref_list, dtype=torch.float32, device=dev)
                                    ).view(M2, 1, 1)
                                    z_cands2 = self.front_end(obs=b_obs2, delta_obs=b_delta2, prev_action=b_act2, prev_reward=b_rew2)
                                    if nhx_cands is not None:
                                        if isinstance(nhx_cands, tuple):
                                            hx_cands2 = tuple(
                                                h_t[:, p_idx_list, :] if h_t.dim() == 3 else h_t[p_idx_list, :]
                                                for h_t in nhx_cands
                                            )
                                        else:
                                            hx_cands2 = nhx_cands[:, p_idx_list, :] if nhx_cands.dim() == 3 else nhx_cands[p_idx_list, :]
                                    else:
                                        hx_cands2 = None

                                    h_cands2, _ = self.trunk(z_cands2, hx=hx_cands2, dones=None)
                                    v_cands2 = self.value_head(h_cands2).squeeze(-1).squeeze(-1)
                                    q2_scores = (
                                        torch.tensor(rew2_list, device=dev)
                                        + (0.99 ** 2) * v_cands2
                                        - torch.tensor(pen2_list, device=dev)
                                    )

                                    # Opção 3 (Horizonte Fantasma de 3º Passo em Rust quando próximo da solução)
                                    solve_3step_act = None
                                    if sim_3step_fn is not None and float(v_cands2.max().item()) > 1.5:
                                        logits3 = self.policy_head.linear(h_cands2).squeeze(1)
                                        _, top_i3 = torch.topk(logits3, k=min(6, n_acts), dim=-1)
                                        for m2_i in range(M2):
                                            if float(v_cands2[m2_i].item()) <= 1.5:
                                                continue
                                            a1_v, a2_v = a1_list[m2_i], act2_list[m2_i]
                                            inv_a2 = ((a2_v + 1) if (a2_v % 2 == 0) else (a2_v - 1)) if n_acts == 12 else -1
                                            for k_idx in range(top_i3.shape[1]):
                                                a3_v = int(top_i3[m2_i, k_idx].item())
                                                if a3_v == inv_a2:
                                                    continue
                                                _, _, done3 = sim_3step_fn(a1_v, a2_v, a3_v, return_obs=False)
                                                if done3:
                                                    solve_3step_act = a1_v
                                                    break
                                            if solve_3step_act is not None:
                                                break

                                    if solve_3step_act is not None:
                                        action = solve_3step_act
                                        value_rerank_triggered = True
                                    else:
                                        best_q2_per_m = torch.full((M,), -1e9, device=dev)
                                        for m2_i, p_i in enumerate(p_idx_list):
                                            if q2_scores[m2_i] > best_q2_per_m[p_i]:
                                                best_q2_per_m[p_i] = q2_scores[m2_i]
                                        q_scores = torch.maximum(q_scores, 0.98 * best_q2_per_m)
                                        best_idx = int(torch.argmax(q_scores).item())
                                        action = valid_cands[best_idx]
                                        value_rerank_triggered = True
                                else:
                                    best_idx = int(torch.argmax(q_scores).item())
                                    action = valid_cands[best_idx]
                                    value_rerank_triggered = True
                            else:
                                best_idx = int(torch.argmax(q_scores).item())
                                action = valid_cands[best_idx]
                                value_rerank_triggered = True

            is_uncertain = (uncertainty >= uncertainty_threshold) or (confidence < confidence_threshold)
            ensemble_fallback_triggered = False

            # Fallback Ativo por Gating de Confiança (A.2)
            if is_uncertain and ensemble_fallback:
                action, _ = ConfidenceGatedEnsemble.resolve_uncertain_action(
                    agent=self,
                    h=h,
                    is_discrete=True,
                    avoid_set=avoid_set,
                    quality_filter=quality_filter,
                )
                ensemble_fallback_triggered = True

            if not hasattr(self, "_action_history"):
                self._action_history = []
            self._action_history.append(action)
            if len(self._action_history) > 8:
                self._action_history.pop(0)

            if not return_decision:
                return action

            latent_val = float(self.value_head(h).squeeze().item()) if return_value else None
            latency_ms = (time.perf_counter() - t0) * 1000.0

            return ReflexDecision(
                action=action,
                confidence=confidence,
                uncertainty=uncertainty,
                margin=margin,
                is_uncertain=is_uncertain,
                entropy=entropy,
                latent_value=latent_val,
                calibration=calib,
                latency_ms=latency_ms,
                ensemble_fallback_triggered=ensemble_fallback_triggered,
                value_rerank_triggered=value_rerank_triggered,
            )
        else:
            if self.action_chunk_size > 1:
                k = self.action_chunk_size
                act_dim = self.policy_head.act_dim
                raw_mu = self.policy_head.mu_net(h).squeeze(0).squeeze(0)
                mu_chunk = raw_mu.view(k, act_dim)
                action_np = mu_chunk.cpu().numpy()

                log_std_clamped = torch.clamp(self.policy_head.log_std, min=-20.0, max=2.0)
                std_1d = torch.exp(log_std_clamped).cpu().numpy()
                std_1d = np.clip(std_1d, a_min=1e-6, a_max=100.0)
                std = np.broadcast_to(std_1d, (k, act_dim))

                # Entropia diferencial e incerteza no chunk
                entropy = float(0.5 * np.mean(np.sum(1.0 + np.log(2.0 * np.pi * (std ** 2)), axis=-1)))
                var = float(np.mean(std ** 2))
                var_scale = 0.5
                uncertainty = float(np.clip(2.0 / (1.0 + np.exp(-var / var_scale)) - 1.0, 0.0, 1.0))
                confidence = float(np.clip(1.0 - uncertainty, 0.0, 1.0))

                if is_auto_calib:
                    if self._adaptive_calibration <= 0.0:
                        calib = 0.0
                    else:
                        calib = float(np.clip(
                            self._adaptive_calibration * (0.6 + 0.4 * uncertainty) + 0.10 * max(0.0, uncertainty - 0.50),
                            0.0,
                            0.80,
                        ))
                    is_deterministic_mode = (calib <= 0.01)
                    eff_scale = calib
                elif calibration is not None:
                    calib = float(np.clip(float(calibration), 0.0, 1.0))
                    is_deterministic_mode = (calib <= 0.0)
                    eff_scale = calib
                else:
                    is_deterministic_mode = deterministic
                    eff_scale = noise_scale if noise_scale > 0.0 else 1.0
                    calib = 0.0 if deterministic else (noise_scale if noise_scale > 0.0 else 1.0)

                if not is_deterministic_mode or (noise_scale > 0.0 and calibration is None and not is_auto_calib):
                    action_np = action_np + eff_scale * std * np.random.randn(*action_np.shape)

                if hasattr(self, "action_space") and isinstance(self.action_space, gym.spaces.Box):
                    action_np = np.clip(action_np, self.action_space.low, self.action_space.high)

                if adaptive_chunk and (uncertainty >= uncertainty_threshold or confidence < confidence_threshold):
                    action_np = action_np[:1]

                is_uncertain = (uncertainty >= uncertainty_threshold) or (confidence < confidence_threshold)
                ensemble_fallback_triggered = False

                if not return_decision:
                    return action_np

                latent_val = float(self.value_head(h).squeeze().item()) if return_value else None
                latency_ms = (time.perf_counter() - t0) * 1000.0

                return ReflexDecision(
                    action=action_np,
                    confidence=confidence,
                    uncertainty=uncertainty,
                    margin=0.0,
                    is_uncertain=is_uncertain,
                    entropy=entropy,
                    latent_value=latent_val,
                    calibration=calib,
                    latency_ms=latency_ms,
                    ensemble_fallback_triggered=ensemble_fallback_triggered,
                )

            mu = self.policy_head.mu_net(h)
            action_np = mu.squeeze(0).squeeze(0).cpu().numpy()

            # Clamping defensivo de log_std / std para evitar underflow numérico e instabilidade
            log_std_clamped = torch.clamp(self.policy_head.log_std, min=-20.0, max=2.0)
            std = torch.exp(log_std_clamped).cpu().numpy()
            std = np.clip(std, a_min=1e-6, a_max=100.0)

            # Entropia diferencial contínua H = 0.5 * sum(1 + ln(2*pi*sigma^2))
            entropy = float(0.5 * np.sum(1.0 + np.log(2.0 * np.pi * (std ** 2))))

            # Variância média das ações contínuas
            var = float(np.mean(std ** 2))
            var_scale = 0.5
            uncertainty = float(np.clip(2.0 / (1.0 + np.exp(-var / var_scale)) - 1.0, 0.0, 1.0))
            confidence = float(np.clip(1.0 - uncertainty, 0.0, 1.0))

            if is_auto_calib:
                if self._adaptive_calibration <= 0.0:
                    calib = 0.0
                else:
                    calib = float(np.clip(
                        self._adaptive_calibration * (0.6 + 0.4 * uncertainty) + 0.10 * max(0.0, uncertainty - 0.50),
                        0.0,
                        0.80,
                    ))
                is_deterministic_mode = (calib <= 0.01)
                eff_scale = calib
            elif calibration is not None:
                calib = float(np.clip(float(calibration), 0.0, 1.0))
                is_deterministic_mode = (calib <= 0.0)
                eff_scale = calib
            else:
                is_deterministic_mode = deterministic
                eff_scale = noise_scale if noise_scale > 0.0 else 1.0
                calib = 0.0 if deterministic else (noise_scale if noise_scale > 0.0 else 1.0)

            # Aplicação de amostragem estocástica ou ruído calibrado (ex: avaliação de locomoção fluida)
            if not is_deterministic_mode or (noise_scale > 0.0 and calibration is None and not is_auto_calib):
                action_np = action_np + eff_scale * std * np.random.randn(*action_np.shape)

            if hasattr(self, "action_space") and isinstance(self.action_space, gym.spaces.Box):
                action_np = np.clip(action_np, self.action_space.low, self.action_space.high)

            is_uncertain = (uncertainty >= uncertainty_threshold) or (confidence < confidence_threshold)
            ensemble_fallback_triggered = False

            # Fallback Ativo por Gating de Confiança Contínuo (A.2)
            if is_uncertain and ensemble_fallback:
                action_np, _ = ConfidenceGatedEnsemble.resolve_uncertain_action(
                    agent=self,
                    h=h,
                    is_discrete=False,
                    quality_filter=quality_filter,
                )
                if hasattr(self, "action_space") and isinstance(self.action_space, gym.spaces.Box):
                    action_np = np.clip(action_np, self.action_space.low, self.action_space.high)
                ensemble_fallback_triggered = True

            if not hasattr(self, "_action_history"):
                self._action_history = []
            self._action_history.append(action_np)
            if len(self._action_history) > 8:
                self._action_history.pop(0)

            if not return_decision:
                return action_np

            latent_val = float(self.value_head(h).squeeze().item()) if return_value else None
            latency_ms = (time.perf_counter() - t0) * 1000.0

            return ReflexDecision(
                action=action_np,
                confidence=confidence,
                uncertainty=uncertainty,
                margin=0.0,
                is_uncertain=is_uncertain,
                entropy=entropy,
                latent_value=latent_val,
                calibration=calib,
                latency_ms=latency_ms,
                ensemble_fallback_triggered=ensemble_fallback_triggered,
            )

    @torch.no_grad()
    def act_with_confidence(
        self,
        obs_dict: Dict[str, Any],
        uncertainty_threshold: float = 0.70,
        confidence_threshold: float = 0.50,
        return_value: bool = False,
        deterministic: bool = True,
        noise_scale: float = 0.0,
        calibration: Optional[Union[float, str]] = None,
        auto_calibrate: bool = False,
        avoid_action: Optional[Union[int, List[int], Set[int]]] = None,
        quality_filter: bool = True,
        ensemble_fallback: bool = False,
        adaptive_chunk: bool = False,
        value_rerank: bool = False,
        value_rerank_top_k: int = 2,
        value_rerank_threshold: float = 0.85,
        env: Optional[Any] = None,
        simulator: Optional[Callable[[int], Tuple[Any, float, bool, Dict[str, Any]]]] = None,
    ) -> ReflexDecision:
        """Executa a decisão reflexiva retornando a estrutura ReflexDecision com telemetria e calibração."""
        decision = self.act_fast(
            obs_dict=obs_dict,
            return_decision=True,
            uncertainty_threshold=uncertainty_threshold,
            confidence_threshold=confidence_threshold,
            return_value=return_value,
            deterministic=deterministic,
            noise_scale=noise_scale,
            calibration=calibration,
            auto_calibrate=auto_calibrate,
            avoid_action=avoid_action,
            quality_filter=quality_filter,
            ensemble_fallback=ensemble_fallback,
            adaptive_chunk=adaptive_chunk,
            value_rerank=value_rerank,
            value_rerank_top_k=value_rerank_top_k,
            value_rerank_threshold=value_rerank_threshold,
            env=env,
            simulator=simulator,
        )
        assert isinstance(decision, ReflexDecision)
        return decision
