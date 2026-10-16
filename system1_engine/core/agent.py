import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution

from system1_engine.core.encoders import ImpalaVisualFrontEnd, VectorFrontEnd
from system1_engine.core.heads import CategoricalPolicyHead, GaussianPolicyHead, ValueHead
from system1_engine.core.trunk import LeanResidualTrunk, System1Trunk


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
    ) -> None:
        super().__init__()
        if isinstance(obs_space, gym.spaces.Dict) and "obs" in obs_space.spaces:
            obs_space = obs_space.spaces["obs"]
        self.obs_space = obs_space
        self.action_space = action_space

        if is_visual is None:
            is_visual = (
                isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3
            )
        self.is_visual = is_visual

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
            )

        # 2. Universal Recurrent Trunk (Can be shared or transferred)
        if trunk is not None:
            self.trunk = trunk
        elif str(trunk_type).lower() in ("lean", "lean_residual", "v2"):
            self.trunk = LeanResidualTrunk(input_dim=337, fast_dim=256, slow_dim=256)
        else:
            self.trunk = System1Trunk(input_dim=337, hidden_dim=256, num_res_blocks=2)

        trunk_hidden_dim = getattr(self.trunk, "hidden_dim", 256)

        # 3. Policy & Value Heads (Scenario-dependent)
        if isinstance(action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.policy_head: Union[CategoricalPolicyHead, GaussianPolicyHead] = (
                CategoricalPolicyHead(in_features=trunk_hidden_dim, n_actions=int(action_space.n))
            )
        elif isinstance(action_space, gym.spaces.Box):
            self.is_discrete = False
            act_dim = int(np.prod(action_space.shape))
            self.policy_head = GaussianPolicyHead(in_features=trunk_hidden_dim, act_dim=act_dim)
        else:
            raise NotImplementedError(f"Unsupported action space: {type(action_space)}")

        self.value_head = ValueHead(in_features=trunk_hidden_dim)

        # Runtime hidden state for act_fast inference and homeostatic auto-calibration
        self.hx: Optional[torch.Tensor] = None
        self._adaptive_calibration: float = 0.0
        self._reward_ema: float = 0.0
        self._stagnation_count: int = 0
        self._action_history: List[int] = []

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
            )

    def forward(
        self,
        obs_dict: Dict[str, Any],
        hx: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[Distribution, torch.Tensor, torch.Tensor]:
        """Unified forward pass returning action distribution, state value, and next hidden state."""
        z_in = self.extract_features(obs_dict)
        h, next_hx = self.trunk(z_in, hx=hx, dones=dones)
        dist = self.policy_head(h)
        val = self.value_head(h)
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
        else:
            log_prob = dist.log_prob(action).sum(dim=-1)

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
    ) -> Union[int, np.ndarray, ReflexDecision]:
        """Ultra-fast single-forward reflex inference.

        Optimized for zero autograd overhead, minimal allocation, and <= 0.8ms latency on CPU.
        Supports deterministic (mean/argmax), calibrated sampling (0.0 to 1.0), and homeostatic
        auto-calibration (auto_calibrate=True or calibration='auto') based on reward momentum.
        If return_decision=True, returns ReflexDecision with uncertainty gating signals and active calibration.
        If ensemble_fallback=True and is_uncertain=True, triggers ConfidenceGatedEnsemble sub-millisecond fallback.
        """
        t0 = time.perf_counter()
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

            is_uncertain = (uncertainty >= uncertainty_threshold) or (confidence < confidence_threshold)
            ensemble_fallback_triggered = False

            # Fallback Ativo por Gating de Confiança (A.2)
            if is_uncertain and ensemble_fallback:
                from system1_engine.core.ensemble import ConfidenceGatedEnsemble
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
            )
        else:
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
                from system1_engine.core.ensemble import ConfidenceGatedEnsemble
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
        )
        assert isinstance(decision, ReflexDecision)
        return decision

    def solve_with_quality_ensemble(
        self,
        env: Any,
        max_steps: int = 40,
        calibrations: Optional[List[Any]] = None,
        is_atomic_cube: bool = True,
    ) -> Tuple[Any, List[Any]]:
        """Executa múltiplos cenários sob diferentes calibrações e seleciona o melhor resultado garantindo qualidade."""
        from system1_engine.core.ensemble import QualityEnsembleEvaluator
        return QualityEnsembleEvaluator.evaluate_scenarios(
            agent=self,
            env=env,
            max_steps=max_steps,
            calibrations=calibrations,
            is_atomic_cube=is_atomic_cube,
        )
