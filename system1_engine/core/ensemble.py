"""Multi-Scenario Quality Ensemble for System 1.

Executes multiple parallel/simultaneous scenario rollouts under different calibration regimes
(e.g., deterministic, calibrated, homeostatic-auto, and exploratory), evaluates trajectory
quality (success, steps-to-solve, cumulative reward, critic latent value, and action purity),
and selects the best high-quality outcome.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import time
from dataclasses import dataclass, field


@dataclass
class TrajectoryResult:
    """Resultado detalhado da execução de uma trajetória com um regime de calibração."""
    scenario_id: int
    calibration: Any
    solved: bool
    steps: int
    cumulative_reward: float
    mean_latent_value: float
    purity_score: float  # [0.0, 1.0] penaliza reversões e oscilações
    actions: List[int]
    decisions: List[Any]
    quality_score: float
    elapsed_ms: float

    def summary(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "calibration": str(self.calibration),
            "solved": self.solved,
            "steps": self.steps,
            "cumulative_reward": round(self.cumulative_reward, 3),
            "mean_latent_value": round(self.mean_latent_value, 3),
            "purity_score": round(self.purity_score, 3),
            "quality_score": round(self.quality_score, 3),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class QualityEnsembleEvaluator:
    """Avaliador e seletor de trajetórias para garantia de qualidade em múltiplos cenários simultâneos.

    [DEPRECATED]: QualityEnsembleEvaluator requer clonagem de ambiente (env.clone()) e simulação
    de trajetórias completas (estilo Sistema 2 / MCTS leve), violando a latência de reflexo puro (< 1ms)
    e o princípio agnóstico a domínio.
    Para reflexos de alta qualidade sob incerteza em tempo real, utilize ConfidenceGatedEnsemble (A.2),
    que opera diretamente no vetor latente h em < 0.1 ms sem interagir com o ambiente.
    Mantido para compatibilidade retroativa com suítes de teste existentes.
    """

    DEFAULT_CALIBRATIONS = [0.0, 0.15, "auto", 0.35]

    @staticmethod
    def compute_purity_score(
        actions: Union[List[Any], np.ndarray],
        is_atomic_cube: bool = False,
        is_discrete: Optional[bool] = None,
    ) -> float:
        """Calcula a pureza da trajetória penalizando reversões imediatas e oscilações redundantes.

        Suporta espaços de ação discretos (inteiros) e contínuos (vetores multidimensionais).
        - Espaço Discreto: penaliza oscilações 2-passos (A -> B -> A), ciclos redundantes e
          (se is_atomic_cube=True) pares reversos imediatos.
        - Espaço Contínuo: penaliza reversões bruscas direcionais (cosine similarity negativa)
          e alto jerk / jittering temporal de 2ª ordem.
        """
        if len(actions) <= 1:
            return 1.0

        if is_discrete is None:
            first_act = actions[0]
            if isinstance(first_act, (int, np.integer)):
                is_discrete = True
            elif isinstance(first_act, (list, tuple, np.ndarray)) and len(np.asarray(first_act).shape) > 0 and np.asarray(first_act).size > 1:
                is_discrete = False
            else:
                try:
                    arr = np.asarray(actions)
                    is_discrete = bool(arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer))
                except Exception:
                    is_discrete = True

        if is_discrete:
            penalties = 0.0
            act_list = [int(np.asarray(a).item()) if isinstance(a, (np.ndarray, np.integer)) else int(a) for a in actions]
            n_acts = len(act_list)
            for i in range(1, n_acts):
                a_prev = act_list[i - 1]
                a_curr = act_list[i]
                if is_atomic_cube:
                    inv_prev = (a_prev + 1) if (a_prev % 2 == 0) else (a_prev - 1)
                    if a_curr == inv_prev:
                        penalties += 1.0
                if i >= 2 and act_list[i] == act_list[i - 2]:
                    # Oscilação A -> B -> A
                    penalties += 0.5
                if i >= 3 and act_list[i] == act_list[i - 3]:
                    # Ciclo redundante de 3 passos
                    penalties += 0.25
            max_possible = max(1.0, float(n_acts - 1))
            return float(np.clip(1.0 - (penalties / max_possible), 0.0, 1.0))
        else:
            # Espaço contínuo: consistência direcional e amortecimento de jerk
            act_arrs = [np.asarray(a, dtype=np.float32).flatten() for a in actions]
            n_acts = len(act_arrs)
            penalties = 0.0
            for i in range(1, n_acts):
                v_prev = act_arrs[i - 1]
                v_curr = act_arrs[i]
                norm_prev = float(np.linalg.norm(v_prev))
                norm_curr = float(np.linalg.norm(v_curr))
                if norm_prev > 1e-6 and norm_curr > 1e-6:
                    cos_sim = float(np.dot(v_prev, v_curr) / (norm_prev * norm_curr))
                    # Penaliza reversão abrupta de torque ou velocidade
                    if cos_sim < -0.2:
                        penalties += 0.5 * abs(cos_sim)
                if i >= 2:
                    v_prev2 = act_arrs[i - 2]
                    jerk = v_curr - 2.0 * v_prev + v_prev2
                    jerk_mag = float(np.linalg.norm(jerk))
                    ref_mag = max(1.0, norm_curr)
                    if jerk_mag > 1.5 * ref_mag:
                        penalties += min(0.5, 0.2 * (jerk_mag / ref_mag))
            max_possible = max(1.0, float(n_acts - 1))
            return float(np.clip(1.0 - (penalties / max_possible), 0.0, 1.0))
    @classmethod
    def evaluate_scenarios(
        cls,
        agent: Any,
        env: Any,
        max_steps: int = 40,
        calibrations: Optional[List[Any]] = None,
        is_atomic_cube: bool = True,
    ) -> Tuple[TrajectoryResult, List[TrajectoryResult]]:
        """Executa múltiplos cenários sobre o mesmo ponto de partida com diferentes calibrações e seleciona o melhor."""
        if calibrations is None:
            calibrations = cls.DEFAULT_CALIBRATIONS


        results: List[TrajectoryResult] = []

        # Determina como clonar o ambiente a partir do estado atual
        can_clone = hasattr(env, "clone") and callable(getattr(env, "clone"))

        for s_id, calib in enumerate(calibrations):
            t0 = time.perf_counter()

            # Cria instância independente do ambiente
            if can_clone:
                sim_env = env.clone()
                obs_dict = {
                    "obs": sim_env.core.get_one_hot() if hasattr(sim_env, "core") else sim_env.reset()[0],
                    "prev_action": 0,
                    "prev_reward": 0.0,
                    "delta_obs": None,
                }
            elif hasattr(env, "core"):
                # Fallback para ambientes de Cubo sem clone explícito
                from system1_engine.env.adapters.rubiks import RubiksCubeEnv
                sim_env = RubiksCubeEnv(
                    render_mode=getattr(env, "render_mode", "rgb_array"),
                    scramble_depth=getattr(env, "scramble_depth", 4),
                    max_steps=max_steps,
                )
                sim_env.reset()
                sim_env.core.state = env.core.state.copy()
                sim_env._steps = 0
                sim_env._prev_score = env.core.get_score()
                obs_dict = {
                    "obs": sim_env.core.get_one_hot(),
                    "prev_action": 0,
                    "prev_reward": 0.0,
                    "delta_obs": None,
                }
            else:
                raise ValueError("Ambiente não suporta clonagem para avaliação em múltiplos cenários.")

            # Reseta estado neural e histórico para cada cenário
            agent.reset_memory()

            actions: List[int] = []
            decisions: List[Any] = []
            cumulative_reward = 0.0
            latent_values: List[float] = []
            solved = False

            curr_obs = obs_dict

            for step in range(max_steps):
                decision = agent.act_fast(
                    curr_obs,
                    calibration=calib,
                    return_decision=True,
                    return_value=True,
                    quality_filter=True,
                )
                act = int(decision.action)
                actions.append(act)
                decisions.append(decision)
                if decision.latent_value is not None:
                    latent_values.append(decision.latent_value)

                next_obs, rew, terminated, truncated, info = sim_env.step(act)
                cumulative_reward += float(rew)

                is_solved = bool(info.get("is_solved", False) or terminated)
                if is_solved:
                    solved = True
                    break
                if truncated:
                    break

                curr_obs = {
                    "obs": next_obs,
                    "prev_action": act,
                    "prev_reward": float(rew),
                    "delta_obs": None,
                }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            purity = cls.compute_purity_score(actions, is_atomic_cube=is_atomic_cube)
            mean_val = float(np.mean(latent_values)) if latent_values else 0.0
            n_steps = len(actions)

            # Função de pontuação de qualidade
            if solved:
                # Prioridade absoluta para resolução, premiando menos passos e alta pureza
                quality_score = 10000.0 - (n_steps * 20.0) + (cumulative_reward * 5.0) + (purity * 100.0)
            else:
                # Caso não resolvido, premia progresso de recompensa, valor latente e pureza
                quality_score = (cumulative_reward * 10.0) + (mean_val * 5.0) + (purity * 50.0) - (n_steps * 2.0)

            traj = TrajectoryResult(
                scenario_id=s_id,
                calibration=calib,
                solved=solved,
                steps=n_steps,
                cumulative_reward=cumulative_reward,
                mean_latent_value=mean_val,
                purity_score=purity,
                actions=actions,
                decisions=decisions,
                quality_score=quality_score,
                elapsed_ms=elapsed_ms,
            )
            results.append(traj)

        # Ordena pelo quality_score decrescente
        sorted_results = sorted(results, key=lambda x: x.quality_score, reverse=True)
        best_result = sorted_results[0]

        return best_result, sorted_results


class ConfidenceGatedEnsemble:
    """Ensemble Ativo por Gating de Confiança para System 1 (A.2).

    Disparado exclusivamente quando o agente detecta 'is_uncertain == True' em act_fast,
    avaliando instantaneamente múltiplas hipóteses térmicas a partir da representação
    latente 'h' do trunk. Executa em tempo sub-milissegundo (< 0.1 ms) sem reexecutar o trunk.
    """

    DEFAULT_CANDIDATE_CALIBRATIONS = [0.0, 0.15, 0.35, 0.70]

    @classmethod
    def resolve_uncertain_action(
        cls,
        agent: Any,
        h: Any,
        is_discrete: bool,
        avoid_set: Optional[Any] = None,
        candidate_calibrations: Optional[List[float]] = None,
        quality_filter: bool = True,
    ) -> Tuple[Union[int, np.ndarray], Dict[str, Any]]:
        """Resolve a ação sob incerteza combinando consenso térmico de calibração e pureza de histórico."""
        if candidate_calibrations is None:
            candidate_calibrations = cls.DEFAULT_CANDIDATE_CALIBRATIONS
        if avoid_set is None:
            avoid_set = set()

        hist = getattr(agent, "_action_history", [])

        if is_discrete:
            import torch
            logits_sq = agent.policy_head.linear(h).squeeze(0).squeeze(0)
            base_probs = torch.softmax(logits_sq, dim=-1)
            n_acts = base_probs.shape[0]

            action_candidates: List[int] = []
            greedy_act = int(torch.argmax(logits_sq).item())
            action_candidates.append(greedy_act)

            for calib in candidate_calibrations:
                if calib <= 0.0:
                    continue
                eff_temp = 1.0 + 0.5 * calib
                scaled_probs = torch.softmax(logits_sq / eff_temp, dim=-1)
                top_k = min(2, n_acts)
                top_indices = torch.topk(scaled_probs, k=top_k).indices
                for idx in top_indices:
                    action_candidates.append(int(idx.item()))

            unique_candidates: List[int] = []
            for a in action_candidates:
                if a not in unique_candidates:
                    unique_candidates.append(a)

            valid_candidates = [a for a in unique_candidates if a not in avoid_set]
            if not valid_candidates:
                valid_candidates = unique_candidates

            best_action = valid_candidates[0]
            best_score = -float("inf")
            candidate_scores: Dict[int, float] = {}

            for a in valid_candidates:
                p = float(base_probs[a].item())
                penalty = 0.0
                if quality_filter and len(hist) >= 1 and n_acts == 12:
                    inv_prev = (hist[-1] + 1) if (hist[-1] % 2 == 0) else (hist[-1] - 1)
                    if a == inv_prev:
                        penalty += 0.40
                if quality_filter and len(hist) >= 2 and a == hist[-2]:
                    penalty += 0.20

                score = p - penalty
                candidate_scores[a] = score
                if score > best_score:
                    best_score = score
                    best_action = a

            meta = {
                "ensemble_fallback": True,
                "selected_action": best_action,
                "candidates": valid_candidates,
                "scores": candidate_scores,
            }
            return best_action, meta

        else:
            mu = agent.policy_head.mu_net(h).squeeze(0).squeeze(0).cpu().numpy()
            best_action = mu
            if len(hist) >= 1 and isinstance(hist[-1], np.ndarray):
                best_action = 0.85 * mu + 0.15 * hist[-1]

            meta = {
                "ensemble_fallback": True,
                "selected_action": best_action,
            }
            return best_action, meta

