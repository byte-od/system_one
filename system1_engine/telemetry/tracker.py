from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class PhaseMetrics:
    """Estrutura com ring buffers (deques de tamanho fixo) em memória RAM para métricas O(1)."""

    # Fase 1: Percepção, Inferência & Decisão Reflexiva
    step_latencies_us: deque = field(default_factory=lambda: deque(maxlen=1000))
    uncertainties: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidences: deque = field(default_factory=lambda: deque(maxlen=1000))
    entropies: deque = field(default_factory=lambda: deque(maxlen=1000))
    calibrations: deque = field(default_factory=lambda: deque(maxlen=1000))
    action_chunk_size: int = 1
    value_rerank: bool = False
    value_reranks_triggered: int = 0

    # Fase 2: Ambiente Físico & Rollout GAE
    fps: float = 0.0
    current_episode_return: float = 0.0
    current_episode_length: int = 0
    episode_returns: deque = field(default_factory=lambda: deque(maxlen=100))
    episode_lengths: deque = field(default_factory=lambda: deque(maxlen=100))
    total_steps: int = 0
    target_steps: Optional[int] = None
    target_return: Optional[float] = None
    best_return: Optional[float] = None
    best_mean_return: Optional[float] = None

    # Fase 3: Otimização PPO & Saúde dos Gradientes
    policy_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    value_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    grad_norms: Dict[str, float] = field(default_factory=dict)
    clip_fractions: deque = field(default_factory=lambda: deque(maxlen=100))
    learning_rate: float = 0.0

    # Métricas de Self-Imitation Learning (SIL) & Contrastive DPO
    sil_trajectories: int = 0
    sil_failures: int = 0
    sil_policy_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    sil_value_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    sil_dpo_losses: deque = field(default_factory=lambda: deque(maxlen=100))

    # Métricas de Currículo (Progresso Dinâmico)
    curriculum_depth: Optional[int] = None
    curriculum_max_depth: Optional[int] = None
    curriculum_success_rate: Optional[float] = None
    curriculum_promotions: int = 0
    curriculum_depth_best_return: Optional[float] = None
    curriculum_depth_best_success_rate: Optional[float] = None
    curriculum_depth_min_success_rate: Optional[float] = None
    successful_depth: Optional[int] = None
    best_return_depth: Optional[int] = None
    dynamic_frontier: bool = True
    curriculum_p_frontier: float = 1.0

    # Estado de Execução
    is_completed: bool = False


class LiveStatsTracker:
    """Coletor centralizado de telemetria de alta frequência com overhead < 2 µs."""

    def __init__(self) -> None:
        self.metrics = PhaseMetrics()
        self._last_step_time = time.perf_counter_ns()
        self._fps_window: deque = deque(maxlen=100)

    def record_inference(
        self,
        latency_us: float,
        uncertainty: float,
        confidence: float,
        entropy: float,
        calibration: float = 0.0,
        value_rerank_triggered: bool = False,
    ) -> None:
        """Registra a fase de reflexo imediato com inserção O(1) sem alocação dinâmica."""
        m = self.metrics
        m.step_latencies_us.append(latency_us)
        m.uncertainties.append(uncertainty)
        m.confidences.append(confidence)
        m.entropies.append(entropy)
        m.calibrations.append(calibration)
        if value_rerank_triggered:
            m.value_reranks_triggered += 1

    def record_env_step(
        self,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        num_steps: int = 1,
    ) -> None:
        """Registra a fase de interação física com o ambiente e metadados de currículo."""
        now = time.perf_counter_ns()
        dt = (now - self._last_step_time) / 1e9
        self._last_step_time = now

        if dt > 0:
            self._fps_window.append(float(num_steps) / dt)
            self.metrics.fps = float(np.mean(self._fps_window))

        m = self.metrics
        m.total_steps += int(num_steps)
        m.current_episode_return += reward
        m.current_episode_length += 1

        if info:
            is_promoted = False
            if "current_depth" in info:
                new_depth = int(info["current_depth"])
                if m.curriculum_depth is not None and new_depth != m.curriculum_depth:
                    # Mudança de profundidade no currículo: reseta pico e extremos da dimensão atual
                    m.curriculum_depth_best_return = None
                    m.curriculum_depth_best_success_rate = None
                    m.curriculum_depth_min_success_rate = None
                    m.curriculum_success_rate = 0.0
                    is_promoted = True
                m.curriculum_depth = new_depth
            if "max_depth" in info:
                m.curriculum_max_depth = info["max_depth"]
            if "curriculum_promotions" in info:
                new_promos = int(info["curriculum_promotions"])
                if new_promos > m.curriculum_promotions:
                    # Promoção explícita de currículo: reseta pico e extremos da dimensão anterior
                    m.curriculum_depth_best_return = None
                    m.curriculum_depth_best_success_rate = None
                    m.curriculum_depth_min_success_rate = None
                    m.curriculum_success_rate = 0.0
                    is_promoted = True
                m.curriculum_promotions = new_promos
            if "curriculum_success_rate" in info:
                rate = float(info["curriculum_success_rate"])
                if is_promoted and rate >= 0.8:
                    # Resíduo da vitória que disparou a promoção no nível anterior: inicializa em 0.0
                    m.curriculum_success_rate = 0.0
                else:
                    m.curriculum_success_rate = rate
                    if m.curriculum_depth_best_success_rate is None or rate > m.curriculum_depth_best_success_rate:
                        m.curriculum_depth_best_success_rate = rate
                    if m.curriculum_depth_min_success_rate is None or rate < m.curriculum_depth_min_success_rate:
                        m.curriculum_depth_min_success_rate = rate
            if "curriculum_p_frontier" in info:
                m.curriculum_p_frontier = float(info["curriculum_p_frontier"])
            if "dynamic_frontier" in info:
                m.dynamic_frontier = bool(info["dynamic_frontier"])

        if done:
            ep_ret = float(m.current_episode_return)
            m.episode_returns.append(ep_ret)
            m.episode_lengths.append(m.current_episode_length)
            if m.best_return is None or ep_ret > m.best_return:
                m.best_return = ep_ret

            # Atualiza pico do currículo atual
            if m.curriculum_depth is not None:
                if m.curriculum_depth_best_return is None or ep_ret > m.curriculum_depth_best_return:
                    m.curriculum_depth_best_return = ep_ret

            rolling_20 = list(m.episode_returns)[-20:]
            if rolling_20:
                cur_mean = float(np.mean(rolling_20))
                # Precedência curricular: um patamar mais profundo tem precedência de rastreamento de pico
                is_deeper = (
                    m.curriculum_depth is not None
                    and m.best_return_depth is not None
                    and m.curriculum_depth > m.best_return_depth
                )
                if m.best_mean_return is None or cur_mean > m.best_mean_return or is_deeper:
                    m.best_mean_return = cur_mean
                    m.best_return_depth = m.curriculum_depth

            m.current_episode_return = 0.0
            m.current_episode_length = 0

    def get_successful_depth(self) -> Optional[int]:
        """Calcula o último nível de profundidade curricular executado com sucesso."""
        m = self.metrics
        if m.curriculum_depth is None:
            return None

        # Se houve promoções no treino, o nível anterior à promoção atual foi concluído com sucesso
        if m.curriculum_promotions > 0:
            if m.curriculum_success_rate is not None and m.curriculum_success_rate >= 0.85:
                return m.curriculum_depth
            return max(1, m.curriculum_depth - 1)
        else:
            return m.curriculum_depth

    def record_training_epoch(
        self,
        policy_loss: float,
        value_loss: float,
        clip_fraction: float,
        grad_norms: Dict[str, float],
        lr: float,
    ) -> None:
        """Registra a fase de atualização dos gradientes e saúde numérica."""
        m = self.metrics
        m.policy_losses.append(policy_loss)
        m.value_losses.append(value_loss)
        m.clip_fractions.append(clip_fraction)
        m.grad_norms = dict(grad_norms)
        m.learning_rate = lr

    def record_sil_step(
        self,
        sil_trajectories: int,
        sil_policy_loss: float = 0.0,
        sil_value_loss: float = 0.0,
        sil_dpo_loss: float = 0.0,
        sil_failures: int = 0,
    ) -> None:
        """Registra o progresso e perdas do Self-Imitation Learning (SIL) e Contrastive DPO."""
        m = self.metrics
        m.sil_trajectories = int(sil_trajectories)
        if sil_failures > 0:
            m.sil_failures = int(sil_failures)
        if sil_policy_loss != 0.0:
            m.sil_policy_losses.append(sil_policy_loss)
        if sil_value_loss != 0.0:
            m.sil_value_losses.append(sil_value_loss)
        if sil_dpo_loss != 0.0:
            m.sil_dpo_losses.append(sil_dpo_loss)

    def snapshot(self) -> Dict[str, Any]:
        """Extrai um resumo numérico instantâneo de todas as três fases."""
        m = self.metrics
        return {
            # Fase 1: Inferência & Ação
            "latency_p50_us": float(np.percentile(m.step_latencies_us, 50)) if m.step_latencies_us else 0.0,
            "latency_p99_us": float(np.percentile(m.step_latencies_us, 99)) if m.step_latencies_us else 0.0,
            "mean_uncertainty": float(np.mean(m.uncertainties)) if m.uncertainties else 0.0,
            "mean_confidence": float(np.mean(m.confidences)) if m.confidences else 0.0,
            "mean_entropy": float(np.mean(m.entropies)) if m.entropies else 0.0,
            "calibration": float(m.calibrations[-1]) if m.calibrations else 0.0,
            "action_chunk_size": int(m.action_chunk_size),
            "value_rerank": bool(m.value_rerank),
            "value_reranks_triggered": int(m.value_reranks_triggered),
            # Fase 2: Ambiente & Rollout
            "fps": m.fps,
            "mean_return_20": float(np.mean(list(m.episode_returns)[-20:])) if m.episode_returns else 0.0,
            "best_return": float(m.best_return) if m.best_return is not None else 0.0,
            "best_mean_return": float(m.best_mean_return) if m.best_mean_return is not None else 0.0,
            "total_steps": m.total_steps,
            "target_steps": m.target_steps,
            "target_return": float(m.target_return) if m.target_return is not None else None,
            "episodes_completed": len(m.episode_returns),
            "current_episode_return": m.current_episode_return,
            "curriculum_depth": m.curriculum_depth,
            "curriculum_max_depth": m.curriculum_max_depth,
            "curriculum_success_rate": m.curriculum_success_rate,
            "curriculum_promotions": m.curriculum_promotions,
            "dynamic_frontier": bool(m.dynamic_frontier),
            "curriculum_p_frontier": float(m.curriculum_p_frontier),
            "curriculum_depth_best_return": float(m.curriculum_depth_best_return) if m.curriculum_depth_best_return is not None else None,
            "curriculum_depth_best_success_rate": float(m.curriculum_depth_best_success_rate) if m.curriculum_depth_best_success_rate is not None else None,
            "curriculum_depth_min_success_rate": float(m.curriculum_depth_min_success_rate) if m.curriculum_depth_min_success_rate is not None else None,
            "successful_depth": self.get_successful_depth(),
            "best_return_depth": m.best_return_depth,
            # Fase 3: Otimização PPO & SIL
            "policy_loss": m.policy_losses[-1] if m.policy_losses else 0.0,
            "value_loss": m.value_losses[-1] if m.value_losses else 0.0,
            "clip_frac": m.clip_fractions[-1] if m.clip_fractions else 0.0,
            "lr": m.learning_rate,
            "grad_norms": dict(m.grad_norms),
            "sil_trajectories": m.sil_trajectories,
            "sil_failures": m.sil_failures,
            "sil_policy_loss": m.sil_policy_losses[-1] if m.sil_policy_losses else 0.0,
            "sil_value_loss": m.sil_value_losses[-1] if m.sil_value_losses else 0.0,
            "sil_dpo_loss": m.sil_dpo_losses[-1] if m.sil_dpo_losses else 0.0,
            # Estado do Ciclo de Vida
            "is_completed": m.is_completed,
        }

    def set_completed(self, completed: bool = True) -> None:
        """Marca o ciclo de vida como concluído para estabilização dos gráficos no painel."""
        self.metrics.is_completed = completed

    def reset(self) -> None:
        """Zera as métricas acumuladas."""
        self.metrics = PhaseMetrics()
        self._last_step_time = time.perf_counter_ns()
        self._fps_window.clear()
