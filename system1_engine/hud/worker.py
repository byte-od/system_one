import argparse
import http.client
import io
import json
import sys
import threading
import time
import urllib.parse
from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
from PIL import Image
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.attention import GradCAMExplainer
from system1_engine.env.adapters import make_game_env
from system1_engine.env.dependencies import make_gym_env_with_auto_install
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry.tracker import LiveStatsTracker
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


class HUDClient:
    """Cliente HTTP interno assíncrono de alto desempenho e zero overhead para o HUD."""

    def __init__(self, server_url: str, initial_vision_mode: str = "normal") -> None:
        parsed = urllib.parse.urlparse(server_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 8050
        self._conn: Optional[http.client.HTTPConnection] = None
        self._vision_mode: str = initial_vision_mode
        self._cube_view_mode: str = "3d"

        # Sincronização assíncrona para zero overhead no loop principal de treino
        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_telemetry: Optional[Dict[str, Any]] = None
        self._worker_busy: bool = False
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()

        # Thread de segundo plano dedicada à compressão JPEG e tráfego de rede
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _get_connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
        return self._conn

    def is_ready_for_frame(self) -> bool:
        """Indica se o worker assíncrono está livre para receber um novo frame.
        Permite que a thread de treino evite chamadas desnecessárias a env.render()."""
        with self._lock:
            return self._pending_frame is None and not self._worker_busy

    def send_telemetry(self, data: Dict[str, Any]) -> None:
        """Enfileira a telemetria mais recente de forma não-bloqueante (< 1 µs)."""
        with self._lock:
            self._pending_telemetry = data
        self._wake_event.set()

    def send_frame(self, frame: np.ndarray) -> None:
        """Submete o frame RGB de forma não-bloqueante (< 1 µs), sem pausar o treino."""
        with self._lock:
            self._pending_frame = frame
        self._wake_event.set()

    def get_vision_mode(self) -> str:
        """Retorna o modo de atenção visual atualmente selecionado no HUD."""
        with self._lock:
            return self._vision_mode

    def get_cube_view_mode(self) -> str:
        """Retorna o modo de visualização 3D/2D para Cubo Mágico."""
        with self._lock:
            return self._cube_view_mode

    def _worker_loop(self) -> None:
        """Loop de fundo para processamento offloaded de telemetria e compressão/envio de vídeo."""
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.05)
            self._wake_event.clear()

            # 1. Envia telemetria pendente se houver
            telem = None
            with self._lock:
                if self._pending_telemetry is not None:
                    telem = self._pending_telemetry
                    self._pending_telemetry = None

            if telem is not None:
                self._dispatch_telemetry(telem)

            # 2. Processa e envia frame pendente se houver
            frame = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None
                    self._worker_busy = True

            if frame is not None:
                try:
                    self._dispatch_frame(frame)
                finally:
                    with self._lock:
                        self._worker_busy = False

    def _dispatch_telemetry(self, data: Dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json", "Content-Length": str(len(payload))}
        for _ in range(2):
            try:
                conn = self._get_connection()
                conn.request("POST", "/api/internal/telemetry", payload, headers)
                res = conn.getresponse()
                raw = res.read()
                if res.status == 200 and raw:
                    try:
                        resp_data = json.loads(raw.decode("utf-8"))
                        if "vision_mode" in resp_data:
                            with self._lock:
                                self._vision_mode = resp_data["vision_mode"]
                        if "cube_view_mode" in resp_data:
                            with self._lock:
                                self._cube_view_mode = resp_data["cube_view_mode"]
                    except Exception:
                        pass
                return
            except Exception:
                self._conn = None

    def _dispatch_frame(self, frame: np.ndarray) -> None:
        try:
            arr = frame
            # Normalização de dimensões: CHW -> HWC se necessário
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)

            img = Image.fromarray(arr)
            # Redimensiona para resolução do viewport se necessário
            if img.width > 640 or img.height > 480:
                img.thumbnail((640, 480))

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=65)
            jpeg_bytes = buf.getvalue()

            headers = {"Content-Type": "image/jpeg", "Content-Length": str(len(jpeg_bytes))}
            for _ in range(2):
                try:
                    conn = self._get_connection()
                    conn.request("POST", "/api/internal/frame", jpeg_bytes, headers)
                    res = conn.getresponse()
                    raw = res.read()
                    if res.status == 200 and raw:
                        try:
                            resp_data = json.loads(raw.decode("utf-8"))
                            if "vision_mode" in resp_data:
                                with self._lock:
                                    self._vision_mode = resp_data["vision_mode"]
                            if "cube_view_mode" in resp_data:
                                with self._lock:
                                    self._cube_view_mode = resp_data["cube_view_mode"]
                        except Exception:
                            pass
                    return
                except Exception:
                    self._conn = None
        except Exception:
            pass

    def close(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def build_hud_env(
    env_id: str,
    scenario: Optional[str] = None,
    render_mode: str = "in_browser",
    is_training: bool = False,
    frame_skip: Optional[int] = None,
    scramble_depth: Optional[int] = None,
    num_envs: Optional[int] = None,
    mixed_replay_rate: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> UniversalS1Wrapper:
    """Instancia o ambiente com suporte a renderização in-browser (rgb_array), janela ou headless no Modo Puro Universal."""
    env_lower = env_id.lower().replace("_", "-")
    if env_lower in ("rubiks", "rubik", "rubiks-macro", "rubiksmacro", "rubik-macro"):
        env_id = "RubiksCubeMacro-v0"
    elif env_lower in ("rubiks-atomic", "rubik-atomic", "rubiksatomic", "rubikatomic"):
        env_id = "RubiksCube-v0"

    if env_id.lower() in ["vizdoom", "native"]:
        scenario_path = scenario or "basic.cfg"
        headless = render_mode != "window"
        fs = frame_skip if frame_skip is not None else 4
        env = make_game_env(
            "native",
            {
                "engine_type": "vizdoom",
                "scenario_path": scenario_path,
                "args": {"headless": headless, "frame_skip": fs},
            },
        )
        return env

    # Gym environments
    gym_render_mode = None
    if render_mode == "in_browser":
        gym_render_mode = "rgb_array"
    elif render_mode == "window":
        gym_render_mode = "human"

    extra_kwargs = {}
    if env_id in ("RubiksCube-v0", "RubiksCubeMacro-v0"):
        if is_training:
            extra_kwargs["curriculum"] = True
            extra_kwargs["min_depth"] = 1
            if scramble_depth is not None and int(scramble_depth) > 0:
                extra_kwargs["max_depth"] = int(scramble_depth)
                extra_kwargs["scramble_depth"] = int(scramble_depth)
            else:
                extra_kwargs["max_depth"] = None
            if mixed_replay_rate is not None:
                extra_kwargs["mixed_replay_rate"] = float(mixed_replay_rate)
            elif config is not None:
                if "mixed_replay_rate" in config:
                    extra_kwargs["mixed_replay_rate"] = float(config["mixed_replay_rate"])
                elif "mixed_replay" in config:
                    extra_kwargs["mixed_replay_rate"] = 0.0 if not config["mixed_replay"] else 0.15
            if config is not None:
                if "dynamic_frontier" in config:
                    extra_kwargs["dynamic_frontier"] = bool(config["dynamic_frontier"])
                elif "proximal_frontier" in config:
                    extra_kwargs["dynamic_frontier"] = bool(config["proximal_frontier"])
                if "proximal_decay" in config:
                    extra_kwargs["proximal_decay"] = float(config["proximal_decay"])
        else:
            extra_kwargs["curriculum"] = False
            if scramble_depth is not None and int(scramble_depth) > 0:
                extra_kwargs["scramble_depth"] = int(scramble_depth)

    if is_training and env_id in ("RubiksCube-v0", "RubiksCubeMacro-v0"):
        from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv
        is_macro = (env_id == "RubiksCubeMacro-v0")
        raw_env = VectorizedRubiksEnv(
            num_envs=int(num_envs) if num_envs is not None else 64,
            is_macro=is_macro,
            render_mode=gym_render_mode,
            **extra_kwargs,
        )
    else:
        raw_env = make_gym_env_with_auto_install(env_id, render_mode=gym_render_mode, **extra_kwargs)
    if env_id == "MountainCar-v0":
        from system1_engine.env.adapters.mountain_car import (
            MountainCarEnergyRewardWrapper,
            MountainCarNormalizedWrapper,
        )
        if is_training:
            raw_env = MountainCarEnergyRewardWrapper(raw_env)
        raw_env = MountainCarNormalizedWrapper(raw_env)
    elif env_id == "FrozenLake-v1":
        from system1_engine.env.adapters.frozenlake import FrozenLakeCurriculumWrapper
        raw_env = FrozenLakeCurriculumWrapper(raw_env, curriculum=is_training)

    return UniversalS1Wrapper(raw_env)


def resolve_compute_device(device_pref: str = "auto", is_training: bool = True) -> torch.device:
    """Resolve o dispositivo de computação (CPU, MPS ou CUDA).

    - 'auto': Se for treino, usa GPU (MPS/CUDA) se disponível para acelerar backprop (~6x mais rápido).
              Se for avaliação/inferência passo-a-passo, usa CPU para latência mínima sem overhead de barramento.
    - 'gpu' / 'cuda' / 'mps': Força o uso do acelerador gráfico disponível (CUDA no Linux/Windows ou MPS no macOS).
    - 'cpu': Força o uso exclusivo de CPU pura universal.
    """
    pref = (device_pref or "auto").strip().lower()
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if pref == "cpu":
        return torch.device("cpu")

    if pref in ("gpu", "cuda", "mps"):
        if pref == "cuda" and has_cuda:
            return torch.device("cuda")
        if pref == "mps" and has_mps:
            return torch.device("mps")
        if has_cuda:
            return torch.device("cuda")
        if has_mps:
            return torch.device("mps")
        print("[Aviso] Acelerador GPU solicitado, mas nenhum dispositivo CUDA/MPS foi detectado. Revertendo para CPU.")
        return torch.device("cpu")

    # pref == 'auto'
    if is_training:
        if has_cuda:
            return torch.device("cuda")
        if has_mps:
            return torch.device("mps")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def run_worker_train(config: Dict[str, Any], hud_client: HUDClient) -> None:
    env_id = config.get("env", "CartPole-v1")
    # Diretriz S1: Treino deve ser sempre headless (100% de capacidade do hardware sem renderização)
    render_mode = "none"
    device_pref = config.get("device", "auto")
    steps = int(config.get("steps", 40000))
    lr = float(config.get("lr", 1e-4))
    entropy_coef = float(config.get("entropy_coef", 0.015))
    target_return = float(config.get("target_return", 475.0))
    rollout_steps = int(config.get("rollout_steps", 2048))
    chunk_length = int(config.get("chunk_length", 32))
    chunk_batch_size = int(config.get("chunk_batch_size", 16))
    save_path = config.get("save")
    load_path = config.get("load")
    transfer_from = config.get("transfer_from")
    checkpoint_to_load = transfer_from or load_path
    freeze_trunk_cfg = config.get("freeze_trunk")
    force_trunk_only = bool(config.get("force_trunk_only", False))

    device = resolve_compute_device(device_pref, is_training=True)

    print(f"=== [HUD Worker] Modo de Treino Iniciado no Ambiente: {env_id} ===")
    print(f"Dispositivo de Execução: {device.type.upper()} (Preferência: {device_pref})")
    print(f"Configuração: steps={steps}, lr={lr}, entropy_coef={entropy_coef}, target_return={target_return}")
    if checkpoint_to_load:
        print(f"Carregamento de pesos para treino a partir de: {checkpoint_to_load}")

    scramble_depth = config.get("scramble_depth")
    if scramble_depth is not None:
        try:
            scramble_depth = int(scramble_depth)
        except (ValueError, TypeError):
            scramble_depth = None

    if env_id.lower().replace("_", "-") in ("rubikscube-v0", "rubikscubemacro-v0", "rubiks", "rubik"):
        if scramble_depth:
            print(f"🎲 Cubo Mágico: Treino com Curriculum Ativo (1 -> {scramble_depth} passos)")
        else:
            print(f"🎲 Cubo Mágico: Treino com Curriculum Aberto (1 -> ∞ passos)")

    env = build_hud_env(
        env_id,
        scenario=config.get("scenario"),
        render_mode=render_mode,
        is_training=True,
        frame_skip=int(config.get("frame_skip", 4)),
        scramble_depth=scramble_depth,
        num_envs=int(config.get("num_envs", 64)),
        config=config,
    )
    trunk_type = config.get("trunk", "lean")
    if checkpoint_to_load and "trunk" not in config:
        detected = KnowledgeTransferManager.detect_trunk_type(checkpoint_to_load)
        if detected:
            trunk_type = detected

    action_chunk_size = int(config.get("action_chunk_size", config.get("act_k", 1)))
    if checkpoint_to_load and "action_chunk_size" not in config and "act_k" not in config:
        detected_k = KnowledgeTransferManager.detect_action_chunk_size(checkpoint_to_load, env.action_space)
        if detected_k > 1:
            action_chunk_size = detected_k
            print(f"🧩 Auto-detectado Action Chunk Size k={action_chunk_size} a partir de {checkpoint_to_load}")

    if action_chunk_size > 1:
        print(f"⚡ Action Chunking Temporal (ACT) ATIVO: k={action_chunk_size} macro-ações coordenadas por passo")

    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type=trunk_type,
        action_chunk_size=action_chunk_size,
    )

    if checkpoint_to_load:
        res = KnowledgeTransferManager.load_for_training(
            agent=agent,
            checkpoint_path=checkpoint_to_load,
            freeze_trunk=freeze_trunk_cfg,
            force_trunk_only=force_trunk_only,
        )
        print(f"Pesos de treino carregados ({res['mode']}): {res['total_params']} parâmetros ativos. Trunk congelado: {res['trunk_frozen']}")

    tracker = LiveStatsTracker()
    tracker.metrics.target_steps = steps
    tracker.metrics.target_return = float(target_return)
    last_telemetry_ts = 0.0
    last_frame_ts = 0.0

    train_calib_cfg = config.get("train_calibration", config.get("exploration_scale", "auto"))
    if str(train_calib_cfg).lower() in ("auto", "adaptive", "homeostatic"):
        train_calibration = "auto"
        print("⚡ Modo de Exploração no Treino: AUTO / HOMEOSTASE DINÂMICA (Auto-correção ativa)")
    else:
        try:
            train_calibration = float(np.clip(float(train_calib_cfg), 0.05, 1.0))
            print(f"🎯 Modo de Exploração no Treino: Fixa em {train_calibration:.2f}")
        except (ValueError, TypeError):
            train_calibration = "auto"
            print("⚡ Modo de Exploração no Treino: AUTO / HOMEOSTASE DINÂMICA (Auto-correção ativa)")

    hindsight_replay = bool(config.get("hindsight_replay", True))
    hindsight_bonus = float(config.get("hindsight_bonus", 2.0))
    hindsight_min_gain = float(config.get("hindsight_min_gain", 0.02))

    if hindsight_replay:
        print(f"🎯 Hindsight Experience Replay (B.2): ATIVO (Bônus={hindsight_bonus:.1f}, Min Gain={hindsight_min_gain:.2f})")
    else:
        print("⚡ Hindsight Experience Replay (B.2): DESATIVADO")

    sil_enabled = bool(config.get("sil_enabled", True))
    sil_coef = float(config.get("sil_coef", 0.08))
    sil_batch_size = int(config.get("sil_batch_size", 8))
    sil_max_trajectories = int(config.get("sil_max_trajectories", 10240))
    sil_max_per_depth = int(config.get("sil_max_per_depth", 2048))
    min_capacity_per_depth = int(config.get("sil_min_capacity_per_depth", 32))
    max_failures_per_depth = int(config.get("max_failures_per_depth", 512))
    min_failures_per_depth = int(config.get("min_failures_per_depth", 16))

    sil_dpo_enabled = bool(config.get("sil_dpo_enabled", True))
    dpo_coef = float(config.get("dpo_coef", 0.05))
    dpo_beta = float(config.get("dpo_beta", 0.10))
    dpo_batch_size = int(config.get("dpo_batch_size", 8))
    progressive_buffer_capacity = bool(config.get("progressive_buffer_capacity", True))

    if sil_enabled:
        print(f"🧠 Self-Imitation Learning (SIL Replay): ATIVO (coef={sil_coef:.2f}, batch={sil_batch_size}, max_trajs={sil_max_trajectories}, max_per_depth={sil_max_per_depth}, progressive={progressive_buffer_capacity})")
        if sil_dpo_enabled:
            print(f"⚖️ Contrastive Online DPO-SIL (c-SIL): ATIVO (coef={dpo_coef:.2f}, beta={dpo_beta:.2f}, batch_pairs={dpo_batch_size})")
    else:
        print("⚡ Self-Imitation Learning (SIL Replay): DESATIVADO")

    sterility_memory_enabled = bool(config.get("sterility_memory_enabled", False))
    if sterility_memory_enabled:
        print("🧠 Memória de Esterilidade Latente: ATIVA")
    else:
        print("⚡ Memória de Esterilidade Latente: DESATIVADA (Pura inferência de alta velocidade)")

    if tracker is not None:
        tracker.metrics.action_chunk_size = action_chunk_size

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        action_chunk_size=action_chunk_size,
        adaptive_chunk=bool(config.get("adaptive_chunk", False)),
        learning_rate=lr,
        rollout_steps=rollout_steps,
        chunk_length=chunk_length,
        chunk_batch_size=chunk_batch_size,
        entropy_coef=entropy_coef,
        tracker=tracker,
        device=device,
        exploration_scale=train_calibration,
        hindsight_replay=hindsight_replay,
        hindsight_bonus=hindsight_bonus,
        hindsight_min_gain=hindsight_min_gain,
        sil_enabled=sil_enabled,
        sil_coef=sil_coef,
        sil_batch_size=sil_batch_size,
        sil_max_trajectories=sil_max_trajectories,
        sil_max_per_depth=sil_max_per_depth,
        min_capacity_per_depth=min_capacity_per_depth,
        sil_dpo_enabled=sil_dpo_enabled,
        dpo_coef=dpo_coef,
        dpo_beta=dpo_beta,
        dpo_batch_size=dpo_batch_size,
        progressive_buffer_capacity=progressive_buffer_capacity,
        max_failures_per_depth=max_failures_per_depth,
        min_failures_per_depth=min_failures_per_depth,
        sterility_memory_enabled=sterility_memory_enabled,
        dynamic_frontier=bool(config.get("dynamic_frontier", True)),
        proximal_decay=float(config.get("proximal_decay", 0.8)),
    )

    raw_env = getattr(env, "unwrapped", env)
    if (
        checkpoint_to_load
        and res.get("mode") in ("warm_start", "net2net_expansion")
        and hasattr(raw_env, "set_curriculum_depth")
        and getattr(raw_env, "curriculum", False)
    ):
        ckpt_info = KnowledgeTransferManager.inspect_checkpoint(checkpoint_to_load)
        ckpt_succ_d = ckpt_info.get("successful_depth")
        ckpt_curr_d = ckpt_info.get("curriculum_depth") or ckpt_info.get("depth")
        ckpt_max_d = max(int(ckpt_succ_d or 1), int(ckpt_curr_d or 1))
        resume_d = int(ckpt_curr_d or ckpt_succ_d or 1)
        if ckpt_max_d > 1:
            raw_env.set_curriculum_depth(resume_d)
            trainer.best_state_depth = raw_env.current_depth
            trainer._rollout_tracked_depth = raw_env.current_depth
            tracker.metrics.curriculum_depth = raw_env.current_depth
            tracker.metrics.curriculum_promotions = raw_env.curriculum_promotions
            if sil_enabled:
                n_warm = trainer.warmup_sil_buffer(
                    up_to_depth=ckpt_max_d,
                    episodes_per_depth=32,
                    calibrate_curriculum=True,
                )
                rates_str = ", ".join(
                    f"D{d}:{sr*100:.0f}%"
                    for d, sr in sorted(getattr(trainer, "last_warmup_solve_rates", {}).items())
                )
                print(
                    f"🔥 [Warm-Start Curricular & SIL] Retomando diretamente em D{raw_env.current_depth} "
                    f"(Promoções consolidadas: {raw_env.curriculum_promotions}) | "
                    f"SIL pré-aquecido com {n_warm} trajetórias âncoras ({rates_str})"
                )
            else:
                print(
                    f"🔥 [Warm-Start Curricular] Retomando diretamente na Profundidade D{raw_env.current_depth} "
                    f"(Promoções consolidadas: {raw_env.curriculum_promotions})"
                )

    explainer = GradCAMExplainer(agent)

    last_ckpt_step = 0
    last_ckpt_promotions = int(getattr(raw_env, "curriculum_promotions", 0))
    best_saved_mean_return = -float("inf")

    def _persist_checkpoint(reason: str = "") -> None:
        if not save_path:
            return
        try:
            best_ret = getattr(trainer, "best_depth_mean_return", None)
            if best_ret is None or best_ret <= -float("inf"):
                best_ret = getattr(trainer, "best_mean_return", None)
            if best_ret is None and tracker and tracker.metrics.best_mean_return is not None:
                best_ret = tracker.metrics.best_mean_return
            if best_ret is None:
                best_ret = getattr(trainer, "final_return", 0.0)

            successful_d = tracker.get_successful_depth() if tracker else None
            best_state_d = getattr(trainer, "best_state_depth", None)
            if best_state_d is None or best_state_d <= 0:
                best_state_d = getattr(tracker.metrics, "best_return_depth", None) if tracker else None
            if best_state_d is None and tracker and tracker.metrics.curriculum_depth is not None:
                best_state_d = tracker.metrics.curriculum_depth

            extra_info = {
                "env_id": env_id,
                "final_return": float(getattr(trainer, "final_return", 0.0)),
                "best_mean_return": float(best_ret),
                "steps": trainer.total_steps,
            }
            if tracker.metrics.curriculum_depth is not None:
                extra_info["curriculum_depth"] = tracker.metrics.curriculum_depth
                extra_info["successful_depth"] = successful_d if successful_d is not None else tracker.metrics.curriculum_depth
                extra_info["depth"] = tracker.metrics.curriculum_depth
            elif config.get("scramble_depth") is not None:

                sd = int(config["scramble_depth"])
                extra_info["scramble_depth"] = sd
                extra_info["successful_depth"] = sd
                extra_info["depth"] = sd

            if tracker.metrics.curriculum_max_depth is not None:
                extra_info["curriculum_max_depth"] = tracker.metrics.curriculum_max_depth
            elif config.get("scramble_depth") is not None:
                extra_info["curriculum_max_depth"] = int(config["scramble_depth"])

            if best_state_d is not None:
                extra_info["best_return_depth"] = int(best_state_d)
                extra_info["best_state_depth"] = int(best_state_d)
            elif getattr(tracker.metrics, "best_return_depth", None) is not None:
                extra_info["best_return_depth"] = tracker.metrics.best_return_depth

            KnowledgeTransferManager.save_checkpoint(
                agent=agent,
                checkpoint_path=save_path,
                extra_info=extra_info,
                auto_sync_web=config.get("auto_sync_web", True),
            )
            tag = f" [{reason}]" if reason else ""
            print(f"💾 Checkpoint persistido com sucesso em: {save_path}{tag}")
        except Exception as e:
            print(f"⚠️ Aviso ao salvar checkpoint ({reason}): {e}")

    def train_callback(cur_steps: int, mean_ret: float) -> None:
        nonlocal last_telemetry_ts, last_ckpt_step, last_ckpt_promotions, best_saved_mean_return
        now = time.time()
        snap = tracker.snapshot()
        snap["grad_norms"] = tracker.metrics.grad_norms
        hud_client.send_telemetry(snap)
        last_telemetry_ts = now

        # Salvamento automático periódico e por marcos (Curriculum Promotion & Novo Pico)
        if save_path:
            cur_promotions = tracker.metrics.curriculum_promotions if tracker else 0
            is_new_level = cur_promotions > last_ckpt_promotions
            if is_new_level:
                # Novo patamar curricular: reinicia o referencial de pico para o novo nível
                best_saved_mean_return = 0.0
            is_periodic = (cur_steps - last_ckpt_step) >= 500_000
            is_new_best = mean_ret > (best_saved_mean_return + 1.5) and cur_steps >= 20_000

            if is_new_level or is_periodic or is_new_best:
                reason_parts = []
                if is_new_level:
                    reason_parts.append(f"Promoção D{tracker.metrics.curriculum_depth}")
                    last_ckpt_promotions = cur_promotions
                if is_new_best:
                    reason_parts.append(f"Novo Pico {mean_ret:.2f}")
                    best_saved_mean_return = mean_ret
                if is_periodic:
                    reason_parts.append(f"Passo {cur_steps}")
                    last_ckpt_step = cur_steps

                _persist_checkpoint(reason=", ".join(reason_parts))

    def step_callback() -> None:
        nonlocal last_telemetry_ts, last_frame_ts
        now = time.time()
        # Telemetria a cada ~100 ms
        if now - last_telemetry_ts >= 0.1:
            snap = tracker.snapshot()
            snap["grad_norms"] = tracker.metrics.grad_norms
            hud_client.send_telemetry(snap)
            last_telemetry_ts = now

        # Renderização in-browser durante treino: amostragem assíncrona (~12-15 FPS)
        # Só solicita env.render() se o worker assíncrono estiver pronto para receber,
        # eliminando completamente qualquer bloqueio ou perda de throughput no treino!
        if render_mode == "in_browser" and (now - last_frame_ts >= 0.08) and hud_client.is_ready_for_frame():
            try:
                cube_vmode = hud_client.get_cube_view_mode()
                if hasattr(env.unwrapped, "set_view_mode"):
                    env.unwrapped.set_view_mode(cube_vmode)
                frame = env.render()
                if frame is not None:
                    try:
                        v_mode = hud_client.get_vision_mode()
                        if v_mode != "normal" and explainer.is_supported():
                            action_descs = getattr(getattr(env, "unwrapped", env), "action_descriptions", None)
                            cam_obs = getattr(env, "latest_obs_dict", None)
                            if cam_obs is not None:
                                heatmap, label = explainer.compute_saliency(
                                    obs_dict=cam_obs,
                                    hx=agent.hx,
                                    mode=v_mode,
                                    action_names=action_descs,
                                )
                                frame = explainer.render_overlay(frame, heatmap, label=label)
                    except Exception:
                        pass
                    hud_client.send_frame(frame)
                    last_frame_ts = now
            except Exception:
                pass

    try:
        final_return = trainer.train(
            max_steps=steps,
            target_return=target_return,
            callback=train_callback,
            step_callback=step_callback,
            verbose=True,
        )

        if save_path:
            _persist_checkpoint("Final")

        tracker.set_completed(True)
        final_snap = tracker.snapshot()
        final_snap["grad_norms"] = tracker.metrics.grad_norms
        hud_client.send_telemetry(final_snap)
        print(f"=== [HUD Worker] Treinamento Concluído! Retorno Final: {final_return:.2f} ===")

    except KeyboardInterrupt:
        print("\n[HUD Worker] ⏹️ Treinamento interrompido. Salvando checkpoint de segurança...")
        if save_path:
            try:
                _persist_checkpoint("Interrupted")
            except Exception as e:
                print(f"[HUD Worker] Aviso ao salvar checkpoint de interrupção: {e}")

    finally:
        explainer.close()
        env.close()


def run_worker_eval(config: Dict[str, Any], hud_client: HUDClient) -> None:
    env_id = config.get("env", "CartPole-v1")
    # Diretriz S1: Avaliação deve ser sempre in-browser assíncrona
    render_mode = "in_browser"
    device_pref = config.get("device", "auto")
    episodes = int(config.get("episodes", 5))
    load_path = config.get("load")
    save_path = config.get("save")
    fps_target = float(config.get("fps", 50.0))
    dt_target = (1.0 / fps_target) if fps_target > 0 else 0.0

    device = resolve_compute_device(device_pref, is_training=False)

    scramble_depth = config.get("scramble_depth")
    if scramble_depth is not None:
        try:
            scramble_depth = int(scramble_depth)
        except (ValueError, TypeError):
            scramble_depth = None

    if env_id.lower().replace("_", "-") in ("rubikscube-v0", "rubikscubemacro-v0", "rubiks", "rubik") and scramble_depth:
        print(f"🎲 Cubo Mágico: Profundidade de Embaralhamento na Avaliação: {scramble_depth} passos")

    env = build_hud_env(
        env_id,
        scenario=config.get("scenario"),
        render_mode=render_mode,
        is_training=False,
        frame_skip=int(config.get("frame_skip", 4)),
        scramble_depth=scramble_depth,
        config=config,
    )
    trunk_type = config.get("trunk", "classic")
    if load_path:
        detected = KnowledgeTransferManager.detect_trunk_type(load_path)
        if detected:
            trunk_type = detected

    action_chunk_size = int(config.get("action_chunk_size", config.get("act_k", 1)))
    if load_path and "action_chunk_size" not in config and "act_k" not in config:
        detected_k = KnowledgeTransferManager.detect_action_chunk_size(load_path, env.action_space)
        if detected_k > 1:
            action_chunk_size = detected_k
            print(f"🧩 Auto-detectado Action Chunk Size k={action_chunk_size} a partir de {load_path}")

    if action_chunk_size > 1:
        print(f"⚡ Action Chunking Temporal (ACT) ATIVO na Avaliação: k={action_chunk_size}")

    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type=trunk_type,
        action_chunk_size=action_chunk_size,
    ).to(device)

    eval_calib_cfg = config.get("eval_calibration", config.get("calibration"))
    eval_action_mode = config.get("eval_action_mode")
    is_continuous = isinstance(env.action_space, gym.spaces.Box)

    if str(eval_calib_cfg).lower() in ("auto", "adaptive", "homeostatic") or eval_action_mode == "auto":
        eval_calibration = "auto"
        eval_det = False
        noise_scale = 0.0
        mode_action_desc = "Auto-Calibração Homeostática (Momentum de Recompensa)"
    elif eval_calib_cfg is not None:
        eval_calibration = float(np.clip(float(eval_calib_cfg), 0.0, 1.0))
        if eval_calibration == 0.0:
            eval_det = True
            noise_scale = 0.0
            mode_action_desc = "Determinística Pura (0.0 — argmax/μ)"
        elif eval_calibration == 1.0:
            eval_det = False
            noise_scale = 1.0
            mode_action_desc = "Estocástica Total (1.0 — Boltzmann T=1.0 / 1.0σ)"
        else:
            eval_det = False
            noise_scale = eval_calibration
            mode_action_desc = f"Estocástica Calibrada ({eval_calibration:.2f} — Boltzmann T={max(0.05, eval_calibration):.2f} / {eval_calibration:.2f}σ)"
    else:
        if eval_action_mode is None:
            if config.get("stochastic", False):
                eval_action_mode = "stochastic"
            elif is_continuous:
                # Em ambientes contínuos (como Ant-v5, HalfCheetah), usa amostragem calibrada por padrão
                # para evitar deadlocks de equilíbrios estáticos em políticas ainda não totalmente convergidas
                eval_action_mode = "calibrated"
            else:
                eval_action_mode = "deterministic"

        if eval_action_mode == "deterministic":
            eval_calibration = 0.0
            eval_det = True
            noise_scale = 0.0
            mode_action_desc = "Determinística Pura (0.0 — argmax/μ)"
        elif eval_action_mode == "calibrated":
            eval_calibration = 0.25 if is_continuous else 0.5
            eval_det = False
            noise_scale = 0.25
            mode_action_desc = f"Estocástica Calibrada ({eval_calibration:.2f}σ)"
        else:  # "stochastic"
            eval_calibration = 1.0
            eval_det = False
            noise_scale = 1.0
            mode_action_desc = "Estocástica Total (1.0 — Exploração σ)"

    is_locomotion = any(
        kw in env_id.lower()
        for kw in ("ant", "halfcheetah", "cheetah", "hopper", "walker", "humanoid")
    )

    ensemble_fallback = bool(config.get("ensemble_fallback", True))
    value_rerank = bool(config.get("value_rerank", False))
    value_rerank_top_k = int(config.get("value_rerank_top_k", 2))
    value_rerank_threshold = float(config.get("value_rerank_threshold", 0.85))
    mode_desc = "Reflexo Neural S1"
    if ensemble_fallback:
        mode_desc += " + Quality Ensemble Fallback (A.2)"
    if value_rerank:
        mode_desc += f" + Value Re-ranking (Top-{value_rerank_top_k} V(s'))"
    print(f"=== [HUD Worker] Modo de Avaliação Iniciado no Ambiente: {env_id} ===")
    print(f"Dispositivo de Execução: {device.type.upper()} (Preferência: {device_pref})")
    print(f"Episódios: {episodes} | Decisão: {mode_desc} | Ação: {mode_action_desc} | FPS Alvo: {fps_target} | Renderização: {render_mode}")

    if load_path:
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        
        compat_err = KnowledgeTransferManager.validate_evaluation_compatibility(agent, state_dict, env_id, load_path)
        if compat_err:
            print(compat_err)
            sys.exit(1)

        agent.load_state_dict(state_dict, strict=False)
        print(f"Pesos carregados com sucesso de: {load_path}")
    else:
        print("[Aviso] Nenhum checkpoint especificado; utilizando pesos aleatórios inicializados.")

    agent.eval()
    explainer = GradCAMExplainer(agent)
    tracker = LiveStatsTracker()
    raw_env_obj = getattr(env, "unwrapped", env)
    max_ep_steps = getattr(env, "_max_episode_steps", None)
    if max_ep_steps is None:
        max_ep_steps = getattr(raw_env_obj, "max_steps", None)
    if max_ep_steps is None:
        max_ep_steps = 40 if "rubik" in env_id.lower() else 500
    tracker.metrics.target_steps = episodes * int(max_ep_steps)
    tracker.metrics.action_chunk_size = action_chunk_size
    tracker.metrics.value_rerank = value_rerank
    last_telemetry_ts = 0.0
    last_frame_ts = 0.0

    try:
        for ep in range(episodes):
            obs_dict, _ = env.reset()
            agent.reset_memory()
            ep_reward = 0.0
            steps = 0
            standstill_steps = 0
            done = False
            win = False
            is_rubiks = "rubik" in env_id.lower()

            while not done:
                t_step_start = time.perf_counter()

                t0 = time.perf_counter_ns()
                decision = agent.act_fast(
                    obs_dict,
                    return_decision=True,
                    deterministic=eval_det,
                    noise_scale=noise_scale,
                    calibration=eval_calibration,
                    ensemble_fallback=ensemble_fallback,
                    adaptive_chunk=bool(config.get("adaptive_chunk", False)),
                    value_rerank=value_rerank,
                    value_rerank_top_k=value_rerank_top_k,
                    value_rerank_threshold=value_rerank_threshold,
                    env=env,
                )
                lat_us = (time.perf_counter_ns() - t0) / 1000.0
                action = decision.action
                tracker.record_inference(
                    latency_us=lat_us,
                    uncertainty=decision.uncertainty,
                    confidence=decision.confidence,
                    entropy=decision.entropy,
                    calibration=decision.calibration,
                    value_rerank_triggered=getattr(decision, "value_rerank_triggered", False),
                )


                obs_dict, reward, terminated, truncated, env_info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                n_chunk_steps = len(action) if hasattr(action, "__len__") and not isinstance(action, (str, bytes)) else 1
                steps += n_chunk_steps
                tracker.record_env_step(
                    reward=reward,
                    done=done,
                    info=env_info if isinstance(env_info, dict) else {},
                    num_steps=n_chunk_steps,
                )

                if is_rubiks:
                    if terminated or (isinstance(env_info, dict) and bool(env_info.get("is_solved", False))):
                        win = True
                    else:
                        try:
                            raw = getattr(getattr(env, "unwrapped", env), "core", None)
                            if raw and hasattr(raw, "is_solved") and raw.is_solved():
                                win = True
                        except Exception:
                            pass
                else:
                    if terminated and not truncated:
                        win = bool(env_info.get("is_success", True) if isinstance(env_info, dict) and "is_success" in env_info else terminated)

                # Standstill Watchdog para ambientes de locomoção contínua (ex: Ant-v5)
                # Se o agente ficar imóvel / travado por muitos passos consecutivos, encerra o episódio
                # imediatamente em vez de ficar 950 frames congelado até o step 1000!
                if is_locomotion and not done and steps > 20:
                    speed = 1.0
                    unwrapped_env = getattr(env, "unwrapped", env)
                    if hasattr(unwrapped_env, "data") and hasattr(unwrapped_env.data, "qvel"):
                        try:
                            xy_vel = unwrapped_env.data.qvel[:2]
                            speed = float(np.linalg.norm(xy_vel))
                        except Exception:
                            speed = 1.0
                    else:
                        delta = obs_dict.get("delta_obs")
                        if delta is not None:
                            speed = float(np.linalg.norm(delta))

                    if speed < 0.02:
                        standstill_steps += 1
                        if standstill_steps >= 40:  # ~0.8s a 50 FPS sem nenhum movimento
                            print(f"[Watchdog Locomoção] Deadlock estático detectado no passo {steps} (velocidade = {speed:.4f} m/s). Reiniciando próximo episódio...")
                            done = True
                    else:
                        standstill_steps = 0

                # Renderização in-browser com taxa adaptativa e zero-overhead
                now = time.time()
                should_render_frame = False
                if render_mode == "in_browser":
                    if dt_target > 0:
                        if (now - last_frame_ts >= max(0.016, dt_target * 0.8)) and hud_client.is_ready_for_frame():
                            should_render_frame = True
                    else:
                        if (now - last_frame_ts >= 0.033) and hud_client.is_ready_for_frame():
                            should_render_frame = True

                if should_render_frame:
                    try:
                        cube_vmode = hud_client.get_cube_view_mode()
                        if hasattr(env.unwrapped, "set_view_mode"):
                            env.unwrapped.set_view_mode(cube_vmode)
                        frame = env.render()
                        if frame is not None:
                            try:
                                is_firing = bool(getattr(getattr(env, "unwrapped", env), "is_attacking", False)) or (
                                    env_id.lower().startswith("vizdoom")
                                    and isinstance(action, (int, np.integer))
                                    and int(action) == 2
                                )
                                if is_firing and isinstance(frame, np.ndarray) and frame.ndim == 3:
                                    frame = frame.copy()
                                    h, w = frame.shape[:2]
                                    cy, cx = h // 2, w // 2
                                    frame[max(0, cy-6):min(h, cy+7), max(0, cx-1):min(w, cx+2)] = [255, 30, 30]
                                    frame[max(0, cy-1):min(h, cy+2), max(0, cx-6):min(w, cx+7)] = [255, 30, 30]

                                v_mode = hud_client.get_vision_mode()
                                if v_mode != "normal" and explainer.is_supported():
                                    action_descs = getattr(getattr(env, "unwrapped", env), "action_descriptions", None)
                                    cam_obs = getattr(env, "latest_obs_dict", obs_dict)
                                    heatmap, label = explainer.compute_saliency(
                                        obs_dict=cam_obs,
                                        hx=agent.hx,
                                        mode=v_mode,
                                        action=action,
                                        action_names=action_descs,
                                    )
                                    frame = explainer.render_overlay(frame, heatmap, label=label)
                            except Exception:
                                pass

                            hud_client.send_frame(frame)
                            last_frame_ts = now
                    except Exception:
                        pass

                # Atualização periódica de telemetria a cada ~50 ms
                if now - last_telemetry_ts >= 0.05:
                    snap = tracker.snapshot()
                    hud_client.send_telemetry(snap)
                    last_telemetry_ts = now


                # Pacing da simulação para suavidade visual no HUD
                if dt_target > 0:
                    elapsed = time.perf_counter() - t_step_start
                    sleep_time = dt_target - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            if is_rubiks and not win:
                try:
                    raw = getattr(getattr(env, "unwrapped", env), "core", None)
                    if raw and hasattr(raw, "is_solved") and raw.is_solved():
                        win = True
                except Exception:
                    pass

            win_str = "\033[92mtrue\033[0m" if win else "\033[91mfalse\033[0m"
            print(f"Episódio {ep + 1}/{episodes} finalizado | Retorno: {ep_reward:.1f} | Passos: {steps} | win: {win_str}")

        if save_path:
            extra_info = {"env_id": env_id, "evaluated_episodes": episodes}
            if tracker.metrics.curriculum_depth is not None:
                extra_info["curriculum_depth"] = tracker.metrics.curriculum_depth
                extra_info["depth"] = tracker.metrics.curriculum_depth
            elif config.get("scramble_depth") is not None:
                extra_info["scramble_depth"] = int(config["scramble_depth"])
                extra_info["depth"] = int(config["scramble_depth"])
            KnowledgeTransferManager.save_checkpoint(
                agent=agent,
                checkpoint_path=save_path,
                extra_info=extra_info,
                auto_sync_web=config.get("auto_sync_web", True),
            )
            print(f"Checkpoint salvo com sucesso em: {save_path}")

        tracker.set_completed(True)
        hud_client.send_telemetry(tracker.snapshot())
        print("=== [HUD Worker] Avaliação Concluída com Sucesso! ===")

    finally:
        explainer.close()
        env.close()


def run_worker_benchmark(config: Dict[str, Any], hud_client: HUDClient) -> None:
    steps = int(config.get("steps", 5000))
    device_pref = config.get("device", "cpu")
    device = resolve_compute_device(device_pref, is_training=False)
    print(f"=== [HUD Worker] Iniciando Benchmark de Latência {device.type.upper()} ({steps} passos) ===")
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    ).to(device)
    agent.eval()
    obs_dict, _ = env.reset()
    agent.reset_memory()

    # Warmup
    for _ in range(50):
        agent.act_fast(obs_dict)

    latencies = []
    tracker = LiveStatsTracker()
    tracker.metrics.target_steps = steps

    t_start = time.time()
    for s in range(steps):
        t0 = time.perf_counter_ns()
        agent.act_fast(obs_dict)
        t1 = time.perf_counter_ns()
        lat_us = (t1 - t0) / 1000.0
        latencies.append(lat_us / 1000.0)  # ms
        tracker.record_inference(latency_us=lat_us, uncertainty=0.01, confidence=0.99, entropy=0.0)

        if s % 500 == 0:
            now = time.time()
            snap = tracker.snapshot()
            hud_client.send_telemetry(snap)

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    p99_ms = float(np.percentile(latencies, 99))

    print(f"Resultado do Benchmark ({steps} passos):")
    print(f"  Latência Média: {avg_ms:.4f} ms ({avg_ms * 1000:.1f} µs)")
    print(f"  P95: {p95_ms:.4f} ms | P99: {p99_ms:.4f} ms")
    status_str = "APROVADO (<= 0.80 ms)" if avg_ms <= 0.8 else "REPROVADO (> 0.80 ms)"
    print(f"  Conformidade com o Orçamento: {status_str}")

    tracker.set_completed(True)
    hud_client.send_telemetry(tracker.snapshot())
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="HUD Execution Worker")
    parser.add_argument("--config", type=str, required=True, help="JSON configuration payload")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8050", help="HUD server URL")
    args = parser.parse_args()

    config = json.loads(args.config)
    mode = config.get("mode", "train")

    hud_client = HUDClient(args.server_url, initial_vision_mode=config.get("vision_mode", "normal"))

    try:
        if mode == "train":
            run_worker_train(config, hud_client)
        elif mode == "run":
            run_worker_eval(config, hud_client)
        elif mode == "benchmark":
            run_worker_benchmark(config, hud_client)
        else:
            print(f"[Erro] Modo desconhecido: '{mode}'")
            sys.exit(1)
    finally:
        hud_client.close()


if __name__ == "__main__":
    main()
