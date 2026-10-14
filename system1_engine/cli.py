import argparse
import sys
import time
from typing import Optional
import gymnasium as gym
import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters import make_game_env
from system1_engine.env.dependencies import make_gym_env_with_auto_install
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry import LiveStatsTracker, S1LiveDashboard, TelemetryServer
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def build_environment(
    env_id: str,
    scenario: Optional[str] = None,
    render: bool = False,
    is_training: bool = False,
    frame_skip: Optional[int] = None,
    scramble_depth: Optional[int] = None,
) -> UniversalS1Wrapper:
    """Instancia o ambiente apropriado (Gymnasium padrão ou Adaptador Nativo ViZDoom) no Modo Puro Universal."""
    env_lower = env_id.lower().replace("_", "-")
    if env_lower in ("rubiks", "rubik", "rubiks-macro", "rubiksmacro", "rubik-macro"):
        env_id = "RubiksCubeMacro-v0"
    elif env_lower in ("rubiks-atomic", "rubik-atomic", "rubiksatomic", "rubikatomic"):
        env_id = "RubiksCube-v0"

    if env_id.lower() in ["vizdoom", "native"]:
        scenario_path = scenario or "basic.cfg"
        fs = frame_skip if frame_skip is not None else 4
        env = make_game_env(
            "native",
            {
                "engine_type": "vizdoom",
                "scenario_path": scenario_path,
                "args": {"headless": not render, "frame_skip": fs},
            },
        )
        return env
    render_mode = "human" if render else None
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
        else:
            extra_kwargs["curriculum"] = False
            if scramble_depth is not None and int(scramble_depth) > 0:
                extra_kwargs["scramble_depth"] = int(scramble_depth)

    if is_training and env_id in ("RubiksCube-v0", "RubiksCubeMacro-v0"):
        from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv
        is_macro = (env_id == "RubiksCubeMacro-v0")
        raw_env = VectorizedRubiksEnv(
            num_envs=64,
            is_macro=is_macro,
            render_mode=render_mode,
            **extra_kwargs,
        )
    else:
        raw_env = make_gym_env_with_auto_install(env_id, render_mode=render_mode, **extra_kwargs)
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
    """Resolve o dispositivo de computação (CPU, MPS ou CUDA)."""
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
        print("[Warning] Requested GPU device not found (no CUDA or MPS available). Falling back to CPU.")
        return torch.device("cpu")

    # pref == "auto"
    if is_training:
        if has_cuda:
            return torch.device("cuda")
        if has_mps:
            return torch.device("mps")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def train_mode(args: argparse.Namespace) -> None:
    device = resolve_compute_device(getattr(args, "device", "auto"), is_training=True)
    print(f"=== Starting Training Mode on {args.env} [Device: {device.type.upper()} (preference: {getattr(args, 'device', 'auto')})] ===")
    env = build_environment(
        args.env,
        scenario=args.scenario,
        render=getattr(args, "render", False),
        is_training=True,
        frame_skip=getattr(args, "frame_skip", None),
        scramble_depth=getattr(args, "scramble_depth", None),
    )

    trunk_type = getattr(args, "trunk", "classic")
    checkpoint_to_load = getattr(args, "load", None) or getattr(args, "transfer_from", None)
    if checkpoint_to_load and getattr(args, "trunk", "classic") == "classic":
        detected = KnowledgeTransferManager.detect_trunk_type(checkpoint_to_load)
        if detected == "lean":
            trunk_type = "lean"

    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type=trunk_type,
    )

    checkpoint_to_load = getattr(args, "load", None) or getattr(args, "transfer_from", None)
    if checkpoint_to_load:
        res = KnowledgeTransferManager.load_for_training(
            agent=agent,
            checkpoint_path=checkpoint_to_load,
            freeze_trunk=args.freeze_trunk,
            force_trunk_only=getattr(args, "force_trunk_only", False),
        )
        print(f"Treino iniciado a partir de checkpoint ({res['mode']}): {res['total_params']} tensores ativos. Tronco congelado: {res['trunk_frozen']}")

    need_tracker = args.live_stats or args.web_panel
    tracker = LiveStatsTracker() if need_tracker else None
    dashboard = S1LiveDashboard(tracker) if args.live_stats else None
    should_open_browser = args.web_panel and not getattr(args, "no_browser", False) and sys.stdin.isatty()
    web_server = (
        TelemetryServer(tracker, host=args.host, port=args.port, open_browser=should_open_browser)
        if args.web_panel and tracker
        else None
    )

    if web_server:
        web_server.start()

    try:
        train_calib_raw = getattr(args, "train_calibration", "auto")
        if str(train_calib_raw).lower() in ("auto", "adaptive", "homeostatic"):
            train_calib = "auto"
        else:
            try:
                train_calib = float(np.clip(float(train_calib_raw), 0.05, 1.0))
            except (ValueError, TypeError):
                train_calib = "auto"

        trainer = RecurrentPPOTrainer(
            agent=agent,
            env=env,
            learning_rate=args.lr,
            rollout_steps=args.rollout_steps,
            chunk_length=args.chunk_length,
            chunk_batch_size=args.chunk_batch_size,
            entropy_coef=args.entropy_coef,
            tracker=tracker,
            device=device,
            exploration_scale=train_calib,
            hindsight_replay=getattr(args, "hindsight_replay", True),
        )

        last_saved_step = [0]
        best_saved_return = [-float("inf")]

        def train_callback(steps: int, mean_ret: float) -> None:
            if dashboard:
                dashboard.update()
            if args.save and (steps - last_saved_step[0] >= 4000 or (mean_ret > best_saved_return[0] and steps - last_saved_step[0] >= 1000)):
                last_saved_step[0] = steps
                best_saved_return[0] = max(best_saved_return[0], mean_ret)
                best_ret = getattr(trainer, "best_mean_return", mean_ret)
                successful_d = tracker.get_successful_depth() if tracker else None
                extra_info = {
                    "env_id": args.env,
                    "final_return": float(mean_ret),
                    "best_mean_return": float(best_ret),
                    "steps": steps,
                }
                if tracker and tracker.metrics.curriculum_depth is not None:
                    extra_info["curriculum_depth"] = tracker.metrics.curriculum_depth
                    extra_info["successful_depth"] = successful_d if successful_d is not None else tracker.metrics.curriculum_depth
                    extra_info["depth"] = extra_info["successful_depth"]
                if tracker and tracker.metrics.curriculum_max_depth is not None:
                    extra_info["curriculum_max_depth"] = tracker.metrics.curriculum_max_depth
                KnowledgeTransferManager.save_checkpoint(
                    agent=agent,
                    checkpoint_path=args.save,
                    extra_info=extra_info,
                )

        if dashboard:
            with dashboard:
                final_return = trainer.train(
                    max_steps=args.steps,
                    target_return=args.target_return,
                    callback=train_callback,
                    verbose=False,
                )
            dashboard.render_once()
        else:
            final_return = trainer.train(
                max_steps=args.steps,
                target_return=args.target_return,
                callback=train_callback,
                verbose=True,
            )

        if args.save:
            if getattr(trainer, "best_state_dict", None) is not None:
                agent.load_state_dict(trainer.best_state_dict)
            best_ret = getattr(trainer, "best_mean_return", None)
            if best_ret is None and tracker and tracker.metrics.best_mean_return is not None:
                best_ret = tracker.metrics.best_mean_return
            if best_ret is None:
                best_ret = final_return

            successful_d = tracker.get_successful_depth() if tracker else None

            extra_info = {
                "env_id": args.env,
                "final_return": float(final_return),
                "best_mean_return": float(best_ret),
                "steps": trainer.total_steps,
            }
            if tracker and tracker.metrics.curriculum_depth is not None:
                extra_info["curriculum_depth"] = tracker.metrics.curriculum_depth
                extra_info["successful_depth"] = successful_d if successful_d is not None else tracker.metrics.curriculum_depth
                extra_info["depth"] = extra_info["successful_depth"]
            if tracker and tracker.metrics.curriculum_max_depth is not None:
                extra_info["curriculum_max_depth"] = tracker.metrics.curriculum_max_depth
            if tracker and getattr(tracker.metrics, "best_return_depth", None) is not None:
                extra_info["best_return_depth"] = tracker.metrics.best_return_depth

            KnowledgeTransferManager.save_checkpoint(
                agent=agent,
                checkpoint_path=args.save,
                extra_info=extra_info,
            )
            print(f"Checkpoint successfully saved to: {args.save}")

        if tracker:
            tracker.set_completed(True)

        if web_server:
            dashboard_url = f"http://{args.host}:{args.port}"
            print("\n" + "=" * 70)
            print("🏁 Treinamento concluído com sucesso!")
            print(f"🌐 Painel Web mantido ativo para inspeção em: {dashboard_url}")
            print("⌨️  Pressione Ctrl+C no terminal para encerrar o servidor...")
            print("=" * 70)
            if not getattr(args, "no_wait", False) and sys.stdin.isatty():
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    print("\nEncerrando servidor web...")

    finally:
        if web_server:
            web_server.stop()
        env.close()


def run_mode(args: argparse.Namespace) -> None:
    device = resolve_compute_device(getattr(args, "device", "auto"), is_training=False)
    print(f"=== Running Agent Evaluation on {args.env} [Device: {device.type.upper()}] ===")
    env = build_environment(
        args.env,
        scenario=args.scenario,
        render=getattr(args, "render", False),
        is_training=False,
        frame_skip=getattr(args, "frame_skip", None),
        scramble_depth=getattr(args, "scramble_depth", None),
    )

    trunk_type = getattr(args, "trunk", "classic")
    if getattr(args, "load", None):
        detected = KnowledgeTransferManager.detect_trunk_type(args.load)
        if detected:
            trunk_type = detected

    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        trunk_type=trunk_type,
    ).to(device)

    if args.load:
        checkpoint = torch.load(args.load, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        
        compat_err = KnowledgeTransferManager.validate_evaluation_compatibility(agent, state_dict, args.env, args.load)
        if compat_err:
            print(compat_err)
            sys.exit(1)

        agent.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {args.load}")

    agent.eval()
    need_tracker = args.live_stats or args.web_panel
    tracker = LiveStatsTracker() if need_tracker else None
    dashboard = S1LiveDashboard(tracker) if args.live_stats else None
    should_open_browser = args.web_panel and not getattr(args, "no_browser", False) and sys.stdin.isatty()
    web_server = (
        TelemetryServer(tracker, host=args.host, port=args.port, open_browser=should_open_browser)
        if args.web_panel and tracker
        else None
    )

    if web_server:
        web_server.start()

    target_fps = args.fps
    if target_fps is None:
        target_fps = 50.0 if (args.web_panel or getattr(args, "render", False)) else 0.0

    dt_target = (1.0 / target_fps) if target_fps > 0 else 0.0

    try:
        eval_calib = getattr(args, "calibration", None)
        def execute_eval_loop() -> None:
            for ep in range(args.episodes):
                obs_dict, _ = env.reset()
                agent.reset_memory()
                ep_reward = 0.0
                steps = 0
                done = False

                while not done:
                    t_step_start = time.perf_counter()
                    if tracker is not None:
                        t0 = time.perf_counter_ns()
                        decision = agent.act_with_confidence(obs_dict, calibration=eval_calib)
                        lat_us = (time.perf_counter_ns() - t0) / 1000.0

                        tracker.record_inference(
                            latency_us=lat_us,
                            uncertainty=decision.uncertainty,
                            confidence=decision.confidence,
                            entropy=decision.entropy,
                        )
                        action = decision.action
                    else:
                        action = agent.act_fast(obs_dict, calibration=eval_calib)

                    obs_dict, reward, terminated, truncated, env_info = env.step(action)
                    done = terminated or truncated

                    if tracker is not None:
                        tracker.record_env_step(
                            reward=reward,
                            done=done,
                            info=env_info if isinstance(env_info, dict) else {},
                        )
                        if dashboard:
                            dashboard.update()

                    ep_reward += reward
                    steps += 1

                    if getattr(args, "render", False) and not args.live_stats:
                        action_labels = {0: "ESQUERDA ⬅️", 1: "DIREITA ➡️", 2: "DISPARO 💥"}
                        act_str = action_labels.get(action, f"AÇÃO {action}")
                        print(f"  [Passo {steps:2d}] {act_str:<15} -> Recompensa: {reward:+.1f}")

                    if dt_target > 0:
                        elapsed = time.perf_counter() - t_step_start
                        sleep_time = dt_target - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                if not args.live_stats:
                    is_rubiks = "rubik" in getattr(args, "env", "").lower()
                    is_victory = bool(terminated) or (isinstance(env_info, dict) and bool(env_info.get("is_solved", False)))
                    if not is_victory and is_rubiks:
                        try:
                            raw = getattr(getattr(env, "unwrapped", env), "core", None)
                            if raw and hasattr(raw, "is_solved") and raw.is_solved():
                                is_victory = True
                        except Exception:
                            pass

                    if is_rubiks:
                        if is_victory:
                            print(f"🏆 [VITÓRIA] Cubo Mágico 3x3 RESOLVIDO com sucesso em {steps} passos! (Retorno: {ep_reward:.2f})")
                            print(f"Episode {ep + 1}/{args.episodes} | Status: 🏆 VITÓRIA (RESOLVIDO) | Return: {ep_reward:.1f} | Steps: {steps}")
                        else:
                            print(f"⏱️ [NÃO RESOLVIDO] Limite de passos atingido ({steps} passos) sem resolução. (Retorno: {ep_reward:.1f})")
                            print(f"Episode {ep + 1}/{args.episodes} | Status: ❌ NÃO RESOLVIDO (Timeout) | Return: {ep_reward:.1f} | Steps: {steps}")
                    else:
                        status_str = " | Status: 🏆 CONCLUÍDO" if terminated else ""
                        print(f"Episode {ep + 1}/{args.episodes}{status_str} | Return: {ep_reward:.1f} | Steps: {steps}")

        if dashboard:
            with dashboard:
                execute_eval_loop()
            dashboard.render_once()
        else:
            execute_eval_loop()

        if tracker:
            tracker.set_completed(True)

        if web_server:
            dashboard_url = f"http://{args.host}:{args.port}"
            print("\n" + "=" * 70)
            print(f"🏁 Avaliação concluída com sucesso! ({args.episodes} episódios)")
            print(f"🌐 Painel Web mantido ativo para inspeção em: {dashboard_url}")
            print("⌨️  Pressione Ctrl+C no terminal para encerrar o servidor...")
            print("=" * 70)
            if not getattr(args, "no_wait", False) and sys.stdin.isatty():
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    print("\nEncerrando servidor web...")

    finally:
        if web_server:
            web_server.stop()
        env.close()


def benchmark_mode(args: argparse.Namespace) -> None:
    device = resolve_compute_device(getattr(args, "device", "cpu"), is_training=False)
    print(f"=== Running act_fast() {device.type.upper()} Latency Benchmark ===")
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
        trunk_type=getattr(args, "trunk", "classic"),
    ).to(device)
    agent.eval()

    obs_dict, _ = env.reset()
    agent.reset_memory()

    # Warmup
    for _ in range(50):
        agent.act_fast(obs_dict)

    latencies = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        agent.act_fast(obs_dict)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    avg_latency = float(np.mean(latencies))
    med_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))

    print(f"Benchmark over {args.steps} sequential steps ({device.type.upper()}):")
    print(f"  Average Latency : {avg_latency:.4f} ms (Budget: <= 0.8000 ms)")
    print(f"  Median Latency  : {med_latency:.4f} ms")
    print(f"  P95 Latency     : {p95_latency:.4f} ms")
    print(f"  P99 Latency     : {p99_latency:.4f} ms")
    print(f"  Min / Max       : {min_latency:.4f} ms / {max_latency:.4f} ms")

    if avg_latency <= 0.8:
        print("[PASS] Latency budget satisfied (<= 0.8 ms).")
    else:
        print(f"[FAIL] Latency budget exceeded: {avg_latency:.4f} ms > 0.8 ms")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal System 1 RL Agent Engine")
    parser.add_argument(
        "--mode",
        choices=["train", "run", "benchmark"],
        default="benchmark",
        help="Operating mode: train, run, or benchmark",
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment ID or 'vizdoom'")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario file for vizdoom/native environment")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu", "mps", "cuda"],
        help="Compute device for neural operations: auto (GPU for training, CPU for inference), gpu, cpu, mps, or cuda (default: auto)",
    )
    parser.add_argument("--steps", type=int, default=40000, help="Training steps or benchmark steps")
    parser.add_argument("--save", type=str, default=None, help="Path to save checkpoint (default: None)")
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to load checkpoint for run/benchmark mode or for continual training / fine-tuning in train mode",
    )
    parser.add_argument(
        "--transfer-from",
        type=str,
        default=None,
        help="Path to load transferable trunk weights for cross-scenario transfer or training warm-start",
    )
    parser.add_argument(
        "--force-trunk-only",
        "--trunk-only",
        action="store_true",
        default=False,
        dest="force_trunk_only",
        help="Force loading only trunk parameters during training even if heads are compatible (default: False)",
    )
    parser.add_argument(
        "--freeze-trunk",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze trunk parameters during transfer or warm-start (default: True for cross-domain, False for warm-start)",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.003, help="Entropy coefficient for PPO exploration (default: 0.003)")
    parser.add_argument(
        "--hindsight-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Habilita Hindsight Experience Replay (HER) On-Policy para quebra de platôs de recompensa esparsa (padrão: True)",
    )
    parser.add_argument(
        "--calibration",
        "--eval-calibration",
        type=float,
        default=None,
        dest="calibration",
        help="Calibração contínua da amostragem na avaliação [0.0 = determinístico puro, 1.0 = estocástico] (padrão: None)",
    )
    parser.add_argument(
        "--train-calibration",
        "--exploration-scale",
        type=str,
        default="auto",
        dest="train_calibration",
        help="Escala de exploração / calibração durante o treino PPO: 'auto' (homeostase dinâmica e auto-correção) ou float [0.05 a 1.0] (padrão: auto)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes for run mode")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument(
        "--target-return",
        type=float,
        default=475.0,
        help="Target moving average return to stop training early",
    )
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Rollout steps per PPO iteration (default: 2048)")
    parser.add_argument("--chunk-length", type=int, default=32, help="Chunk sequence length for BPTT (default: 32)")
    parser.add_argument("--chunk-batch-size", type=int, default=16, help="Chunk batch size (default: 16)")
    parser.add_argument(
        "--live-stats",
        action="store_true",
        default=False,
        help="Enable real-time Rich terminal live telemetry dashboard",
    )
    parser.add_argument(
        "--web-panel",
        action="store_true",
        default=False,
        help="Launch real-time web telemetry dashboard (HTTP/SSE on port 8050)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="HTTP port for web telemetry server (default: 8050)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for web telemetry server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Cadência em FPS para o modo run (padrão: 50.0 quando --web-panel está ativo, ou 0 para velocidade máxima desimpedida)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        default=False,
        help="Não aguarda confirmação com Ctrl+C ao término da execução com --web-panel",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Não abre o navegador automaticamente ao iniciar com --web-panel",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Habilita visualização gráfica do ambiente em tempo real na tela (render_mode='human')",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Frame skip para ambientes de motor nativo (padrão: 1 ao renderizar, 4 em headless)",
    )
    parser.add_argument(
        "--scramble-depth",
        type=int,
        default=None,
        help="Número de passos/profundidade de embaralhamento do Cubo Mágico (sem limite aberto, ex: 1, 3, 10, 20...)",
    )
    parser.add_argument(
        "--trunk",
        type=str,
        default="classic",
        choices=["classic", "lean"],
        help="Tipo de tronco neural recorrente: 'classic' (GRU 256 + 2x ResMLP 256, ~830k) ou 'lean' (LeanResidualTrunk 512 Multi-Timescale, ~1.6M)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "run":
        run_mode(args)
    elif args.mode == "benchmark":
        benchmark_mode(args)


if __name__ == "__main__":
    main()
