import glob
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.parse
from typing import Any, Dict, List, Optional

from system1_engine.hud.frame_buffer import VideoFrameBuffer
from system1_engine.hud.runner import HUDProcessRunner
from system1_engine.transfer.manager import KnowledgeTransferManager


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Servidor TCP multi-thread com liberação rápida de portas e supressão de erros de desconexão."""

    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        exc_type, _, _ = sys.exc_info()
        if exc_type in (
            ConnectionResetError,
            BrokenPipeError,
            ConnectionAbortedError,
            TimeoutError,
        ):
            return
        super().handle_error(request, client_address)


class HUDServer:
    """Servidor Web do HUD Operacional do System 1 Engine."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8050,
        open_browser: bool = True,
        auth_token: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.auth_token = auth_token or os.environ.get("S1_AUTH_TOKEN")
        self.runner = HUDProcessRunner()
        self.frame_buffer = VideoFrameBuffer()
        self._lock = threading.Lock()
        self.latest_telemetry: Dict[str, Any] = {
            "fps": 0.0,
            "total_steps": 0,
            "mean_return_20": 0.0,
            "best_return": 0.0,
            "best_mean_return": 0.0,
            "latency_p50_us": 0.0,
            "latency_p99_us": 0.0,
            "mean_confidence": 0.0,
            "mean_uncertainty": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "grad_norms": {"FrontEnd": 0.0, "Trunk": 0.0, "PolicyHead": 0.0},
        }
        self.vision_mode: str = "normal"
        self.cube_view_mode: str = "3d"

        self.server: Optional[ThreadedTCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # Caminho do template HTML do dashboard
        self.html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")

    def get_dashboard_html(self) -> str:
        """Lê o template HTML do disco dinamicamente para permitir recarregamento imediato (Cmd+R)."""
        if os.path.exists(self.html_path):
            try:
                with open(self.html_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return "<h1>Dashboard template not found</h1>"

    def scan_checkpoints(self) -> List[str]:
        """Varre o diretório do projeto em busca de modelos e checkpoints .pt disponíveis."""
        candidates = glob.glob("*.pt") + glob.glob("**/*.pt", recursive=True)
        valid = set()
        for p in candidates:
            # Ignora pastas virtuais, git e caches
            if any(part.startswith(".") or part == ".venv" or part == "build" for part in p.split(os.sep)):
                continue
            valid.add(p)
        return sorted(list(valid))

    def scan_checkpoints_detailed(self) -> List[Dict[str, Any]]:
        """Retorna detalhes enriquecidos (tamanho em MB, data, ambiente de origem e tensores) de cada checkpoint."""
        candidates = self.scan_checkpoints()
        results = []
        for path in candidates:
            try:
                stat = os.stat(path)
                size_mb = stat.st_size / (1024 * 1024)
                mtime_str = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(stat.st_mtime))
                meta = KnowledgeTransferManager.inspect_checkpoint(path)
                results.append({
                    "path": path,
                    "filename": os.path.basename(path),
                    "size_mb": round(size_mb, 2),
                    "mtime": mtime_str,
                    "env_id": meta.get("env_id"),
                    "obs_dim": meta.get("obs_dim"),
                    "act_dim": meta.get("act_dim"),
                    "final_return": meta.get("final_return"),
                    "best_mean_return": meta.get("best_mean_return"),
                    "steps": meta.get("steps"),
                    "curriculum_depth": meta.get("curriculum_depth"),
                    "curriculum_max_depth": meta.get("curriculum_max_depth"),
                    "successful_depth": meta.get("successful_depth"),
                    "best_return_depth": meta.get("best_return_depth"),
                    "depth": meta.get("depth"),
                    "trunk_type": meta.get("trunk_type", "classic"),
                })
            except Exception:
                results.append({"path": path, "filename": os.path.basename(path), "size_mb": 0.0, "mtime": "—"})
        return results

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Detecta aceleradores de hardware disponíveis (Apple Silicon MPS ou NVIDIA CUDA)."""
        mps_available = False
        cuda_available = False
        device_name = "CPU Universal"
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if cuda_available:
                name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
                device_name = f"NVIDIA CUDA ({name})"
            elif mps_available:
                device_name = "Apple Silicon GPU (MPS)"
            else:
                device_name = "CPU Universal"
        except Exception:
            pass
        return {
            "mps_available": mps_available,
            "cuda_available": cuda_available,
            "gpu_available": mps_available or cuda_available,
            "device_name": device_name,
        }

    @staticmethod
    def get_gym_environments() -> List[Dict[str, Any]]:
        """Retorna a lista estruturada e categorizada de ambientes do Gymnasium disponíveis com documentação."""
        try:
            import gymnasium as gym
        except Exception:
            return []

        doc_urls = {
            "cartpole": "https://gymnasium.farama.org/environments/classic_control/cart_pole/",
            "mountaincar": "https://gymnasium.farama.org/environments/classic_control/mountain_car/",
            "mountaincarcontinuous": "https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/",
            "pendulum": "https://gymnasium.farama.org/environments/classic_control/pendulum/",
            "acrobot": "https://gymnasium.farama.org/environments/classic_control/acrobot/",
            "lunarlander": "https://gymnasium.farama.org/environments/box2d/lunar_lander/",
            "lunarlandercontinuous": "https://gymnasium.farama.org/environments/box2d/lunar_lander/",
            "bipedalwalker": "https://gymnasium.farama.org/environments/box2d/bipedal_walker/",
            "bipedalwalkerhardcore": "https://gymnasium.farama.org/environments/box2d/bipedal_walker/",
            "carracing": "https://gymnasium.farama.org/environments/box2d/car_racing/",
            "blackjack": "https://gymnasium.farama.org/environments/toy_text/blackjack/",
            "frozenlake": "https://gymnasium.farama.org/environments/toy_text/frozen_lake/",
            "cliffwalking": "https://gymnasium.farama.org/environments/toy_text/cliff_walking/",
            "taxi": "https://gymnasium.farama.org/environments/toy_text/taxi/",
            "ant": "https://gymnasium.farama.org/environments/mujoco/ant/",
            "halfcheetah": "https://gymnasium.farama.org/environments/mujoco/half_cheetah/",
            "hopper": "https://gymnasium.farama.org/environments/mujoco/hopper/",
            "humanoid": "https://gymnasium.farama.org/environments/mujoco/humanoid/",
            "humanoidstandup": "https://gymnasium.farama.org/environments/mujoco/humanoidstandup/",
            "invertedpendulum": "https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/",
            "inverteddoublependulum": "https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/",
            "reacher": "https://gymnasium.farama.org/environments/mujoco/reacher/",
            "swimmer": "https://gymnasium.farama.org/environments/mujoco/swimmer/",
            "walker2d": "https://gymnasium.farama.org/environments/mujoco/walker2d/",
            "pusher": "https://gymnasium.farama.org/environments/mujoco/pusher/",
        }

        defaults = {
            "CartPole-v1": {"target_return": 475.0, "entropy_coef": 0.005, "steps": 40000},
            "Acrobot-v1": {"target_return": -100.0, "entropy_coef": 0.01, "steps": 50000},
            "Pendulum-v1": {"target_return": -200.0, "entropy_coef": 0.005, "steps": 60000},
            "MountainCar-v0": {"target_return": -110.0, "entropy_coef": 0.03, "steps": 80000},
            "LunarLander-v3": {"target_return": 200.0, "entropy_coef": 0.01, "steps": 100000},
            "LunarLander-v2": {"target_return": 200.0, "entropy_coef": 0.01, "steps": 100000},
            "BipedalWalker-v3": {"target_return": 300.0, "entropy_coef": 0.01, "steps": 200000},
            "FrozenLake-v1": {"target_return": 0.9, "entropy_coef": 0.02, "steps": 30000},
            "Ant-v5": {"target_return": 4000.0, "entropy_coef": 0.005, "steps": 300000},
            "Ant-v4": {"target_return": 4000.0, "entropy_coef": 0.005, "steps": 300000},
            "HalfCheetah-v5": {"target_return": 4000.0, "entropy_coef": 0.005, "steps": 300000},
            "HalfCheetah-v4": {"target_return": 4000.0, "entropy_coef": 0.005, "steps": 300000},
            "Hopper-v5": {"target_return": 3000.0, "entropy_coef": 0.005, "steps": 300000},
            "Hopper-v4": {"target_return": 3000.0, "entropy_coef": 0.005, "steps": 300000},
            "Humanoid-v5": {"target_return": 5000.0, "entropy_coef": 0.005, "steps": 500000},
            "Walker2d-v5": {"target_return": 4000.0, "entropy_coef": 0.005, "steps": 300000},
            "RubiksCubeMacro-v0": {"target_return": 12.0, "entropy_coef": 0.01, "steps": 30000},
            "RubiksCube-v0": {"target_return": 12.0, "entropy_coef": 0.003, "steps": 40000, "chunk_length": 32, "rollout_steps": 2048, "trunk": "lean"},
        }

        # Garante registro dos ambientes customizados
        try:
            import system1_engine.env  # noqa: F401
        except Exception:
            pass

        results = []
        registered_keys = sorted(list(gym.envs.registry.keys()))
        for env_id in registered_keys:
            if env_id.startswith(("GymV21", "GymV26", "phys2d", "tabular")):
                continue

            entry = gym.envs.registry[env_id]
            entry_point = str(entry.entry_point) if entry.entry_point else ""

            if "rubiks" in entry_point.lower() or "rubik" in env_id.lower():
                category = "Cubo Mágico"
                type_tag = "Macro / Combinatório" if "macro" in env_id.lower() else "Atômico / 3D"
            elif "classic_control" in entry_point:
                category = "Classic Control"
                type_tag = "Física Clássica"
            elif "box2d" in entry_point:
                category = "Box2D"
                type_tag = "Física 2D / Dinâmica"
            elif "toy_text" in entry_point:
                category = "Toy Text"
                type_tag = "Discreto / Tabular"
            elif "mujoco" in entry_point:
                category = "MuJoCo"
                type_tag = "Robótica Contínua 3D"
            elif "atari" in entry_point or "ale" in entry_point.lower():
                category = "Atari (ALE)"
                type_tag = "Visual 2D / Retro"
            else:
                category = "Outros"
                type_tag = "Gymnasium"

            clean_id = env_id.split("-")[0].lower()
            doc_link = doc_urls.get(clean_id, "https://gymnasium.farama.org/environments/")
            preset = defaults.get(env_id, {})

            results.append({
                "id": env_id,
                "category": category,
                "type_tag": type_tag,
                "doc_url": doc_link,
                "target_return": preset.get("target_return"),
                "entropy_coef": preset.get("entropy_coef"),
                "steps": preset.get("steps"),
            })

        return results

    def update_telemetry(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self.latest_telemetry.update(data)

    def reset_telemetry(self) -> None:
        """Reseta todos os valores de telemetria acumulados para zero."""
        with self._lock:
            self.latest_telemetry = {
                "fps": 0.0,
                "total_steps": 0,
                "mean_return_20": 0.0,
                "best_return": 0.0,
                "best_mean_return": 0.0,
                "latency_p50_us": 0.0,
                "latency_p99_us": 0.0,
                "mean_confidence": 0.0,
                "mean_uncertainty": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "grad_norms": {"FrontEnd": 0.0, "Trunk": 0.0, "PolicyHead": 0.0},
            }

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.latest_telemetry)

    @staticmethod
    def get_local_ip() -> Optional[str]:
        """Detecta o endereço IP na rede local (Wi-Fi ou Ethernet)."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return None

    @staticmethod
    def get_tailscale_ip() -> Optional[str]:
        """Detecta o endereço IP IPv4 da interface Tailscale caso ativa."""
        import shutil
        import subprocess
        ts_bin = shutil.which("tailscale")
        if ts_bin:
            try:
                res = subprocess.run([ts_bin, "ip", "-4"], capture_output=True, text=True, timeout=1.0)
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout.strip().splitlines()[0].strip()
            except Exception:
                pass
        return None

    def start(self) -> None:
        if self._running:
            return

        server_instance = self
        runner = self.runner
        frame_buffer = self.frame_buffer
        stop_event = self._stop_event

        class HUDRequestHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle(self) -> None:
                try:
                    super().handle()
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError, OSError):
                    pass

            def _is_authorized(self) -> bool:
                if not server_instance.auth_token:
                    return True
                auth_hdr = self.headers.get("Authorization", "")
                if auth_hdr.startswith("Bearer "):
                    token = auth_hdr.split("Bearer ", 1)[1].strip()
                    if token == server_instance.auth_token:
                        return True
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                if "token" in params and params["token"][0] == server_instance.auth_token:
                    return True
                return False

            def _send_unauthorized(self) -> None:
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                err_msg = b'{"error": "Unauthorized: invalid or missing authentication token"}'
                self.send_header("Content-Length", str(len(err_msg)))
                self.send_header("WWW-Authenticate", 'Bearer realm="System1HUD"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(err_msg)

            def do_OPTIONS(self) -> None:
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_HEAD(self) -> None:
                self.do_GET()

            def do_GET(self) -> None:
                parsed_url = urllib.parse.urlparse(self.path)
                clean_path = parsed_url.path

                if not self._is_authorized():
                    self._send_unauthorized()
                    return

                if clean_path in ("/", "/index.html"):
                    encoded_html = server_instance.get_dashboard_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(encoded_html)))
                    self.end_headers()
                    self.wfile.write(encoded_html)

                elif clean_path == "/video_feed":
                    # MJPEG Video Streaming Endpoint de Alta Eficiência
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    try:
                        last_count = -1
                        while not stop_event.is_set():
                            frame, count = frame_buffer.wait_for_frame_change(last_count, timeout=1.0)
                            header = (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(frame)).encode("ascii") + b"\r\n\r\n"
                            )
                            self.wfile.write(header + frame + b"\r\n")
                            self.wfile.flush()
                            last_count = count
                            time.sleep(0.01)  # Respiro anti-flooding
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass

                elif clean_path in ("/api/frame", "/api/snapshot"):
                    # Frame Snapshot Endpoint (captura instantânea de frame único)
                    frame = frame_buffer.get_frame()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.send_header("X-Frame-Count", str(frame_buffer.get_frame_count()))
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(frame)

                elif clean_path == "/stream":
                    # SSE Real-time Telemetry & Log Stream
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    client_log_cursor = 0
                    try:
                        while not stop_event.is_set():
                            new_logs, client_log_cursor = runner.get_logs(client_log_cursor)
                            snap = server_instance.get_telemetry_snapshot()
                            runner_state = runner.get_state()

                            payload = {
                                "telemetry": snap,
                                "runner": runner_state,
                                "new_logs": new_logs,
                                "frame_count": frame_buffer.get_frame_count(),
                                "cube_view_mode": server_instance.cube_view_mode,
                            }

                            data_line = f"data: {json.dumps(payload)}\n\n"
                            self.wfile.write(data_line.encode("utf-8"))
                            self.wfile.flush()
                            time.sleep(0.08)  # ~12 Hz
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass


                elif clean_path == "/api/state":
                    resp = {
                        "runner": runner.get_state(),
                        "hardware": server_instance.get_hardware_info(),
                        "checkpoints": server_instance.scan_checkpoints(),
                        "checkpoints_detailed": server_instance.scan_checkpoints_detailed(),
                        "telemetry": server_instance.get_telemetry_snapshot(),
                        "vision_mode": server_instance.vision_mode,
                        "cube_view_mode": server_instance.cube_view_mode,
                    }
                    encoded_resp = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded_resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(encoded_resp)

                elif clean_path == "/api/checkpoints":
                    resp = {
                        "checkpoints": server_instance.scan_checkpoints(),
                        "checkpoints_detailed": server_instance.scan_checkpoints_detailed(),
                    }
                    encoded_resp = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded_resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(encoded_resp)

                elif clean_path == "/api/gym_environments":
                    resp = {
                        "environments": server_instance.get_gym_environments(),
                        "docs_url": "https://gymnasium.farama.org/environments/",
                    }
                    encoded_resp = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded_resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(encoded_resp)

                elif clean_path in ("/web", "/web/", "/web/index.html"):
                    # Web HUD (ONNX WebAssembly Standalone Evaluation)
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    web_html_path = os.path.join(project_root, "web", "index.html")
                    if os.path.exists(web_html_path):
                        with open(web_html_path, "rb") as f:
                            web_bytes = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(web_bytes)))
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(web_bytes)
                    else:
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()

                elif clean_path in ("/web/system1-eval.js", "/system1-eval.js"):
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    js_path = os.path.join(project_root, "web", "system1-eval.js")
                    if os.path.exists(js_path):
                        with open(js_path, "rb") as f:
                            js_bytes = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/javascript; charset=utf-8")
                        self.send_header("Content-Length", str(len(js_bytes)))
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(js_bytes)
                    else:
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()

                elif clean_path.startswith("/models/") or clean_path.startswith("/web/models/"):
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    model_file = os.path.basename(clean_path)
                    target_file = os.path.join(project_root, "web", "models", model_file)
                    if os.path.exists(target_file) and os.path.isfile(target_file):
                        with open(target_file, "rb") as f:
                            file_bytes = f.read()
                        content_type = "application/json; charset=utf-8" if model_file.endswith(".json") else "application/octet-stream"
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", str(len(file_bytes)))
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(file_bytes)
                    else:
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()

                elif clean_path.startswith("/pkg/") or clean_path.startswith("/web/pkg/"):
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    pkg_file = os.path.basename(clean_path)
                    target_file = os.path.join(project_root, "web", "pkg", pkg_file)
                    if os.path.exists(target_file) and os.path.isfile(target_file):
                        with open(target_file, "rb") as f:
                            file_bytes = f.read()
                        if pkg_file.endswith(".wasm"):
                            content_type = "application/wasm"
                        elif pkg_file.endswith(".js"):
                            content_type = "application/javascript; charset=utf-8"
                        elif pkg_file.endswith(".json"):
                            content_type = "application/json; charset=utf-8"
                        else:
                            content_type = "text/plain; charset=utf-8"
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", str(len(file_bytes)))
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(file_bytes)
                    else:
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()


                elif clean_path == "/api/web_models":
                    try:
                        from system1_engine.core.onnx_exporter import generate_web_manifest
                        manifest = generate_web_manifest()
                    except Exception as err:
                        manifest = {"error": str(err), "models": {}}
                    encoded_resp = json.dumps(manifest).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded_resp)))
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(encoded_resp)

                elif clean_path in ("/manifest.json", "/manifest.webmanifest"):
                    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
                    if os.path.exists(manifest_path):
                        with open(manifest_path, "rb") as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/manifest+json; charset=utf-8")
                        self.send_header("Content-Length", str(len(content)))
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(content)
                    else:
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()

                elif clean_path == "/sw.js":
                    sw_path = os.path.join(os.path.dirname(__file__), "sw.js")
                    if os.path.exists(sw_path):
                        with open(sw_path, "rb") as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/javascript; charset=utf-8")
                        self.send_header("Content-Length", str(len(content)))
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(content)
                    else:
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()

                elif clean_path in ("/icon-192.png", "/icon-512.png", "/favicon.ico"):
                    filename = clean_path.lstrip("/")
                    icon_path = os.path.join(os.path.dirname(__file__), filename)
                    if os.path.exists(icon_path):
                        with open(icon_path, "rb") as f:
                            content = f.read()
                        mime = "image/x-icon" if filename.endswith(".ico") else "image/png"
                        self.send_response(200)
                        self.send_header("Content-Type", mime)
                        self.send_header("Content-Length", str(len(content)))
                        self.send_header("Cache-Control", "public, max-age=86400")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(content)
                    else:
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()

                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            def do_POST(self) -> None:
                parsed_url = urllib.parse.urlparse(self.path)
                clean_path = parsed_url.path

                # Proteção de endpoints de IPC interno: somente localhost
                if clean_path in ("/api/internal/telemetry", "/api/internal/frame"):
                    client_ip = self.client_address[0]
                    if client_ip not in ("127.0.0.1", "::1", "localhost", "testclient"):
                        self.send_response(403)
                        self.send_header("Content-Type", "application/json")
                        err_bytes = b'{"error": "Forbidden: internal IPC endpoint accessible only from localhost"}'
                        self.send_header("Content-Length", str(len(err_bytes)))
                        self.end_headers()
                        self.wfile.write(err_bytes)
                        return
                elif not self._is_authorized():
                    self._send_unauthorized()
                    return

                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)

                if clean_path == "/api/action":
                    try:
                        data = json.loads(body.decode("utf-8"))
                        action = data.get("action")
                        if action == "start":
                            cfg = data.get("config", {})
                            if "vision_mode" in cfg:
                                server_instance.vision_mode = cfg["vision_mode"]
                            if "cube_view_mode" in cfg:
                                server_instance.cube_view_mode = cfg["cube_view_mode"]
                            mode_label = cfg.get("mode", "run").upper()
                            env_name = cfg.get("env", "Ambiente")
                            frame_buffer.reset_placeholder("⚡ SYSTEM 1 ENGINE", f"INICIANDO {env_name} [{mode_label}]...")
                            url = f"http://{server_instance.host}:{server_instance.port}"
                            success, msg = runner.start(cfg, url)
                            resp = {"success": success, "message": msg}

                        elif action == "stop":
                            success, msg = runner.stop()
                            frame_buffer.reset_placeholder("⚡ SYSTEM 1 ENGINE", "TAREFA INTERROMPIDA PELO USUÁRIO")
                            resp = {"success": success, "message": msg}
                        elif action in ("clear", "reset"):
                            runner.clear_logs()
                            server_instance.reset_telemetry()
                            frame_buffer.reset_placeholder("⚡ SYSTEM 1 ENGINE", "CENTRAL PRONTA / AGUARDANDO TREINO")
                            resp = {"success": True, "message": "Estado, logs e telemetria resetados com sucesso."}
                        elif action == "set_vision_mode":
                            mode = data.get("mode", "normal")
                            server_instance.vision_mode = mode
                            resp = {"success": True, "vision_mode": mode}
                        elif action == "set_cube_view_mode":
                            mode = data.get("mode", "3d")
                            server_instance.cube_view_mode = mode
                            resp = {"success": True, "cube_view_mode": mode}
                        else:
                            resp = {"success": False, "message": f"Ação desconhecida: {action}"}
                    except Exception as err:
                        resp = {"success": False, "message": str(err)}

                    encoded_action = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded_action)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(encoded_action)

                elif clean_path == "/api/internal/telemetry":
                    try:
                        data = json.loads(body.decode("utf-8"))
                        server_instance.update_telemetry(data)
                        resp_body = json.dumps({
                            "vision_mode": server_instance.vision_mode,
                            "cube_view_mode": server_instance.cube_view_mode,
                        }).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(resp_body)))
                        self.end_headers()
                        self.wfile.write(resp_body)
                    except Exception:
                        self.send_response(400)
                        self.send_header("Content-Length", "0")
                        self.end_headers()

                elif clean_path == "/api/internal/frame":
                    frame_buffer.update_frame(body)
                    resp_body = json.dumps({
                        "vision_mode": server_instance.vision_mode,
                        "cube_view_mode": server_instance.cube_view_mode,
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)

                elif clean_path == "/api/web_models/sync":
                    try:
                        from system1_engine.core.onnx_exporter import sync_all_web_models, generate_web_manifest
                        sync_all_web_models()
                        manifest = generate_web_manifest()
                        resp = {"success": True, "message": "Modelos web sincronizados com sucesso", "manifest": manifest}
                    except Exception as err:
                        resp = {"success": False, "message": f"Erro na sincronização: {err}"}
                    encoded_resp = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded_resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(encoded_resp)

                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            def log_message(self, format: str, *args) -> None:
                sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]} - {format % args}\n")
                sys.stderr.flush()

        self._stop_event.clear()
        self.server = ThreadedTCPServer((self.host, self.port), HUDRequestHandler)
        self._running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        local_host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        dashboard_url = f"http://{local_host}:{self.port}"
        if self.auth_token:
            dashboard_url += f"/?token={self.auth_token}"

        print("\n" + "=" * 70)
        print("🎮 SYSTEM 1 ENGINE — VISUAL CONTROL HUD INICIALIZADO")
        print(f"🌐 Local (este Mac):      {dashboard_url}")

        local_wifi_ip = self.get_local_ip()
        if local_wifi_ip and (self.host in ("0.0.0.0", "::") or self.host == local_wifi_ip):
            wifi_url = f"http://{local_wifi_ip}:{self.port}"
            if self.auth_token:
                wifi_url += f"/?token={self.auth_token}"
            print(f"🏠 Rede Wi-Fi (Tablet/Cel): {wifi_url}")

        ts_ip = self.get_tailscale_ip()
        if ts_ip and (self.host in ("0.0.0.0", "::") or self.host == ts_ip):
            ts_url = f"http://{ts_ip}:{self.port}"
            if self.auth_token:
                ts_url += f"/?token={self.auth_token}"
            print(f"🔒 Remoto (Tailscale VPN):  {ts_url}")
        print("=" * 70)

        if self.open_browser:
            def _open() -> None:
                time.sleep(0.2)
                try:
                    import webbrowser
                    webbrowser.open(dashboard_url)
                except Exception:
                    pass

            threading.Thread(target=_open, daemon=True).start()

    def stop(self) -> None:
        if self._running:
            self._stop_event.set()
            self._running = False
            self.runner.stop()
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
                self.server = None

    def __enter__(self) -> "HUDServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
