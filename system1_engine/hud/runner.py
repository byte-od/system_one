import collections
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class HUDProcessRunner:
    """Gerenciador de subprocesso para treinos e avaliações disparados pelo HUD ou detectados externamente."""

    def __init__(self, max_log_history: int = 500) -> None:
        self.max_log_history = max_log_history
        self._lock = threading.RLock()
        self.process: Optional[subprocess.Popen] = None
        self.external_pid: Optional[int] = None
        self.external_log_path: Optional[str] = None
        self.status: str = "IDLE"  # IDLE, RUNNING, STOPPED, COMPLETED, ERROR
        self.current_config: Dict[str, Any] = {}
        self.logs: collections.deque[str] = collections.deque(maxlen=max_log_history)
        self.log_counter = 0
        self._reader_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_requested = False

    def is_running(self) -> bool:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                return True
            if self.external_pid is not None:
                try:
                    os.kill(self.external_pid, 0)
                    return True
                except OSError:
                    return False
            return False

    def attach_external(
        self,
        pid: int,
        log_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Conecta o runner a um processo de treino já em execução externa."""
        with self._lock:
            try:
                os.kill(pid, 0)
            except OSError:
                return False, f"Processo PID {pid} não encontrado ou já finalizado."

            self._stop_requested = False
            self.external_pid = pid
            self.external_log_path = log_path
            self.current_config = config or {
                "mode": "train",
                "env": "RubiksCube-v0",
                "trunk": "lru",
                "steps": 100000000,
                "scar_memory": "hybrid_v2",
            }
            self.status = "RUNNING"
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] 🔗 Detectado e conectado ao treino ativo em segundo plano (PID {pid})...")
            self.log_counter += 1

        if log_path and os.path.exists(log_path):
            self._reader_thread = threading.Thread(target=self._tail_external_log, args=(log_path,), daemon=True)
            self._reader_thread.start()

        self._monitor_thread = threading.Thread(target=self._monitor_external_pid, args=(pid,), daemon=True)
        self._monitor_thread.start()

        return True, f"Treino externo (PID {pid}) acoplado com sucesso ao HUD."

    @staticmethod
    def _is_valid_training_log(log_path: Optional[str]) -> bool:
        """Verifica se um caminho de log corresponde de fato a uma sessão de treino do System 1."""
        if not log_path or not os.path.exists(log_path):
            return False
        try:
            if os.path.getsize(log_path) < 100:
                return False
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(8192)
                # Rejeita logs do próprio servidor HUD ou de outros comandos
                if "VISUAL CONTROL HUD INICIALIZADO" in head:
                    return False
                return any(k in head for k in (
                    "run_hybrid_v2", "Universal S1 Agent", "marathon", "Maratona",
                    "INICIANDO MARATONA", "RecurrentPPOTrainer", "[Step "
                ))
        except Exception:
            return False

    def auto_detect_external_training(self) -> bool:
        """Verifica se há algum script de treino do System 1 rodando no sistema e conecta automaticamente."""
        with self._lock:
            # Se já está conectado ao PID e possui log válido, mantém conexão ativa
            if self.is_running() and self.external_log_path and self._is_valid_training_log(self.external_log_path):
                return True

        try:
            res = subprocess.run(["ps", "-eo", "pid,command"], capture_output=True, text=True, timeout=2.0)
            if res.returncode != 0:
                return False

            target_pid = None
            target_cmd = ""
            for line in res.stdout.splitlines():
                if "python" in line and ("run_hybrid_v2" in line or "run_three_mechanisms" in line or "marathon" in line or "scratch/run_" in line):
                    parts = line.strip().split(None, 1)
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[0])
                            if pid != os.getpid():
                                target_pid = pid
                                target_cmd = parts[1]
                                break
                        except ValueError:
                            continue

            if target_pid is None:
                return False

            # Localiza o arquivo de log mais recente que seja autenticamente do treino (e não do HUD)
            log_candidates = glob.glob(os.path.expanduser("~/.gemini/antigravity/brain/*/.system_generated/tasks/task-*.log"))
            sorted_candidates = sorted(log_candidates, key=os.path.getmtime, reverse=True)
            latest_log = None
            for cand in sorted_candidates:
                if self._is_valid_training_log(cand):
                    latest_log = cand
                    break

            cfg = {
                "mode": "train",
                "env": "RubiksCube-v0",
                "trunk": "lru",
                "steps": 100000000,
                "scar_memory": "hybrid_v2",
                "cmd": target_cmd,
            }

            with self._lock:
                if self.external_pid == target_pid:
                    # Já associado ao PID, atualiza ponteiro de log se estava incorreto
                    if latest_log and self.external_log_path != latest_log:
                        self.external_log_path = latest_log
                        self._reader_thread = threading.Thread(target=self._tail_external_log, args=(latest_log,), daemon=True)
                        self._reader_thread.start()
                    return True

            success, _ = self.attach_external(target_pid, latest_log, cfg)
            return success
        except Exception:
            return False

    def _tail_external_log(self, log_path: str) -> None:
        """Lê o arquivo de log do processo externo de forma contínua."""
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                # Carrega as últimas linhas iniciais
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 32768))
                lines = f.readlines()
                with self._lock:
                    for l in lines[-100:]:
                        clean = l.rstrip()
                        if clean:
                            self.logs.append(clean)
                            self.log_counter += 1

                while not self._stop_requested and self.is_running():
                    line = f.readline()
                    if line:
                        clean = line.rstrip()
                        if clean:
                            with self._lock:
                                self.logs.append(clean)
                                self.log_counter += 1
                    else:
                        time.sleep(0.2)
        except Exception:
            pass

    def _monitor_external_pid(self, pid: int) -> None:
        """Monitora a vida do processo externo até sua finalização."""
        while not self._stop_requested:
            try:
                os.kill(pid, 0)
                time.sleep(1.0)
            except OSError:
                break

        with self._lock:
            if self._stop_requested:
                self.status = "STOPPED"
            else:
                self.status = "COMPLETED"
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] 🏁 Treino externo finalizado com sucesso.")
                self.log_counter += 1
            self.external_pid = None

    def start(self, config: Dict[str, Any], server_url: str) -> Tuple[bool, str]:
        """Inicia um novo processo filho de execução em segundo plano."""
        with self._lock:
            if self.is_running():
                return False, "Já existe uma execução em andamento. Interrompa-a antes de iniciar outra."

            self._stop_requested = False
            self.external_pid = None
            self.external_log_path = None
            self.current_config = config
            self.status = "RUNNING"
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] ▶️ Iniciando tarefa ({config.get('mode', 'train')})...")
            self.log_counter += 1

            cmd = [
                sys.executable,
                "-u",  # Unbuffered stdout/stderr
                "-m",
                "system1_engine.hud.worker",
                "--config",
                json.dumps(config),
                "--server-url",
                server_url,
            ]

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy(),
                )
            except Exception as e:
                self.status = "ERROR"
                err_msg = f"Falha ao iniciar subprocesso: {e}"
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] ❌ {err_msg}")
                self.log_counter += 1
                return False, err_msg

        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

        self._monitor_thread = threading.Thread(target=self._monitor_process, daemon=True)
        self._monitor_thread.start()

        return True, "Execução iniciada com sucesso."

    def stop(self) -> Tuple[bool, str]:
        """Interrompe a execução atual com encerramento gracioso via SIGINT / SIGTERM."""
        with self._lock:
            if not self.is_running():
                self.status = "IDLE"
                return True, "Nenhuma execução ativa para interromper."

            self._stop_requested = True
            proc = self.process
            ext_pid = self.external_pid

        if proc is not None:
            # Tenta encerrar graciosamente via SIGINT (Ctrl+C simulado)
            try:
                proc.send_signal(signal.SIGINT)
            except OSError:
                pass

            # Aguarda até 2.0s pelo término limpo
            t0 = time.time()
            while time.time() - t0 < 2.0:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)

            # Força SIGTERM caso ainda esteja ativo
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
                time.sleep(0.2)

            # Último recurso: SIGKILL
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass

        elif ext_pid is not None:
            try:
                os.kill(ext_pid, signal.SIGINT)
            except OSError:
                pass

        with self._lock:
            self.status = "STOPPED"
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] ⏹️ Execução interrompida pelo usuário.")
            self.log_counter += 1

        return True, "Execução interrompida."

    def _read_stdout(self) -> None:
        """Lê linhas de stdout do processo filho e alimenta o histórico de logs."""
        proc = self.process
        if proc is None or proc.stdout is None:
            return

        for line in iter(proc.stdout.readline, ""):
            line_clean = line.rstrip()
            if line_clean:
                with self._lock:
                    self.logs.append(line_clean)
                    self.log_counter += 1
        proc.stdout.close()

    def _monitor_process(self) -> None:
        """Monitora o encerramento do processo e atualiza o estado correspondente."""
        proc = self.process
        if proc is None:
            return

        exit_code = proc.wait()

        with self._lock:
            if self._stop_requested:
                self.status = "STOPPED"
            elif exit_code == 0:
                self.status = "COMPLETED"
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] 🏁 Tarefa concluída com sucesso (código 0).")
                self.log_counter += 1
            else:
                self.status = "ERROR"
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠️ Processo finalizado com erro (código {exit_code}).")
                self.log_counter += 1

    def clear_logs(self) -> None:
        """Limpa o buffer histórico de logs acumulados."""
        with self._lock:
            self.logs.clear()
            self.log_counter = 0

    def get_logs(self, since_index: int = 0) -> Tuple[List[str], int]:
        """Retorna as linhas de log acumuladas desde um índice específico."""
        with self._lock:
            all_logs = list(self.logs)
            total = self.log_counter
            start_offset = max(0, total - len(all_logs))
            slice_start = max(0, since_index - start_offset)
            return all_logs[slice_start:], total

    def get_state(self) -> Dict[str, Any]:
        """Retorna o estado operacional consolidado do executor."""
        with self._lock:
            active_pid = self.process.pid if self.process is not None else self.external_pid
            return {
                "status": self.status,
                "is_running": self.is_running(),
                "pid": active_pid,
                "config": self.current_config,
                "total_logs": self.log_counter,
            }
