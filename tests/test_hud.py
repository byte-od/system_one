import http.client
import json
import time
import numpy as np
import pytest

from system1_engine.hud.frame_buffer import VideoFrameBuffer
from system1_engine.hud.runner import HUDProcessRunner
from system1_engine.hud.server import HUDServer


def test_video_frame_buffer_placeholder_and_update():
    buf = VideoFrameBuffer(width=320, height=240)
    frame0 = buf.get_frame()
    assert isinstance(frame0, bytes)
    assert len(frame0) > 100
    # Check JPEG header
    assert frame0[:2] == b"\xff\xd8"

    dummy_jpeg = b"\xff\xd8\xff\xe0testjpeg"
    buf.update_frame(dummy_jpeg)
    assert buf.get_frame() == dummy_jpeg

    # Test wait_for_frame_change
    cur_frame, count = buf.wait_for_frame_change(last_count=-1, timeout=0.1)
    assert cur_frame == dummy_jpeg
    assert count > 0

    buf.reset_placeholder("TEST TITLE", "TEST SUBTITLE")
    frame1 = buf.get_frame()
    assert frame1[:2] == b"\xff\xd8"


def test_hud_server_lifecycle_and_endpoints():
    server = HUDServer(host="127.0.0.1", port=8995, open_browser=False)
    with server:
        time.sleep(0.1)
        conn = http.client.HTTPConnection("127.0.0.1", 8995, timeout=2.0)

        # 1. Test GET /
        conn.request("GET", "/")
        res = conn.getresponse()
        assert res.status == 200
        html = res.read().decode("utf-8")
        assert "SYSTEM 1 ENGINE" in html
        assert "Visual Control HUD" in html
        assert "openGymCatalogModal()" in html
        assert "https://gymnasium.farama.org/environments/" in html
        assert "btn-mini-mode" in html
        assert "openMiniFloatingWindow()" in html
        assert "mini-mode-toolbar" in html
        assert "train-status-cards-container" in html

        # 2. Test GET /api/state
        conn.request("GET", "/api/state")
        res = conn.getresponse()
        assert res.status == 200
        state = json.loads(res.read().decode("utf-8"))
        assert "runner" in state
        assert "hardware" in state
        assert "device_name" in state["hardware"]
        assert "gpu_available" in state["hardware"]
        assert "checkpoints" in state
        assert "checkpoints_detailed" in state
        assert "telemetry" in state
        assert state["runner"]["status"] == "IDLE"

        # 2.1 Test GET /api/checkpoints
        conn.request("GET", "/api/checkpoints")
        res = conn.getresponse()
        assert res.status == 200
        ckpt_resp = json.loads(res.read().decode("utf-8"))
        assert "checkpoints" in ckpt_resp
        assert "checkpoints_detailed" in ckpt_resp
        assert isinstance(ckpt_resp["checkpoints_detailed"], list)

        # 2.2 Test GET /api/gym_environments
        conn.request("GET", "/api/gym_environments")
        res = conn.getresponse()
        assert res.status == 200
        gym_resp = json.loads(res.read().decode("utf-8"))
        assert "environments" in gym_resp
        assert "docs_url" in gym_resp
        assert len(gym_resp["environments"]) > 30
        assert any(e["id"] == "CartPole-v1" for e in gym_resp["environments"])
        assert any("doc_url" in e and "category" in e for e in gym_resp["environments"])

        # 3. Test POST /api/internal/telemetry
        telemetry_payload = json.dumps({"fps": 60.5, "total_steps": 1234}).encode("utf-8")
        conn.request(
            "POST",
            "/api/internal/telemetry",
            telemetry_payload,
            {"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        res.read()
        snap = server.get_telemetry_snapshot()
        assert snap["fps"] == 60.5
        assert snap["total_steps"] == 1234

        # 4. Test POST /api/internal/frame
        dummy_frame = b"\xff\xd8\xff\xe0dummydata"
        conn.request(
            "POST",
            "/api/internal/frame",
            dummy_frame,
            {"Content-Type": "image/jpeg"},
        )
        res = conn.getresponse()
        assert res.status == 200
        res.read()
        assert server.frame_buffer.get_frame() == dummy_frame

        # 4.1 Test GET /api/frame snapshot endpoint
        conn.request("GET", "/api/frame")
        res = conn.getresponse()
        assert res.status == 200
        assert res.getheader("Content-Type") == "image/jpeg"
        assert res.read() == dummy_frame

        # 4.2 Test GET /api/web_models
        conn.request("GET", "/api/web_models")
        res = conn.getresponse()
        assert res.status == 200
        web_models_resp = json.loads(res.read().decode("utf-8"))
        assert "models" in web_models_resp
        assert "rubiks_atomic" in web_models_resp["models"]

        # 4.3 Test GET /web and /web/system1-eval.js
        conn.request("GET", "/web")
        res = conn.getresponse()
        assert res.status == 200
        web_html = res.read().decode("utf-8")
        assert "System 1 Engine" in web_html
        assert "LIVE EVALUATION" in web_html

        conn.request("GET", "/web/system1-eval.js")
        res = conn.getresponse()
        assert res.status == 200
        res.read()

        # 4.4 Test POST /api/web_models/sync
        conn.request("POST", "/api/web_models/sync")
        res = conn.getresponse()
        assert res.status == 200
        sync_resp = json.loads(res.read().decode("utf-8"))
        assert sync_resp["success"] is True

        # 5. Test POST /api/action stop
        action_payload = json.dumps({"action": "stop"}).encode("utf-8")
        conn.request("POST", "/api/action", action_payload, {"Content-Type": "application/json"})
        res = conn.getresponse()
        assert res.status == 200
        action_resp = json.loads(res.read().decode("utf-8"))
        assert action_resp["success"] is True

        # 6. Test POST /api/action clear/reset
        clear_payload = json.dumps({"action": "clear"}).encode("utf-8")
        conn.request("POST", "/api/action", clear_payload, {"Content-Type": "application/json"})
        res = conn.getresponse()
        assert res.status == 200
        clear_resp = json.loads(res.read().decode("utf-8"))
        assert clear_resp["success"] is True

        conn.close()


def test_hud_process_runner_start_and_stop():
    runner = HUDProcessRunner()
    assert not runner.is_running()
    assert runner.get_state()["status"] == "IDLE"

    # Start a benchmark task with few steps
    config = {
        "mode": "benchmark",
        "steps": 100,
    }
    success, msg = runner.start(config, "http://127.0.0.1:8995")
    assert success is True
    assert runner.is_running()

    # Wait briefly and verify logs appear
    time.sleep(0.4)
    logs, count = runner.get_logs(0)
    assert count > 0

    # Stop runner
    stop_ok, stop_msg = runner.stop()
    assert stop_ok is True
    assert not runner.is_running()
    assert runner.get_state()["status"] in ["STOPPED", "COMPLETED"]

    # Clear logs
    runner.clear_logs()
    cleared_logs, cleared_count = runner.get_logs(0)
    assert len(cleared_logs) == 0
    assert cleared_count == 0


def test_hud_e2e_evaluation_and_in_browser_rendering(tmp_path):
    save_ckpt = str(tmp_path / "test_saved_agent_trained.pt")
    server = HUDServer(host="127.0.0.1", port=8996, open_browser=False)
    with server:
        config = {
            "mode": "run",
            "env": "CartPole-v1",
            "episodes": 1,
            "fps": 0.0,
            "render_mode": "in_browser",
            "inference_mode": "confidence",
            "save": save_ckpt,
        }
        success, msg = server.runner.start(config, "http://127.0.0.1:8996")
        assert success is True

        t0 = time.time()
        while server.runner.is_running() and (time.time() - t0 < 8.0):
            time.sleep(0.1)

        state = server.runner.get_state()
        assert state["status"] == "COMPLETED"
        assert not server.runner.is_running()

        # Verifica se frames de vídeo foram transmitidos para o buffer MJPEG
        current_frame = server.frame_buffer.get_frame()
        assert current_frame[:2] == b"\xff\xd8"
        assert current_frame != server.frame_buffer._default_frame

        # Verifica se telemetria foi atualizada
        snap = server.get_telemetry_snapshot()
        assert snap["total_steps"] > 0

        # Verifica logs gerados e mensagem de salvamento
        logs, count = server.runner.get_logs(0)
        assert count > 0
        assert any("Episódio 1/1 finalizado" in line for line in logs)
        assert any("Checkpoint salvo com sucesso em:" in line for line in logs)

        # Verifica se o arquivo de checkpoint realmente foi salvo no disco
        import os
        assert os.path.exists(save_ckpt)
        assert os.path.getsize(save_ckpt) > 1000


def test_resolve_compute_device():
    import torch
    from system1_engine.hud.worker import resolve_compute_device

    # CPU mode should always return cpu
    dev_cpu = resolve_compute_device("cpu", is_training=True)
    assert dev_cpu.type == "cpu"

    # Evaluation in auto mode should return cpu (lowest single-step latency)
    dev_eval_auto = resolve_compute_device("auto", is_training=False)
    assert dev_eval_auto.type == "cpu"

    # Auto mode during training should select GPU (mps or cuda) if available
    has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    dev_train_auto = resolve_compute_device("auto", is_training=True)
    if has_gpu:
        assert dev_train_auto.type in ("mps", "cuda")
    else:
        assert dev_train_auto.type == "cpu"


def test_hud_device_selection_in_training():
    server = HUDServer(host="127.0.0.1", port=8997, open_browser=False)
    with server:
        config = {
            "mode": "train",
            "env": "CartPole-v1",
            "device": "cpu",
            "steps": 64,
            "rollout_steps": 32,
            "chunk_length": 8,
            "chunk_batch_size": 4,
            "render_mode": "none",
        }
        success, msg = server.runner.start(config, "http://127.0.0.1:8997")
        assert success is True

        t0 = time.time()
        while server.runner.is_running() and (time.time() - t0 < 8.0):
            time.sleep(0.1)

        state = server.runner.get_state()
        assert state["status"] == "COMPLETED"

        logs, count = server.runner.get_logs(0)
        assert count > 0
        assert any("Dispositivo de Execução: CPU" in line for line in logs)


def test_hud_device_selection_gpu_training():
    import torch
    has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if not has_gpu:
        pytest.skip("No GPU accelerator (MPS or CUDA) available on this machine.")

    server = HUDServer(host="127.0.0.1", port=8998, open_browser=False)
    with server:
        config = {
            "mode": "train",
            "env": "CartPole-v1",
            "device": "gpu",
            "steps": 64,
            "rollout_steps": 32,
            "chunk_length": 8,
            "chunk_batch_size": 4,
            "render_mode": "none",
        }
        success, msg = server.runner.start(config, "http://127.0.0.1:8998")
        assert success is True

        t0 = time.time()
        while server.runner.is_running() and (time.time() - t0 < 8.0):
            time.sleep(0.1)

        state = server.runner.get_state()
        assert state["status"] == "COMPLETED"

        logs, count = server.runner.get_logs(0)
        assert count > 0
        expected_dev = "CUDA" if torch.cuda.is_available() else "MPS"
        assert any(f"Dispositivo de Execução: {expected_dev}" in line for line in logs)


def test_hud_carracing_visual_training():
    """Verify that HUD worker trains visual continuous environment CarRacing-v3 without dimension error."""
    server = HUDServer(host="127.0.0.1", port=8999, open_browser=False)
    with server:
        config = {
            "mode": "train",
            "env": "CarRacing-v3",
            "device": "cpu",
            "steps": 32,
            "rollout_steps": 16,
            "chunk_length": 8,
            "chunk_batch_size": 2,
            "render_mode": "none",
        }
        success, msg = server.runner.start(config, "http://127.0.0.1:8999")
        assert success is True

        t0 = time.time()
        while server.runner.is_running() and (time.time() - t0 < 15.0):
            time.sleep(0.2)

        state = server.runner.get_state()
        assert state["status"] == "COMPLETED"

        logs, count = server.runner.get_logs(0)
        assert count > 0
        assert not any("RuntimeError" in line for line in logs)
        assert any("CarRacing-v3" in line for line in logs)


def test_hud_auth_token_protection():
    """Verify that when auth_token is set, unauthenticated requests return 401."""
    import urllib.request
    import urllib.error

    server = HUDServer(host="127.0.0.1", port=9001, open_browser=False, auth_token="supersecret123")
    with server:
        # 1. Sem token -> 401 Unauthorized
        try:
            urllib.request.urlopen("http://127.0.0.1:9001/api/state")
            assert False, "Deveria ter lançado HTTPError 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401

        # 2. Token incorreto no cabeçalho -> 401 Unauthorized
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:9001/api/state",
                headers={"Authorization": "Bearer wrong_token"},
            )
            urllib.request.urlopen(req)
            assert False, "Deveria ter lançado HTTPError 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401

        # 3. Token correto no cabeçalho Bearer -> 200 OK
        req_valid = urllib.request.Request(
            "http://127.0.0.1:9001/api/state",
            headers={"Authorization": "Bearer supersecret123"},
        )
        with urllib.request.urlopen(req_valid) as resp:
            assert resp.status == 200

        # 4. Token correto via query parameter ?token= -> 200 OK
        with urllib.request.urlopen("http://127.0.0.1:9001/api/state?token=supersecret123") as resp:
            assert resp.status == 200


def test_hud_carracing_in_browser_evaluation_streaming():
    """Verify that evaluating CarRacing-v3 (continuous action space) in in_browser mode streams frames to the buffer."""
    server = HUDServer(host="127.0.0.1", port=9002, open_browser=False)
    with server:
        config = {
            "mode": "run",
            "env": "CarRacing-v3",
            "episodes": 1,
            "fps": 0.0,
            "render_mode": "in_browser",
            "inference_mode": "confidence",
        }
        success, msg = server.runner.start(config, "http://127.0.0.1:9002")
        assert success is True

        # Wait until at least 1 frame arrives in the buffer (should be within 3-4s)
        t0 = time.time()
        frame_received = False
        while time.time() - t0 < 10.0:
            cur_frame = server.frame_buffer.get_frame()
            if cur_frame != server.frame_buffer._default_frame and cur_frame[:2] == b"\xff\xd8":
                frame_received = True
                break
            time.sleep(0.1)

        # Stop runner cleanly
        server.runner.stop()

        assert frame_received, "Expected live game frame from CarRacing-v3 to arrive in HUD frame buffer"


def test_hud_frozenlake_training():
    """Verify that HUD worker trains discrete observation environment FrozenLake-v1 without AssertionErrors."""
    server = HUDServer(host="127.0.0.1", port=9003, open_browser=False)
    with server:
        config = {
            "mode": "train",
            "env": "FrozenLake-v1",
            "device": "cpu",
            "steps": 32,
            "rollout_steps": 16,
            "chunk_length": 8,
            "chunk_batch_size": 2,
            "render_mode": "none",
        }
        success, msg = server.runner.start(config, "http://127.0.0.1:9003")
        assert success is True

        t0 = time.time()
        while server.runner.is_running() and (time.time() - t0 < 10.0):
            time.sleep(0.1)

        state = server.runner.get_state()
        assert state["status"] == "COMPLETED"

        logs, count = server.runner.get_logs(0)
        assert count > 0
        assert not any("AssertionError" in line for line in logs)
        assert any("FrozenLake-v1" in line for line in logs)


def test_hud_ant_evaluation_with_eval_action_mode():
    """Verify that evaluating Ant-v5 with eval_action_mode='calibrated' runs without errors."""
    server = HUDServer(host="127.0.0.1", port=9004, open_browser=False)
    with server:
        config = {
            "mode": "run",
            "env": "Ant-v5",
            "episodes": 1,
            "fps": 0.0,
            "render_mode": "none",
            "eval_action_mode": "calibrated",
            "inference_mode": "fast",
        }
        success, msg = server.runner.start(config, "http://127.0.0.1:9004")
        assert success is True

        t0 = time.time()
        while server.runner.is_running() and (time.time() - t0 < 8.0):
            time.sleep(0.1)

        # Stop cleanly if still running
        server.runner.stop()

        logs, count = server.runner.get_logs(0)
        assert count > 0
        assert any("Ant-v5" in line for line in logs)
        assert any("Estocástica Calibrada" in line for line in logs)


def test_tracker_curriculum_depth_min_max_reset():
    """Garante que a mudança de dimensão/profundidade curricular reseta min/max de retorno e win rate."""
    from system1_engine.telemetry.tracker import LiveStatsTracker

    tracker = LiveStatsTracker()

    # D1: modelo atinge 100% de vitórias
    tracker.record_env_step(
        reward=10.0,
        done=True,
        info={"current_depth": 1, "curriculum_success_rate": 1.0, "curriculum_promotions": 0},
    )
    snap1 = tracker.snapshot()
    assert snap1["curriculum_depth"] == 1
    assert snap1["curriculum_success_rate"] == 1.0
    assert snap1["curriculum_depth_best_success_rate"] == 1.0

    # Promoção para D2: novos patamares devem resetar registros anteriores
    tracker.record_env_step(
        reward=1.0,
        done=False,
        info={"current_depth": 2, "curriculum_promotions": 1},
    )
    snap2 = tracker.snapshot()
    assert snap2["curriculum_depth"] == 2
    assert snap2["curriculum_promotions"] == 1
    assert snap2["curriculum_success_rate"] == 0.0
    assert snap2["curriculum_depth_best_success_rate"] is None
    assert snap2["curriculum_depth_min_success_rate"] is None
    assert snap2["curriculum_depth_best_return"] is None

    # Novos passos em D2: registram apenas a performance de D2
    tracker.record_env_step(
        reward=2.5,
        done=True,
        info={"current_depth": 2, "curriculum_success_rate": 0.35, "curriculum_promotions": 1},
    )
    snap3 = tracker.snapshot()
    assert snap3["curriculum_depth"] == 2
    assert snap3["curriculum_success_rate"] == 0.35
    assert snap3["curriculum_depth_best_success_rate"] == 0.35
    assert snap3["curriculum_depth_min_success_rate"] == 0.35

    # Promoção para D3 onde o pacote de info ainda contém a taxa residual de 1.0 da vitória de D2:
    # Não deve contaminar a dimensão nova D3 com o teto de 100% da anterior
    tracker.record_env_step(
        reward=10.0,
        done=True,
        info={"current_depth": 3, "curriculum_promotions": 2, "curriculum_success_rate": 1.0},
    )
    snap4 = tracker.snapshot()
    assert snap4["curriculum_depth"] == 3
    assert snap4["curriculum_promotions"] == 2
    assert snap4["curriculum_success_rate"] == 0.0
    assert snap4["curriculum_depth_best_success_rate"] is None
    assert snap4["curriculum_depth_min_success_rate"] is None

    # Avanço para D10: uma taxa inicial espúria de 1.0 (1 amostra) não deve registrar 100% como best
    tracker.record_env_step(
        reward=1.0,
        done=False,
        info={"current_depth": 10, "curriculum_promotions": 9},
    )
    tracker.record_env_step(
        reward=2.0,
        done=True,
        info={"current_depth": 10, "curriculum_promotions": 9, "curriculum_success_rate": 1.0},
    )
    snap5 = tracker.snapshot()
    assert snap5["curriculum_depth"] == 10
    # O pico de 1.0 é ignorado em D10 inicial
    assert snap5["curriculum_depth_best_success_rate"] is None or snap5["curriculum_depth_best_success_rate"] < 0.95

    # Taxa real madura subsequente (ex: 15%):
    tracker.record_env_step(
        reward=2.0,
        done=True,
        info={"current_depth": 10, "curriculum_promotions": 9, "curriculum_success_rate": 0.15},
    )
    snap6 = tracker.snapshot()
    assert snap6["curriculum_depth_best_success_rate"] == 0.15


def test_hud_server_telemetry_curriculum_spike_sanitization():
    """Garante que update_telemetry sanitiza picos de 100% decorrentes de janela com 1 amostra em D5+."""
    from system1_engine.hud.server import HUDServer
    server = HUDServer(host="127.0.0.1", port=9005, open_browser=False)

    server.update_telemetry({
        "curriculum_depth": 10,
        "curriculum_promotions": 9,
        "curriculum_success_rate": 0.12,
        "curriculum_depth_best_success_rate": 1.0,  # Spike vindo de amostragem unitária
    })
    snap = server.get_telemetry_snapshot()
    assert snap["curriculum_depth"] == 10
    # O valor 1.0 é sanitizado para a taxa real observada (0.12)
    assert snap["curriculum_depth_best_success_rate"] == 0.12







