import json
import socket
import time
import urllib.request
import pytest

from system1_engine.telemetry.server import TelemetryServer
from system1_engine.telemetry.tracker import LiveStatsTracker


def test_telemetry_server_lifecycle_and_http_get():
    """Valida inicialização, requisição HTTP GET na home e encerramento limpo."""
    tracker = LiveStatsTracker()
    port = 8951
    server = TelemetryServer(tracker=tracker, host="127.0.0.1", port=port)

    server.start()
    assert server._running is True

    try:
        # Aguarda subida do socket
        time.sleep(0.1)

        req = urllib.request.Request(f"http://127.0.0.1:{port}/")
        with urllib.request.urlopen(req, timeout=2.0) as response:
            assert response.status == 200
            content_type = response.headers.get("Content-Type")
            assert "text/html" in content_type
            html = response.read().decode("utf-8")
            assert "SYSTEM 1 ENGINE" in html
            assert "chartLatency" in html
            assert "EventSource" in html

    finally:
        server.stop()
        assert server._running is False


def test_telemetry_server_api_metrics_endpoint():
    """Valida o endpoint REST /api/metrics para consumo direto de JSON."""
    tracker = LiveStatsTracker()
    tracker.record_inference(latency_us=190.5, uncertainty=0.15, confidence=0.85, entropy=0.4)
    tracker.record_env_step(reward=15.0, done=True)
    tracker.record_training_epoch(
        policy_loss=0.032,
        value_loss=0.85,
        clip_fraction=0.12,
        grad_norms={"FrontEnd": 0.2, "Trunk": 1.5, "PolicyHead": 2.2},
        lr=5e-4,
    )

    port = 8952
    with TelemetryServer(tracker=tracker, host="127.0.0.1", port=port) as server:
        time.sleep(0.1)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/metrics")
        with urllib.request.urlopen(req, timeout=2.0) as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
            assert data["latency_p50_us"] == 190.5
            assert data["mean_confidence"] == pytest.approx(0.85, abs=1e-5)
            assert data["mean_return_20"] == 15.0
            assert data["policy_loss"] == 0.032
            assert "FrontEnd" in data["grad_norms"]
            assert data["grad_norms"]["Trunk"] == 1.5


def test_telemetry_server_sse_stream():
    """Valida o streaming SSE (/stream) emitindo eventos formatados data: {...}."""
    tracker = LiveStatsTracker()
    tracker.record_inference(latency_us=170.0, uncertainty=0.05, confidence=0.95, entropy=0.1)
    tracker.record_env_step(reward=25.0, done=False)

    port = 8953
    with TelemetryServer(tracker=tracker, host="127.0.0.1", port=port, refresh_hz=30.0) as server:
        time.sleep(0.1)
        s = socket.create_connection(("127.0.0.1", port), timeout=3.0)
        s.sendall(b"GET /stream HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")

        raw_data = b""
        start_time = time.time()
        while (b"data: " not in raw_data or b"\n\n" not in raw_data.split(b"data: ", 1)[-1]) and (time.time() - start_time) < 3.0:
            chunk = s.recv(4096)
            if not chunk:
                break
            raw_data += chunk
        s.close()

        text = raw_data.decode("utf-8", errors="replace")
        assert "Content-Type: text/event-stream" in text or "content-type: text/event-stream" in text.lower()
        assert "data: " in text

        # Extrai linha JSON do evento SSE
        for line in text.splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[len("data: ") :].strip())
                assert payload["latency_p50_us"] == 170.0
                assert payload["mean_confidence"] == pytest.approx(0.95, abs=1e-5)
                assert payload["total_steps"] == 1
                break


def test_telemetry_server_context_manager_and_idempotency():
    """Garante suporte seguro a context manager e idempotência de start/stop."""
    tracker = LiveStatsTracker()
    port = 8954
    server = TelemetryServer(tracker=tracker, host="127.0.0.1", port=port)

    # Idempotência de start
    server.start()
    server.start()
    assert server._running is True

    # Idempotência de stop
    server.stop()
    server.stop()
    assert server._running is False
