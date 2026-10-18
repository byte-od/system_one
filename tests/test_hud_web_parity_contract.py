"""Testes formais do Contrato de Paridade HUD-Web.

Garante que qualquer evolução de arquitetura, modo de inferência, homeostase ou telemetria
seja obrigatoriamente implementada tanto no HUD Python quanto nos runtimes Web
(web/index.html, web/system1-eval.js e portfolio/src/lib/system1.ts).
"""

from pathlib import Path
import pytest
import inspect
from system1_engine.hud.contract import HUDWebParityContract
from system1_engine.core.agent import UniversalS1Agent, ReflexDecision
from system1_engine.telemetry.tracker import LiveStatsTracker


PROJECT_ROOT = Path(__file__).parent.parent
PORTFOLIO_ROOT = PROJECT_ROOT.parent / "portfolio"


def test_contract_homeostasis_python_agent():
    """Garante que o UniversalS1Agent implementa todas as regras da especificação homeostática."""
    agent_src = inspect.getsource(UniversalS1Agent)
    errors = HUDWebParityContract.validate_homeostasis_code(agent_src, label="Python UniversalS1Agent")
    assert not errors, f"Violações do contrato no agente Python: {errors}"


def test_contract_homeostasis_web_eval_js():
    """Garante que o runtime web (web/system1-eval.js) implementa as mesmas constantes homeostáticas."""
    web_js_path = PROJECT_ROOT / "web" / "system1-eval.js"
    assert web_js_path.exists(), "web/system1-eval.js não encontrado"
    code = web_js_path.read_text(encoding="utf-8")
    errors = HUDWebParityContract.validate_homeostasis_code(code, label="web/system1-eval.js")
    assert not errors, f"Violações do contrato em web/system1-eval.js: {errors}"


def test_contract_homeostasis_portfolio_ts():
    """Se o portfólio estiver presente, garante paridade estrita em portfolio/src/lib/system1.ts."""
    portfolio_ts = PORTFOLIO_ROOT / "src" / "lib" / "system1.ts"
    if portfolio_ts.exists():
        code = portfolio_ts.read_text(encoding="utf-8")
        errors = HUDWebParityContract.validate_homeostasis_code(code, label="portfolio/src/lib/system1.ts")
        assert not errors, f"Violações do contrato em portfolio/src/lib/system1.ts: {errors}"


def test_contract_telemetry_schema_parity():
    """Garante que ReflexDecision, TelemetryTracker e Web Runtime reportam os campos obrigatórios."""
    # 1. ReflexDecision em Python
    decision_fields = ReflexDecision.__dataclass_fields__.keys()
    for field in ("latency_ms", "confidence", "uncertainty", "calibration"):
        assert field in decision_fields, f"Campo mandatório '{field}' ausente em ReflexDecision"

    # 2. TelemetryTracker snapshot em Python
    tracker = LiveStatsTracker()
    tracker.record_inference(latency_us=200.0, uncertainty=0.1, confidence=0.9, entropy=0.1, calibration=0.35)
    snap = tracker.snapshot()
    assert "calibration" in snap, "Campo 'calibration' ausente no snapshot da telemetria"
    assert snap["calibration"] == 0.35

    # 3. Web Runtime
    web_js_code = (PROJECT_ROOT / "web" / "system1-eval.js").read_text(encoding="utf-8")
    assert "calibration:" in web_js_code, "web/system1-eval.js não retorna a métrica 'calibration'"


def test_contract_action_modes_ui_parity():
    """Garante paridade dos modos de ação e telemetria na UI do HUD Python e do Web Hub."""
    hud_html = (PROJECT_ROOT / "system1_engine" / "hud" / "dashboard.html").read_text(encoding="utf-8")
    web_html = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")

    # Ambas as interfaces devem suportar todos os modos de ação
    for mode in ("auto", "calibrated", "deterministic", "stochastic"):
        assert mode in hud_html, f"Modo de ação '{mode}' ausente no HUD Python (dashboard.html)"

    # Ambas as UIs devem exibir o botão/atalho Auto
    assert "Auto" in hud_html, "Controle 'Auto' ausente no HUD Python"
    assert "Auto" in web_html, "Controle 'Auto' ausente no Web Hub (index.html)"

    # Ambas as UIs devem conter o card de telemetria de calibração
    assert "CALIBRAÇÃO" in hud_html or "Calibração" in hud_html, "Card de calibração ausente no HUD Python"
    assert "CALIBRAÇÃO" in web_html or "metric-calib" in web_html, "Card de calibração ausente no Web Hub"

    # Se portfólio existir, valida também a interface React
    portfolio_tsx = PORTFOLIO_ROOT / "src" / "components" / "SystemOneHudDemo.tsx"
    if portfolio_tsx.exists():
        p_code = portfolio_tsx.read_text(encoding="utf-8")
        assert "isAutoCalibrate" in p_code, "Auto-calibração ausente no portfólio React"
        assert "CALIBRAÇÃO" in p_code or "CALIBRATION" in p_code, "Card de calibração ausente no portfólio React"


def test_contract_manifest_and_models_parity():
    """Garante que todos os modelos exigidos pelo contrato estão no manifesto e existem como arquivos .onnx."""
    manifest_path = PROJECT_ROOT / "web" / "models" / "manifest.json"
    errors = HUDWebParityContract.validate_manifest(manifest_path)
    assert not errors, f"Violações do contrato no manifesto de modelos: {errors}"


def test_contract_checkpoint_naming_convention():
    """Garante que a convenção formal exige 'trained', 'cerberus' ou 'aura' no nome de todos os checkpoints."""
    # 1. Regra de validação formal do contrato
    assert HUDWebParityContract.validate_checkpoint_convention("s1_cartpole_trained.pt")
    assert HUDWebParityContract.validate_checkpoint_convention("s1_ant_v5_trained_v3.pt")
    assert HUDWebParityContract.validate_checkpoint_convention("s1_rubikscube_v0_cerberus_v1.pt")
    assert HUDWebParityContract.validate_checkpoint_convention("s1_rubikscube_v0_aura.pt")
    assert HUDWebParityContract.validate_checkpoint_convention("s1_rubikscube_v0_trained.pt")
    assert not HUDWebParityContract.validate_checkpoint_convention("s1_cartpole.pt")
    assert not HUDWebParityContract.validate_checkpoint_convention("s1_cartpole_checkpoint.pt")

    # 2. HUD Python (dashboard.html) deve sugerir estritamente nomes com 'trained'
    dashboard_html = (PROJECT_ROOT / "system1_engine" / "hud" / "dashboard.html").read_text(encoding="utf-8")
    assert "s1_cartpole_trained.pt" in dashboard_html
    assert "const suffix = 'trained';" in dashboard_html

    # 3. Web Hub (index.html) deve exibir o checkpoint de origem PyTorch
    web_html = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "active-ckpt-source" in web_html
    assert "Origem PyTorch (.pt)" in web_html

    # 4. Manifesto de modelos web deve vincular checkpoints de origem contendo 'trained', 'cerberus' ou 'aura'
    manifest_path = PROJECT_ROOT / "web" / "models" / "manifest.json"
    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, model_info in data.get("models", {}).items():
        source = model_info.get("checkpoint_source")
        assert source, f"Modelo {key} não declara checkpoint_source"
        assert any(kw in source.lower() for kw in ("trained", "cerberus", "aura")), f"Checkpoint de origem {source} para {key} não possui 'trained', 'cerberus' ou 'aura'"


def test_contract_javascript_syntax_cleanliness():
    """Garante que nenhum script JavaScript no HUD (dashboard.html) ou Web Hub (index.html) possui erros de sintaxe."""
    import re
    import shutil
    import subprocess
    import tempfile

    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js não disponível no ambiente para validação de sintaxe JS")

    files_to_check = [
        PROJECT_ROOT / "system1_engine" / "hud" / "dashboard.html",
        PROJECT_ROOT / "web" / "index.html",
    ]

    for fpath in files_to_check:
        assert fpath.exists(), f"Arquivo {fpath} não encontrado"
        content = fpath.read_text(encoding="utf-8")
        scripts = re.findall(r"<script(?![^>]*src=)[^>]*>(.*?)</script>", content, re.DOTALL)
        for idx, s in enumerate(scripts):
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tf:
                tf.write(s)
                temp_name = tf.name
            try:
                res = subprocess.run([node_bin, "-c", temp_name], capture_output=True, text=True)
                assert res.returncode == 0, f"Erro de sintaxe JS no script #{idx} de {fpath.name}:\n{res.stderr}"
            finally:
                Path(temp_name).unlink(missing_ok=True)


