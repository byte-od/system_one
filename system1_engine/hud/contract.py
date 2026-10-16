"""HUD-Web Parity Contract.

Define rigorosamente as especificações e constantes obrigatórias que devem ser
implementadas em perfeita paridade tanto no HUD Python quanto no Web Hub (JavaScript/TypeScript).

Qualquer alteração em modos de inferência, hiperparâmetros de calibração homeostática,
esquema de telemetria ou catálogo de modelos deve satisfazer este contrato, validado
automaticamente via suíte de testes (tests/test_hud_web_parity_contract.py).
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Any
import json
from pathlib import Path


@dataclass(frozen=True)
class HomeostasisSpec:
    """Hiperparâmetros universais da regulação termodinâmica do System 1."""
    min_calibration: float = 0.0
    max_calibration: float = 0.80
    stagnation_threshold: int = 2
    heating_rate: float = 0.10
    cooling_rate: float = 0.20
    momentum_alpha: float = 0.10
    relief_threshold: float = 0.01


@dataclass(frozen=True)
class TelemetrySpec:
    """Métricas obrigatórias em tempo real retornadas em cada reflexo."""
    required_fields: tuple = ("latency", "confidence", "uncertainty", "calibration")


@dataclass(frozen=True)
class CheckpointSpec:
    """Convenção oficial de nomenclatura e auto-resolução de checkpoints do System 1."""
    required_keywords: tuple = ("trained", "cerberus", "aura")
    pattern: str = r"^s1_.+_(?:trained|cerberus|aura)(?:_v\d+)?\.pt$"
    description: str = "Checkpoints PyTorch (.pt) salvos e treinados devem conter 'trained' ou linhagem canônica ('aura', 'cerberus') no nome."


class HUDWebParityContract:
    """Contrato formal de paridade entre o HUD Python e o Web Hub."""

    VERSION = "1.0.0"

    HOMEOSTASIS = HomeostasisSpec()
    TELEMETRY = TelemetrySpec()
    CHECKPOINT = CheckpointSpec()

    ACTION_MODES = {
        "auto": "Auto-calibração homeostática por momentum de recompensa",
        "calibrated": "Amostragem estocástica contínua calibrada [0.0 a 1.0]",
        "deterministic": "Modo determinístico puro (argmax para discreto, média μ para contínuo)",
        "stochastic": "Amostragem estocástica total da política (1.0σ ou Boltzmann T=1.0)",
    }

    MANDATORY_MODELS = {
        "rubiks_atomic",
        "rubiks_macro",
        "cartpole",
        "acrobot",
        "mountaincar",
        "lunarlander",
        "ant_v5",
    }

    @classmethod
    def validate_homeostasis_code(cls, source_code: str, label: str = "Web Source") -> List[str]:
        """Inspeciona o código-fonte (JS/TS/Python) e garante que as constantes homeostáticas estão presentes."""
        errors = []
        spec = cls.HOMEOSTASIS

        # Valida limiar de estagnação
        if f">={spec.stagnation_threshold}" not in source_code.replace(" ", "") and f">={spec.stagnation_threshold}" not in source_code:
            errors.append(f"[{label}] Limiar de estagnação ({spec.stagnation_threshold}) não encontrado no código.")

        # Valida taxa de aquecimento
        if f"{spec.heating_rate:.2f}" not in source_code and f"{spec.heating_rate}" not in source_code:
            errors.append(f"[{label}] Taxa de aquecimento ({spec.heating_rate}) não encontrada no código.")

        # Valida taxa de resfriamento
        if f"{spec.cooling_rate:.2f}" not in source_code and f"{spec.cooling_rate}" not in source_code:
            errors.append(f"[{label}] Taxa de resfriamento ({spec.cooling_rate}) não encontrada no código.")

        # Valida teto máximo de calibração
        if f"{spec.max_calibration:.2f}" not in source_code and f"{spec.max_calibration}" not in source_code:
            errors.append(f"[{label}] Limite máximo de calibração ({spec.max_calibration}) não encontrado no código.")

        # Valida limiar de alívio / progresso
        if f"{spec.relief_threshold}" not in source_code and f"{spec.relief_threshold:.2f}" not in source_code:
            errors.append(f"[{label}] Limiar de alívio de recompensa ({spec.relief_threshold}) não encontrado no código.")

        return errors

    @classmethod
    def validate_manifest(cls, manifest_path: Path) -> List[str]:
        """Garante que o manifesto web contém todos os modelos mandatórios e metadados corretos."""
        errors = []
        if not manifest_path.exists():
            return [f"Manifesto não encontrado em {manifest_path}"]

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return [f"Erro ao ler JSON do manifesto: {e}"]

        models = data.get("models", {})
        missing = cls.MANDATORY_MODELS - set(models.keys())
        if missing:
            errors.append(f"Modelos mandatórios ausentes no manifesto: {missing}")

        models_dir = manifest_path.parent
        for key, m in models.items():
            filename = m.get("file")
            if not filename:
                errors.append(f"Modelo '{key}' não declara o campo 'file'.")
                continue
            onnx_file = models_dir / filename
            if not onnx_file.exists():
                errors.append(f"Arquivo ONNX {filename} não existe em {models_dir}.")

        return errors

    @classmethod
    def validate_checkpoint_convention(cls, checkpoint_filename: str) -> bool:
        """Valida se o nome do arquivo de checkpoint segue a convenção de conter 'trained' ou 'cerberus'."""
        fn = Path(checkpoint_filename).name.lower()
        return any(kw in fn for kw in cls.CHECKPOINT.required_keywords) and fn.endswith(".pt")

