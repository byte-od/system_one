"""Testes Unitários do Motor de Calibração Residual (Fase 2 do MoS 70B)."""

import math
import pytest
import torch
import torch.nn as nn
from scar_cache.real_70b import Real70BMoSMLP
from scar_cache.calibration import ScarCalibrationEngine, CalibrationConfig
from scar_cache.convert_qwen import factorize_svd


class DenseTeacherMLP(nn.Module):
    """Bloco SwiGLU Denso de Referência para calibração."""

    def __init__(self, in_features: int, intermediate: int, gw=None, uw=None, dw=None):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, intermediate, bias=False)
        self.up_proj = nn.Linear(in_features, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, in_features, bias=False)

        if gw is not None:
            self.gate_proj.weight.data.copy_(gw)
        if uw is not None:
            self.up_proj.weight.data.copy_(uw)
        if dw is not None:
            self.down_proj.weight.data.copy_(dw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def test_scar_calibration_reduces_loss():
    torch.manual_seed(42)
    hidden_size = 256
    intermediate_size = 768
    true_rank = 64
    truncated_rank = 16

    base_gw = torch.randn(intermediate_size, hidden_size)
    base_uw = torch.randn(intermediate_size, hidden_size)
    base_dw = torch.randn(hidden_size, intermediate_size)

    # Resíduo estruturado com rank verdadeiro 64
    diff_gw = (torch.randn(intermediate_size, true_rank) @ torch.randn(true_rank, hidden_size)) / true_rank**0.5
    diff_uw = (torch.randn(intermediate_size, true_rank) @ torch.randn(true_rank, hidden_size)) / true_rank**0.5
    diff_dw = (torch.randn(hidden_size, true_rank) @ torch.randn(true_rank, intermediate_size)) / true_rank**0.5

    teacher = DenseTeacherMLP(
        hidden_size,
        intermediate_size,
        gw=base_gw + 0.2 * diff_gw,
        uw=base_uw + 0.2 * diff_uw,
        dw=base_dw + 0.2 * diff_dw,
    )

    # Inicialização SVD truncada para rank 16
    Ag, Bg = factorize_svd(0.2 * diff_gw, rank=truncated_rank)
    Au, Bu = factorize_svd(0.2 * diff_uw, rank=truncated_rank)
    Ad, Bd = factorize_svd(0.2 * diff_dw, rank=truncated_rank)

    base_gate_param = nn.Parameter(base_gw.clone())
    student = Real70BMoSMLP(
        group_base_gate=base_gate_param,
        group_base_up=nn.Parameter(base_uw.clone()),
        group_base_down=nn.Parameter(base_dw.clone()),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rank=truncated_rank,
    )
    student.gate_proj.load_scar_weights(Ag, Bg)
    student.up_proj.load_scar_weights(Au, Bu)
    student.down_proj.load_scar_weights(Ad, Bd)

    # Dados de calibração
    calibration_x = torch.randn(4, 32, hidden_size)

    config = CalibrationConfig(
        learning_rate=2e-4,
        steps_per_layer=50,
        verbose=False,
    )
    engine = ScarCalibrationEngine(config)

    metrics = engine.calibrate_layer(
        teacher_mlp=teacher,
        mos_mlp=student,
        calibration_inputs=calibration_x,
    )

    # Validações
    assert metrics["final_mse"] <= metrics["init_mse"], "O MSE pós-calibração deve ser menor ou igual ao inicial!"
    assert metrics["final_cosine"] >= metrics["init_cosine"], "A similaridade de cosseno deve melhorar ou estabilizar!"
    assert metrics["improvement_pct"] > 5.0, f"Melhoria de MSE ({metrics['improvement_pct']:.1f}%) deve ser > 5%!"

    # Verifica que a base compartilhada permaneceu 100% inalterada
    assert torch.equal(student.gate_proj.base_weight, base_gate_param), "A base compartilhada NÃO deve mudar!"


def test_calibrate_group_pipeline():
    torch.manual_seed(42)
    hidden_size = 128
    intermediate_size = 384
    true_rank = 32
    rank = 16
    num_layers = 4

    base_gw = torch.randn(intermediate_size, hidden_size)
    base_uw = torch.randn(intermediate_size, hidden_size)
    base_dw = torch.randn(hidden_size, intermediate_size)

    teachers = []
    students = []

    base_gate_p = nn.Parameter(base_gw.clone())
    base_up_p = nn.Parameter(base_uw.clone())
    base_down_p = nn.Parameter(base_dw.clone())

    for l in range(num_layers):
        if l == 0:
            t = DenseTeacherMLP(hidden_size, intermediate_size, gw=base_gw, uw=base_uw, dw=base_dw)
            s = Real70BMoSMLP(base_gate_p, base_up_p, base_down_p, hidden_size, intermediate_size, rank=rank)
            with torch.no_grad():
                s.gate_proj.B.zero_()
                s.up_proj.B.zero_()
                s.down_proj.B.zero_()
        else:
            diff_g = (torch.randn(intermediate_size, true_rank) @ torch.randn(true_rank, hidden_size)) / true_rank**0.5
            diff_u = (torch.randn(intermediate_size, true_rank) @ torch.randn(true_rank, hidden_size)) / true_rank**0.5
            diff_d = (torch.randn(hidden_size, true_rank) @ torch.randn(true_rank, intermediate_size)) / true_rank**0.5

            t = DenseTeacherMLP(
                hidden_size, intermediate_size,
                gw=base_gw + 0.1 * diff_g,
                uw=base_uw + 0.1 * diff_u,
                dw=base_dw + 0.1 * diff_d,
            )

            Ag, Bg = factorize_svd(0.1 * diff_g, rank=rank)
            Au, Bu = factorize_svd(0.1 * diff_u, rank=rank)
            Ad, Bd = factorize_svd(0.1 * diff_d, rank=rank)

            s = Real70BMoSMLP(base_gate_p, base_up_p, base_down_p, hidden_size, intermediate_size, rank=rank)
            s.gate_proj.load_scar_weights(Ag, Bg)
            s.up_proj.load_scar_weights(Au, Bu)
            s.down_proj.load_scar_weights(Ad, Bd)

        teachers.append(t)
        students.append(s)

    calibration_x = torch.randn(2, 16, hidden_size)
    config = CalibrationConfig(steps_per_layer=25, learning_rate=1e-4, verbose=False)
    engine = ScarCalibrationEngine(config)

    results = engine.calibrate_group(teachers, students, calibration_x)

    assert len(results) == num_layers
    assert results[0]["status"] == "base_layer_frozen"
    for r in results[1:]:
        assert r["final_cosine"] > 0.95, f"Camada {r['layer']} similaridade {r['final_cosine']} abaixo de 0.95"


if __name__ == "__main__":
    test_scar_calibration_reduces_loss()
    test_calibrate_group_pipeline()
    print("✅ TODOS OS TESTES DO MOTOR DE CALIBRAÇÃO (FASE 2) PASSARAM COM SUCESSO!")
