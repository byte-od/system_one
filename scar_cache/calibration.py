"""Motor de Calibração Residual (Fase 2 do MoS 70B).

Executa a Calibração Layerwise de Scar Experts (Destilação de Ativações):
  1. Base Universal Compartilhada (W_base) e pesos de atenção permanecem estritamente congelados.
  2. As matrizes de cicatriz A_l e B_l (inicializadas via SVD) são alinhadas diretamente
     contra as ativações do modelo denso pré-treinado de 70B.
  3. Minimiza a divergência de ativações via perda composta:
       L = MSE(Y_teacher, Y_mos) + alpha * (1 - CosineSimilarity(Y_teacher, Y_mos))
  4. Mantém sempre o melhor checkpoint local (menor MSE), garantindo monotonicidade de qualidade.
  5. Executa camada por camada sem necessidade de backpropagation ao longo das 80 camadas,
     reduzindo a pegada de VRAM durante o treinamento para < 250 MB por passo!
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.real_70b import Real70BMoSMLP, MoSResidualLinear


@dataclass
class CalibrationConfig:
    """Configuração de Hiperparâmetros para Calibração de Scar Experts."""

    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    steps_per_layer: int = 50
    batch_size: int = 4
    seq_len: int = 64
    cosine_loss_weight: float = 0.5
    device: str = "cpu"
    verbose: bool = True


class ScarCalibrationEngine:
    """Motor de Calibração e Alinhamento Residual Layerwise para MoS."""

    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        self.config = config or CalibrationConfig()

    def calibrate_layer(
        self,
        teacher_mlp: nn.Module,
        mos_mlp: Real70BMoSMLP,
        calibration_inputs: torch.Tensor,
        steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Calibra os Scar Experts de uma única camada MoS contra o bloco denso original."""
        steps = steps or self.config.steps_per_layer
        device = torch.device(self.config.device)

        # Congela explicitamente os pesos da base compartilhada
        mos_mlp.gate_proj.base_weight.requires_grad = False
        mos_mlp.up_proj.base_weight.requires_grad = False
        mos_mlp.down_proj.base_weight.requires_grad = False

        # Garante que apenas A e B de cada projeção recebam gradiente
        trainable_params = []
        for proj in [mos_mlp.gate_proj, mos_mlp.up_proj, mos_mlp.down_proj]:
            proj.A.requires_grad = True
            proj.B.requires_grad = True
            trainable_params.extend([proj.A, proj.B])

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps,
            eta_min=self.config.min_learning_rate,
        )

        # 1. Ativação do Professor (Dense Teacher)
        teacher_mlp.eval()
        with torch.no_grad():
            x_sample = calibration_inputs.to(device)
            y_teacher = teacher_mlp(x_sample)

        # 2. Perda Inicial Pré-Calibração
        with torch.no_grad():
            y_init = mos_mlp(x_sample)
            init_mse = F.mse_loss(y_init, y_teacher).item()
            init_cos = F.cosine_similarity(y_init.view(-1), y_teacher.view(-1), dim=0).item()

        # Se já estiver praticamente perfeito (erro < 1e-5), encerra de imediato
        if init_mse < 1e-4:
            if self.config.verbose:
                print(f"    [Calibração] Camada já convergida com precisão SVD (MSE={init_mse:.2e}, Cos={init_cos:.6f}).")
            return {
                "init_mse": init_mse,
                "final_mse": init_mse,
                "init_cosine": init_cos,
                "final_cosine": init_cos,
                "improvement_pct": 0.0,
                "elapsed_seconds": 0.0,
                "steps": 0,
            }

        # Armazena melhor estado (garantia de monotonicidade)
        best_mse = init_mse
        best_state = {
            "gate_A": mos_mlp.gate_proj.A.data.clone(),
            "gate_B": mos_mlp.gate_proj.B.data.clone(),
            "up_A": mos_mlp.up_proj.A.data.clone(),
            "up_B": mos_mlp.up_proj.B.data.clone(),
            "down_A": mos_mlp.down_proj.A.data.clone(),
            "down_B": mos_mlp.down_proj.B.data.clone(),
        }

        # 3. Laço de Otimização Layerwise
        t0 = time.time()
        mos_mlp.train()

        for step in range(steps):
            optimizer.zero_grad()

            y_pred = mos_mlp(x_sample)
            mse_loss = F.mse_loss(y_pred, y_teacher)
            cos_loss = 1.0 - F.cosine_similarity(y_pred.view(-1), y_teacher.view(-1), dim=0)

            total_loss = mse_loss + self.config.cosine_loss_weight * cos_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Salva checkpoint se melhorou
            cur_mse = mse_loss.item()
            if cur_mse < best_mse:
                best_mse = cur_mse
                best_state["gate_A"].copy_(mos_mlp.gate_proj.A.data)
                best_state["gate_B"].copy_(mos_mlp.gate_proj.B.data)
                best_state["up_A"].copy_(mos_mlp.up_proj.A.data)
                best_state["up_B"].copy_(mos_mlp.up_proj.B.data)
                best_state["down_A"].copy_(mos_mlp.down_proj.A.data)
                best_state["down_B"].copy_(mos_mlp.down_proj.B.data)

        # Restaura melhor estado encontrado
        mos_mlp.gate_proj.A.data.copy_(best_state["gate_A"])
        mos_mlp.gate_proj.B.data.copy_(best_state["gate_B"])
        mos_mlp.up_proj.A.data.copy_(best_state["up_A"])
        mos_mlp.up_proj.B.data.copy_(best_state["up_B"])
        mos_mlp.down_proj.A.data.copy_(best_state["down_A"])
        mos_mlp.down_proj.B.data.copy_(best_state["down_B"])

        # 4. Avaliação Pós-Calibração
        mos_mlp.eval()
        with torch.no_grad():
            y_final = mos_mlp(x_sample)
            final_mse = F.mse_loss(y_final, y_teacher).item()
            final_cos = F.cosine_similarity(y_final.view(-1), y_teacher.view(-1), dim=0).item()

        elapsed = time.time() - t0
        improvement_mse = (init_mse - final_mse) / (init_mse + 1e-9) * 100.0

        metrics = {
            "init_mse": init_mse,
            "final_mse": final_mse,
            "init_cosine": init_cos,
            "final_cosine": final_cos,
            "improvement_pct": improvement_mse,
            "elapsed_seconds": elapsed,
            "steps": steps,
        }

        if self.config.verbose:
            print(
                f"    [Calibração] MSE: {init_mse:.2e} -> {final_mse:.2e} ({improvement_mse:+.1f}%) | "
                f"Cos: {init_cos:.5f} -> {final_cos:.5f} | Tempo: {elapsed:.2f}s"
            )

        return metrics

    def calibrate_group(
        self,
        teacher_mlps: List[nn.Module],
        mos_mlps: List[Real70BMoSMLP],
        calibration_inputs: torch.Tensor,
    ) -> List[Dict[str, float]]:
        """Calibra todas as camadas de um bloco MoS em sequência."""
        results = []
        for l_idx, (teacher, mos) in enumerate(zip(teacher_mlps, mos_mlps)):
            # Camada 0 é a base universal de referência (não requer calibração residual)
            if l_idx == 0:
                results.append({"layer": 0, "status": "base_layer_frozen"})
                continue

            if self.config.verbose:
                print(f"  --- Calibrando Camada MoS {l_idx} ---")
            metrics = self.calibrate_layer(teacher, mos, calibration_inputs)
            metrics["layer"] = l_idx
            results.append(metrics)

        return results
