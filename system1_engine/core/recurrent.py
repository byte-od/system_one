"""Linear Recurrent Units (LRU) - Recorrência Linear Complexa Diagonal (Orvieto et al., ICML 2023).

Substitui as portas saturáveis do GRU (sigmóide/tanh) por uma dinâmica puramente linear
e diagonal no corpo dos números complexos com decomposição analítica real float32:
  h_t = diag(lambda) h_{t-1} + gamma * (B x_t)
  y_t = Re(C h_t) + D x_t

Vantagens Arquiteturais:
  1. Zero saturação de portas e ausência de desvanecimento de gradientes em longos horizontes (T >= 50).
  2. Passo recorrente de inferência O(1) ultrarrápido em CPU e WebAssembly (multiplicações elemento a elemento).
  3. Totalmente compatível com exportação ONNX (Opset 11+) e ONNX Runtime Web via decomposição real pura float32.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRecurrentUnit(nn.Module):
    """Módulo Linear Recurrent Unit (LRU) com parametrização polar e decomposição real float32.

    Args:
        d_in: Dimensão do vetor de entrada x_t.
        d_out: Dimensão do vetor de saída y_t.
        d_hidden: Quantidade de autovalores / modos complexos (cada modo tem componente real e imaginária).
                  A dimensão total do estado oculto real é 2 * d_hidden.
        r_min: Raio polar mínimo dos autovalores (|lambda| >= r_min), define a menor escala temporal.
        r_max: Raio polar máximo dos autovalores (|lambda| <= r_max), define a maior escala temporal.
        dropout: Taxa de dropout opcional na saída.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int = 128,
        r_min: float = 0.85,
        r_max: float = 0.999,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.state_dim = 2 * d_hidden

        # Parametrização polar estável de autovalores no interior estrito do disco unitário:
        # lambda_j = exp(-nu_j + i * theta_j), com nu_j > 0 => |lambda_j| = exp(-nu_j) < 1.0.
        u1 = torch.rand(d_hidden)
        u2 = torch.rand(d_hidden)
        nu_init = -0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)
        theta_init = u2 * 2.0 * math.pi

        # nu_log garante nu = exp(nu_log) > 0 para qualquer valor nos reais (estabilidade contratante)
        self.nu_log = nn.Parameter(torch.log(nu_init))
        self.theta = nn.Parameter(theta_init)

        # Fator de normalização de variância de entrada gamma = sqrt(1 - |lambda|^2)
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1.0 - torch.exp(-2.0 * nu_init))))

        # Decomposição real das matrizes de projeção complexas (B e C)
        # B = B_real + i * B_imag
        self.b_real = nn.Parameter(torch.randn(d_hidden, d_in) / math.sqrt(2 * d_in))
        self.b_imag = nn.Parameter(torch.randn(d_hidden, d_in) / math.sqrt(2 * d_in))

        # C = C_real + i * C_imag
        self.c_real = nn.Parameter(torch.randn(d_out, d_hidden) / math.sqrt(d_hidden))
        self.c_imag = nn.Parameter(torch.randn(d_out, d_hidden) / math.sqrt(d_hidden))

        # Projeção residual direta (Highway D: d_in -> d_out)
        self.d = nn.Parameter(torch.randn(d_out, d_in) / math.sqrt(d_in))
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def get_lambda_real_imag(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calcula os componentes reais e imaginários de lambda e o escalonador gamma."""
        nu = torch.exp(self.nu_log)
        exp_neg_nu = torch.exp(-nu)
        lam_r = exp_neg_nu * torch.cos(self.theta)
        lam_i = exp_neg_nu * torch.sin(self.theta)
        gamma = torch.exp(self.gamma_log)
        return lam_r, lam_i, gamma

    def get_initial_hx(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Retorna estado oculto inicial canônico com formato [1, B, state_dim]."""
        return torch.zeros(
            1, batch_size, self.state_dim,
            device=device,
            dtype=dtype or torch.float32,
        )

    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Passo recorrente individual O(1) otimizado para inferência CPU / WASM.

        Args:
            x_t: Entrada no passo t com formato [B, d_in].
            h_prev: Estado oculto anterior com formato [B, 2 * d_hidden].

        Returns:
            Tuple de:
              - y_t: Saída no passo t com formato [B, d_out].
              - next_h: Próximo estado oculto com formato [B, 2 * d_hidden].
        """
        # Desempacota partes real e imaginária do estado anterior
        h_prev_r = h_prev[:, :self.d_hidden]
        h_prev_i = h_prev[:, self.d_hidden:]

        lam_r, lam_i, gamma = self.get_lambda_real_imag()

        # Projeção de entrada espacial
        bx_r = F.linear(x_t, self.b_real) * gamma
        bx_i = F.linear(x_t, self.b_imag) * gamma

        # Recorrência linear diagonal complexa: (lam_r + i * lam_i) * (h_r + i * h_i) + (bx_r + i * bx_i)
        h_t_r = (lam_r * h_prev_r - lam_i * h_prev_i) + bx_r
        h_t_i = (lam_r * h_prev_i + lam_i * h_prev_r) + bx_i

        # Projeção de saída: Re(C * h_t) + D * x_t
        cy = F.linear(h_t_r, self.c_real) - F.linear(h_t_i, self.c_imag)
        dx = F.linear(x_t, self.d)
        y_t = self.dropout(self.norm(cy + dx))

        next_h = torch.cat([h_t_r, h_t_i], dim=-1)
        return y_t, next_h

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processamento de sequências temporais completas [B, T, d_in].

        Args:
            x: Sequência de entrada [B, T, d_in].
            hx: Estado oculto inicial [1, B, 2 * d_hidden] ou [B, 2 * d_hidden].
            dones: Máscara temporal de fim de episódio [B, T] para evitar contaminação cross-trajectory.

        Returns:
            Tuple de:
              - y_seq: Sequência de saídas [B, T, d_out].
              - next_hx: Estado oculto final [1, B, 2 * d_hidden].
        """
        B, T, D = x.shape
        assert D == self.d_in, f"Dimensão de entrada esperada {self.d_in}, recebido {D}"

        if hx is not None:
            curr_h = hx.squeeze(0) if hx.dim() == 3 else hx
        else:
            curr_h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)

        outs = []
        for t in range(T):
            if dones is not None:
                mask = (1.0 - dones[:, t].float()).unsqueeze(-1)
                curr_h = curr_h * mask

            y_t, curr_h = self.step(x[:, t], curr_h)
            outs.append(y_t)

        y_seq = torch.stack(outs, dim=1)
        next_hx = curr_h.unsqueeze(0)
        return y_seq, next_hx
