"""Universal Relational Perceptual Front-End (Pilar 4 do AURA).

Implementa Atenção Relacional Linear O(N) com Kernelização Positiva
(Katharopoulos et al., ICML 2020) sobre slots latentes universais.

Propriedades Arquiteturais:
  1. 100% Agnóstico de Domínio: opera sobre qualquer vetor de observação s in R^d_obs
     (Cubo de Rubik, CartPole, LunarLander, MuJoCo, etc.) sem heurísticas humanas.
  2. Complexidade Linear O(S): ao invés de atenção quadrática O(S^2), associa
     phi(K)^T @ V primeiro, gerando matrizes de projeção de dimensão constante
     e execução em nanossegundos em CPU/WASM.
  3. Kernel Positivo: phi(x) = ELU(x) + 1.0 garante garantia estrita de não-negatividade
     sem necessidade de computação exponencial cara de Softmax no gargalo temporal.
  4. 100% Exportável para ONNX: utiliza operadores universais (MatMul, Add, Div, Elu).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRelationalAttention(nn.Module):
    """Atenção Linear Multi-Head com Kernel Positivo phi(x) = ELU(x) + 1.0.

    Complexidade O(S * d_h^2) em vez de O(S^2 * d_h), ideal para inferência
    sub-milissegundo em CPU, WebAssembly e aceleração Metal/MPS.
    """

    def __init__(self, embed_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), f"embed_dim ({embed_dim}) deve ser divisível por num_heads ({num_heads})"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self._qkv_cache = None

    def train(self, mode: bool = True) -> "LinearRelationalAttention":
        super().train(mode)
        self._qkv_cache = None
        return self

    def _get_eval_qkv_weight(self) -> torch.Tensor:
        ver = (
            self.q_proj.weight._version,
            self.k_proj.weight._version,
            self.v_proj.weight._version,
            self.q_proj.weight.device,
            self.q_proj.weight.dtype,
        )
        if self._qkv_cache is not None and self._qkv_cache[0] == ver:
            return self._qkv_cache[1]
        w_qkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        self._qkv_cache = (ver, w_qkv)
        return w_qkv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processa tensor de slots com formato [B, T, S, D] ou [B, S, D]."""
        orig_shape = x.shape
        if x.dim() == 4:
            b, t, s, d = x.shape
            x = x.view(b * t, s, d)
        n, s, d = x.shape

        if not self.training and not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            w_qkv = self._get_eval_qkv_weight()
            qkv = F.linear(x, w_qkv).view(n, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # Projeções Lineares [N, S, H, head_dim] -> [N, H, S, head_dim]
            q = (
                self.q_proj(x)
                .view(n, s, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                self.k_proj(x)
                .view(n, s, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(x)
                .view(n, s, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        # Kernelização Positiva: phi(x) = ELU(x) + 1.0 >= 0
        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0

        # Contração Chave-Valor KV: [N, H, head_dim, head_dim]
        # (phi_k^T @ v tem dimensão constante head_dim x head_dim, independente de S)
        kv = torch.matmul(phi_k.transpose(-2, -1), v)

        # Contração Consulta-Contexto (Numerador): [N, H, S, head_dim]
        num = torch.matmul(phi_q, kv)

        # Denominador de Normalização: [N, H, S, 1]
        k_sum = phi_k.sum(dim=-2, keepdim=True)  # [N, H, 1, head_dim]
        denom = torch.matmul(phi_q, k_sum.transpose(-2, -1))

        # Saída normalizada com proteção numérica
        out = num / (denom + 1e-6)

        # Re-combinação das cabeças: [N, S, embed_dim]
        out = out.transpose(1, 2).contiguous().view(n, s, d)
        out = self.out_proj(out)

        if len(orig_shape) == 4:
            out = out.view(orig_shape)
        return out


class UniversalRelationalStateEncoder(nn.Module):
    """Universal Relational Perceptual State Encoder (Pilar 4 do AURA).

    Combina projeção direta com Atenção Relacional Linear sobre S slots latentes
    universais para descobrir correlações estruturais e simetrias de forma autodidata.

    Arquitetura:
      1. Ramo Direto (Direct Highway):
         obs in R^obs_dim -> Linear(obs_dim, out_dim) -> GELU -> LayerNorm -> z_base
      2. Ramo Relacional (Universal Slots + Linear Attention):
         - Slot Projection: Linear(obs_dim, num_slots * slot_dim) -> [B, T, S, slot_dim]
         - Learnable Slot Identity Bias: slot_embed in R^[1, 1, S, slot_dim]
         - Linear Self-Attention O(S): slot_norm1 -> LinearRelationalAttention
         - Slot FFN: slot_norm2 -> FFN(slot_dim -> slot_dim * 2 -> slot_dim)
         - Aggregation: Linear(num_slots * slot_dim, out_dim)
      3. Fusão Residual e Normalização:
         z_s = LayerNorm(z_base + z_rel) in R^out_dim (256)
    """

    def __init__(
        self,
        obs_dim: int,
        out_dim: int = 256,
        num_slots: int = 8,
        slot_dim: int = 32,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # 1. Ramo Direto / Highway (retrocompatível com state_encoder.net)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

        # 2. Ramo Relacional com Slots Latentes
        self.slot_proj = nn.Linear(obs_dim, num_slots * slot_dim)
        self.slot_embed = nn.Parameter(
            torch.randn(1, 1, num_slots, slot_dim) * 0.02
        )
        self.slot_norm1 = nn.LayerNorm(slot_dim)
        self.attn = LinearRelationalAttention(
            embed_dim=slot_dim, num_heads=num_heads
        )
        self.slot_norm2 = nn.LayerNorm(slot_dim)
        self.slot_ffn = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.slot_out_proj = nn.Linear(num_slots * slot_dim, out_dim)

        # 3. Normalização de Fusão
        self.combine_norm = nn.LayerNorm(out_dim)
        self.scar_memory = None
        self._last_delta_slots = None
        self._last_resonance = None
        self._obs_proj_cache = None

    def train(self, mode: bool = True) -> "UniversalRelationalStateEncoder":
        super().train(mode)
        self._obs_proj_cache = None
        return self

    def _get_eval_obs_proj(self):
        l0 = self.net[0]
        ver = (
            l0.weight._version,
            l0.bias._version,
            self.slot_proj.weight._version,
            self.slot_proj.bias._version,
            l0.weight.device,
            l0.weight.dtype,
        )
        if self._obs_proj_cache is not None and self._obs_proj_cache[0] == ver:
            return self._obs_proj_cache[1], self._obs_proj_cache[2]
        w_both = torch.cat([l0.weight, self.slot_proj.weight], dim=0)
        b_both = torch.cat([l0.bias, self.slot_proj.bias], dim=0)
        self._obs_proj_cache = (ver, w_both, b_both)
        return w_both, b_both

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Processa vetor de observação [B, T, obs_dim] ou [B, obs_dim]."""
        is_2d = obs.dim() == 2
        if is_2d:
            obs = obs.unsqueeze(1)

        b, t, d = obs.shape
        assert (
            d == self.obs_dim
        ), f"Dimensão de entrada esperada {self.obs_dim}, recebido {d}"

        if not self.training and not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            w_both, b_both = self._get_eval_obs_proj()
            both = F.linear(obs, w_both, b_both)
            z_base = self.net[2](self.net[1](both[..., :self.out_dim]))
            slots = both[..., self.out_dim:].view(b, t, self.num_slots, self.slot_dim)
        else:
            # 1. Ramo Direto
            z_base = self.net(obs)  # [B, T, out_dim]
            # 2. Ramo Relacional com Atenção Linear sobre Slots
            slots = self.slot_proj(obs).view(b, t, self.num_slots, self.slot_dim)

        slots = slots + self.slot_embed
        slots = slots + self.attn(self.slot_norm1(slots))
        slots = slots + self.slot_ffn(self.slot_norm2(slots))

        # Modulação por Cicatriz de Gradiente (Fast-Weights / Hybrid O(1))
        if self.scar_memory is not None and getattr(self.scar_memory, "has_fast_weights", False):
            slots, delta = self.scar_memory.modulate_slots(slots)
            self._last_delta_slots = delta
            if getattr(self.scar_memory, "mode", "") == "hybrid" and delta is not None:
                self._last_resonance = self.scar_memory.compute_resonance_hybrid(delta)
        else:
            self._last_delta_slots = None
            self._last_resonance = None

        # Achatamento e projeção relacional
        self._last_slots = slots
        slots_flat = slots.view(b, t, self.num_slots * self.slot_dim)
        z_rel = self.slot_out_proj(slots_flat)

        # 3. Fusão Residual
        out = self.combine_norm(z_base + z_rel)

        if is_2d:
            out = out.squeeze(1)
        return out
