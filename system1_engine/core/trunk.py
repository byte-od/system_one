from typing import Optional, Tuple
import torch
import torch.nn as nn

from system1_engine.core.recurrent import LinearRecurrentUnit


class ResMLPBlock(nn.Module):
    """Residual Multi-Layer Perceptron Block.

    Structure:
      x + Linear(GELU(Linear(LayerNorm(x))))
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return res + x


class System1Trunk(nn.Module):
    """System 1 Recurrent Trunk.

    Universal & transferable reflexive core with latent recurrence:
      - nn.LayerNorm(337)
      - nn.GRU(337 -> 256, batch_first=True)
      - 2x ResMLP Blocks (256)

    Strictly decoupled from front-ends and decision heads.
    Can be frozen and transferred across environments.
    """

    def __init__(
        self,
        input_dim: int = 337,
        hidden_dim: int = 256,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if input_dim > 337:
            self.input_norm = nn.LayerNorm(337)
            self.compass_input_norm = nn.LayerNorm(input_dim - 337)
        else:
            self.input_norm = nn.LayerNorm(input_dim)
            self.compass_input_norm = None
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.res_blocks = nn.Sequential(
            *[ResMLPBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

    def get_initial_hx(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Returns standard zero-initialized hidden state of shape [1, B, hidden_dim]."""
        return torch.zeros(
            1, batch_size, self.hidden_dim,
            device=device,
            dtype=dtype or torch.float32,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through recurrent trunk.

        Args:
            x: Input tensor of shape [B, T, 337] or [B, 337].
            hx: Hidden state of GRU [1, B, 256].
            dones: Episode boundary mask [B, T]. If provided, zeroes hx at episode resets.

        Returns:
            Tuple of:
              - h: Latent representations [B, T, 256]
              - next_hx: Updated GRU hidden state [1, B, 256]
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)  # [B, 1, 337]

        b, t, d = x.shape
        assert d == self.input_dim, (
            f"Expected input dimension {self.input_dim}, got {d}"
        )

        if hx is not None:
            assert hx.shape == (1, b, self.hidden_dim), (
                f"Expected hx shape {(1, b, self.hidden_dim)}, got {hx.shape}"
            )
        else:
            hx = torch.zeros(1, b, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Normalize input
        if self.compass_input_norm is not None:
            x_norm = torch.cat([
                self.input_norm(x[:, :, :337]),
                self.compass_input_norm(x[:, :, 337:]),
            ], dim=-1)
        else:
            x_norm = self.input_norm(x)

        if dones is None or not dones.any():
            # Fast vectorized unroll
            gru_out, next_hx = self.gru(x_norm, hx)
        else:
            # Temporal masking unroll to prevent cross-trajectory state leakage
            assert dones.shape == (b, t), (
                f"Expected dones shape {(b, t)}, got {dones.shape}"
            )
            step_outs = []
            curr_h = hx
            for step in range(t):
                # Mask hidden state immediately before step t
                mask = (1.0 - dones[:, step].float()).view(1, b, 1)
                curr_h = curr_h * mask
                step_x = x_norm[:, step : step + 1, :]
                step_out, curr_h = self.gru(step_x, curr_h)
                step_outs.append(step_out)

            gru_out = torch.cat(step_outs, dim=1)
            next_hx = curr_h

        # ResMLP processing
        h = self.res_blocks(gru_out)
        assert h.shape == (b, t, self.hidden_dim), (
            f"Expected h shape {(b, t, self.hidden_dim)}, got {h.shape}"
        )

        if is_2d:
            h = h.squeeze(1)

        return h, next_hx


class LeanResidualTrunk(nn.Module):
    """Universal Lean Deep Residual Recurrent Trunk (~1.5M parameters).

    Combines:
      1. LayerNorm(input_dim)
      2. Layer 1 GRU (Fast sensory recurrence): input_dim (337) -> fast_dim (256)
      3. Layer 2 GRU (Temporal abstraction): fast_dim (256) -> slow_dim (256)
         Multi-timescale temporal features concatenated into R^(fast_dim + slow_dim) = R^512
      4. Identity Residual Highway: Linear(input_dim, hidden_dim)
         Direct skip connection allowing unhindered gradient flow:
         z_comb = LayerNorm([h_fast, h_slow] + highway(x_norm))
      5. High-Capacity ResMLPBlock(hidden_dim):
         z_out = ResMLPBlock(512)(z_comb)

    State Management:
      Accepts hidden state hx of shape [1, B, 512], [2, B, 256], or [B, 512].
      Returns next_hx with canonical shape [1, B, 512].
    """

    def __init__(
        self,
        input_dim: int = 337,
        fast_dim: int = 256,
        slow_dim: int = 256,
        highway_bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.hidden_dim = fast_dim + slow_dim

        if input_dim > 337:
            self.input_norm = nn.LayerNorm(337)
            self.compass_input_norm = nn.LayerNorm(input_dim - 337)
        else:
            self.input_norm = nn.LayerNorm(input_dim)
            self.compass_input_norm = None
        self.fast_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=fast_dim,
            batch_first=True,
        )
        self.slow_gru = nn.GRU(
            input_size=fast_dim,
            hidden_size=slow_dim,
            batch_first=True,
        )
        self.highway = nn.Linear(input_dim, self.hidden_dim, bias=highway_bias)
        self.combine_norm = nn.LayerNorm(self.hidden_dim)
        self.res_block = ResMLPBlock(self.hidden_dim)

    def get_initial_hx(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Returns standard zero-initialized hidden state of shape [1, B, hidden_dim]."""
        return torch.zeros(
            1, batch_size, self.hidden_dim,
            device=device,
            dtype=dtype or torch.float32,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through lean deep residual recurrent trunk.

        Args:
            x: Input tensor of shape [B, T, 337] or [B, 337].
            hx: Hidden state [1, B, 512] or [2, B, 256].
            dones: Episode boundary mask [B, T]. If provided, zeroes hx at episode resets.

        Returns:
            Tuple of:
              - h: Latent representations [B, T, 512] (or [B, 512] if 2D input).
              - next_hx: Updated GRU hidden state [1, B, 512].
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)  # [B, 1, 337]

        b, t, d = x.shape
        assert d == self.input_dim, (
            f"Expected input dimension {self.input_dim}, got {d}"
        )

        if hx is not None:
            if hx.shape == (1, b, self.hidden_dim):
                h_fast = hx[:, :, : self.fast_dim].contiguous()
                h_slow = hx[:, :, self.fast_dim :].contiguous()
            elif hx.shape == (2, b, self.fast_dim):
                h_fast = hx[0:1].contiguous()
                h_slow = hx[1:2].contiguous()
            elif hx.shape == (b, self.hidden_dim):
                hx_3d = hx.view(1, b, self.hidden_dim)
                h_fast = hx_3d[:, :, : self.fast_dim].contiguous()
                h_slow = hx_3d[:, :, self.fast_dim :].contiguous()
            else:
                raise ValueError(
                    f"Expected hx shape {(1, b, self.hidden_dim)} or {(2, b, self.fast_dim)}, got {hx.shape}"
                )
        else:
            h_fast = torch.zeros(1, b, self.fast_dim, device=x.device, dtype=x.dtype)
            h_slow = torch.zeros(1, b, self.slow_dim, device=x.device, dtype=x.dtype)

        # Normalize sensory input
        if self.compass_input_norm is not None:
            x_norm = torch.cat([
                self.input_norm(x[:, :, :337]),
                self.compass_input_norm(x[:, :, 337:]),
            ], dim=-1)
        else:
            x_norm = self.input_norm(x)

        if dones is None or not dones.any():
            # Fast vectorized unroll across both GRU stages
            out_fast, next_h_fast = self.fast_gru(x_norm, h_fast)
            out_slow, next_h_slow = self.slow_gru(out_fast, h_slow)
        else:
            # Temporal masking unroll to prevent cross-trajectory state leakage
            assert dones.shape == (b, t), (
                f"Expected dones shape {(b, t)}, got {dones.shape}"
            )
            fast_outs = []
            slow_outs = []
            curr_hf = h_fast
            curr_hs = h_slow
            for step in range(t):
                mask = (1.0 - dones[:, step].float()).view(1, b, 1)
                curr_hf = curr_hf * mask
                curr_hs = curr_hs * mask
                step_x = x_norm[:, step : step + 1, :]
                step_of, curr_hf = self.fast_gru(step_x, curr_hf)
                step_os, curr_hs = self.slow_gru(step_of, curr_hs)
                fast_outs.append(step_of)
                slow_outs.append(step_os)

            out_fast = torch.cat(fast_outs, dim=1)
            out_slow = torch.cat(slow_outs, dim=1)
            next_h_fast = curr_hf
            next_h_slow = curr_hs

        # Multi-timescale recurrent fusion: concatenate fast and slow features [B, T, 512]
        h_rec = torch.cat([out_fast, out_slow], dim=-1)

        # Identity residual highway projection directly from sensory input [B, T, 512]
        h_high = self.highway(x_norm)

        # Fused representation with LayerNorm
        h_comb = self.combine_norm(h_rec + h_high)

        # High-capacity ResMLP non-linear processing
        h = self.res_block(h_comb)
        assert h.shape == (b, t, self.hidden_dim), (
            f"Expected h shape {(b, t, self.hidden_dim)}, got {h.shape}"
        )

        if is_2d:
            h = h.squeeze(1)

        # Unified next hidden state [1, B, 512]
        next_hx = torch.cat([next_h_fast, next_h_slow], dim=-1)
        return h, next_hx


class LRUResidualTrunk(nn.Module):
    """Universal Deep Linear Recurrent Residual Trunk (~1.14M parameters).

    Pilar 3 do AURA (Orvieto et al., ICML 2023 - DeepMind).
    Substitui os GRUs por Linear Recurrent Units (LRU) com parametrização
    polar e decomposição analítica real float32:

      1. LayerNorm(input_dim) (com suporte transparente a Tri-Compass)
      2. Layer 1 LRU (Fast sensory recurrence): input_dim -> fast_dim (256)
         r in [0.80, 0.95] (alta reatividade temporal, dinâmica de 1 a 10 passos)
      3. Layer 2 LRU (Strategic temporal abstraction): fast_dim -> slow_dim (256)
         r in [0.95, 0.999] (memória estratégica profunda, traços de 20 a 500 passos)
      4. Identity Residual Highway: Linear(input_dim, hidden_dim)
         z_comb = LayerNorm([y_fast, y_slow] + highway(x_norm))
      5. High-Capacity ResMLPBlock(hidden_dim):
         z_out = ResMLPBlock(512)(z_comb)

    State Management:
      Accepts hidden state hx of shape [1, B, 512] or [B, 512].
      Returns next_hx with canonical shape [1, B, 512].
      100% compatível com a interface do LeanResidualTrunk e exportação ONNX Opset 11+.
    """

    def __init__(
        self,
        input_dim: int = 337,
        fast_dim: int = 256,
        slow_dim: int = 256,
        highway_bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.hidden_dim = fast_dim + slow_dim

        if input_dim > 337:
            self.input_norm = nn.LayerNorm(337)
            self.compass_input_norm = nn.LayerNorm(input_dim - 337)
        else:
            self.input_norm = nn.LayerNorm(input_dim)
            self.compass_input_norm = None

        self.fast_lru = LinearRecurrentUnit(
            d_in=input_dim,
            d_out=fast_dim,
            d_hidden=fast_dim // 2,
            r_min=0.80,
            r_max=0.95,
        )
        self.slow_lru = LinearRecurrentUnit(
            d_in=fast_dim,
            d_out=slow_dim,
            d_hidden=slow_dim // 2,
            r_min=0.95,
            r_max=0.999,
        )
        self.highway = nn.Linear(input_dim, self.hidden_dim, bias=highway_bias)
        self.combine_norm = nn.LayerNorm(self.hidden_dim)
        self.res_block = ResMLPBlock(self.hidden_dim)
        self._xnorm_cache = None

    def train(self, mode: bool = True) -> "LRUTrunk":
        super().train(mode)
        self._xnorm_cache = None
        return self

    def _get_eval_xnorm_fused(self):
        flru = self.fast_lru
        hw = self.highway
        ver = (
            flru.nu_log._version, flru.theta._version, flru.gamma_log._version,
            flru.b_real._version, flru.b_imag._version, flru.c_real._version,
            flru.c_imag._version, flru.d._version,
            hw.weight._version, hw.bias._version if hw.bias is not None else 0,
            hw.weight.device, hw.weight.dtype,
        )
        if self._xnorm_cache is not None and self._xnorm_cache[0] == ver:
            return self._xnorm_cache[1:]
        f_lam_r, f_lam_i, w_flru_in, w_flru_out = flru._get_eval_fused()
        w_all = torch.cat([w_flru_in, hw.weight], dim=0)
        if hw.bias is not None:
            b_all = torch.cat([torch.zeros(w_flru_in.shape[0], device=hw.weight.device, dtype=hw.weight.dtype), hw.bias], dim=0)
        else:
            b_all = None
        self._xnorm_cache = (ver, f_lam_r, f_lam_i, w_all, b_all, w_flru_out)
        return f_lam_r, f_lam_i, w_all, b_all, w_flru_out

    def get_initial_hx(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Returns standard zero-initialized hidden state of shape [1, B, hidden_dim]."""
        return torch.zeros(
            1, batch_size, self.hidden_dim,
            device=device,
            dtype=dtype or torch.float32,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through deep linear recurrent residual trunk.

        Args:
            x: Input tensor of shape [B, T, D] or [B, D].
            hx: Hidden state [1, B, 512] or [B, 512].
            dones: Episode boundary mask [B, T].

        Returns:
            Tuple of:
              - h: Latent representations [B, T, 512] (or [B, 512] if 2D input).
              - next_hx: Updated LRU hidden state [1, B, 512].
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)  # [B, 1, D]

        b, t, d = x.shape
        assert d == self.input_dim, (
            f"Expected input dimension {self.input_dim}, got {d}"
        )

        if hx is not None:
            if hx.dim() == 2:
                hx = hx.unsqueeze(0)
            assert hx.shape == (1, b, self.hidden_dim), (
                f"Expected hx shape {(1, b, self.hidden_dim)}, got {hx.shape}"
            )
            h_fast = hx[:, :, :self.fast_dim]
            h_slow = hx[:, :, self.fast_dim:]
        else:
            h_fast = torch.zeros(1, b, self.fast_dim, device=x.device, dtype=x.dtype)
            h_slow = torch.zeros(1, b, self.slow_dim, device=x.device, dtype=x.dtype)

        # Normalize sensory input
        if self.compass_input_norm is not None:
            x_norm = torch.cat([
                self.input_norm(x[:, :, :337]),
                self.compass_input_norm(x[:, :, 337:]),
            ], dim=-1)
        else:
            x_norm = self.input_norm(x)

        if t == 1 and dones is None and not self.training and not torch.is_grad_enabled() and not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            f_lam_r, f_lam_i, w_all, b_all, w_flru_out = self._get_eval_xnorm_fused()
            xn_2d = x_norm.squeeze(1)
            xh_proj = torch.nn.functional.linear(xn_2d, w_all, b_all)
            dh = self.fast_lru.d_hidden
            dh2 = 2 * dh
            fd_end = dh2 + self.fast_dim
            fbx_r = xh_proj[:, :dh]
            fbx_i = xh_proj[:, dh:dh2]
            fdx = xh_proj[:, dh2:fd_end]
            h_high_2d = xh_proj[:, fd_end:]

            hf_2d = h_fast.squeeze(0)
            fhr, fhi = hf_2d[:, :dh], hf_2d[:, dh:]
            next_hf_2d = torch.cat([f_lam_r * fhr - f_lam_i * fhi + fbx_r, f_lam_r * fhi + f_lam_i * fhr + fbx_i], dim=-1)
            out_fast_2d = self.fast_lru.norm(torch.nn.functional.linear(next_hf_2d, w_flru_out) + fdx)

            out_slow_2d, next_hs_2d = self.slow_lru.step(out_fast_2d, h_slow.squeeze(0))
            h_comb = self.combine_norm(torch.cat([out_fast_2d, out_slow_2d], dim=-1) + h_high_2d)
            h_2d = self.res_block(h_comb)
            next_hx = torch.cat([next_hf_2d, next_hs_2d], dim=-1).unsqueeze(0)
            return (h_2d if is_2d else h_2d.unsqueeze(1)), next_hx

        # Recurrent propagation through cascaded LRU stages
        out_fast, next_hf = self.fast_lru(x_norm, h_fast, dones=dones)
        out_slow, next_hs = self.slow_lru(out_fast, h_slow, dones=dones)

        # Multi-timescale fusion: concatenate fast and slow features [B, T, 512]
        h_rec = torch.cat([out_fast, out_slow], dim=-1)

        # Identity residual highway projection directly from sensory input [B, T, 512]
        h_high = self.highway(x_norm)

        # Fused representation with LayerNorm
        h_comb = self.combine_norm(h_rec + h_high)

        # High-capacity ResMLP non-linear processing
        h = self.res_block(h_comb)
        assert h.shape == (b, t, self.hidden_dim), (
            f"Expected h shape {(b, t, self.hidden_dim)}, got {h.shape}"
        )

        if is_2d:
            h = h.squeeze(1)

        # Unified next hidden state [1, B, 512]
        next_hx = torch.cat([next_hf, next_hs], dim=-1)
        return h, next_hx


