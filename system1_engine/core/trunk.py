from typing import Optional, Tuple
import torch
import torch.nn as nn


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

