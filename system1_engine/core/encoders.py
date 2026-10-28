from typing import Any, Optional, Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """Encodes raw vector state s_t into z_s in R^256."""

    def __init__(self, obs_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.shape[-1] == self.obs_dim, (
            f"Expected state dimension {self.obs_dim}, got {obs.shape[-1]}"
        )
        return self.net(obs)


class DeltaEncoder(nn.Module):
    """Encodes state derivative Delta s_t = s_t - s_{t-1} into z_Delta in R^64."""

    def __init__(self, obs_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, delta_obs: torch.Tensor) -> torch.Tensor:
        assert delta_obs.shape[-1] == self.obs_dim, (
            f"Expected delta dimension {self.obs_dim}, got {delta_obs.shape[-1]}"
        )
        return self.net(delta_obs)


class ActionEncoder(nn.Module):
    """Encodes previous action a_{t-1} into z_a in R^16."""

    def __init__(
        self, action_space: gym.spaces.Space, embed_dim: int = 16
    ) -> None:
        super().__init__()
        self.action_space = action_space
        self.embed_dim = embed_dim

        if isinstance(action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.act_dim = int(action_space.n)
            self.encoder = nn.Embedding(self.act_dim, embed_dim)
        elif isinstance(action_space, gym.spaces.Box):
            self.is_discrete = False
            self.act_dim = int(np.prod(action_space.shape))
            self.encoder = nn.Linear(self.act_dim, embed_dim)
        else:
            raise NotImplementedError(
                f"Action space {type(action_space)} not supported. Use Discrete or Box."
            )

    def forward(self, prev_action: torch.Tensor) -> torch.Tensor:
        if self.is_discrete:
            # Handle shapes [B, T] or [B, T, 1] -> [B, T]
            if prev_action.dim() == 3 and prev_action.shape[-1] == 1:
                prev_action = prev_action.squeeze(-1)
            action_indices = prev_action.long()
            return self.encoder(action_indices)
        else:
            if prev_action.dim() == 2 and self.act_dim == 1:
                prev_action = prev_action.unsqueeze(-1)
            assert prev_action.shape[-1] == self.act_dim, (
                f"Expected action dim {self.act_dim}, got {prev_action.shape[-1]}"
            )
            return self.encoder(prev_action.float())


class RewardEncoder(nn.Module):
    """Encodes previous reward r_{t-1} into z_r in R^1."""

    def __init__(self, fixed: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
        if fixed:
            self.linear.weight.requires_grad = False

    def forward(self, prev_reward: torch.Tensor) -> torch.Tensor:
        if prev_reward.dim() == 1:
            prev_reward = prev_reward.view(-1, 1, 1)
        elif prev_reward.dim() == 2:
            prev_reward = prev_reward.unsqueeze(-1)
        assert prev_reward.shape[-1] == 1, (
            f"Expected reward tensor with last dim 1, got shape {prev_reward.shape}"
        )
        return self.linear(prev_reward.float())


class VectorFrontEnd(nn.Module):
    """Universal Vector Perceptual Front-End.

    Supports:
      1. Triangulação Desacoplada (tri_compass=True):
         s_t           (obs_dim)  -> z_s  in R^256
         Delta_local   (obs_dim)  -> z_l  in R^64 (velocidade instantânea)
         Delta_origin  (obs_dim)  -> z_o  in R^64 (deslocamento da origem)
         Delta_goal    (obs_dim)  -> z_g  in R^64 (atrator geodésico c-SIL/DPO)
         a_{t-1}       (act_dim)  -> z_a  in R^16
         r_{t-1}       (1)        -> z_r  in R^1
         Concatena em R^465 e aplica LayerNorm(465).
      2. Modo Clássico (tri_compass=False):
         Concatena em R^337 e aplica LayerNorm(337).
    """

    def __init__(
        self,
        obs_dim: int,
        action_space: gym.spaces.Space,
        reward_fixed: bool = False,
        tri_compass: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.tri_compass = tri_compass

        self.state_encoder = StateEncoder(obs_dim, out_dim=256)
        if tri_compass:
            self.local_delta_encoder = DeltaEncoder(obs_dim, out_dim=64)
            self.origin_delta_encoder = DeltaEncoder(obs_dim, out_dim=64)
            self.goal_delta_encoder = DeltaEncoder(obs_dim, out_dim=64)
            self.out_dim = 256 + 64 + 16 + 1 + 64 + 64  # 465: core (337) + compass (128)
            self.out_norm = nn.LayerNorm(337)
            self.compass_norm = nn.LayerNorm(128)
        else:
            self.delta_encoder = DeltaEncoder(obs_dim, out_dim=64)
            self.out_dim = 256 + 64 + 16 + 1  # 337
            self.out_norm = nn.LayerNorm(337)
            self.compass_norm = None

        self.action_encoder = ActionEncoder(action_space, embed_dim=16)
        self.reward_encoder = RewardEncoder(fixed=reward_fixed)

    def forward(
        self,
        obs: torch.Tensor,
        delta_obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        local_delta: Optional[torch.Tensor] = None,
        origin_delta: Optional[torch.Tensor] = None,
        goal_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Defensive dimensionality checks and alignment to [B, T, ...]
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if delta_obs.dim() == 2:
            delta_obs = delta_obs.unsqueeze(1)
        if prev_action.dim() == 1:
            prev_action = prev_action.unsqueeze(1)
        elif prev_action.dim() == 2 and not isinstance(self.action_space, gym.spaces.Discrete):
            prev_action = prev_action.unsqueeze(1)
        if prev_reward.dim() == 1:
            prev_reward = prev_reward.unsqueeze(1).unsqueeze(-1)
        elif prev_reward.dim() == 2:
            prev_reward = prev_reward.unsqueeze(-1)

        b, t = obs.shape[0], obs.shape[1]
        z_s = self.state_encoder(obs)           # [B, T, 256]

        if self.tri_compass:
            if local_delta is not None and origin_delta is not None and goal_delta is not None:
                d_loc = local_delta if local_delta.dim() == 3 else local_delta.unsqueeze(1)
                d_orig = origin_delta if origin_delta.dim() == 3 else origin_delta.unsqueeze(1)
                d_goal = goal_delta if goal_delta.dim() == 3 else goal_delta.unsqueeze(1)
            elif delta_obs.shape[-1] >= 3 * self.obs_dim:
                d_loc = delta_obs[..., :self.obs_dim]
                d_orig = delta_obs[..., self.obs_dim : 2 * self.obs_dim]
                d_goal = delta_obs[..., 2 * self.obs_dim : 3 * self.obs_dim]
            else:
                d_loc = delta_obs[..., :self.obs_dim]
                d_orig = delta_obs[..., :self.obs_dim]
                d_goal = torch.zeros_like(d_loc)

            z_l = self.local_delta_encoder(d_loc)   # [B, T, 64]
            z_o = self.origin_delta_encoder(d_orig) # [B, T, 64]
            z_g = self.goal_delta_encoder(d_goal)   # [B, T, 64]
            z_a = self.action_encoder(prev_action)  # [B, T, 16]
            z_r = self.reward_encoder(prev_reward)  # [B, T, 1]

            # Core features (337d) normalized with out_norm (identical to single-channel)
            z_core = torch.cat([z_s, z_l, z_a, z_r], dim=-1)
            out_core = self.out_norm(z_core)

            # Decoupled compass features (128d) normalized with compass_norm
            z_comp = torch.cat([z_o, z_g], dim=-1)
            out_comp = self.compass_norm(z_comp)

            out = torch.cat([out_core, out_comp], dim=-1) # [B, T, 465]
        else:
            z_d = self.delta_encoder(delta_obs[..., :self.obs_dim]) # [B, T, 64]
            z_a = self.action_encoder(prev_action)                  # [B, T, 16]
            z_r = self.reward_encoder(prev_reward)                  # [B, T, 1]

            z = torch.cat([z_s, z_d, z_a, z_r], dim=-1)             # [B, T, 337]
            out = self.out_norm(z)

        assert out.shape[-1] == self.out_dim, f"Expected {self.out_dim} concat dimensions, got {out.shape[-1]}"
        assert out.shape == (b, t, self.out_dim), f"Expected shape {(b, t, self.out_dim)}, got {out.shape}"
        return out


class _ImpalaConvBlock(nn.Module):
    """Residual convolutional block for IMPALA architecture."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2 residual convolutions with GroupNorm (equivalent to LayerNorm over channels for spatial)
        self.res1_norm = nn.GroupNorm(1, out_channels)
        self.res1_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.res2_norm = nn.GroupNorm(1, out_channels)
        self.res2_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)

        # Res 1
        res = x
        x = self.act(self.res1_norm(x))
        x = self.res1_conv(x)
        x = x + res

        # Res 2
        res = x
        x = self.act(self.res2_norm(x))
        x = self.res2_conv(x)
        x = x + res

        return x


class ImpalaVisualEncoder(nn.Module):
    """Encodes stacked frames [s_t, s_{t-1}] (channels: 2*C) directly to z_vis in R^320."""

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 320,
        channels: tuple[int, int, int] = (16, 32, 32),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        self.block1 = _ImpalaConvBlock(in_channels, channels[0])
        self.block2 = _ImpalaConvBlock(channels[0], channels[1])
        self.block3 = _ImpalaConvBlock(channels[1], channels[2])

        # Compute flatten size with dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            flatten_dim = int(np.prod(x.shape[1:]))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((11, 11))
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(flatten_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # visual_obs shape: [B, T, 2*C, H, W] or [B, 2*C, H, W]
        orig_dim = visual_obs.dim()
        if orig_dim == 4:
            visual_obs = visual_obs.unsqueeze(1)
        b, t, c, h, w = visual_obs.shape
        assert c == self.in_channels, (
            f"Expected {self.in_channels} channels (2*C), got {c}"
        )

        # Flatten B and T to process through CNN
        x = visual_obs.view(b * t, c, h, w)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        x = x.view(b * t, -1)
        z = self.head(x)
        z_vis = z.view(b, t, self.out_dim)

        return z_vis


class ImpalaVisualFrontEnd(nn.Module):
    """Universal Visual Perceptual Front-End.

    Maps:
      [s_t, s_{t-1}] (2*C, 84, 84) -> z_vis in R^320
      a_{t-1}                     -> z_a   in R^16
      r_{t-1}                     -> z_r   in R^1
    Concatenates into R^337 and applies LayerNorm(337).
    """

    def __init__(
        self,
        in_channels: int,
        action_space: gym.spaces.Space,
        reward_fixed: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.action_space = action_space

        self.visual_encoder = ImpalaVisualEncoder(in_channels=in_channels, out_dim=320)
        self.action_encoder = ActionEncoder(action_space, embed_dim=16)
        self.reward_encoder = RewardEncoder(fixed=reward_fixed)
        self.out_dim = 337
        self.out_norm = nn.LayerNorm(337)

    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        delta_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # delta_obs is ignored in visual front-end as channels already contain [s_t, s_{t-1}]
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
        if prev_action.dim() == 1:
            prev_action = prev_action.unsqueeze(1)
        elif prev_action.dim() == 2 and not isinstance(self.action_space, gym.spaces.Discrete):
            prev_action = prev_action.unsqueeze(1)
        if prev_reward.dim() == 1:
            prev_reward = prev_reward.unsqueeze(1).unsqueeze(-1)
        elif prev_reward.dim() == 2:
            prev_reward = prev_reward.unsqueeze(-1)

        b, t = obs.shape[0], obs.shape[1]

        z_vis = self.visual_encoder(obs)        # [B, T, 320]
        z_a = self.action_encoder(prev_action)  # [B, T, 16]
        z_r = self.reward_encoder(prev_reward)  # [B, T, 1]

        z = torch.cat([z_vis, z_a, z_r], dim=-1) # [B, T, 337]
        assert z.shape[-1] == 337, f"Expected 337 concat dimensions, got {z.shape[-1]}"

        out = self.out_norm(z)
        assert out.shape == (b, t, 337), f"Expected shape {(b, t, 337)}, got {out.shape}"
        return out
