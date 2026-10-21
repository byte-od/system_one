from typing import Tuple, Union
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class CategoricalPolicyHead(nn.Module):
    """Categorical policy head for discrete action spaces with Action Chunking Temporal (ACT) support."""

    def __init__(
        self,
        in_features: int = 256,
        n_actions: int = 2,
        action_chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_actions = n_actions
        self.action_chunk_size = max(1, int(action_chunk_size))
        self.linear = nn.Linear(in_features, n_actions * self.action_chunk_size)

    def forward(self, h: torch.Tensor) -> Categorical:
        raw_logits = self.linear(h)
        if self.action_chunk_size > 1:
            chunk_shape = (*h.shape[:-1], self.action_chunk_size, self.n_actions)
            logits = raw_logits.view(chunk_shape)
        else:
            logits = raw_logits
        return Categorical(logits=logits)

    def get_action_and_log_prob(
        self, h: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(h)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        if self.action_chunk_size > 1:
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.mean(dim=-1)
        return action, log_prob, entropy


class GaussianPolicyHead(nn.Module):
    """Diagonal Gaussian policy head for continuous action spaces with Action Chunking Temporal (ACT) support."""

    def __init__(
        self,
        in_features: int = 256,
        act_dim: int = 1,
        init_log_std: float = 0.0,
        action_chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.act_dim = act_dim
        self.action_chunk_size = max(1, int(action_chunk_size))
        self.mu_net = nn.Linear(in_features, act_dim * self.action_chunk_size)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, h: torch.Tensor) -> Normal:
        raw_mu = self.mu_net(h)
        log_std = torch.clamp(self.log_std, min=-20.0, max=2.0)
        if self.action_chunk_size > 1:
            mu = raw_mu.view(*h.shape[:-1], self.action_chunk_size, self.act_dim)
            if log_std.ndim == 2:
                std = log_std.exp().view(*([1] * (mu.dim() - 2)), self.action_chunk_size, self.act_dim).expand_as(mu)
            else:
                std = log_std.exp().view(*([1] * (mu.dim() - 1)), self.act_dim).expand_as(mu)
        else:
            mu = raw_mu
            std = log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def get_action_and_log_prob(
        self, h: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(h)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        if self.action_chunk_size > 1:
            log_prob = dist.log_prob(action).sum(dim=(-2, -1))
            entropy = dist.entropy().sum(dim=(-2, -1)) / self.action_chunk_size
        else:
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class ValueHead(nn.Module):
    """Critic head estimating expected return V(s)."""

    def __init__(self, in_features: int = 256) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)
