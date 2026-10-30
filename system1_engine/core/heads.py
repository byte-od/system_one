from typing import Optional, Tuple, Union
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


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Função logarítmica simétrica contínua e diferenciável em ℝ."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(y: torch.Tensor) -> torch.Tensor:
    """Inversa analítica da função symlog."""
    return torch.sign(y) * (torch.exp(torch.abs(y)) - 1.0)


def two_hot_encode(
    scalar_targets: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """Converte um tensor escalar contínuo em uma distribuição two-hot sobre os bins discretos."""
    device = scalar_targets.device
    num_bins = bins.shape[0]

    # Garante que os alvos estejam dentro da faixa dos bins
    clamped = scalar_targets.clamp(min=bins[0].item(), max=bins[-1].item())

    # Localiza o índice do bin imediatamente inferior via bucketize
    bin_idx = torch.bucketize(clamped, bins) - 1
    bin_idx = bin_idx.clamp(min=0, max=num_bins - 2)

    lower_bin = bins[bin_idx]
    upper_bin = bins[bin_idx + 1]

    # Pesos lineares complementares que somam exatamente 1.0
    weight_upper = (clamped - lower_bin) / (upper_bin - lower_bin + 1e-8)
    weight_lower = 1.0 - weight_upper

    # Cria o tensor two-hot esparso
    target_dist = torch.zeros(*scalar_targets.shape, num_bins, device=device)
    target_dist.scatter_(-1, bin_idx.unsqueeze(-1), weight_lower.unsqueeze(-1))
    target_dist.scatter_(-1, (bin_idx + 1).unsqueeze(-1), weight_upper.unsqueeze(-1))

    return target_dist


class ValueHead(nn.Module):
    """Cabeça de Crítico Categórica Two-Hot Symlog (DreamerV3 / MuZero style).

    Substitui a regressão MSE escalar por entropia cruzada sobre partição de bins symlog.
    Elimina explosão de gradientes (gradientes estritamente limitados em [-1, 1]) e é invariante à escala.
    """

    def __init__(
        self,
        in_features: int = 256,
        num_bins: int = 255,
        v_min: float = -50.0,
        v_max: float = 500.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max

        # Bins uniformemente distribuídos no espaço symlog
        bins = torch.linspace(
            symlog(torch.tensor(float(v_min))),
            symlog(torch.tensor(float(v_max))),
            num_bins,
        )
        self.register_buffer("bins", bins)

        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Linear(in_features, num_bins),
        )
        self._last_logits: Optional[torch.Tensor] = None

    @property
    def last_logits(self) -> Optional[torch.Tensor]:
        return self._last_logits

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Retorna o valor escalar contínuo decodificado V(s) com shape (*h.shape[:-1], 1)."""
        logits = self.net(h)
        self._last_logits = logits
        probs = torch.softmax(logits, dim=-1)
        expected_symlog = torch.sum(probs * self.bins, dim=-1, keepdim=True)
        return symexp(expected_symlog)

    def forward_with_logits(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna (valor_escalar, logits, entropia_incerteza)."""
        logits = self.net(h)
        self._last_logits = logits
        probs = torch.softmax(logits, dim=-1)
        expected_symlog = torch.sum(probs * self.bins, dim=-1, keepdim=True)
        expected_val = symexp(expected_symlog)
        entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-8)), dim=-1)
        return expected_val, logits, entropy

    def loss(self, logits: torch.Tensor, continuous_targets: torch.Tensor) -> torch.Tensor:
        """Calcula perda de Cross-Entropy contra a projeção Two-Hot dos alvos contínuos."""
        target_symlog = symlog(continuous_targets)
        target_dist = two_hot_encode(target_symlog, self.bins)
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(target_dist * log_probs).sum(dim=-1).mean()
