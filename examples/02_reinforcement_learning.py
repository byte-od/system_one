"""Exemplo 02: SCache em Reinforcement Learning (RL) com Recompensa Esparsa.

Demonstra como o SCache substitui integralmente buffers de replay episódico de gigabytes
por uma matriz associativa constante O(1) de 4.25 KB, consolidando vitórias raras instantaneamente.
"""

import torch
import torch.nn as nn
from scar_cache import SCache


class RLActorCritic(nn.Module):
    def __init__(self, state_dim: int = 32, action_dim: int = 6):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Backbone de features
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        # SCache integrado nas features latentes
        self.scar_cache = SCache(slot_dim=32, num_slots=1, mode="hybrid", scale=0.2, resonance_beta=0.25)

        # Cabeças de política (Actor) e valor (Critic)
        self.actor_head = nn.Linear(32, action_dim)
        self.critic_head = nn.Linear(32, 1)

    def forward(self, state: torch.Tensor):
        z = self.encoder(state)
        # Modulação Fast-Weights + Gating de Manifold
        z_guided, delta, affinity = self.scar_cache.modulate(z)

        logits = self.actor_head(z_guided)
        value = self.critic_head(z_guided)

        # Modulação analítica de nitidez dos logits por ressonância
        if delta is not None:
            res = self.scar_cache.compute_resonance(delta)
            logits = self.scar_cache.modulate_logits(logits, res)

        return logits, value, z


def main():
    print("=== SCache: Exemplo 02 - Reinforcement Learning (RL) ===")
    policy = RLActorCritic(state_dim=32, action_dim=6)

    # 1. Agente interage com ambiente e obtém estado
    state = torch.randn(1, 32)
    logits, value, z = policy(state)
    probs = torch.softmax(logits, dim=-1)
    print(f"Probabilidades de ação iniciais (alta entropia):\n{probs.detach().numpy().round(3)}")

    # 2. Suponha que após uma busca ou golpe de sorte, o agente atinge a meta rara (Recompensa +1.0)
    # Em vez de guardar 1.000.000 de transições em um buffer de replay de 2 GB:
    reward = 1.0
    td_surprise = 2.5  # TD-error alto / surpresa elevada de Shannon + Bellman

    # Criamos um target de vitória para guiar a descida de gradiente no espaço latente
    best_action = torch.tensor([4])
    loss = nn.functional.cross_entropy(logits, best_action)

    # Calcula gradiente no espaço latente z
    grad_z = -torch.autograd.grad(loss, z, retain_graph=True)[0]

    # Consolida a cicatriz no SCache O(1)
    policy.scar_cache.record_scar(z.detach(), grad_z.detach(), surprise=td_surprise)
    print(f"\n[Evento de Vitória Rara Registrado no SCache]")
    print(f"Cicatrizes gravadas: {policy.scar_cache.scars_count.item()}")
    print(f"Pegada de memória estrita: {policy.scar_cache.memory_bytes} bytes (vs ~2 GB em Replay Buffer)")

    # 3. Agente revisita o mesmo estado ou estado vizinho no manifold
    new_logits, new_val, _ = policy(state)
    new_probs = torch.softmax(new_logits, dim=-1)
    print(f"\nProbabilidades de ação após a cicatriz (priorização instantânea da rota vitoriosa):")
    print(f"{new_probs.detach().numpy().round(3)}")
    print(f"Probabilidade da ação vencedora (ação {best_action.item()}): de {probs[0, 4]:.3f} -> {new_probs[0, 4]:.3f}")


if __name__ == "__main__":
    main()
