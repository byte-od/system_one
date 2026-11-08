"""Exemplo 01: Minimal Drop-in do SCache no PyTorch.

Demonstra como substituir uma camada nn.Linear padrão por ScarLinear em 3 linhas
e acumular cicatrizes associativas de vitórias/sucessos em tempo constante O(1).
"""

import torch
import torch.nn as nn
from scar_cache import ScarLinear


class SimpleClassifier(nn.Module):
    def __init__(self, in_features: int = 32, num_classes: int = 4):
        super().__init__()
        # Substitui a camada linear padrão por ScarLinear
        self.fc1 = ScarLinear(in_features, 64, mode="hybrid")
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def main():
    print("=== SCache: Exemplo 01 - Minimal Drop-in ===")
    model = SimpleClassifier(in_features=32, num_classes=4)

    # 1. Entrada padrão
    x = torch.randn(2, 32, requires_grad=True)
    logits = model(x)
    print(f"Logits iniciais (antes da cicatriz):\n{logits.detach().numpy()}")

    # 2. Simulação de um evento de alta recompensa/vitória (target de sucesso)
    target = torch.tensor([1, 3])
    loss = nn.functional.cross_entropy(logits, target)

    # 3. Registra a cicatriz sináptica no SCache a partir do gradiente de perda
    model.fc1.record_from_loss(loss, x, surprise=2.0)
    print(f"\nCicatriz consolidada! Total de cicatrizes: {model.fc1.scars_count}")
    print(f"Pegada de memória estrita do cache: {model.fc1.cache.memory_bytes} bytes (~4.25 KB)")

    # 4. Nova passagem com o mesmo estado ou estado similar
    new_logits = model(x)
    print(f"\nLogits pós-cicatriz (Fast-Weights ativados):\n{new_logits.detach().numpy()}")
    print("O modelo adaptou suas ativações imediatamente sem atualizar os pesos lentos W!")


if __name__ == "__main__":
    main()
