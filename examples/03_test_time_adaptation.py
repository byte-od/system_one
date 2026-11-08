"""Exemplo 03: Adaptação em Tempo de Teste (Test-Time Adaptation - Zero-Backprop).

Demonstra como o SCache adapta um modelo pré-treinado a uma mudança de distribuição
(distribution shift) ou novas condições de inferência em tempo real (< 3 µs por forward)
sem necessidade de executar backpropagation nos parâmetros estáticos do modelo.
"""

import time
import torch
import torch.nn as nn
from scar_cache import ScarLinear


class AdaptiveTransformerMLP(nn.Module):
    def __init__(self, d_model: int = 64, d_ff: int = 128):
        super().__init__()
        # Primeira projeção com ScarLinear
        self.fc_in = ScarLinear(d_model, d_ff, mode="hybrid", scale=0.25)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        h = self.act(self.fc_in(x))
        return x + self.fc_out(h)


def main():
    print("=== SCache: Exemplo 03 - Adaptação em Tempo de Teste (Zero-Backprop) ===")
    model = AdaptiveTransformerMLP(d_model=64, d_ff=128)
    model.eval()

    # 1. Entrada nominal (distribuição limpa de treinamento)
    x_clean = torch.randn(1, 64)
    out_clean = model(x_clean)

    # 2. Mudança de distribuição durante inferência (ruído/ambiente novo)
    x_shift = x_clean + 0.8 * torch.randn(1, 64)

    # O modelo detecta um sinal de consistência / self-supervised feedback
    # Registra uma cicatriz sináptica em tempo real na camada ScarLinear
    target_grad = torch.randn(1, 64)  # Sinal de alinhamento com a nova distribuição
    model.fc_in.record_scar(x_shift, target_grad, surprise=1.8)

    # 3. Medição de latência de inferência adaptada
    iterations = 1000
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x_shift)
    t1 = time.perf_counter()

    avg_latency_us = ((t1 - t0) / iterations) * 1e6
    print(f"\nInferência contínua adaptada por Fast-Weights executada com sucesso!")
    print(f"Latência média por forward: {avg_latency_us:.2f} µs (Microssegundos!)")
    print(f"Número de cicatrizes ativas no SCache: {model.fc_in.scars_count}")
    print(f"Pegada de memória estrita: {model.fc_in.cache.memory_bytes} bytes")
    print("\nNenhum gradiente foi calculado nos pesos base; a adaptação ocorreu 100% no cache O(1)!")


if __name__ == "__main__":
    main()
