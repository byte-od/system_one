"""Script para baixar o Qwen2.5-3B-Instruct com monitoramento de progresso."""

import time
import os
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2.5-3B-Instruct"
print(f"Iniciando o download de {model_id}...")
print("Os pesos somam aproximadamente ~6.18 GB (em formato safetensors bfloat16).")

t0 = time.time()
path = snapshot_download(
    repo_id=model_id,
    repo_type="model",
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
)
dt = time.time() - t0

print(f"\n✓ Download concluído com sucesso em {dt/60:.2f} minutos!")
print(f"Salvo localmente em: {path}")
