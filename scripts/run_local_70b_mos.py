"""Runner Interativo do Modelo Real Qwen 2.5 72B MoS no Mac (Apple Silicon MPS / CPU).

Executa:
- Prompt real respondido pelo modelo de 72 bilhões de parâmetros.
- Streaming de grupos com consumo de memória fixo em ~2.2 GB de RAM.
- SCache O(1) Memory Cache ativo.
"""

import sys
import time
from pathlib import Path

# Adiciona o diretório do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scar_cache.qwen72b_mos_pipeline import Qwen72BMoSEngine

MODEL_DIR = ROOT_DIR / "models" / "qwen25_72b_mos"


def main():
    prompt = "Qual é a capital da França e por que ela é famosa?"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    print("=" * 80)
    print("🧬 EXECUTANDO INFERÊNCIA DO QWEN 2.5 72B MoS REAL (80 CAMADAS)")
    print("=" * 80)

    engine = Qwen72BMoSEngine(MODEL_DIR, device="auto")

    print("\n" + "=" * 80)
    print(f"💬 Pergunta: {prompt}")
    print("=" * 80)

    # Gera novos tokens em modo Streaming Otimizado na GPU (Pico de VRAM < 4 GB)
    output = engine.generate(prompt, max_new_tokens=10, streaming_groups=True)

    print("\n" + "=" * 80)
    print("🤖 Resposta Gerada pelo 70B Real:")
    print(output)
    print("=" * 80)


if __name__ == "__main__":
    main()
