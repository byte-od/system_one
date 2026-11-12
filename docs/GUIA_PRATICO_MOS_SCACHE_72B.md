# Guia Prático Oficial — Executando `Qwen2.5-72B-Instruct` (`72.71B`) Localmente com `MoS v2 + SCache O(1)`

Este guia mostra passo a passo como baixar, auditar e executar localmente o modelo **`Qwen/Qwen2.5-72B-Instruct` (`72.71 bilhões` de parâmetros, `80` camadas, `145.4 GB` em BF16 denso)** em notebooks e desktops convencionais (**Apple Silicon M1/M2/M3/M4 de `8 GB` / `16 GB+`**, **GPUs NVIDIA de `8 GB` a `24 GB+`** ou **CPU**) usando a arquitetura genuína:

1. **Mixture of Scars (`MoS v2` Canônico):** Decomposição de cada projeção linear (`240` matrizes SwiGLU MLP `[29568, 8192]` + `320` projeções de Atenção GQA) em **Base Simétrica `Block-64 INT4`** acoplada a **Especialistas Residuais de Baixo Posto ponderados por ativação (`ASVD`, `rank=48` MLP / `rank=32` Atenção)**, preservando canais outliers emergentes (`>6.7B`) em **20 grupos de 4 camadas (`1.79 GiB` por shard)**.
2. **Scar-Cache (`SCacheDynamicCache` $\mathcal{O}(1)$):** Substituição do KV Cache linear $\mathcal{O}(T)$ por uma janela física fixa combinada a **memória associativa ortogonal online de Widrow-Hoff** (`ΔBytes = +0.00 B` para qualquer comprimento de contexto $L > W$).

---

## 1. Requisitos de Hardware e Dimensionamento Real

| Plataforma Local | Modo Recomendado | Memória Ativa por Grupo (`MoS`) | Memória de Contexto (`SCache O(1)`, `W=256, S=24`) | Espaço em Disco SSD |
|---|---|---|---|---|
| **Apple Silicon (`8 GB` / `16 GB` M1–M4)** | `--device mps` (Streaming) | **`1.79 GiB`/shard** (`68.57 MiB` pós-unload) | **`128.51 MiB` constantes** (`68.51 MiB` em `W=64`) | `38.2 GiB` livres |
| **Apple Silicon (`64 GB+` Max / Ultra)** | `--device mps --resident` | `38.12 GiB` residentes em Unified RAM | **`128.51 MiB` constantes** (`+0.00 B` em `4k–128k` tok) | `38.2 GiB` livres |
| **GPU Consumidora NVIDIA (`8 GB–24 GB`)** | `--device cuda` (Streaming) | **`1.79 GiB`/shard VRAM** | **`128.51 MiB` VRAM constantes** | `38.2 GiB` livres |
| **Desktop / Servidor (`64 GB+` RAM / Multi-GPU)** | `--device cuda --resident` | `38.12 GiB` residentes | **`128.51 MiB` constantes** | `38.2 GiB` livres |

> **Telemetria Real Verificada em Apple Silicon M2 (`16.0 GB` Unified RAM, `MPS` — [`results/scale_72b_audit/apple_m2_live_72b_generation.json`](../results/scale_72b_audit/apple_m2_live_72b_generation.json)):** Em máquinas de `8 GB` / `16 GB` onde `38.12 GiB` excedem a RAM física disponível, o motor executa as `80` camadas em `20` grupos de `4` camadas (`1.79 GiB`/shard), liberando cada grupo após o forward (**`0.51 GiB` de pico de RSS Python** e **`68.57 MiB` alocados no `MPS` após unload de cada grupo**). No MacBook Apple M2 (`16 GB`), o prefill em lote de `154` tokens nas `80` camadas rodou a **`2.87 tok/s` (`53.70 s`)**, mantendo o `SCache O(1)` (`W=64, S=24`) travado em **`71,838,720 B` (`68.51 MiB`) com `0 B` de deriva em todos os passos de decodificação** e recuperando com exatidão o código evictado **`7942-OMEGA`** (`107` tokens fora da janela). Na decodificação batch-1 em streaming de disco (`16 GB` RAM), a latência é limitada pela leitura sequencial dos `38.12 GiB` no SSD NVMe (`44.04 s/tok`); em máquinas com `>= 48 GB` de RAM/VRAM unificada, basta adicionar `--resident` para eliminar `100%` do I/O de disco por token.

---

## 2. Instalação Rápida (`< 1 minuto`)

```bash
git clone https://github.com/byte-od/mixture-of-scars.git
cd mixture-of-scars
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e . "transformers>=5.0.0" "safetensors>=0.4.3" "accelerate>=0.34.0" "huggingface_hub>=0.26.0"
```

---

## 3. Download Público Oficial do Checkpoint `Qwen2.5-72B-Instruct MoS v2` (`38.12 GiB`)

Todos os `22` shards `.safetensors` (`mos_embed_tokens.safetensors`, `mos_group_00.safetensors` .. `mos_group_19.safetensors`, `mos_lm_head.safetensors`) possuem **`< 1.80 GiB` cada** e assinatura criptográfica **SHA-256** verificada automaticamente no manifesto [`mos_manifest.json`](../models/qwen25_72b_mos_v2/mos_manifest.json).

### Opção A — Download Direto HTTPS (Zero Login / Com Retomada Automática `Range`)
```bash
python -m scar_cache.cli_72b \
  --download \
  --model-dir ./models/qwen25_72b_mos_v2
```
*(Se a conexão oscilar, basta rodar o mesmo comando novamente: shards já verificados por SHA-256 são pulados instantaneamente e shards parciais retomam do byte exato via `HTTP 206 Range`).*

### Opção B — Download Direto via Navegador / `curl` / `aria2c` Paralelo (Zero Login)
Todos os `29` arquivos (`mos_manifest.json`, `6` arquivos de tokenizador/configuração e os `22` shards `.safetensors`) estão disponíveis publicamente em:
- **Página Oficial da Release:** https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos
- **Padrão de URL Direta CDN (`HTTP 206 Range`):** `https://github.com/byte-od/mixture-of-scars/releases/download/v1.2.0-qwen72b-mos/<nome_do_arquivo>`

Após baixar manualmente ou via gerenciador paralelo (`aria2c -x 16`) para `./models/qwen25_72b_mos_v2`, rode `--download` para validar instantaneamente os `22` checksums **SHA-256** antes de executar!

---

## 4. Executando Localmente (`CLI` Interativo e Prompt Direto)

### 4.1 Pergunta Única (`--prompt`)
```bash
python -m scar_cache.cli_72b \
  --model-dir ./models/qwen25_72b_mos_v2 \
  --prompt "Implemente em Python o algoritmo de Dijkstra com fila de prioridade e explique a complexidade:" \
  --max-new-tokens 128 \
  --temperature 0.0
```

### 4.2 Chat Multi-Turno no Terminal com Telemetria `SCache O(1)` ao Vivo (`--chat`)
```bash
python -m scar_cache.cli_72b \
  --model-dir ./models/qwen25_72b_mos_v2 \
  --chat \
  --window-size 256 \
  --num-slots 24
```
Comandos internos durante o chat:
- Digite sua mensagem normalmente para continuar a conversa acumulando contexto na memória associativa `SCache` com **crescimento físico de `0.00 Bytes`** após preencher a janela `W`.
- Digite `reset` para limpar as cicatrizes associativas do `SCache`.
- Digite `exit` ou `quit` para encerrar.

### 4.3 Ajustando a Fronteira 3D de Pareto (`Memória x Qualidade x Velocidade`)
Você pode controlar diretamente o ponto de operação do `SCache O(1)` nas `80` camadas (`8` KV heads, `head_dim=128`) via flags:
- **Ultra-Compacto (`64.73 MiB` em `S=12` / `68.51 MiB` em `S=24`):** `--window-size 64 --num-slots 24`
- **Equilibrado Padrão (`128.51 MiB` exatos em 80 camadas):** `--window-size 256 --num-slots 24`
- **Alta Retenção de Contexto Longo (`211.03 MiB` exatos em 80 camadas):** `--window-size 512 --num-slots 32`

---

## 5. Uso Programático em Python (`CanonicalMoSEngine` + `SCacheDynamicCache`)

```python
from scar_cache.cli_72b import CanonicalMoSEngine

# Inicializa o motor 72.7B em modo streaming (< 1.95 GiB por grupo ativo em MPS/CUDA)
engine = CanonicalMoSEngine(
    model_dir="./models/qwen25_72b_mos_v2",
    device="auto",       # Seleciona automaticamente cuda:0 -> mps -> cpu
    streaming=True,      # Use False em máquinas com >= 48 GB de RAM/VRAM livre
)

# Cria o cache associativo estritamente constante O(1) (Widrow-Hoff Fast Weights)
scache = engine.make_scache(window_size=256, num_slots=24, num_sinks=4)

resposta, scache = engine.generate_stream(
    prompt="Explique por que canais outliers emergentes em SwiGLU exigem cicatrizes SVD de baixo posto:",
    scache=scache,
    max_new_tokens=128,
    temperature=0.0,
)

print(f"Bytes físicos constantes do SCache: {scache.total_memory_bytes:,} B")
```

---

## 6. Como Converter Qualquer Modelo `Qwen2` / `Qwen2.5` (`0.5B` → `72B`) Você Mesmo

Toda a pipeline de conversão canônica é 100% aberta e reprodutível via [`scripts/export_canonical_72b_mos.py`](../scripts/export_canonical_72b_mos.py):

### Teste Local Instantâneo em `0.5B` ou `7B`:
```bash
python scripts/export_canonical_72b_mos.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --out-dir ./models/qwen25_7b_mos_v2 \
  --layers-per-group 4 \
  --mlp-rank 48 \
  --attn-rank 32

python -m scar_cache.cli_72b \
  --model-dir ./models/qwen25_7b_mos_v2 \
  --chat
```

### Conversão Completa do `Qwen/Qwen2.5-72B-Instruct` (Único Comando ou Multi-Partes):
```bash
python scripts/export_canonical_72b_mos.py \
  --model-id Qwen/Qwen2.5-72B-Instruct \
  --out-dir ./models/qwen25_72b_mos_v2 \
  --layers-per-group 4 \
  --mlp-rank 48 \
  --attn-rank 32
```
