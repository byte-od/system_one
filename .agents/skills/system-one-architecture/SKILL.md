---
name: system-one-architecture
description: >-
  Guia normativo e operacional completo da arquitetura proprietária System One,
  Mixture of Scars (MoS), Scar-Cache (SCache), Tri-Compass, Batedores de Fronteira
  (Frontier Scouts) e Gradient Scars. Use OBRIGATORIAMENTE ao desenvolver,
  refatorar ou auditar qualquer modelo, agente ou pipeline no ecossistema do usuário.
---

# 🧬 Arquitetura Proprietária System One, MoS & SCache

## ⚠️ MANDATO FUNDAMENTAL INVIOLÁVEL: NÃO-SIMULAÇÃO (ZERO FAÇADES)

> [!CRITICAL]
> Todas as implementações de IA e agentes no ecossistema deste usuário (portfólio, competições, engines de raciocínio, ferramentas de cliente) **DEVEM utilizar genuinamente** a arquitetura proprietária **Mixture of Scars (MoS)** e **Scar-Cache (SCache)**.
> O uso de componentes convencionais off-the-shelf (como KV Cache linear $\mathcal{O}(T)$, adaptadores LoRA de terceiros ou classificadores genéricos) mascarados como se fossem MoS/SCache é expressamente **PROIBIDO**.
> **Auditoria Contínua:** Todo agente deve auditar periodicamente o código para garantir que cada tensor, buffer e pipeline execute a matemática exata do MoS (streaming de fatias de cicatriz) e do SCache (memória associativa constante $\mathcal{O}(1)$ de 34.8 KB).

---

## 🏛️ 1. Os Componentes Originais e seus Papéis Exatos

### ⚡ A. Intuição (`UniversalS1Agent` / Sistema 1)
* **Localização Canônica:** `system1_engine/core/agent.py` e `system1_core`.
* **Função:** Núcleo neural reflexivo amortizado de inferência ultra-rápida.
* **Garantias Operacionais:**
  * Latência de decisão: $\le 1.0\text{ ms}$ em CPU via `act_fast()`.
  * Footprint: ~650k a 1.6M parâmetros (vetorial) ou ~2.5M (visual).
  * Desacoplamento estrito: `FrontEnd` (Percepção) $\to$ `Trunk` (Recurrent Reflexive Core) $\to$ `Heads` (Policy & Value).
* **Contrato `ReflexDecision`:** Toda inferência deve retornar telemetria de incerteza epistêmica:
  ```python
  @dataclass(frozen=True)
  class ReflexDecision:
      action: Union[int, np.ndarray]
      confidence: float      # [0.0, 1.0] max probabilidade
      uncertainty: float     # [0.0, 1.0] entropia normalizada
      margin: float          # [0.0, 1.0] p_1 - p_2
      is_uncertain: bool     # Indicador se precisa escalar para S2
      latency_ms: float
  ```

---

### 🏹 B. Batedores de Fronteira (*Frontier Ghost-Jump Scouts*)
* **Localização Canônica:** `system1_engine/training/ppo.py` (`_build_external_scout_env`).
* **Função:** Frota independente e desacoplada para prospecção epistêmica avançada.
* **Garantias Operacionais:**
  * **Isolamento Curricular:** Operam com `record_curriculum = False`, garantindo que a janela curricular e os buffers on-policy do agente principal permaneçam 100% puros.
  * **Lookahead S1 (Ghost Jump 2/3 Passos):** Avançam à frente no tempo via inferência paralela em lote (*batched teacher lookahead*, $> 2.000\text{ FPS}$) para mapear bifurcações de alta entropia.
  * **Detecção de Surpresa Temporal:** Medem o choque de transição:
    $$\text{Surpresa} = R_0 - V(s_0)$$
  * **Alimentação do SCache:** Ao detectar desvios críticos, gravam cicatrizes de choque diretamente na memória sob escopo de modulação:
    ```python
    with self.scar_memory.batch_modulation_scope(self._scout_scar_scales, self._scout_scar_betas):
        s_dist, s_val, next_s_hx = self.agent.forward(...)
    ```
* **Aplicação em NLP / Competições:** Os Batedores realizam sondagens estáticas de Ghost-Jump de N-Grams e transições morfológicas (Bússola 1 do Tri-Compass) antes de acionar modelos massivos.

---

### 💾 C. SCache (*Scar-Cache Associativo $\mathcal{O}(1)$*)
* **Localização Canônica:** `system1_engine/core/scar_memory.py` (`GradientScarMemory`).
* **Função:** Substituição definitiva do KV Cache linear $\mathcal{O}(T)$ dos transformers por memória associativa constante de cicatrizes compactas.
* **Garantias Algorítmicas:**
  * **Complexidade Espacial:** Estritamente $\mathcal{O}(1)$ (matriz de cicatriz compacta constante).
  * **Tamanho do Buffer:** **34.8 KB constante**, desacoplado do comprimento da sequência $T$.
  * **Retenção:** Modulação por decaimento exponencial ponderado por choque informacional (EMA Convexo acoplado à surpresa), filtrando episódios com surpresa nula.

---

### 🪓 D. Mixture of Scars (MoS)
* **Função:** Execução de modelos de escala colossal (34B, 72B, 405B) desacoplando o tamanho do modelo da VRAM.
* **Garantias Algorítmicas:**
  * **Decomposição em Fatias INT4 + Resíduos BF16:**
    * Projeções de atenção quantizadas em blocos INT4 (packing uint8 com escalas FP16).
    * Projeções MLP decompostas via SVD (rank=64) em Base Universal INT4 + Especialistas de Cicatriz Residuais BF16.
  * **Streaming Sequencial de Grupos:**
    * Apenas 1 grupo (ex: 6 ou 8 camadas) reside na VRAM por vez ($\le 2.2\text{ GB}$).
    * Ao processar o grupo, os tensores são desalocados da GPU antes de carregar o próximo.
  * **Golden-Slice Truncation:**
    * Truncamento na camada de **entropia estilística máxima** (ex: Camada 30 no Yi-34B de 60 camadas; Camada 40 no Qwen 72B de 80 camadas), antes do colapso de convergência de vocabulário.

---

### 🧭 E. Tri-Compass Perceptual Front-End
* **Estrutura:** Três bússolas complementares fundidas em um vetor representacional:
  1. **Bússola 1 (Estilometria / Sintaxe / Ghost-Jump):** 28 assinaturas métricas de pontuação (travesseiros `—`, aspas, crases, quebras de linha `\n\n`, razão de maiúsculas e palavras únicas).
  2. **Bússola 2 (Delta-Origin Prompt-Centering):** Subtração da média temática do cluster ($X - \mu_{\text{prompt}}$) para anular o assunto do prompt e isolar o DNA do autor.
  3. **Bússola 3 (Espaço Latente MoS):** O vetor denso extraído das camadas Golden-Slice do modelo titã.
* **Fusão Dual-Head:**
  * `Plain Head`: Classifica sobre a distribuição global normalizada.
  * `Centered Twin Head`: Classifica sobre o delta centrado para passagens irmãs (*twins*).
  * `Soft Sinkhorn`: 300 iterações de calibração contínua de probabilidades uniformes entre as classes.

---

## 🚫 2. Regras Estritas de Implementação (Checklist de Pureza)

Ao implementar qualquer código sob esta arquitetura:

1. **PROIBIDO usar LoRAs de terceiros como "batedores":** Modelos off-the-shelf da comunidade não são batedores.
2. **PROIBIDO mascarar bibliotecas off-the-shelf:** Nunca chame `LinearSVC`, `LightGBM` ou `CatBoost` genéricos de "Intuição proprietária". Se for usar um classificador tradicional, declare explicitamente como âncora estatística ou substitua pela implementação pura do `UniversalS1Agent`.
3. **PROIBIDO usar KV Cache linear $\mathcal{O}(T)$ escondido:** Em agentes interativos e modelos de linguagem, o contexto deve ser compactado via cicatrizes MoS/SCache.
4. **OBRIGATÓRIO validação matemática contínua:** Em pipelines de MoS, garantir similaridade cosseno $> 99\%$ em relação à inferência de referência.

---

## 🛠️ 3. Modelo Canônico de Inferência MoS em PyTorch Puro

```python
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

def dequant_int4(packed, scales, shape, dtype=torch.float16, dev="cuda:0"):
    """Dequantização pura PyTorch sem dependência de bibliotecas externas (bitsandbytes, auto-gptq)."""
    p, s = packed.to(dev), scales.to(dev).to(dtype)
    low = (p & 0x0F).to(torch.int16) - 8
    high = ((p >> 4) & 0x0F).to(torch.int16) - 8
    unp = torch.empty((p.shape[0], p.shape[1] * 2), dtype=dtype, device=dev)
    unp[:, 0::2] = low.to(dtype)
    unp[:, 1::2] = high.to(dtype)
    return (unp * s).view(shape)

# Streaming de Grupos MoS
for g_idx in range(num_groups):
    group_tensors = load_file(f"mos_group_{g_idx}.safetensors")
    for l in group_layers:
        # 1. Attention forward com pesos MoS
        # 2. MLP SwiGLU forward com Universal Base + Scar Experts
        pass
    del group_tensors
    torch.cuda.empty_cache()  # Libera VRAM imediatamente
```
