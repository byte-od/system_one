# Roadmap & Especificação Arquitetural: Substituição do KV Cache por SCache Multi-Slot

> **Objetivo Primário:** Eliminar o KV Cache de Transformers e LLMs ($O(T)$ em VRAM, dezenas de gigabytes) substituindo-o por um substrato de Memória Associativa Tensorial Multi-Slot com Regra Delta e PCGrad ($O(1)$ estrito, 34 KB residente no L1 Cache).

---

## 🏛️ Os 5 Pilares que Devem Ter Sucesso

Para que o SCache substitua o KV Cache em tarefas de linguagem natural e raciocínio sequencial sem perder acurácia, os seguintes pilares matemáticos devem ser obrigatoriamente satisfeitos:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SCACHE MULTI-SLOT (34 KB NO L1)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Pilar 1] Tensorização Multi-Slot (8x32x32) + PCGrad Local por Slot        │
│            • Elimina cancelamento cruzado entre sub-rotinas                 │
│            • Capacidade multiplicada: 256 direções ortogonais               │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Pilar 2] Regra Delta de Sobrescrita Exata (Widrow-Hoff Anti-Ghosting)      │
│            • ΔW = β · (v - W k) k^T                                         │
│            • Sobrescreve fatos antigos sem deixar fantasmas aditivos        │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Pilar 3] Roteamento Competitivo por Slot (Competitive Ingestion)          │
│            • Cada token escolhe dinamicamente o slot de maior afinidade     │
│            • Especialização funcional espontânea (sintaxe vs fatos vs metas)│
├─────────────────────────────────────────────────────────────────────────────┤
│  [Pilar 4] Decaimento Temporal Heterogêneo Multi-Escala                     │
│            • γ_k diferenciado por slot (curto, médio e longo prazo)         │
│            • Preserva ordem e hierarquia temporal sem estourar parâmetros   │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Pilar 5] Modo Dual de Execução (Inferência O(1) + Treino Paralelo)         │
│            • Inferência: < 5 µs recorrente passo a passo                    │
│            • Treinamento: Parallel Associative Scan para GPUs modernas      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Pilar 1: Tensorização Multi-Slot ($K \times D \times D$) com PCGrad por Slot
* **Desafio:** Uma matriz 2D ($32 \times 32$) faz a média de todos os slots, causando cancelamento de direções opostas.
* **Solução:** Expandir para um tensor de ordem 3:
  $$W_{scar} \in \mathbb{R}^{K \times D \times D}$$
  onde cada slot $k \in \{0, \dots, K-1\}$ possui sua própria bússola consensual $g_{target}^{(k)} \in \mathbb{R}^D$ e centróide de manifold $z_{target}^{(k)} \in \mathbb{R}^D$.
* **Critério de Sucesso:**
  - O gradiente do Slot $i$ é ortogonalizado apenas contra $g_{target}^{(i)}$, sem interferência no Slot $j$.
  - Consumo total: exatamente $34\text{ KB}$ em float32 (ou $17\text{ KB}$ em float16), cabendo 100% no Cache L1.

---

### Pilar 2: A Regra Delta de Sobrescrita Exata (Anti-Ghosting)
* **Desafio:** A acumulação Hebbiana pura ($W \leftarrow W + g z^T$) adiciona informação indefinidamente. Se um fato é retificado no texto, o valor antigo permanece sobreposto, gerando alucinações.
* **Solução:** Implementar a **Regra Delta**:
  $$e_t^{(k)} = \mathbf{v}_t^{(k)} - W^{(k)} \mathbf{k}_t^{(k)}$$
  $$\Delta W^{(k)} = \beta_t^{(k)} e_t^{(k)} (\mathbf{k}_t^{(k)})^T$$
  Onde $e_t^{(k)}$ é o erro de recuperação da memória atual. Se a chave $\mathbf{k}_t$ já possui um valor antigo em $W$, ele é subtraído com precisão cirúrgica antes de registrar o novo.
* **Critério de Sucesso:**
  - Teste de Retificação de Fatos: Gravar `Chave="Capital" -> Valor="Curitiba"` e depois `Chave="Capital" -> Valor="Madri"`. Ao consultar `Chave="Capital"`, a recuperação deve convergir para `"Madri"` com erro residual $< 10^{-4}$.

---

### Pilar 3: Roteamento Competitivo por Slot (Competitive Ingestion)
* **Desafio:** O fluxo de tokens de texto não possui rótulo pré-definido de qual slot deve receber cada representação.
* **Solução:** Mecanismo de despacho diferenciável via projeção de chaves de slot:
  $$\alpha_t^{(k)} = \text{softmax}\left(\frac{\mathbf{x}_t \cdot \mathbf{q}_k}{\sqrt{D}}\right)$$
  Onde $\alpha_t^{(k)}$ atua como a taxa de escrita seletiva no slot $k$.
* **Critério de Sucesso:**
  - Especialização espontânea observável: tokens sintáticos concentrados em slots de decaimento rápido, entidades e instruções em slots de decaimento longo.

---

### Pilar 4: Decaimento Temporal Heterogêneo Multi-Escala
* **Desafio:** Manter o histórico de 100.000 tokens sem perder a sensibilidade para as últimas 3 palavras.
* **Solução:** Vetor diagonal de decaimento $\boldsymbol{\gamma} = [\gamma_0, \dots, \gamma_{K-1}]$:
  $$W_{t+1}^{(k)} = \gamma_k W_t^{(k)} + \Delta W_t^{(k)}$$
  com valores escalonados geometricamente:
  $$\gamma_k = \exp\left(-\frac{\lambda}{2^k}\right)$$
  - Slots 0-1: $\gamma \approx 0.50$ (memória ultrarrápida / últimos 5 tokens).
  - Slots 2-4: $\gamma \approx 0.95$ (parágrafo atual).
  - Slots 5-7: $\gamma \approx 0.9999$ (objetivo, regras de sistema, contexto macro).
* **Critério de Sucesso:**
  - Capacidade simultânea de lembrar instruções do início do texto e manter concordância gramatical com o token imediatamente anterior.

---

### Pilar 5: Modo Dual de Execução (Inferência $O(1)$ + Treino Paralelo)
* **Desafio:** O loop recorrente token a token é rápido na inferência ($< 5\ \mu\text{s}$), mas inviabiliza o treinamento massivo em clusters de GPU se não for paralelizável no tempo.
* **Solução:** Formulação da recorrência como um **Parallel Associative Scan**:
  Como a operação $W_t = \gamma_t W_{t-1} + \Delta W_t$ é estritamente associativa, ela pode ser computada em $O(\log T)$ passos de tempo em GPU durante o treinamento e em $O(1)$ na inferência.
* **Critério de Sucesso:**
  - Idênticos resultados numéricos entre o modo paralelo (treino) e o modo recorrente (inferência) com tolerância de precisão de float32 ($10^{-5}$).

---

## 📈 Metas de Benchmark para Aposentadoria do KV Cache

1. **Memória de Cache por Camada:** Redução de **~140 MB/usuário** para **34 KB/usuário fixo** ($4.000\times$ menor).
2. **Latência de Geração:** Constante do token 1 ao token 1.000.000 (eliminando o *Time-to-First-Token* e *Time-per-Output-Token* degradados por leitura de VRAM).
3. **Teste da Agulha no Palheiro (*Needle-in-a-Haystack*):** Acurácia de recuperação $> 98\%$ em contextos de $64\text{k}$ tokens para dados roteados aos slots de longo prazo com Regra Delta.
