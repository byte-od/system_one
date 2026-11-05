# Arquitetura de Lookahead em Funil Guiado por Intuição & Cicatrizes + Otimização Extrema de CPU

**Data:** 05/11/2026  
**Modelo de Referência:** `s1_rubiks_hybrid_pure_30m_final.pt` (30M passos, Tabula Rasa Pura, Fronteira D9, 321.675 cicatrizes puras em $O(1)$ / 4,25 KB)

---

## 1. Benchmark Comparativo Base: Reflexo Puro vs. 1-Step Lookahead (D9 a D20)

Avaliação pareada com **100 episódios por profundidade** (sementes idênticas `seed = 100000 * d + ep`, totalizando 1.200 cubos por método):

| Profundidade | Reflexo Puro (%) | Média Passos | 1-Step Lookahead (%) | Média Passos | Delta Absoluto | Ganho Relativo |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D09** | **22%** (22/100) | 12.0p | **31%** (31/100) | 12.3p | **+9%** | +40.9% |
| **D10** | **13%** (13/100) | 13.8p | **21%** (21/100) | 13.3p | **+8%** | +61.5% |
| **D11** | **5%** (5/100) | 8.6p | **11%** (11/100) | 14.5p | **+6%** | **+120.0%** |
| **D12** | **2%** (2/100) | 21.0p | **4%** (4/100) | 19.0p | **+2%** | **+100.0%** |
| **D13** | **2%** (2/100) | 19.0p | **2%** (2/100) | 19.0p | +0% | 0.0% |
| **D14** | **0%** (0/100) | -- | **1%** (1/100) | 12.0p | **+1%** | *(novo)* |
| **D15–D16** | **0%** | -- | **0%** | -- | +0% | -- |
| **D17** | **0%** (0/100) | -- | **1%** (1/100) | 31.0p | **+1%** | *(novo)* |
| **D18–D20** | **0%** | -- | **0%** | -- | +0% | -- |
| **TOTAL** | **3.67%** (44/1200) | -- | **5.92%** (71/1200) | -- | **+27 cubos** | **+61.4%** |

---

## 2. Perfil de Custo Performático (CPU vs. GPU MPS) e o Segredo dos Clones Fantasmas

### 2.1 Comparativo de Latência (Pré-Otimização de Grafo)
Para inferência reativa passo a passo ($batch=1$), a **CPU é 5,4× a 6,4× mais rápida que a GPU (MPS)** porque executa diretamente no cache L1/L2 sem o overhead fixo de ~2,5 ms de despacho de command buffers do driver Metal:

| Modo de Inferência | Plataforma | Latência Média | Mediana (p50) | Mínimo | Throughput Real |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Reflexo Puro** | **CPU** | **0.495 ms** | **0.485 ms** | **0.439 ms** | **2.020 decisões/s** |
| Reflexo Puro | GPU (MPS) | 3.198 ms | 3.043 ms | 2.669 ms | 313 decisões/s |
| **1-Step Lookahead (Top-2)** | **CPU** | **1.491 ms** | **1.576 ms** | **0.456 ms** | **671 decisões/s** |
| **1-Step Lookahead (Top-4)** | **CPU** | **1.521 ms** | **1.624 ms** | **0.444 ms** | **657 decisões/s** |

### 2.2 Por que os Clones Fantasmas levam apenas 0,04 ms a 0,06 ms?
1. **Reversibilidade Algébrica Zero-Copy (`simulate_step`):** O simulador não faz `deepcopy` do ambiente. Ele aplica o movimento no próprio cubo (`apply_atomic(a)`), extrai a observação e aplica imediatamente o movimento inverso (`apply_atomic(inv_a)`).
2. **Estado Compacto em Cache L1 (54 inteiros = 216 bytes):** O núcleo em Rust (`system1_core`) e a tabela `PERMUTATIONS` apenas permutam 20 inteiros por giro (~10 µs por clone).
3. **Assimetria Física vs. Neural:** Os 4 clones realizam ~2.000 operações inteiras simples, enquanto a avaliação neural do Piloto e do Juiz executa ~1,5 milhão de FLOPs.

---

## 3. Os 3 Gargalos Algorítmicos e a Tríade Intuição + Cicatriz + Lookahead

### 3.1 Gargalos do 1-Step Lookahead Clássico
1. **Descarte da Intuição do Ator (`argmax(q_scores)` puro):** Ignora a probabilidade $\pi_\theta(a|s)$ do Ator na decisão final, permitindo que flutuações de milésimos no Crítico derrubem ações de alta convicção.
2. **Cegueira a Loops de Estado (Ausência de Memória Tabu):** Evita apenas inversões imediatas ($U \to U'$), permitindo ciclos de 4, 6 ou 8 passos que revisitam estados idênticos no mesmo episódio.
3. **Miopia de 1 Passo em Comutadores:** Penaliza manobras onde o passo $t+1$ desalinha peças temporariamente para encaixá-las no passo $t+2$.

### 3.2 As 3 Inovações Guiadas por Intuição e Cicatriz
1. **Bússola de Familiaridade no Futuro ($\text{Confiança}(s')$ a Custo Zero):**
   - Passar o estado latente dos clones `h_cands` também pela cabeça do Piloto (`policy_head.linear(h_cands)`, +0,005 ms).
   - Futuros onde o Piloto apresenta alta confiança (baixa entropia) indicam entrada em uma **bacia de cicatrizes já dominada**.
2. **Cicatriz Instantânea em Voo (Tabu Hash de Estados + Poda Pré-Neural):**
   - Estados visitados no episódio são registrados em um conjunto de hashes de 64 bits (`_visited_state_hashes`).
   - Clones que pisam em estados já visitados **morrem imediatamente durante a simulação física**, antes de chamar a rede neural do Juiz.
3. **Salto Fantasma de 2º Passo (2-Step Ghost Jump):**
   - Usando o golpe instintivo previsto em $s'$ ($\hat{a}' = \arg\max \pi(\cdot|s')$), os clones sobreviventes testam o 2º passo no simulador rápido para detectar vitórias em 2 movimentos (`done_2`) e ganhos de alinhamento pós-manobra.

### 3.3 Fórmula Unificada de Pontuação
$$\text{Score}(a_i) = Q_{\text{Juiz}}(s, a_i) + \alpha \cdot \ln \pi(a_i | s) + \beta \cdot \text{Confiança}(s'_i) + \gamma_2 \cdot \Delta\text{GhostJump}(s'_i, \hat{a}'_i) - \text{CicatrizTabu}(s'_i)$$

---

## 4. Otimização Extrema de CPU (< 0,25 ms Reflexo / < 0,80 ms Lookahead)

1. **Fusão Matricial e Cache de Autovalores na LRU (`LinearRecurrentUnit`):**
   - Cachear $(\lambda_r, \lambda_i, \gamma)$ e as matrizes fundidas $W_{\text{in}} = [\gamma B_{\text{real}}; \gamma B_{\text{imag}}; D]$ e $W_{\text{out}} = [C_{\text{real}}, -C_{\text{imag}}]$ em modo `eval()`, reduzindo 5 chamadas `F.linear` e 4 exponenciais/trigonométricas para apenas 2 multiplicações matriciais por camada.
2. **Fusão dos Codificadores Diferenciais (`Tri-Compass DeltaEncoder`):**
   - Processar `[local_delta, origin_delta, goal_delta]` de forma vetorizada e enxuta.
3. **Simulação em Lote sem Overhead de Dicionários + Poda de Clones Mortos:**
   - Cortar clones que batem em cicatrizes locais antes do forward pass do Crítico; se restar apenas 1 clone válido, pular integralmente o segundo forward neural.
4. **Execução sob `torch.inference_mode()`:**
   - Desativa rastreamento de view/versionamento de tensores em C++, reduzindo a latência de despacho na CPU.
