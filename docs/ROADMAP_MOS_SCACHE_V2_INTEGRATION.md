# 🗺️ Roadmap Oficial de Evolução Científica e Engenharia (`v1.4` → `v2.x`): System One + MoS + SCache

> [!IMPORTANT]
> **MANDATO INEGOCIÁVEL DE INTEGRIDADE ARQUITETURAL (ZERO FACHADAS):**  
> Todo avanço descrito neste roadmap deve operar genuinamente sobre a matemática auditada do **Universal System 1**, **Mixture of Scars (MoS)** e **Scar-Cache (`SCache O(1)`)** — com auditoria recursiva de ponteiros de memória (`0.00 B` de discrepância), testes de causalidade em voo e zero uso de caches lineares `O(T)` ou wrappers convencionais mascarados.

---

## 🧬 1. De Onde Viemos: A Equação-Mãe das Cicatrizes (`v1.3.0` Consolidado)

Originada no motor reflexivo [`system1_engine/core/scar_memory.py`](../system1_engine/core/scar_memory.py) (`34.8 KB` no Cache L1, latência CPU `0.23 ms`) e expandida em escala colossal no repositório irmão [`mixture-of-scars`](https://github.com/byte-od/mixture-of-scars) (`DOI v1.3.0: 10.5281/zenodo.24172784` · `Concept DOI: 10.5281/zenodo.24144512`), toda a arquitetura obedece ao **Princípio Unificado da Cicatriz**:

$$F(\mathbf{x}) = \underbrace{F_{\text{base}}(\mathbf{x})}_{\text{Tecido Base Enxuto}} + \sum_{k} \underbrace{\operatorname{ReLU}\!\left(\frac{\mathbf{x}^\top \mathbf{z}_k}{\|\mathbf{x}\|_2 \|\mathbf{z}_k\|_2 + \epsilon}\right)}_{\text{Cone Hemisférico (Vazamento 0.000000)}} \cdot \underbrace{\Delta_{\text{scar}, k}(\mathbf{x})}_{\text{Correção Residual de Erro / Surpresa}}$$

### Marcos já provados e publicados (`v1.3.0`):
* **Reflexo `System 1` Pura CPU:** `0.230 ms` (P50) navegando variedades de `232 Bilhões` de estados sem busca em árvore.
* **Memória `SCache O(1)` em `32.868` Tokens:** `100%` de recall *Passkey / Needle-in-a-Haystack* com crescimento físico de **`+0.00 Bytes`** (`22.71 MiB` vs. `1.155,8 MiB` do KV Cache normal) e execução **`5,84×` mais rápida** (`300,8` vs. `51,5 tok/s`).
* **Titã `72.71B` (`Qwen2.5-72B-Instruct MoS v2`):** Neutralização de *outliers* SwiGLU de **`195.720×`** em `38.12 GiB` (`100%` Top-5 agreement; `-96,37%` em banco de 8 especialistas), com kernels fundidos em registradores para **Apple Metal (`50,38×` speedup)** e **NVIDIA CUDA Triton (`W4A16+ASVD`)**.

---

## 🚀 2. Horizonte 1 — Unificação `System 1 + System 2` e Barreira dos `24 GB` (`v1.4`)

```mermaid
flowchart LR
    S1["⚡ System 1 Scout (0.5B MoS + SCache)<br>1.176,9 tok/s · Ghost-Jump 6–8 tokens"] -->|Rascunho + Confiança| S2["👑 System 2 Verifier (72.71B MoS 2-Bit/4-Bit)<br>Passe Único em Lote M<=8 nos Registradores GPU"]
    S2 -->|Aceite Exato / Rollback O(1) SCache| Out["Saída 100% Idêntica ao 72B<br>3x a 5x Mais Rápida!"]
```

### Fase 1.1: *Ghost-Jump Speculative Decoding* (`Universal System 1` + `72.71B MoS`)
* **Objetivo:** Unir fisicamente o motor de Batedores de Fronteira (`system_one`) ao motor `CanonicalMoSEngine` (`72.71B`).
* **Mecânica:**
  1. Um Batedor leve (`0.5B MoS + SCache` rodando a `>1.100 tok/s`) avança `6 a 8` tokens à frente guiado pela telemetria de margem/incerteza do `ReflexDecision`.
  2. O Titã `72.71B MoS` verifica todos os `8` tokens em **um único varrimento de memória** (`M <= 8` no kernel fundido Metal/Triton), já que ler os pesos uma vez para `1` ou `8` tokens tem quase o mesmo custo de banda.
  3. Em caso de divergência, aplica-se *rollback* ortogonal atômico no `SCache`.
* **Meta Mensurável:** **`3× a 5×` de ganho real em `tok/s`** na geração do `72B`, mantendo `0.0` perda matemática de qualidade.

### Fase 1.2: `MoS Sub-2-Bit + ASVD` (`72.71B` em `~19–21 GiB`)
* **Objetivo:** Quebrar a barreira física de `24 GB` (RTX 3090 / 4090) e `32 GB` (Apple M2 Pro).
* **Mecânica:** Como as cicatrizes `ASVD` já isolam os canais *outliers* de `195.720×`, comprimir a base saudável de `4 bits` (`38.12 GiB`) para **`2 bits` / `1.58-bit`** com cicatrizes `rank 96–128` (`~1.6 GiB`).
* **Meta Mensurável:** Modelo `72.71B` completo cabendo **100% residente em placas de `24 GB` e Macs de `32 GB`** com `0 MB/tok` de leitura de SSD e `>= 98,5%` de acordo Top-5.

---

## 🧠 3. Horizonte 2 — A Próxima Descoberta Científica (`v2.0`)

### Fase 2.1: "O Ciclo do Sono Artificial" (`SCache O(1)` $\to$ `MoS` Permanente)
* **Inspiração Neurocientífica:** Durante a vigília, o **Hipocampo** (`SCache O(1)`) retém eventos de alta surpresa ($\|\mathbf{v} - \mathbf{W}_{\text{scar}}\hat{\mathbf{k}}\|_2$). Durante o repouso ("sono"), o sistema destila automaticamente os slots saturados do `SCache` em micro-especialistas permanentes `ScarExpert` ($\mathbf{B}_k \mathbf{A}_k$, `32–64 KB`) acoplados a cones direcionais `ReLU` $\mathbf{z}_k$ no **Neocórtex** (`MoS`), limpando o `SCache` para o próximo ciclo.
* **Resultado:** Aprendizado contínuo vitalício (*Lifelong Learning*) em memória `O(1)` com **zero esquecimento catastrófico** (`0.000000` de vazamento fora do hemisfério do cone).

### Fase 2.2: Alinhamento de Permutação Entre Camadas (*Sinkhorn Re-Basin*)
* **Mecânica:** Reaproveitar o calibrador bistocástico **Soft Sinkhorn** do Tri-Compass (`SKILL.md`) para alinhar a permutação de neurônios entre camadas adjacentes ($\min_{\mathbf{P}} \|\mathbf{W}_{\ell+1} - \mathbf{P}\mathbf{W}_\ell\mathbf{P}^\top\|_F$) antes da fatoração `ASVD`.
* **Resultado:** Reduzir a divergência KL de compartilhamento entre camadas distintas de `0.78` para **`< 0.08`** sem retreinar o modelo base do zero.

---

## 🗄️ 4. Horizonte 3 — Armazenamento de Dados, Velocidade de Busca e `WebGPU` (`v2.1+`)

1. **Velocidade de Busca (*Scar-Search* & Cartuchos Pós-RAG `.scache`):**
   * **Filtro Geométrico L1 (`34.8 KB`):** Podar `50%–95%` de partições vetoriais irrelevantes em nanossegundos dentro do Cache L1 da CPU via calota esférica `ReLU` antes de acessar RAM/SSD.
   * **Indexação Guiada por Surpresa:** Filtrar `90%+` de texto repetitivo/boilerplate via erro de Widrow-Hoff, gerando índices vetoriais **`20×–50×` menores** focados exclusivamente nas "agulhas".
   * **Cartuchos `.scache` (`mos-digest`) + Multi-Agulha `128k–1M`:** Pré-digerir repositórios inteiros uma única vez em cartuchos portáteis de consulta `O(1)` com paginação hierárquica para dezenas de fatos exatos simultâneos.
2. **Armazenamento de Dados (*Scar-Storage* & *Machine Unlearning*):**
   * **`scar-zip` (`SCache + ANS` Sem Perdas):** Compressor *lossless* de longo alcance e RAM constante capaz de capturar repetições separadas por dezenas de gigabytes sem travar em `O(T^2)`.
   * **`mos-git`:** Versionamento de checkpoints e tensores científicos com `1` base canônica + diffs de cicatriz (`-96%+` de disco).
   * **Desaprendizado Cirúrgico `< 1 ms` (LGPD/GDPR):** Apagamento ortogonal exato via $\mathbf{W}_{\text{scar}}(\mathbf{I} - \hat{\mathbf{k}}\hat{\mathbf{k}}^\top)$ sem retreino.
3. **Execução Universal no Navegador (`WebGPU / WGSL`) e `Flash-SCache`:**
   * Implementação real dos shaders `WGSL` para descompressão `INT4/2-bit + ASVD` em registradores e atualização Widrow-Hoff no cliente, cumprindo integralmente a diretriz do `AGENTS.md`.

---

## 🌍 5. Horizonte 4 — Aplicações Transdisciplinares Prioritárias

* **Robótica Embarcada & Drones (VLAs):** Adaptação motora e proprioceptiva em `< 1 ms` a piso escorregadio ou desgaste mecânico via cicatrizes de torque/atrito em `34.8 KB`.
* **Vídeo 24/7, Óculos AR & *World Models*:** Memória visual contínua sem explosão `O(T^2)` (`+0 Bytes/hora` para cenas estáticas, gravando apenas mudanças de estado sem fantasmas).
* **Nuvem Multi-Inquilino (*SaaS*):** `1` base `72B MoS` servindo milhares de cartuchos corporativos isolados com `-96%+` de VRAM.
* **Biossensores Médicos (`34.8 KB` em SRAM) & Cibersegurança:** Detecção de arritmias raras na borda e imunidade matemática contra inundação de ruído `500×` (*DoS*).
* **Genômica Longa, Proteômica & Telemetria Satelital:** Varredura `O(1)` de cromossomos inteiros e transmissão apenas de pacotes de surpresa ($\|\mathbf{e}_t\|_2 > \tau$).
