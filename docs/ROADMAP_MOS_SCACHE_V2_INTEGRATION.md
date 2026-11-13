# 🗺️ Roadmap Oficial de Evolução Científica e Engenharia (`v1.4` → `v3.x`): System One + MoS + SCache

> [!IMPORTANT]
> **MANDATO INEGOCIÁVEL DE INTEGRIDADE ARQUITETURAL (ZERO FACHADAS):**  
> Todo avanço descrito neste roadmap deve operar genuinamente sobre a matemática auditada do **Universal System 1**, **Mixture of Scars (MoS)** e **Scar-Cache (`SCache O(1)`)** — com auditoria recursiva de ponteiros de memória (`0.00 B` de discrepância), testes de causalidade em voo e zero uso de caches lineares `O(T)` ou wrappers convencionais mascarados.

---

## 🧬 1. De Onde Viemos: A Equação-Mãe das Cicatrizes (`v3.0.0` Consolidado)

Originada no motor reflexivo [`system1_engine/core/scar_memory.py`](../system1_engine/core/scar_memory.py) (`34.8 KB` no Cache L1, latência CPU `0.23 ms`) e expandida em escala colossal no repositório irmão [`mixture-of-scars`](https://github.com/byte-od/mixture-of-scars) (`DOI v3.0.0: 10.5281/zenodo.24172784` · `Concept DOI: 10.5281/zenodo.24144512`), toda a arquitetura obedece ao **Princípio Unificado da Cicatriz**:

$$F(\mathbf{x}) = \underbrace{F_{\text{base}}(\mathbf{x})}_{\text{Tecido Base Enxuto}} + \sum_{k} \underbrace{\operatorname{ReLU}\!\left(\frac{\mathbf{x}^\top \mathbf{z}_k}{\|\mathbf{x}\|_2 \|\mathbf{z}_k\|_2 + \epsilon}\right)}_{\text{Cone Hemisférico (Vazamento 0.000000)}} \cdot \underbrace{\Delta_{\text{scar}, k}(\mathbf{x})}_{\text{Correção Residual de Erro / Surpresa}}$$

### Marcos já provados e publicados (`v1.3.0`):
* **Reflexo `System 1` Pura CPU:** `0.230 ms` (P50) navegando variedades de `232 Bilhões` de estados sem busca em árvore.
* **Memória `SCache O(1)` em `32.868` Tokens:** `100%` de recall *Passkey / Needle-in-a-Haystack* com crescimento físico de **`+0.00 Bytes`** (`22.71 MiB` vs. `1.155,8 MiB` do KV Cache normal) e execução **`5,84×` mais rápida** (`300,8` vs. `51,5 tok/s`).
* **Titã `72.71B` (`Qwen2.5-72B-Instruct MoS v2`):** Neutralização de *outliers* SwiGLU de **`195.720×`** em `38.12 GiB` (`100%` Top-5 agreement; `-96,37%` em banco de 8 especialistas), com kernels fundidos em registradores para **Apple Metal (`50,38×` speedup)** e **NVIDIA CUDA Triton (`W4A16+ASVD`)**.

---

## ✅ 2. Horizonte 1 — CONCLUÍDO E AUDITADO (`v1.4.0`, `51/51` Testes Passando)

```mermaid
flowchart LR
    S1["⚡ System 1 Scout (0.5B + SCache O(1))<br>Ghost-Jump Adaptativo via ReflexDecision"] -->|Rascunho + Confiança/Margem| S2["👑 System 2 Verifier (3B / 72.71B Sub-2-Bit 21.89 GiB)<br>Verificação em Passe Único + Rollback Atômico O(1)"]
    S2 -->|KL = 0.0000 · 0 B Discrepância| Out["✅ Auditado: 1,89x–1,92x Speedup Real (MPS)<br>72.71B < 24 GB VRAM (RTX 3090/4090 & Mac 32 GB)"]
```

### ✅ Fase 1.1: *Ghost-Jump Speculative Decoding* (`System 1` + `System 2` + Rollback Atômico `SCache O(1)`) — **ENTREGUE**
* **Implementação Canônica:** `scar_cache/speculative_ghost_jump.py`, `SCacheDynamicCache.speculative_capture()` em `scar_cache/hf_cache.py` e suporte CLI `mos-run --draft-model` em `scar_cache/cli_72b.py`.
* **Resultados Empíricos Auditados (`results/horizon1_v1_4_audit.json`):**
  * **`1,89× – 1,92×` de aceleração real medida** na geração em Apple M2 (`MPS`, `Qwen2.5-0.5B` Scout + `Qwen2.5-3B` Verifier: `12,50 → 23,65 tok/s` em código Python e `12,14 → 23,30 tok/s` em demonstração matemática; `78,8%–79,2%` de aceitação, **`4,80–5,05` tokens/varredura**).
  * **Paridade Matemática Exata (`KL = 0.0000`):** `100,0%` de identidade token a token (`96/96`) em decodificação gulosa + amostragem estocástica exata de Leviathan–Kalman–Matias + **`0 Bytes` de discrepância** na auditoria recursiva `untyped_storage().data_ptr()`.

### ✅ Fase 1.2: `MoS Sub-2-Bit + ASVD` (`72.71B` em `20,65 GiB` Shards / `21,89 GiB` VRAM) — **ENTREGUE**
* **Implementação Canônica:** `scar_cache/sub2bit_mos.py` (`2.25 bpw` código simétrico exato IEEE `beta = 5/16` + Oráculo Analítico SwiGLU zero-dados + Escudo GQA `k_proj/v_proj` INT4 + CLI `mos-transcode-sub2bit`) e kernels CUDA Triton fundidos `W2A16+ASVD` em `scar_cache/triton_mos_gemm.py`.
* **Resultados Empíricos Auditados:**
  * **Quebra da Barreira de `24 GB` e `32 GB`:** Transcodificação real do `mos_group_19.safetensors` reduziu cada grupo de 4 camadas de `1,79 GiB → 1,0325 GiB` (`1,734×` menor) → **`20,65 GiB`** nos 20 grupos (`22,97 GiB` disco total `< 32 GB` Apple M2 Pro; **`21,89 GiB` de VRAM residente** `< 24 GB` RTX 3090 / 4090 / L4!).
  * **Escudo contra *Outliers* de `195.720×`:** Na `Layer 79 down_proj` (`8192 × 29568`) do `Qwen2.5-72B`, atingiu **`cos = 0,99911`** (`erro = 0,0421` vs. `0,5430` do `2-bit` ingênuo — **`12,9×` menos erro a `2,49 bpw`**) e **`cos = 0,9633`** através de 4 camadas profundas consecutivas (`L76..L79`); no `Qwen2.5-3B-Instruct`, derrubou o colapso do `2-bit` ingênuo em **`12.290×`** (`PPL = 1.351.758,2 → 109,99`, `79,4%–91,1%` Top-5 Recall).

---

## ✅ 3. Horizonte 2 & Consolidação `v3.0.0` — CONCLUÍDO E AUDITADO (`v3.0.0`, `84/84` Testes Passando em `25` Suítes)

### ✅ Fase 2.1 + `ScarStore v2` + CLI `mos-store` (Etapas 1–6 + Red-Team `0.5B vs 3B`): Transposição `72.71B → 0.5B & 3B` em Forma Fechada, `HeadExpandedScarStore` (`140,00 KB = 35×4KB` `mmap`), Streaming Real `HTTP/1.1 206` & Apagamento Forense `0x00` em SSD — **ENTREGUE E AUDITADO**
* **Implementação Canônica & Auditorias (`scar-cache`):** `scar_cache/scar_grafting.py`, `scar_cache/scar_store_v2.py` (`Etapas 1 & 2`), `scar_cache/dual_depth_bridge.py` (`Etapa 3`), `scar_cache/head_expanded_attention.py` (`Etapas 4 & 5: HeadExpandedScarStore`), `docs/SCAR_STORE_V2_ARCHITECTURE.md`, `results/scar_store_v2_stage5_3b_vs_0p5b_audit.json` e `results/adversarial_0p5b_vs_3b_audit.json`.
* **Resultados Empíricos Auditados (`Qwen2.5-72B-Instruct MoS [72.71B] → Qwen2.5-0.5B [61,25 KB] & Qwen2.5-3B [140,00 KB]`):**
  * **Geometria Perfeita de Páginas SO no Núcleo `3B` (`0 B` de *padding*):** Cada `HeadExpandedScarCapsule` no `Qwen2.5-3B-Instruct` (`36L`, `16 × 128` cabeças Q, $d = 2048$) ocupa **exatos `143.360 B` (`140,00 KB` = `35 × 4.096 B` páginas `mmap` do SO, `1.015.469×` menor que o `72B` BF16)** sobre `4.194.304` células Kronecker (`512 KB` bitmap CPU), elevando a concordância inédita Top-1 contra o `72.71B` de `94,44%` (`0.5B`) para **`98,61%` (`71/72` tokens exatos; `KL = 0,0727 / 0,0066 / 0,0079` — `6,11×` menor KL médio!)** com **`36/36` blocos internos `torch.equal = True`**.
  * **Auditoria Adversarial *Red-Team* (`6/6` Ataques Aprovados em Ambos os Núcleos):** `0,0 B` de deriva no `SCache O(1)` (`0,0 KB` de discrepância em `untyped_storage().data_ptr()`), `0/6` falsos positivos em armadilhas lexicais (`0 B` lidos em `mmap`), imunidade total a `1.560` tokens de boato falso `"999 Joules"` sob *chunked prefill* (`3B` emite `kT \ln 2` exato), síntese tripla simultânea cristalina num único *rollout* (`96,99%` de especialização entre as `16` cabeças do `3B`), **`100,0%` Top-1 (`KL = 0,0029` no `3B`)** com `MoS` (`3,41×` MLP) + `SCache O(1)` + `ScarStore` simultâneos, e **`unlearn()` cirúrgico em `10,6–18,0 µs` (`0,011–0,018 ms`)**!
  * **Zero Backpropagation / Zero SGD (`~0,90 s` por cápsula):** Sintetiza em forma fechada cápsulas portáteis `.safetensors` de `225,75 KB` FP16 (`64,75 KB` equivalente `INT4`, índice L1 de `5,25 KB` — **`629.052,7×` menor** transferência ativa que os `135,43 GiB` do `72B` BF16).
  * **Superação da Barreira Linear $d = 896$ via Cones Esféricos `ReLU`:** Pelo próprio teorema de empacotamento esférico de Kabatiansky-Levenshtein em $\mathbb{S}^{895}$, os cones `ReLU` disjuntos (`tau = 0.62`, `strictly_disjoint_at_anchors = True`) evitam a saturação linear de `~896` edições estáticas, mantendo **`0.0000000000` (`torch.equal = True`)** de interferência fora do cone em 6 prompts de controle e **`< 2 ms` de `unlearn()` cirúrgico**.
  * **Generalização em Paráfrases Inéditas + Autoaceleração `Ghost-Jump`:** No teorema de *Immerman-Szelepcsényi*, atingiu **`100,0%` de concordância Top-1** com o `72.71B` (`KL: 0,48 → 0,03`, redução de **`18,5×`**) e elevou a aceitação do `Ghost-Jump` contra o `72.71B` de **`26,0% → 100,0%`** (`44,7% → 100,0%` em *Landauer* e `13,0% → 60,0%` em *Kabatiansky-Levenshtein*).
  * **Etapa 6 (`v2.3.0` — Produto `mos-store` CLI/SDK + `HTTP/1.1 206` + Wipe Forense `0x00`, `results/mos_store_v2_3_cli_http206_audit.json`):** CLI unificada de 7 comandos (`mos-store build|inspect|slice|serve|query|sleep-bridge|unlearn`), paginação em soquete TCP real RFC 7233 alinhada a páginas SO de `4.096 B` (`Range: bytes=17629184-17772543`, `143.360 B = 35 × 4 KB` em `58,3 ms`) e sobrescrita criptográfica *in-place* **`0x00`** em SSD (`147.456 B` + `fsync` em `7,64 ms`, `torch.equal = True` ao reabrir).

### ✅ Fase 2.2: *Sinkhorn Monomial–Procrustes Re-Basin* & Lei de Ortogonalidade de Marchenko-Pastur — **ENTREGUE E AUDITADO (`v2.4.0`)**
* **Implementação Canônica & Auditoria (`scar-cache`):** `scar_cache/sinkhorn_rebasin.py` (`mos-rebasin` CLI), `scripts/audit_sinkhorn_rebasin_v2_4.py`, `tests/test_sinkhorn_rebasin.py` (`5/5` testes) e `results/sinkhorn_rebasin_v2_4_audit.json`.
* **Mecânica Algébrica & *Tiled Log-Sinkhorn*:** Explora o grupo de calibre monomial exato da SwiGLU $\mathcal{G}_{\text{SwiGLU}} = \mathrm{GL}_1(\mathbb{R})^{d_{\text{ff}}} \rtimes S_{d_{\text{ff}}}$ (`FP32 max_abs_diff = 5,25e-06`), **Fused Single-Pass Online Dual-Potential Log-Sinkhorn** (`< 350 MiB` de VRAM transiente em `72B+`, `57×` menos memória GPU), grade `Block-64 INT4` compartilhada por autovetor principal de Gram `2×2` (`mesmo untyped_storage().data_ptr()`, **`0 B` de discrepância** em CPU/MPS/CUDA) e dobra `100%` das escalas $\alpha_i$ e RMS intermediária diretamente nas escalas `FP16` (`transposed_grid=True`, **`0` FLOPs extras em inferência**).
* **Resultados Empíricos Auditados (`0.5B`, `3B` e `72.71B`):**
  * **KPI `< 0.08` Superado no `Qwen2.5-3B-Instruct` (`5,09×` de compressão = `-80,37%` vs FP16):** No par `L22-L23`, a base ingênua marca `KL = 0,1190`, enquanto o **Sinkhorn Monomial atinge `KL = 0,0395 < 0,08` (`3,01×` menor, `99,77%` Top-5)** e o **Sinkhorn–Procrustes crava `KL = 0,0353 < 0,08` (`3,37×` menor)**! Em **`12` camadas consecutivas (`L18..L29` = `1/3` da rede, `1.548 → 303,92 MiB`)**, o compartilhamento ingênuo colapsa (`KL = 6,4634`), enquanto o Sinkhorn Monomial mantém **`KL = 0,4403` (`14,68×` menor KL, `95,02%` Top-5)**!
  * ***Git Re-Basin* de Especialistas Permutados (`0.5B` `L12`):** Recupera cosseno `0,0678 → 0,9817` e derruba o KL em **`56,36×` (`0,1578 → 0,0028`, `97,41%` Top-1, `100,0%` Top-5)**; no par crítico `0.5B L4-L5` (Limitação #1 da Seção 5 do *preprint*), reduz o KL em `2,50×` (`0,5722 → 0,2290`).
  * **Titã `Qwen2.5-72B` (`29.568 × 8.192`, compressão `6,226×`: `2.772,0 → 445,21 MiB`) & Lei de Marchenko-Pastur:** Atinge ganhos escalares de `51,11×` (`L40-L41`) e `109,21×` (`L78-L79`) e eleva o cosseno de saída da `Layer 79` (sob *outliers* SwiGLU de `195.720×`) de `0,1626 → 0,8835` (`2,73×` menos erro), revelando que blocos residuais adjacentes pré-treinados convergem para quase-ortogonalidade de Marchenko-Pastur ($\mathbb{E}[\max_j \cos] = 0,0633\text{–}0,1228$ vs. piso Gaussiano `0,0501`) — estabelecendo rigorosamente a fronteira ótima entre *Sinkhorn Re-Basin* (bancos multi-especialistas e pares interiores seletivos) e bases independentes `Sub-2-Bit MoS` (`2,25 bpw`, `cos = 0,99911` em `20,65 GiB`).

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
