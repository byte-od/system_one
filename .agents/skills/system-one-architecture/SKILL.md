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

### 🧩 F. VQ4-Scar (`Micro-Cicatrizes Vetoriais 4D + Cura Residual INT4-ASVD` — Nativo & WebGPU)
* **Localização Canônica:** `scar_cache/vq4_scar_mos.py` (Nativo PyTorch/Metal/CUDA) + `scar_cache/webgpu/mos_scache_kernels.wgsl` (`vq4_equilibrate_fwht64` e `mos_fused_vq4_int4scar_gemv` em WGSL 1.0).
* **Função:** Substituir a quantização escalar Gaussiana legada em **todo o ecossistema** (tanto no motor **Nativo** quanto no motor **Web**) por dicionários direcionais 4D acoplados a cicatrizes residuais comprimidas.
* **Garantias Algorítmicas:**
  * **`LUT256` (`256 × vec4<f16>` = `2 KB` por matriz / `4 KB` em `var<workgroup>`):** Entrada `0` travada no vetor nulo exato `(0,0,0,0)` + espigas esparsas + reticulado $D_4$ + GPU Lloyd-Max ponderado por energia $\alpha_b^2$.
  * **Equilibração de Canal (`d_fp16`) + `FWHT-64` na Entrada:** Neutraliza *outliers* de ativação antes da rotação Hadamard, aplicando a transformada apenas uma vez por vetor de entrada $x$.
  * **Cura Residual em 2 Passos (*2-Pass Residual Scar Healing*):** Comprime a própria cicatriz `ASVD` em `INT4` (`3,76x` menor que `FP16`) ajustando o 2º passo diretamente sobre o erro residual $(W - \widehat{W}_{\text{vq4}})$.

---

### 🌱 G. ScarNativeLM v5.0 (`Arquitetura Fundacional Pura de Cicatrizes — Tabula Rasa` — Nativo & 22 Shaders WebGPU)
* **Localização Canônica:** `scar_native/` (`model.py`, `webgpu_v5.py`, `tokenizer.py`, `data.py`, CLI `mos-scar-native`) + Shaders `#21` (`scarnative_widrow_hoff_evict_update`) e `#22` (`scarnative_o1_hybrid_gqa_attention`) em `scar_cache/webgpu/mos_scache_kernels.wgsl` + `portfolio/src/lib/scarNativeEngine.ts` (`ScarNativeLab`).
* **Função:** Eliminar o *Piso Gaussiano Pré-Treinado* treinando modelos fundacionais **100% do zero (`step = 0`, zero camadas `nn.Linear`, zero pesos externos)** diretamente na variedade de cicatrizes:
  1. **`ScarNativeLinear` (`D4-LUT256 VQ4-STE` + `FWHT-64` + `d_eq` + `INT4-STE`):** Zero choque de quantização pós-treino (`0.000000` KL ao empacotar em binário).
  2. **`DifferentiableSCacheAttention`:** Atenção híbrida diferenciável ponta a ponta em espaço estritamente $\mathcal{O}(1)$ (`Δ = 0 B` em qualquer horizonte $T \to \infty$) com matriz associativa Widrow-Hoff pré-RoPE $W_{\text{scar}}$ e paridade exata (`cos > 0.99999`) entre PyTorch e WebGPU WGSL 1.0.
  3. **`LanguageNullspaceProjector` ($P_{\perp\text{lang}}$) + `DetachedScarStore`:** Tronco 100% livre de fatos mutáveis + cartuchos `INT4` RRR fechados (`0` SGD no tronco, alinhados a páginas de `4 KB`) cunhados apenas em inglês e disparando em `PT/EN/ES/FR`, com desaprendizagem criptográfica `0x00` GDPR instantânea (`ΔPPL = 0.0000`).

---

### 🚀 H. ScarNativeLM v6.0 (`Shared-Stage MoS, Chunkwise WY & Dual-Tier O(1) SCache` — Nativo & 26 Shaders WebGPU)
* **Localização Canônica:** `scar_native/` (`chunkwise_wy.py`, `v6_bpe_tokenizer.py`, `model_v6.py`, `mos_digest.py`, `webgpu_v6.py`, `data_v6.py`, CLI `mos-digest`) + Shaders `#23` (`scarnative_v6_mos_router_topk_relu`), `#24` (`scarnative_v6_mos_bscar_accumulate`), `#25` (`scarnative_v6_hierarchical_evict_update`) e `#26` (`scarnative_v6_hierarchical_o1_gqa_attention`) em `scar_cache/webgpu/mos_scache_kernels.wgsl` + `portfolio/src/lib/scarNativeV6Engine.ts` (`ScarNativeLab.tsx`).
* **Função:** Escalar a fundação *Tabula Rasa* para velocidade de pré-treino Tensor-Core, vocabulário byte-level BPE poliglota sem `<unk>`, sub-1 GiB de pico via *Sequential Stage-Streaming* e abstenção epistêmica nativa:
  1. **`Chunkwise WY Widrow-Hoff` (`chunkwise_wy.py`):** Varredura paralela em blocos de tamanho $C$ resolvendo $(I + L_{ab})^{-1}$ triangular inferior unitário em Tensor Cores (`>12x` mais rápida que a recorrência sequencial em `A100` com paridade `cos > 0.999999` em forward + backward).
  2. **`SharedStageUniversalBase` + `ScarMoSExpertBank` (`Top-k ReLU INT4-STE`):** As camadas de cada estágio compartilham um único substrato `D4-LUT256 VQ4` enquanto mantêm cicatrizes de identidade leves e especialistas direcionais `MoS` esparsos, reduzindo `v6-Frontier` (`96L, d=8192, 128K ctx`) de `205.01 GiB` para **`4.36 GiB`** (`MoS-LRU`, `46.99x`) ou **`0.91 GiB`** em *Sequential Stage-Streaming* (`diff = 0.00e+00` vs modo residente).
  3. **`HierarchicalSCacheAttentionV6` (`Dual-Tier O(1)`):** Combina `RoPE Sinks + Sliding Ring + Surprise-Promoted RoPE Micro-Pages + Widrow-Hoff W_scar` em espaço estritamente constante (`Δ = +0 B` para $T \to \infty$), preservando cópia literal exata de dígitos/chaves e memória semântica contínua.
  4. **`ProcrustesLanguageNullspace` ($R_{\text{lang}} \in SO(D) + P_{\perp\text{lang}}$) + `mos-digest` + `<abstain>`:** Compila diretórios inteiros de documentos Markdown em inglês em cartuchos `.scarstore` (`12 KB = 3 × 4 KB`, `Zero-SGD RRR`) que roteiam automaticamente em `PT/EN/ES/FR` via dobramento ortogonal $(h R_{\text{lang}}) A_{\text{scar}}^\top = h (A_{\text{scar}} R_{\text{lang}}^\top)^\top$, com *wipe* criptográfico `0x00` GDPR por entidade ou documento inteiro (`max |Δlogits| = 0.0`) e emissão nativa do token `<abstain>` fora do cone factual.

---

### 🌌 I. PointScarFlow v1.0 (`Gerador 3D Contínuo de Cicatrizes — Zero Catálogo` — Nativo & 12 Shaders WebGPU)
* **Localização Canônica:** `scar_native/point_scar_flow.py` + `scar_native/point_scar_flow.wgsl` (`12` shaders `@compute` WGSL 1.0) + `portfolio/src/lib/pointScarFlowEngine.ts` + `portfolio/src/components/JarvisFloatingPointStage.tsx`.
* **Função:** Geração generativa 3D contínua `Tabula Rasa` (`N = 1024` pontos 3D + normais/energia em `4` passos Euler OT Rectified Flow) condicionada diretamente pelo estado latente `Layer-23` $h_{23} \in \mathbb{R}^{2048}$ do `Qwen2.5-3B` no navegador (`3.87 MiB` em `VQ4-Scar + INT4-STE`, zero catálogo fixo de primitivas, invariância poliglota `PT/EN/ES/FR` `cos > 0.90`).

---

### ⚡ J. ScarCodec Biortogonal (`v6.3`) & ScarField (`v7.5-Fused` — Pós-Transformer Zero-Legado + `@triton.jit` + Fechamento Anti-Stuttering)
* **Localização Canônica:** `scar_codec/` (`numpy_engine.py`, `nd_e8.py`, `core.py`, `c_kernels.c`), `scar_field/` (`triton_kernels.py`, `proper_time_bus.py`, `cayley_geodesic.py`, `e8_lattice.py`, `byte_shock_simplex.py`, `riemannian_muon.py`, `s1_scouts.py`, `engine.py`) e `docs/SCARFIELD_V7_5_FUSED_AND_V8_MACRO_SCOUT_FLEET.md`.
* **Garantias Biortogonais (`ŨᵀU = I`, `Δ = +0 B`):**
  1. **`BiorthogonalObliqueProjector` ($\mathbf{P}_{\text{biorth}} = I - U_J \widetilde{U}_J^\top$, $\widetilde{U}_J^\top U_S = 0$):** Anula interferência de `-60 dB` (`1.000×`) sem atenuar sinais oblíquos não-ortogonais (`100%` retenção vs `28%` em `P_⊥` a `31,8°`).
  2. **Leitura Dual Biortogonal em `SCache O(1)` (`34.816 B`) & `BiorthogonalShockBank` (`ScarField`):** Resolve o sistema de Gram dos slots ativos $(S S^\top + \lambda I)^{-1}$, eliminando *crosstalk* sob superposição oblíqua (`cos = 0,9822–0,9908` no EEG `PhysioNet`, `cos = 0,9999` / `159×` menos erro no Ônibus Espacial NASA) e isolando polos `.uscar` irmãos com wipe `0x00` GDPR exato.
  3. **Desacoplamento Suave–Choque–Reticulado + Escala Dual (`biorthogonal=True`):** Melhora todos os 10 arquivos científicos públicos sem alterar 1 byte do formato `.scar` (`+8,56 dB` no sismo `Tohoku Mw 9.1`, `-27,11%` bytes no Holter `MIT-BIH ±50 µV`, `0,8821 °C` no oceano 3D `NOAA`, `44–72,5 MB/s` em C99 Conway–Sloane + `Gate G0`).
* **Diretrizes `ScarField v7.5-Fused` (`699,8 ms/passo` na `A100`, `12,74 ms` Batedor `6:1`, `52,3 ms` Cura $E_8$):**
  * **Kernels `@triton.jit` (`scar_field/triton_kernels.py`):** Conway–Sloane $\mathcal{O}(8)$ Gosset $E_8$ em registradores (`0` divergência vs C99), `Tsallis-1.5` sem `sort` invariante a deslocamento (`< 3e-7`), `RiemannianMuon` fundido em 2 passagens de HBM e onda multi-escala por rotação do dicionário `[256, D]`.
  * **Adjunto Simplético Analítico Zero-Autograd + Inversa `Chunkwise WY` em Blocos `2×2`:** `0` chamadas `torch.autograd.grad` em `_ReversibleLeapfrogAutograd`, adjunto Rank-2 `[U, V]` sem alocar `p_mat/q_mat` e `_fast_block_unit_tril_inv` (`||T|| <= 1,0`, erro `< 5e-7` até `clen = 256`).
  * **Fechamento Causal de Choque Anti-Stuttering (`Δ = +0 B`):** FIR causal de 1 passo (`shock_closure_decay = 0,85`) + harmônicos ímpares de repetição (`enable_repeat_phase = True`, shifts `5, 13, 21`) + amortecedor de ciclo em `state.recent_bytes`, mantendo `cos > 0,999999` de paridade de streaming.

---

### 🛸 K. Frota de Macro-Batedores Geodésicos & Super-Kernels Vetorizados (`ScarField v8.0 / v8.1-Velocity / v8.2-Overnight-Sovereign`)
* **Localização Canônica:** `scar_field/macro_scout_fleet.py`, `scar_field/batched_super_fleet.py`, `scripts/colab_train_scar_field_v8_1_velocity.py`, `scripts/colab_train_scar_field_v8_2_overnight.py`, `results/scar_field_v8_2_overnight_audit.json` e `docs/SCARFIELD_V7_5_FUSED_AND_V8_MACRO_SCOUT_FLEET.md`.
* **Princípio Físico & Validação Empírica em `NVIDIA A100` (`.sfield.scar` fixo em `9,527 MiB`, `455.504 B` `O(1)` `Δ = +0 B`, `mmap cos = 1,000000`, normalização hipersférica por padrão sem `RMSNorm`):**
  1. **Super-Kernels Vetorizados em Stream Único (`BatchedScarFieldSuperEngine` / `VectorizedMacroScoutFleet`):** Empilha todos os `M = 4` especialistas em `[M, O, I]` / `[M, B, T, D]` (`torch.bmm` `BF16` Tensor Cores + `@triton.jit` Conway–Sloane/Tsallis-1.5 + compactação retangular `c_max` `S1/S2 + Gate G0` + conjugação simplética exata $M_{\text{fwd}}^{-T} \equiv J M_{\text{rev}}^{-1} J^T$ em `cayley_geodesic.py`), alcançando **`635,8 ms/passo`** (`2,89× > v7.3`) em `17,73 GiB` VRAM com **`0` retentativas OOM do alocador**.
  2. **Margem Simétrica Bidirecional `C^1` Zero-Leak (`prose_noncode_first_mask` + `code_def_first_mask`) & Síntese Python Executável (`v8.2`):** Na maratona de `15.760` passos novos (`32.816` acumulados, `29.849` conflitos geodésicos projetados), zerou completamente vazamentos de `def ...` em prosa (**`0/8` nos 4 idiomas `PT/EN/ES/FR`**), validou **`2/2 (100%)` funções Python em `ast.parse` + `exec()` funcional ao vivo** (indentação PEP-8 `4/8/12` espaços em `engine.py` com `Δ = +0 B`), bateu recordes históricos em **`Code = 1,6368 BPB` (`72,18%` Top-1)**, **`FR = 2,3633 BPB`**, **`PT = 2,4421 BPB`**, **`ES = 2,4551 BPB`** e **`EN Top-1 = 63,48%`** (`1,9796 BPB`), e estendeu o Passkey Biortogonal `O(1)` até **`T = 4.096` bytes (`64×` o anel sensorial: `100%` até `T=1024`, `87,5%` em `1536`, `75,0%` em `2048`, `40,62%` em `3072` e `12,50%` exato / `51,04%` por dígito em `4096` vs `0,0%` sem Banco Biortogonal e `0,0%` Dense)**.
  3. **Enxame Giant-GEMM sobre Cristal $E_8$ Compartilhado (`VectorizedSharedCrystalSwarm`):** Concatena `S = 8..16` trabalhadores de cicatriz (`37,45×` menos parâmetros mutáveis por batedor) em **1 único GEMM `BF16` `[S*N, I] @ [I, O]`** sem gradiente de peso `[O, I]` e com identidade de traço zero-`ys`, atingindo **`560,8 ms/passo` (`S=8`, `3,27× > v7.3`)** e **`549,4 ms/passo` (`S=16`, `3,34× > v7.3`, `11.183 B/s`)** com `0` OOM retries e **`Δ = +0 B`** no `.sfield.scar`.

---

### 🪐 L. Escala Soberana Multi-Bucket & Elasticidade Geodésica (`ScarField v9.0-Scale` & `v9.1-Elastic` — `Edge` + `Core MoS`)
* **Localização Canônica:** `scar_field/proper_time_bus.py` (`build_spherical_hash_bucket_poles`, `register_interscale_stiefel_bridge`), `scar_field/geodesic_expansion.py`, `scar_field/elastic_mos.py`, `scar_field/engine.py`, `scripts/colab_train_scar_field_v9_0_scale.py` e `docs/SCARFIELD_V7_5_FUSED_AND_V8_MACRO_SCOUT_FLEET.md` (§7).
* **Postura WebGPU (`AGENTS.md` §3):** Porte `WGSL/WebGPU` do `ScarField` explicitamente agendado para após o congelamento dos operadores `v9.0` na âncora `ScarField-Edge` (`60M`, `9,527 MiB`).
* **Garantias de Escala & Elasticidade Imposto-Zero:**
  1. **Banco Biortogonal $\mathcal{O}(1)$ Multi-Bucket de Sylvester–Hadamard (`16 × 32 = 512` slots, `Δ = +0 B`):** Quebra a barreira $M \le d_{\text{key}} = 128$ resolvendo Gram SPD `[32 × 32]` independente por octante com portão esférico $C^1$ local (`0,0` ruído dual cruzado) + `enable_percept_s1_gate` + Ponte de Stiefel `.uscar` inter-escala (`512D → 1024D → 2048D+`, wipe `0x00` GDPR exato).
  2. **Imersão Geodésica Isométrica Unidirecional (`Edge → Core`, Zero Retrabalho & `0%` Imposto *Matryoshka*):** Expande `60M → 285M → 1,5B → 7B` via esfera $\ell_2$ ($\|[h, \mathbf{0}]\|_2 = \|h\|_2 \equiv 1$), permutação de blocos `FWHT-64`, vácuo `0x00` do `Gosset E8` e quebra de sela em Stiefel ($\gamma_k = 0 \implies \Delta\text{scores} = 0,0$ no passo `0` e $\nabla_{\gamma_k} \neq 0$ no passo `1`), liberando mistura global irrestrita em seguida.
  3. **Duas Âncoras Físicas Elásticas (`ScarField-Edge` `9,53 MiB` + `ScarField-Core MoS` `40,6 MiB..1,12 GiB`):** Trocam de capacidade *por pergunta/byte* sem alterar `d_bus` (via marchas de ciclos simpléticos `1..12` + bancos de cicatrizes `MoS` ternários sob demanda) e suportam elevação sem perdas de `KalmanBusState` (`Edge → Core`).

---

## 🚫 2. Regras Estritas de Implementação (Checklist de Pureza)

Ao implementar qualquer código sob esta arquitetura:

1. **PROIBIDO usar LoRAs de terceiros como "batedores":** Modelos off-the-shelf da comunidade não são batedores.
2. **PROIBIDO mascarar bibliotecas off-the-shelf:** Nunca chame `LinearSVC`, `LightGBM` ou `CatBoost` genéricos de "Intuição proprietária". Se for usar um classificador tradicional, declare explicitamente como âncora estatística ou substitua pela implementação pura do `UniversalS1Agent`.
3. **PROIBIDO usar KV Cache linear $\mathcal{O}(T)$ escondido:** Em agentes interativos e modelos de linguagem, o contexto deve ser compactado via cicatrizes MoS/SCache.
4. **OBRIGATÓRIO validação matemática contínua:** Em pipelines de MoS, garantir similaridade cosseno $> 99\%$ em relação à inferência de referência.
5. **OBRIGATÓRIO escopo e testes duplos (`Nativo` E `WebGPU`):** Mudanças arquiteturais nos operadores de cicatriz (`MoS`, `VQ4-Scar`, `SCache`, `GoldenSlice P_⊥lang`) valem para **todo o ecossistema** (servidor/desktop Nativo e navegador Web) e DEVEM ser testadas em **ambos** os motores com paridade numérica verificada (`cosine > 0.99999`) e avaliação poliglota *held-out* (`PT, EN, ES, FR`).

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
