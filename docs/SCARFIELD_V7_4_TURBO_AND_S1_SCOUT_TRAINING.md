# Especificação Arquitetural `ScarField v7.4-Turbo` & Treino Guiado por Batedores `System 1` (*Scout-Guided Geodesic Training*)

> **Data:** 17/11/2026  
> **Status:** Diagnóstico Forense Concluído (`A100` + Profiler Local) & Plano Normativo de Engenharia (`Zero-Legacy / Zero-Facade`)  
> **Escopo:** Eliminar o gargalo de `47,2×` por passo medido no treino do `ScarField v7.3` (`1.836,4 ms/passo` vs `38,9 ms/passo` do `DenseByteTransformer` na `NVIDIA A100-SXM4-40GB`) combinando **4 otimizações exatas de Tensor Cores** com **3 mecanismos nativos de Batedores de Fronteira do `System 1`**.

---

## 1. Diagnóstico Forense Cronometrado (`ScarField v7.3` — `59,98M` Parâmetros Esféricos)

Na execução auditada na `NVIDIA A100-SXM4-40GB` (`scar-cache/results/scar_field_v7_3_biorth_chat_audit.json`), `840` passos consumiram `1.542,6 s` (`1.836,4 ms/passo` $\approx 3.507\text{ bytes/s}$), enquanto `2.400` passos do `DenseByteTransformer` (`33,4M` params, `63,77 MiB` FP16) levaram `93,3 s` (`38,9 ms/passo` $\approx 158.000\text{ bytes/s}$).

A cronometragem isolada órgão a órgão revelou onde residem **98,8%** do custo computacional:

| Órgão do `ScarField v7.3` | Parâmetros | % dos Pesos | % do Tempo de Passo | Diagnóstico |
| :--- | :---: | :---: | :---: | :--- |
| `1. ByteShockFrontEnd` (`V=256`, ondas `lags=1..24`, surpresa `S1`) | `327.691` | `0,55%` | **`0,1%`** | **`673×` mais rápido que o tronco `Cayley`** |
| `2. ChannelSelectiveKalmanScarBus` (`Chunkwise WY` + `Biorth` + `Ring`) | `1.190.475` | `1,98%` | **`1,0%`** | Memória $\mathcal{O}(1)$ completa (`444,8 KiB`, `Δ = +0 B`) é levíssima |
| `3. SphericalCapSimplexHead` (`Tsallis-1.5`, zero `Softmax`) | `328.197` | `0,55%` | **`0,1%`** | Projeção quadrática analítica na calota é imediata |
| **`4. 3× ReversibleSymplecticCayleyFlow` (`Fwd + Bwd`)** | **`58.138.572`** | **`96,92%`** | **`70,7%`** (`31,9%` fwd + `38,8%` bwd) | **Gargalo primário (LU `cuSOLVER` + `FP32` `E8` recomputado `4×`)** |
| **`5. RiemannianMuon` (5 passos Newton-Schulz + `SVD` periódico)** | *(sobre `58,14M`)* | — | **`28,1%`** (+ `2,97 s` a cada `heal`) | **Gargalo secundário (`Newton-Schulz` em `FP32` sobre `[12288 × 832]`)** |

---

## 2. Os 4 Gargalos de Kernel/Hardware Identificados (`Pilar Turbo`)

1. **80 Fatorações LU com Pivoteamento (`torch.linalg.solve` / `cuSOLVER`) por Passo em `cayley_woodbury_step`:**
   - Os 3 fluxos reversíveis executam `K = 2 + 4 + 4 = 10` passos × `2` geradores = `20` chamadas no `forward`, `20` na inversão analítica `-dt` e `40` no `autograd.grad` reversível (`[B·T, 32, 32]` em `float32`), disparando kernels `getrf_batched` sequenciais sem Tensor Cores.
   - **Correção Exata (`Cayley-Schulz / SPD Neumann` em `BF16`):** Como $\Omega = U V^\top - V U^\top$ é antissimétrica, $(I + \alpha \Omega)^{-1} = (I - \alpha \Omega)(I + \alpha^2 \Omega^\top \Omega)^{-1}$ reduz-se a uma matriz **Simétrica Positiva Definida** com autovalores $\ge 1$, invertida até precisão de máquina com **3 multiplicações `bmm` puras em Tensor Cores** (`0` chamadas ao `cuSOLVER`).

2. **6 Matrizes Gigantes Desfatoradas `base_e8` (`[12.288 × 832]` e `[9.216 × 896]` = `55,4M` Pesos) Forçadas em `FP32` e Recomputadas `4×` no `Backward`:**
   - `E8SphericalLinear.forward` converte `x.float()` (`FP32`) e, dentro do loop `for k in range(K-1, -1, -1)` de `_ReversibleLeapfrogAutograd.backward`, re-executa `project_rows_to_unit_sphere` + `fwht64_rademacher` + GEMM de gradiente de pesos **4 vezes por fluxo** sobre pesos que não mudam entre `k=0..3`.
   - **Correção Exata:**
     - **(a)** Executar GEMMs `E8` e os 5 passos quinticos de Newton-Schulz do `RiemannianMuon` em **`BF16` Tensor Cores** (`312 TFLOPS` na `A100` vs `19,5 TFLOPS` FP32);
     - **(b)** Pré-computar `w_ste_rot` **1 única vez por passo** fora do loop `k` e acumular $\sum_k (\nabla y_k)^\top x_{\text{rot}, k}$ em um único GEMM de gradiente por matriz;
     - **(c)** Compartilhar o substrato universal `E8` por estágio (`Shared-Stage E8`, espelhando o `ScarNative v6.0`) ou fatorar os geradores simpléticos `(U, V)`.

3. **Cura Periódica `heal_all_e8_scars` via `SVD` Completa (`[12.288 × 832]`):**
   - Substituir `torch.linalg.svd` densa completa por **Subspace / Randomized Block Power Iteration de rank `r = 8`** (`2` multiplicações em `BF16`, `< 3 ms` em vez de `2.974 ms`).

4. **Correção de Escalares de 1 Elemento no `RiemannianMuon` (`SignSGD` Indevido):**
   - Para tensores escalares (`p.numel() == 1`, como `biorth_gain` e `cap_adapt_gain`), `u_s / sqrt(mean(u_s^2))` reduzia-se a `±1.0` com passo fixo `scalar_lr`, zerando `biorth_gain` nos passos ímpares curtos sem passkey. Usar segundo momento escalar suave escalona o passo proporcionalmente à magnitude real do gradiente.

---

## 3. Aceleração Nativa por Batedores de Fronteira do `System 1` (*Scout-Guided Training*)

Como `ByteShockFrontEnd` + `Kalman/Biorth Bus` consomem apenas **`1,1%`** do passo (**`> 90×` mais leves que os fluxos `Cayley S2`**), os próprios órgãos do **System 1** atuam como **Batedores de Fronteira (*Frontier Scouts*)** durante o treino em **3 níveis complementares**:

### Nível A — Funil Intra-Sequência `G0` (*Scout-Gated Sparse Token Compaction* no Tronco `S2`)
- **Princípio (idêntico ao `Gate G0` do `System 1` em `LOOKAHEAD_INTUITION_SCAR_ARCHITECTURE.md`):**
  Em prosa natural (`PT/EN/ES/FR`) e código-fonte, **`75% a 80%` dos bytes UTF-8** possuem baixa surpresa reflexiva (`s_t < θ_S2`, alta confiança no `ByteShockFrontEnd` + `SCache O(1)`, ex.: continuações morfológicas `"ção"`, `"ment"`, indentação, palavras funcionais).
- **Execução Vetorizada Exata:**
  1. O `ByteShockFrontEnd` + `ChannelSelectiveKalmanScarBus` varrem **100% dos bytes `[B, T]`** em paralelo (`1,1%` do tempo), mantendo o tempo próprio relativístico $\tau_t$ e a memória causal $\mathcal{O}(1)$ 100% íntegros.
  2. Um **Gate de Batedor `G0`** seleciona apenas os índices de fronteira/choque $\mathcal{I}_{\text{shock}} = \{(b, t) : s_{b,t} \ge \theta_{\text{scout}} \text{ ou máscara SFT crítica}\}$ (`~20%–25%` dos bytes!).
  3. Somente o sub-lote compactado `h_shock = gather(h, I_shock)` (`[N_shock, D]`, **`4× a 5×` menor!**) atravessa os passos simpléticos profundos `Cayley S2` (`flow` + `extra_flows`), retornando via `scatter` residual para a esfera unitária $\mathbb{S}^{D-1}$.
- **Impacto:** Reduz em **`~4×`** os FLOPs do tronco `Cayley` e concentra 100% da capacidade dos cicatrizes geodésicas `S2` nas bifurcações reais (inícios de tokens semânticos, dígitos de `PIN`, operadores lógicos), eliminando a diluição de gradiente por bytes triviais.

### Nível B — Prospecção Assíncrona de Corpus `8:1` (*Frontier Ghost-Jump Batch Scouting*)
- **Princípio (idêntico ao `_build_external_scout_env` do `ppo.py` no `System 1`):**
  Em vez de alimentar o otimizador com janelas aleatórias cegas (onde grande parte do texto aberto já é trivial ou redundante), uma frota leve de **Batedores `S1` sem gradiente** (`ByteShockFrontEnd` + kernel C99/SIMD / `BF16` rápido) sonda `M × B` janelas candidatas (`ex.: 8 × 16 = 128` janelas) a **`> 200.000 bytes/s`**, pontuando a **Surpresa Epistêmica Líquida** (choque alto no modelo atual, filtrando ruído puro irreduzível).
- **Isolamento Curricular:** Apenas as `Top-B` janelas de fronteira informacional entram no passo completo `Fwd + Bwd + RiemannianMuon`, elevando em **`3× a 5×`** o ganho de validação por passo de GPU.

### Nível C — Destilação Anti-Loop por Salto Fantasma Zero-Copy (*Ghost-Jump Lookahead SFT*)
- Nas posições de diálogo/código onde a margem da calota `Tsallis-1.5` cai abaixo de `uncertainty_margin_thresh` (`0,24`), `3` Batedores *Ghost-Jump* zero-copy (`bus.step_pulse(..., mutate_state=False)`) simulam 2–3 passos à frente avaliados pelo `Juiz` $V(h_{\text{world}})$ e pelo `RepulsiveScarRing`, penalizando imediatamente ciclos degenerativos (repetição de palavras/identificadores) durante o SFT.

---

## 4. Meta Quantitativa Auditável para o `ScarField v7.4-Turbo`

| Métrica na `NVIDIA A100-SXM4-40GB` | `ScarField v7.3` (Atual Medido) | Meta `ScarField v7.4-Turbo + Batedores S1` |
| :--- | :---: | :---: |
| **Tempo por Passo (`B=16, T=384`)** | `1.836,4 ms` | **`45 – 75 ms` (`25× a 40×` mais rápido)** |
| **Throughput de Treino (`bytes/s`)** | `~3.500 B/s` | **`> 100.000 B/s`** |
| **Volume Processado em 30 min (`A100`)** | `~6,4 MB` (`840` passos) | **`> 180 MB` (`> 25.000` passos)** |
| **Garantias Arquiteturais** | `0` Legado · `1,0 bpw E8` · `O(1)` Biortogonal | **100% Idênticas (`0` Fachadas, `Δ = +0 B`)** |
