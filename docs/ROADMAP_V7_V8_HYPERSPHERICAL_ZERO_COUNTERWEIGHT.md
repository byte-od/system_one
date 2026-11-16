# Roadmap Arquitetural v7.0 / v8.0 — Arquitetura Hipersférica Limpa ("Zero Contra-Pesos") & Universal ScarStore

> **Documento Espelho Canônico:** Sincronizado com `/Users/juliocesarreisfilho/Projects/scar-cache/docs/ROADMAP_V7_V8_HYPERSPHERICAL_ZERO_COUNTERWEIGHT.md`.  
> **Diretriz:** Eliminar todos os cálculos compensatórios herdados da era de adaptação de Transformers euclidianos irrestritos (`v1–v4`), garantindo que pesos, ativações, memória `SCache O(1)` e cartuchos `.uscar` nasçam, treinem e operem em escala estritamente normalizada desde `t = 0`.

---

## Resumo Executivo dos 4 Pilares (`v7.0` -> `v8.0`)

1. **Fluxo Geodésico na Hipersfera Unitária ($\|h_l\|_2 \equiv 1$ em todas as camadas):**
   Substitui a soma residual euclidiana irrestrita ($h_{l+1} = h_l + \Delta h$, que inflava normas de `1` para `> 7.500`) por interpolação esférica $h_{l+1} = \operatorname{Norm}(h_l + \alpha_l (\operatorname{Norm}(f_l(h_l)) - h_l))$, eliminando `RMSNorm` parametrizada e `coord_rms`.
2. **Pesos Unitários `Spherical-D4` em 1 Único Dispatch WGSL:**
   Linhas de peso mantidas em $\mathbb{S}^{D-1}$ desde `t = 0` impedem matematicamente a formação de *outliers* de canal por concentração de medida, aposentando `d_eq`, `FWHT-64` e escalas `FP16` por bloco.
3. **`SCache O(1)` Contrativo por Construção ($\|W_{\text{scar}}\|_2 \le 1$):**
   Com $\|k_t\|_2 = \|v_t\|_2 = \|q_t\|_2 \equiv 1$ nativos, a atualização Widrow-Hoff nunca diverge nem exige *clamps* ou contra-pesos manuais.
4. **Variedade Produto Desembaraçada ($\mathbb{S}_{\text{syn}} \times \mathbb{S}_{\text{sem}}$) & `Universal ScarStore (.uscar)`:**
   Separa estruturalmente fluência/idioma ($\mathbb{S}_{\text{syn}}$) de fatos/conceitos ($\mathbb{S}_{\text{sem}}$ com perda de uniformidade hipersférica), aposentando `Procrustes R_lang`, `Ridge-LDA`, `U_lang` e `z_dual`, e viabilizando cartuchos universais `.uscar` independentes de versão e dimensão do modelo.
