# 🏆 Kaggle Competition Submission Template: Mixture of Scars (MoS)
### Como Rodar Raciocínio de 70B dentro de GPUs de 16 GB (Tesla T4)

Este diretório fornece o template pronto para submeter modelos baseados em **Mixture of Scars (MoS) + SCache** em competições do Kaggle.

---

## 🎯 Por que o MoS é uma Vantagem Competitiva Injusta no Kaggle?

Nas competições do Kaggle (como *Kaggle AI Olympiad*, *ARC Prize*, *LLM Science Exam*, *LLM 20 Questions*):
1. **A Restrição:** Os notebooks de submissão rodam **sem internet** e com limite estrito de **16 GB de VRAM** (Tesla T4) e 9 horas de tempo limite.
2. **O Limite dos Concorrentes:** Todos são obrigados a usar modelos pequenos de **7B ou 8B** (Llama-3-8B, Gemma-2-9B) porque um 70B tradicional dá `CUDA Out Of Memory` instantâneo.
3. **O Diferencial MoS:** Nós conseguimos instanciar a profundidade e a inteligência de uma arquitetura de **72 Camadas (escala 70B)** consumindo apenas **4.2 GB de VRAM**!
4. **O Diferencial SCache:** Usando `SCacheDynamicCache`, o contexto longo não consome RAM linearmente, garantindo que o notebook nunca crashe por memória, mesmo processando perguntas de milhares de tokens.

---

## 🚀 Como Submeter em uma Competição:

1. **Criar um Dataset Privado no Kaggle com os Pesos e o Wheel:**
   * Baixe o repositório como wheel: `pip wheel git+https://github.com/byte-od/mixture-of-scars.git`
   * Faça upload no Kaggle Datasets: `my-mos-tools`
2. **No seu Notebook de Submissão:**
   ```python
   # Instale o pacote offline a partir do dataset privado
   !pip install --no-index --find-links=/kaggle/input/my-mos-tools mixture_of_scars

   # Execute o pipeline de submissão
   from kaggle_submission_template.submission_pipeline import run_kaggle_submission
   run_kaggle_submission(
       test_csv_path="/kaggle/input/competicao/test.csv",
       output_csv_path="/kaggle/working/submission.csv"
   )
   ```
3. **Clique em "Save Version" -> "Save & Run All (Commit)".**
4. **Na aba de submissões da competição, selecione a versão gerada e envie!**
