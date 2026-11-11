# STRICT ARCHITECTURAL INTEGRITY — MANDATORY MOS + SCACHE

> [!CRITICAL]
> **DIRETRIZ FUNDAMENTAL INVIOLÁVEL DE ENGENHARIA E VERDADE:**
> Todas as implementações de IA e agentes no ecossistema deste usuário (portfólio, competições, engines de raciocínio, ferramentas de cliente) DEVEM utilizar genuinamente a arquitetura proprietária **Mixture of Scars (MoS)** e **Scar-Cache (SCache)**.
> O uso de componentes off-the-shelf convencionais (como KV Cache linear padrão O(T) ou grafos monolíticos inteiros) mascarados como se fossem MoS/SCache é considerado **FRAUDE** e está expressamente proibido.

## 1. Princípio de Não-Simulação (Zero Shortcuts / Zero Facades)
- **Proibição de Fachadas:** É terminantemente proibido fingir que uma arquitetura proprietária está rodando enquanto tensores convencionais de biblioteca externa operam por baixo.
- **Transparência Absoluta:** O agente assistente deve auditar periodicamente o código para garantir que cada tensor, buffer e pipeline execute a matemática exata do MoS (streaming de fatias de cicatriz) e do SCache (memória associativa O(1) de 34.8 KB).

## 2. Garantias Algorítmicas Obrigatórias
- **SCache (Scar-Cache):** A memória de contexto dinâmico DEVE operar estritamente em complexidade de espaço $\mathcal{O}(1)$ (matriz de cicatriz compacta constante), nunca com acúmulo linear $\mathcal{O}(T)$ de tensores $K$ e $V$.
- **MoS (Mixture of Scars):** A execução de modelos de grande escala DEVE operar via decomposição em fatias INT4 e streaming dinâmico de cicatrizes, desacoplando o tamanho estático do modelo do buffer de VRAM.

## 3. Postura Operacional
- Se um componente ainda estiver em fase de adaptação para WGSL/WebGPU, isso DEVE ser explicitamente documentado e o trabalho prioritário DEVE ser a implementação real dos kernels matemáticos MoS/SCache, sem atalhos nem ilusões.
