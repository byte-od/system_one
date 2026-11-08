# Qwen 2.5 72B - Mixture of Scars (MoS) Compressed

Official compressed checkpoint of **Qwen 2.5 72B Instruct** powered by the **Mixture of Scars (MoS)** architecture and **SCache O(1)** memory.

## Architecture Highlights
- **Universal Base Weight Sharing:** The 80 deep transformer layers are grouped into 10 base clusters (1 base per 8 layers).
- **Low-Rank Scar Experts ($r=64$):** Residual divergences between layers are factored via Truncated SVD + Layerwise Activation Distillation into compact $A \in \mathbb{R}^{d_{\text{in}} \times 64}$ and $B \in \mathbb{R}^{64 \times d_{\text{out}}}$ matrices.
- **VRAM Footprint:** Fits in **under 12 GB of VRAM** (INT4 Base + BF16 Scars), enabling full 70B-scale inference on a single 16 GB GPU (Nvidia Tesla T4, RTX 4080, Apple Silicon 16GB).
- **Context Scalability:** Eliminates the linear $O(N)$ KV Cache explosion via **SCache $O(1)$**, keeping attention memory constant at 2.66 MB for 128k context tokens.

Developed by **Julio Cesar Reis Filho** & **Byte-OD Research Lab (2026)**.
GitHub: [https://github.com/byte-od/mixture-of-scars](https://github.com/byte-od/mixture-of-scars)
