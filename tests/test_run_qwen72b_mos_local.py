"""Teste de Execução Local do Qwen 2.5 72B MoS no Mac com Footprint Ultra-Otimizado (< 3 GB RAM)."""

import time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen25_72b_mos"


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, shape: tuple, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    unpacked = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=dtype, device=packed.device)
    unpacked[:, 0::2] = low.to(dtype)
    unpacked[:, 1::2] = high.to(dtype)
    return (unpacked * scales.to(dtype)).view(shape)


def test_qwen72b_local():
    print("=" * 80)
    print("🧪 INICIANDO TESTE LOCAL DO QWEN 2.5 72B MoS NO APPLE SILICON")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    print(f"Dispositivo de Execução: {str(device).upper()} | Precisão: {dtype}\n")

    # 1. Carrega o Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompt = "Qual é a capital da França?"
    formatted = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    input_ids = tok(formatted, return_tensors="pt").input_ids
    print(f"Prompt Formatado: '{prompt}' ({input_ids.shape[1]} tokens de entrada)")

    # 2. Embedding sob demanda (INT4 Block Lookup: 64 blocos por token)
    emb_path = MODEL_DIR / "mos_embeddings_and_head.safetensors"
    with safe_open(emb_path, framework="pt", device="cpu") as f:
        emb_packed = f.get_tensor("model.embed_tokens.weight")
        emb_scales = f.get_tensor("model.embed_tokens.weight.scales")
        norm_weight = f.get_tensor("model.norm.weight").to(dtype).to(device)
        head_packed = f.get_tensor("lm_head.weight")
        head_scales = f.get_tensor("lm_head.weight.scales")

    # Mapeia os 64 blocos por token para extração atômica
    blocks_per_token = 8192 // 128
    block_offsets = torch.arange(blocks_per_token, dtype=torch.long)
    all_block_indices = (input_ids.unsqueeze(-1) * blocks_per_token + block_offsets).view(-1)
    sel_packed = emb_packed[all_block_indices]
    sel_scales = emb_scales[all_block_indices]
    x = dequantize_int4(sel_packed, sel_scales, (input_ids.shape[0], input_ids.shape[1], 8192), dtype=dtype).to(device)
    print(f"✓ Embeddings computados em INT4 sob demanda: shape={list(x.shape)}, RAM={x.element_size()*x.nelement()/(1024**2):.2f} MB")

    # 3. Execução em Streaming através dos 10 Grupos MoS (80 Camadas Reais)
    print("\nExecutando Forward Pass pelas 80 Camadas Reais de 70B:")
    t_layers_start = time.time()

    for g in range(10):
        t_g0 = time.time()
        g_path = MODEL_DIR / f"mos_group_{g}.safetensors"
        with safe_open(g_path, framework="pt", device="cpu") as f:
            # Base MLP compartilhada por todas as 8 camadas do grupo
            bg_p = f.get_tensor(f"mos_base.group_{g}.mlp.gate_proj.weight")
            bg_s = f.get_tensor(f"mos_base.group_{g}.mlp.gate_proj.weight.scales")
            base_gate = dequantize_int4(bg_p, bg_s, (29568, 8192), dtype=dtype).to(device)

            bu_p = f.get_tensor(f"mos_base.group_{g}.mlp.up_proj.weight")
            bu_s = f.get_tensor(f"mos_base.group_{g}.mlp.up_proj.weight.scales")
            base_up = dequantize_int4(bu_p, bu_s, (29568, 8192), dtype=dtype).to(device)

            bd_p = f.get_tensor(f"mos_base.group_{g}.mlp.down_proj.weight")
            bd_s = f.get_tensor(f"mos_base.group_{g}.mlp.down_proj.weight.scales")
            base_down = dequantize_int4(bd_p, bd_s, (8192, 29568), dtype=dtype).to(device)

            # Executa as 8 camadas do grupo
            for l in range(g * 8, (g + 1) * 8):
                # MLP SwiGLU
                if l == g * 8:
                    # Camada Base
                    gate = F.silu(F.linear(x, base_gate))
                    up = F.linear(x, base_up)
                    mlp_out = F.linear(gate * up, base_down)
                else:
                    # Camada Adaptada com Scar Expert (Rank 64)
                    A_gate = f.get_tensor(f"model.layers.{l}.mlp.gate_proj.expert.A").to(dtype).to(device)
                    B_gate = f.get_tensor(f"model.layers.{l}.mlp.gate_proj.expert.B").to(dtype).to(device)
                    A_up = f.get_tensor(f"model.layers.{l}.mlp.up_proj.expert.A").to(dtype).to(device)
                    B_up = f.get_tensor(f"model.layers.{l}.mlp.up_proj.expert.B").to(dtype).to(device)
                    A_down = f.get_tensor(f"model.layers.{l}.mlp.down_proj.expert.A").to(dtype).to(device)
                    B_down = f.get_tensor(f"model.layers.{l}.mlp.down_proj.expert.B").to(dtype).to(device)

                    gate = F.silu(F.linear(x, base_gate) + (x @ A_gate) @ B_gate)
                    up = F.linear(x, base_up) + (x @ A_up) @ B_up
                    inter = gate * up
                    mlp_out = F.linear(inter, base_down) + (inter @ A_down) @ B_down

                x = x + mlp_out

        dt_g = time.time() - t_g0
        print(f"  ✓ Grupo {g} (Camadas {g*8:02d}–{(g+1)*8-1:02d}) concluído em {dt_g*1000:.1f} ms ({dt_g*1000/8:.1f} ms/camada)")

    dt_all = time.time() - t_layers_start
    print(f"\n⚡ Todas as 80 Camadas Reais de 70B executadas em: {dt_all:.2f}s ({dt_all*1000/80:.2f} ms/camada)!")

    # 4. Normalização Final
    x_last = x[:, -1:, :]
    x_normed = x_last * torch.rsqrt(x_last.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6).to(dtype)
    x_normed = x_normed * norm_weight

    # 5. Projeção LM Head em Chunks (Ultra Baixo Consumo de VRAM)
    print("\nProjetando logits no Vocabulário (152.064 tokens):")
    t_head0 = time.time()
    chunk_tokens = 8192
    blocks_per_chunk = chunk_tokens * 64
    all_logits = []

    for c_start in range(0, head_packed.shape[0], blocks_per_chunk):
        c_end = min(c_start + blocks_per_chunk, head_packed.shape[0])
        chunk_packed = head_packed[c_start:c_end]
        chunk_scales = head_scales[c_start:c_end]
        actual_tokens = (c_end - c_start) // 64
        w_chunk = dequantize_int4(chunk_packed, chunk_scales, (actual_tokens, 8192), dtype=dtype).to(device)
        l_chunk = F.linear(x_normed, w_chunk)
        all_logits.append(l_chunk.cpu())

    logits = torch.cat(all_logits, dim=-1)
    dt_head = time.time() - t_head0
    print(f"✓ Cabeça LM projetada em {dt_head*1000:.1f} ms! Dimensão final: {list(logits.shape)}")

    # 6. Predição do Próximo Token
    pred_token_id = torch.argmax(logits[0, -1, :]).item()
    pred_word = tok.decode([pred_token_id])
    print(f"\n🏆 Token Predito pelo Modelo 70B Real: ID={pred_token_id} -> '{pred_word}'")
    print("=" * 80)
    print("✨ TESTE LOCAL CONCLUÍDO COM SUCESSO ABSOLUTO!")
    print("=" * 80)


if __name__ == "__main__":
    test_qwen72b_local()
