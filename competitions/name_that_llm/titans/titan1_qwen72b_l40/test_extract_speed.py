import time
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_DIR = Path("/Users/juliocesarreisfilho/Projects/scar-cache/models/qwen25_72b_mos")
DATA_PATH = Path("/Users/juliocesarreisfilho/Projects/scar-cache/competitions/name_that_llm/data/train.csv")

def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    unpacked = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=dtype, device=packed.device)
    unpacked[:, 0::2] = low.to(dtype)
    unpacked[:, 1::2] = high.to(dtype)
    return (unpacked * scales.to(dtype)).view(shape)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].head(10).tolist()
    
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    
    # Pre-tokenize all 10 texts
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    input_ids = enc.input_ids
    attn_mask = enc.attention_mask
    B, S = input_ids.shape
    print(f"Batch size: {B}, Seq len: {S}")
    
    # 1. Embedding lookup
    t0 = time.time()
    emb_path = MODEL_DIR / "mos_embeddings_and_head.safetensors"
    with safe_open(emb_path, framework="pt", device="cpu") as f:
        emb_packed = f.get_tensor("model.embed_tokens.weight")
        emb_scales = f.get_tensor("model.embed_tokens.weight.scales")
    
    blocks_per_token = 8192 // 128
    block_offsets = torch.arange(blocks_per_token, dtype=torch.long)
    all_block_indices = (input_ids.unsqueeze(-1) * blocks_per_token + block_offsets).view(-1)
    sel_packed = emb_packed[all_block_indices]
    sel_scales = emb_scales[all_block_indices]
    x = dequantize_int4(sel_packed, sel_scales, (B, S, 8192), dtype=torch.float32).to(device)
    print(f"Embedding done in {time.time()-t0:.2f}s, x shape: {x.shape}")
    
    # 2. Stream Groups 0..4 (40 layers)
    t_layers = time.time()
    for g in range(5):
        tg0 = time.time()
        g_path = MODEL_DIR / f"mos_group_{g}.safetensors"
        with safe_open(g_path, framework="pt", device="cpu") as f:
            bg_p = f.get_tensor(f"mos_base.group_{g}.mlp.gate_proj.weight")
            bg_s = f.get_tensor(f"mos_base.group_{g}.mlp.gate_proj.weight.scales")
            base_gate = dequantize_int4(bg_p, bg_s, (29568, 8192), dtype=torch.float32).to(device)

            bu_p = f.get_tensor(f"mos_base.group_{g}.mlp.up_proj.weight")
            bu_s = f.get_tensor(f"mos_base.group_{g}.mlp.up_proj.weight.scales")
            base_up = dequantize_int4(bu_p, bu_s, (29568, 8192), dtype=torch.float32).to(device)

            bd_p = f.get_tensor(f"mos_base.group_{g}.mlp.down_proj.weight")
            bd_s = f.get_tensor(f"mos_base.group_{g}.mlp.down_proj.weight.scales")
            base_down = dequantize_int4(bd_p, bd_s, (8192, 29568), dtype=torch.float32).to(device)

            for l in range(g * 8, (g + 1) * 8):
                # Layer RMSNorm
                x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                if l == g * 8:
                    gate = F.silu(F.linear(x_norm, base_gate))
                    up = F.linear(x_norm, base_up)
                    mlp_out = F.linear(gate * up, base_down)
                else:
                    A_gate = f.get_tensor(f"model.layers.{l}.mlp.gate_proj.expert.A").float().to(device)
                    B_gate = f.get_tensor(f"model.layers.{l}.mlp.gate_proj.expert.B").float().to(device)
                    A_up = f.get_tensor(f"model.layers.{l}.mlp.up_proj.expert.A").float().to(device)
                    B_up = f.get_tensor(f"model.layers.{l}.mlp.up_proj.expert.B").float().to(device)
                    A_down = f.get_tensor(f"model.layers.{l}.mlp.down_proj.expert.A").float().to(device)
                    B_down = f.get_tensor(f"model.layers.{l}.mlp.down_proj.expert.B").float().to(device)

                    gate = F.silu(F.linear(x_norm, base_gate) + (x_norm @ A_gate) @ B_gate)
                    up = F.linear(x_norm, base_up) + (x_norm @ A_up) @ B_up
                    inter = gate * up
                    mlp_out = F.linear(inter, base_down) + (inter @ A_down) @ B_down

                x = x + mlp_out
        print(f"Group {g} (8 layers) done in {time.time()-tg0:.2f}s")
    
    total_time = time.time() - t_layers
    print(f"Total 40 layers for {B} samples done in {total_time:.2f}s ({total_time/B:.2f}s per sample)")
    
    # Masked mean pooling
    mask = attn_mask.unsqueeze(-1).float().to(device)
    pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    print(f"Pooled shape: {pooled.shape}, is_finite: {torch.isfinite(pooled).all().item()}")

if __name__ == "__main__":
    main()
