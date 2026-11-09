import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer

from scar_cache.qwen72b_mos_pipeline import Qwen72BMoSGroup, dequantize_int4

MODEL_DIR = Path("/Users/juliocesarreisfilho/Projects/scar-cache/models/qwen25_72b_mos")
DATA_PATH = Path("/Users/juliocesarreisfilho/Projects/scar-cache/competitions/name_that_llm/data/train.csv")

def precompute_freqs_cis(position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype):
    dim = 128; theta = 1000000.0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim))
    freqs = torch.outer(position_ids.reshape(-1).float(), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cis = torch.cat([freqs_cis, freqs_cis], dim=-1)
    cos = freqs_cis.cos().to(dtype).reshape(position_ids.shape[0], 1, position_ids.shape[1], dim)
    sin = freqs_cis.sin().to(dtype).reshape(position_ids.shape[0], 1, position_ids.shape[1], dim)
    return cos, sin

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    print(f"Device: {device}, dtype: {dtype}")

    df = pd.read_csv(DATA_PATH)
    N = 64
    texts = df["text"].head(N).tolist()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    
    max_len = 128
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = enc.input_ids
    attn_mask = enc.attention_mask.unsqueeze(-1)
    
    # 1. Initialize CPU resident tensor
    t0 = time.time()
    X_cpu = torch.empty((N, max_len, 8192), dtype=dtype, device="cpu")
    emb_file = MODEL_DIR / "mos_embeddings_and_head.safetensors"
    with safe_open(emb_file, framework="pt", device="cpu") as f:
        emb_p = f.get_tensor("model.embed_tokens.weight")
        emb_s = f.get_tensor("model.embed_tokens.weight.scales")
    bpt = 8192 // 128
    boff = torch.arange(bpt, dtype=torch.long)
    all_bidx = (input_ids.unsqueeze(-1) * bpt + boff).view(-1)
    X_cpu[:] = dequantize_int4(emb_p[all_bidx], emb_s[all_bidx], (N, max_len, 8192), dtype=dtype, target_device=torch.device("cpu"))
    del emb_p, emb_s
    print(f"Initialized CPU buffer for {N} samples in {time.time()-t0:.2f}s ({X_cpu.element_size()*X_cpu.nelement()/(1024**2):.1f} MB)")

    batch_size = 16
    num_batches = N // batch_size

    # Precompute RoPE positions
    pos = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    cos, sin = precompute_freqs_cis(pos, device, dtype)

    # 2. Sequential Group Execution with strict 1-group memory footprint
    t_groups_all = time.time()
    for g in range(5):
        tg = time.time()
        grp = Qwen72BMoSGroup(g, MODEL_DIR / f"mos_group_{g}.safetensors", device, dtype=dtype)
        grp.load()

        with torch.no_grad():
            for b in range(num_batches):
                s_idx = b * batch_size
                e_idx = s_idx + batch_size
                x_b = X_cpu[s_idx:e_idx].to(device)

                for l in range(grp.start_layer, grp.end_layer):
                    x_b = grp.forward_layer(l, x_b, cos, sin)

                X_cpu[s_idx:e_idx] = x_b.cpu()

        grp.unload()
        torch.mps.empty_cache()
        print(f"  ✓ Group {g} (Layers {g*8}..{g*8+7}) completed in {time.time()-tg:.2f}s")

    total_fwd = time.time() - t_groups_all
    print(f"\n⚡ All 40 layers for {N} samples completed in {total_fwd:.2f}s! ({total_fwd/N:.4f}s per sample)")

    # 3. Masked Mean Pooling
    den = attn_mask.float().sum(dim=1).clamp(min=1)
    pooled = (X_cpu.float() * attn_mask.float()).sum(dim=1) / den
    rms = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    Z = (pooled * rms).numpy()
    print(f"✓ Final pooled features shape: {Z.shape}, is_finite: {np.isfinite(Z).all()}")

if __name__ == "__main__":
    main()
