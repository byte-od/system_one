import time
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
    texts = df["text"].head(32).tolist()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    
    # Pre-load all 5 groups (Groups 0..4 = 40 layers)
    t_load = time.time()
    groups = []
    for g in range(5):
        grp = Qwen72BMoSGroup(g, MODEL_DIR / f"mos_group_{g}.safetensors", device, dtype=dtype)
        grp.load()
        groups.append(grp)
        print(f"  ✓ Group {g} loaded ({time.time()-t_load:.2f}s)")
    print(f"All 5 groups (40 layers) resident in memory in {time.time()-t_load:.2f}s")

    # Load embeddings lookup
    emb_file = MODEL_DIR / "mos_embeddings_and_head.safetensors"
    with safe_open(emb_file, framework="pt", device="cpu") as f:
        emb_p = f.get_tensor("model.embed_tokens.weight")
        emb_s = f.get_tensor("model.embed_tokens.weight.scales")
    bpt = 8192 // 128
    boff = torch.arange(bpt, dtype=torch.long)

    # Run 2 batches of 16
    batch_size = 16
    max_len = 128
    t_fwd_all = time.time()

    for b in range(2):
        tb = time.time()
        b_texts = texts[b*batch_size : (b+1)*batch_size]
        enc = tok(b_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = enc.input_ids
        attn_mask = enc.attention_mask.unsqueeze(-1).to(device)
        bsz, seq_len = input_ids.shape

        all_bidx = (input_ids.unsqueeze(-1) * bpt + boff).view(-1)
        x = dequantize_int4(emb_p[all_bidx], emb_s[all_bidx], (bsz, seq_len, 8192), dtype=dtype, target_device=device)

        pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)
        cos, sin = precompute_freqs_cis(pos, device, dtype)

        with torch.no_grad():
            for grp in groups:
                for l in range(grp.start_layer, grp.end_layer):
                    x = grp.forward_layer(l, x, cos, sin)

            den = attn_mask.sum(dim=1).clamp(min=1)
            pooled = (x.float() * attn_mask.float()).sum(dim=1) / den
            rms = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            res = (pooled * rms).cpu().numpy()

        print(f"Batch {b+1}/2 (16 samples, 40 layers) done in {time.time()-tb:.3f}s! Result shape: {res.shape}")

    total_dt = time.time() - t_fwd_all
    print(f"\n⚡ Total time for 32 samples across 40 layers: {total_dt:.3f}s ({total_dt/32:.4f}s per sample)!")

if __name__ == "__main__":
    main()
