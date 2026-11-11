import time
import torch
import pandas as pd
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer

from scar_cache.qwen72b_mos_pipeline import Qwen72BMoSGroup, dequantize_int4

def precompute_freqs_cis(position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype):
    dim = 128; theta = 1000000.0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim))
    freqs = torch.outer(position_ids.reshape(-1).float(), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cis = torch.cat([freqs_cis, freqs_cis], dim=-1)
    cos = freqs_cis.cos().to(dtype).reshape(position_ids.shape[0], 1, position_ids.shape[1], dim)
    sin = freqs_cis.sin().to(dtype).reshape(position_ids.shape[0], 1, position_ids.shape[1], dim)
    return cos, sin

MODEL_DIR = Path("/Users/juliocesarreisfilho/Projects/scar-cache/models/qwen25_72b_mos")
DATA_PATH = Path("/Users/juliocesarreisfilho/Projects/scar-cache/competitions/name_that_llm/data/train.csv")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    print(f"Device: {device}, dtype: {dtype}")

    df = pd.read_csv(DATA_PATH)
    texts = df["text"].head(16).tolist()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc.input_ids
    attn_mask = enc.attention_mask.unsqueeze(-1)
    bsz, seq_len = input_ids.shape
    print(f"Batch: {bsz}, Seq: {seq_len}")

    emb_file = MODEL_DIR / "mos_embeddings_and_head.safetensors"
    with safe_open(emb_file, framework="pt", device="cpu") as f:
        emb_p = f.get_tensor("model.embed_tokens.weight")
        emb_s = f.get_tensor("model.embed_tokens.weight.scales")
    
    bpt = 8192 // 128
    boff = torch.arange(bpt, dtype=torch.long)
    all_bidx = (input_ids.unsqueeze(-1) * bpt + boff).view(-1)
    x = dequantize_int4(emb_p[all_bidx], emb_s[all_bidx], (bsz, seq_len, 8192), dtype=dtype, target_device=device)
    print(f"Embedding initialized: {x.shape}")

    pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)
    cos, sin = precompute_freqs_cis(pos, device, dtype)

    # Load and test Group 0 (layers 0..7)
    g0 = Qwen72BMoSGroup(0, MODEL_DIR / "mos_group_0.safetensors", device, dtype=dtype)
    g0.load()
    print("Group 0 loaded successfully!")

    t0 = time.time()
    for l in range(8):
        x = g0.forward_layer(l, x, cos, sin)
    dt = time.time() - t0
    print(f"Full 8-layer Attention + MoS MLP forward pass done in {dt:.3f}s! ({dt/bsz:.4f}s per sample)")
    print(f"Output x is_finite: {torch.isfinite(x).all().item()}")

if __name__ == "__main__":
    main()
