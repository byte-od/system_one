import time
import torch
import pandas as pd
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_DIR = Path("/Users/juliocesarreisfilho/Projects/scar-cache/models/qwen25_72b_mos")
DATA_PATH = Path("/Users/juliocesarreisfilho/Projects/scar-cache/competitions/name_that_llm/data/train.csv")

def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, shape: tuple, dtype: torch.dtype = torch.float16, target_device: str = "cpu") -> torch.Tensor:
    dev = target_device
    p = packed.to(dev)
    s = scales.to(dev).to(dtype)
    low = (p & 0x0F).to(torch.int16) - 8
    high = ((p >> 4) & 0x0F).to(torch.int16) - 8
    unpacked = torch.empty((p.shape[0], p.shape[1] * 2), dtype=dtype, device=dev)
    unpacked[:, 0::2] = low.to(dtype)
    unpacked[:, 1::2] = high.to(dtype)
    return (unpacked * s).view(shape)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    print(f"Device: {device}, dtype: {dtype}")

    df = pd.read_csv(DATA_PATH)
    texts = df["text"].head(64).tolist()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    
    # Pre-load group 0 once into MPS
    t_load = time.time()
    g_path = MODEL_DIR / "mos_group_0.safetensors"
    with safe_open(g_path, framework="pt", device="cpu") as f:
        bg_p = f.get_tensor("mos_base.group_0.mlp.gate_proj.weight")
        bg_s = f.get_tensor("mos_base.group_0.mlp.gate_proj.weight.scales")
        base_gate = dequantize_int4(bg_p, bg_s, (29568, 8192), dtype, device)

        bu_p = f.get_tensor("mos_base.group_0.mlp.up_proj.weight")
        bu_s = f.get_tensor("mos_base.group_0.mlp.up_proj.weight.scales")
        base_up = dequantize_int4(bu_p, bu_s, (29568, 8192), dtype, device)

        bd_p = f.get_tensor("mos_base.group_0.mlp.down_proj.weight")
        bd_s = f.get_tensor("mos_base.group_0.mlp.down_proj.weight.scales")
        base_down = dequantize_int4(bd_p, bd_s, (8192, 29568), dtype, device)
    print(f"Group 0 loaded to {device} in {time.time()-t_load:.2f}s")

    # Run 2 batches of 32
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    input_ids = enc.input_ids
    B, S = input_ids.shape
    x = torch.randn(B, S, 8192, dtype=dtype, device=device)
    
    t_run = time.time()
    for l in range(8):
        x_norm = x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6).to(dtype)
        gate = torch.nn.functional.silu(torch.nn.functional.linear(x_norm, base_gate))
        up = torch.nn.functional.linear(x_norm, base_up)
        mlp_out = torch.nn.functional.linear(gate * up, base_down)
        x = x + mlp_out
    
    dt = time.time() - t_run
    print(f"8 layers for 64 samples done in {dt:.2f}s ({dt/64:.3f}s per sample)")

if __name__ == "__main__":
    main()
