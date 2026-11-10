import os
import sys
import json
import time
import glob
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["KAGGLE_API_TOKEN"] = "KGAT_51b5f3784b45c50dd053aa54332d3faa"

model_id = "unsloth/mistral-large-instruct-2407-bnb-4bit"
out_dir = Path("/content/mistral_large_123b_l44_backbone")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🚀 TITAN 3: MISTRAL LARGE 2 (123B) L44 MoS BACKBONE SETUP & KAGGLE UPLOAD")
print("=" * 80, flush=True)

# 1. Config (num_hidden_layers = 44)
print("[1/5] Preparando config.json (44 camadas)...", flush=True)
cfg_file = hf_hub_download(repo_id=model_id, filename="config.json", local_dir=str(out_dir))
with open(cfg_file, "r") as f:
    cfg = json.load(f)
cfg["num_hidden_layers"] = 44
cfg["use_cache"] = False
with open(out_dir / "config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("✓ config.json configurado com num_hidden_layers=44", flush=True)

# 2. Tokenizer
print("[2/5] Baixando arquivos de tokenização...", flush=True)
for tok_f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    try:
        hf_hub_download(repo_id=model_id, filename=tok_f, local_dir=str(out_dir))
        print(f"  ✓ {tok_f}")
    except Exception as e:
        print(f"  Tokenizer file {tok_f}: {e}")

# 3. Filter index
print("[3/5] Filtrando model.safetensors.index.json para as camadas 0..43...", flush=True)
idx_file = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json", local_dir=str(out_dir))
with open(idx_file, "r") as f:
    idx_data = json.load(f)

weight_map = idx_data["weight_map"]
filtered_weight_map = {}
needed_shards = set()

for k, shard in weight_map.items():
    if k.startswith("model.layers."):
        layer_idx = int(k.split(".")[2])
        if layer_idx < 44:
            filtered_weight_map[k] = shard
            needed_shards.add(shard)
    elif not k.startswith("lm_head."):
        filtered_weight_map[k] = shard
        needed_shards.add(shard)

filtered_idx = {
    "metadata": idx_data.get("metadata", {}),
    "weight_map": filtered_weight_map
}
with open(out_dir / "model.safetensors.index.json", "w") as f:
    json.dump(filtered_idx, f, indent=2)

shards_list = sorted(needed_shards)
print(f"✓ Shards necessários ({len(shards_list)}): {shards_list}", flush=True)

# 4. Download needed shards
print(f"[4/5] Baixando {len(shards_list)} shards MoS Golden-Slice (~36.75 GB)...", flush=True)
t0 = time.time()
for s in shards_list:
    target = out_dir / s
    if target.exists() and target.stat().st_size > 1024**3:
        print(f"  ✓ {s} já presente ({target.stat().st_size / (1024**3):.2f} GB)", flush=True)
    else:
        print(f"  ⬇ Baixando {s}...", flush=True)
        t_s = time.time()
        hf_hub_download(repo_id=model_id, filename=s, local_dir=str(out_dir))
        sz = target.stat().st_size / (1024**3)
        print(f"  ✓ {s} concluído ({sz:.2f} GB) em {time.time()-t_s:.1f}s", flush=True)
print(f"✓ Todos os shards Golden-Slice baixados com sucesso em {time.time()-t0:.1f}s!", flush=True)

# 5. Metadata for Kaggle dataset
print("[5/5] Publicando dataset no Kaggle...", flush=True)
meta = {
    "title": "NTL Titan 3 Mistral Large 123B L44 Backbone",
    "id": "thetod/name-that-llm-titan3-mistral-backbone",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(out_dir / "dataset-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

t_up = time.time()
res = subprocess.run(["kaggle", "datasets", "create", "-p", str(out_dir)], capture_output=True, text=True)
print("Returncode:", res.returncode, flush=True)
print("STDOUT:", res.stdout, flush=True)
print("STDERR:", res.stderr, flush=True)
print(f"✓ Upload finalizado em {time.time()-t_up:.1f}s!", flush=True)
