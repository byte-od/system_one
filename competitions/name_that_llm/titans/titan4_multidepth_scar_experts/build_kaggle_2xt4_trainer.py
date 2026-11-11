#!/usr/bin/env python3
"""
Gera o Kernel Privado Automatizado de Treinamento GPU (`2x T4`) no Kaggle para o Titan 4:
  `thetod/ntl-titan4-qwen3emb-mos-scar-trainer`
Monta offline (`enable_internet: false`):
  - Modelo oficial Kaggle: `qwen-lm/qwen-3-embedding/Transformers/4b/1` (36 camadas, 8 profundidades)
  - Competição: `name-that-llm` (`train.csv`)
  - Datasets estendidos: `nishantdahal/ntl-handwriting-blend` (`pool_v4.parquet`),
    `prvsiyan/prvsiyans-extended-name-that-llm-dataset` (`passages.csv`)
Treina 100% na infraestrutura GPU do Kaggle (Estágio 1 + 5-Fold Estágio 2 Multi-Multiplicidade)
e exporta `titan4_qwen3_emb_8b_fold0..4_scars.safetensors` + `meta_titan4_qwen3_emb_8b.json` + OOF `.npy`
diretamente em `/kaggle/working` para acoplamento imediato via `kernel_sources`!
"""

import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CORE_PY = (THIS_DIR / "train_two_stage_mos_scar_experts_colab.py").read_text()

KAGGLE_DIR = THIS_DIR / "kaggle_trainer"
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

RUNNER_SRC = f'''#!/usr/bin/env python3
import glob
import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

TRAINER_SCRIPT = Path("/tmp/train_two_stage_mos_scar_experts_colab.py")
TRAINER_SCRIPT.write_text({repr(CORE_PY)})

def find_one(pattern: str) -> str:
    hits = sorted(glob.glob(f"/kaggle/input/**/{{pattern}}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"Could not find {{pattern}} under /kaggle/input")
    return hits[0]

# Localiza modelo Qwen3-Embedding montado via model_sources
cfg_hits = [
    p for p in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
    if any(k in p.lower() for k in ("qwen-3-embedding", "qwen3-embedding", "qwen3_emb"))
]
if not cfg_hits:
    raise FileNotFoundError(f"Qwen3-Embedding config.json not found! Inputs: {{glob.glob('/kaggle/input/*/*')}}")

model_dir = os.path.dirname(cfg_hits[0])
train_csv = find_one("train.csv")
pool_v4 = find_one("pool_v4.parquet")
passages_csv = find_one("passages.csv")

print("=== KAGGLE 2xT4 TITAN-4 TWO-STAGE MoS SCAR TRAINER ===", flush=True)
print("  Model Dir   :", model_dir, flush=True)
print("  Train CSV   :", train_csv, flush=True)
print("  Pool v4     :", pool_v4, flush=True)
print("  Passages CSV:", passages_csv, flush=True)

cmd = [
    sys.executable,
    str(TRAINER_SCRIPT),
    "--model-id", model_dir,
    "--train-csv", train_csv,
    "--pool-v4", pool_v4,
    "--passages-csv", passages_csv,
    "--out-dir", "/kaggle/working",
    "--quant", "fp16",
    "--stage1-max-rows", "12600",
    "--stage1-epochs", "1",
    "--stage2-epochs", "4",
    "--n-folds", "5",
    "--max-folds-run", "5",
    "--batch-size", "16",
    "--max-len", "384",
    "--rank", "16",
]
subprocess.check_call(cmd)
'''

(KAGGLE_DIR / "train_titan4_kaggle_2xt4.py").write_text(RUNNER_SRC)
(KAGGLE_DIR / "kernel-metadata.json").write_text(
    json.dumps(
        {
            "id": "thetod/ntl-titan4-qwen3emb-mos-scar-trainer",
            "title": "NTL Titan4 Qwen3Emb MoS Scar Trainer",
            "code_file": "train_titan4_kaggle_2xt4.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [
                "nishantdahal/ntl-handwriting-blend",
                "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
            ],
            "kernel_sources": [],
            "competition_sources": ["name-that-llm"],
            "model_sources": ["qwen-lm/qwen-3-embedding/Transformers/4b/1"],
        },
        indent=2,
    )
)
print("✓ Generated Kaggle 2xT4 trainer package in:", KAGGLE_DIR)
