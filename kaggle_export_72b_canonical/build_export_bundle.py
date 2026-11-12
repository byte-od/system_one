#!/usr/bin/env python3
"""
Builds the 3-Part Standalone Kaggle Dual-T4 Exporters for Canonical Generative `Qwen2.5-72B-Instruct` (`MoS v2`):
  - `part1`: `--start-group 0 --end-group 7 --no-head` (`mos_embed_tokens` + `mos_group_00..06` = ~13.7 GiB < 19.5 GiB Kaggle cap)
  - `part2`: `--start-group 7 --end-group 14 --no-embed --no-head` (`mos_group_07..13` = ~12.5 GiB < 19.5 GiB Kaggle cap)
  - `part3`: `--start-group 14 --end-group 20 --no-embed` (`mos_group_14..19` + `mos_lm_head` + tokenizer/config = ~11.9 GiB < 19.5 GiB Kaggle cap)
"""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PARTS = [
    {
        "dir": "part1",
        "slug": "thetod/qwen2-5-72b-canonical-mos-v2-part-1",
        "title": "Qwen2.5 72B Canonical MoS v2 Part 1",
        "script": "run_export_part1.py",
        "extra_flags": '["--start-group", "0", "--end-group", "7", "--no-head"]',
    },
    {
        "dir": "part2",
        "slug": "thetod/qwen2-5-72b-canonical-mos-v2-part-2",
        "title": "Qwen2.5 72B Canonical MoS v2 Part 2",
        "script": "run_export_part2.py",
        "extra_flags": '["--start-group", "7", "--end-group", "14", "--no-embed", "--no-head"]',
    },
    {
        "dir": "part3",
        "slug": "thetod/qwen2-5-72b-canonical-mos-v2-part-3",
        "title": "Qwen2.5 72B Canonical MoS v2 Part 3",
        "script": "run_export_part3.py",
        "extra_flags": '["--start-group", "14", "--end-group", "20", "--no-embed"]',
    },
]


def main() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(REPO / "scripts" / "export_canonical_72b_mos.py", arcname="scripts/export_canonical_72b_mos.py")
    b64_payload = base64.b64encode(buf.getvalue()).decode("ascii")

    for part in PARTS:
        pdir = HERE / part["dir"]
        pdir.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": part["slug"],
            "title": part["title"],
            "code_file": part["script"],
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": ["qwen-lm/qwen2.5/transformers/72b-instruct/1"],
        }
        (pdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        runner = f'''#!/usr/bin/env python3
"""Standalone Kaggle GPU Exporter — {part["title"]}"""
import base64
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "transformers>=5.0.0", "accelerate>=0.34.0", "safetensors>=0.4.3", "huggingface_hub>=0.26.0"
])

WORK_PKG = Path("/tmp/mos_v2_exporter")
WORK_PKG.mkdir(parents=True, exist_ok=True)
PAYLOAD_B64 = "{b64_payload}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD_B64)), "r") as zf:
    zf.extractall(WORK_PKG)

out_dir = Path("/kaggle/working/qwen25_72b_mos_v2")
cmd = [
    sys.executable,
    str(WORK_PKG / "scripts" / "export_canonical_72b_mos.py"),
    "--model-id", "Qwen/Qwen2.5-72B-Instruct",
    "--out-dir", str(out_dir),
    "--layers-per-group", "4",
    "--mlp-rank", "48",
    "--attn-rank", "32",
] + {part["extra_flags"]}

env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["PYTHONUNBUFFERED"] = "1"
print("[EXEC]", " ".join(cmd))
subprocess.check_call(cmd, env=env)
'''
        (pdir / part["script"]).write_text(runner, encoding="utf-8")
        print(f"Built {pdir / part['script']} + kernel-metadata.json")


if __name__ == "__main__":
    main()
