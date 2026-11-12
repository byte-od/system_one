#!/usr/bin/env python3
"""Bundles verified `scar_cache` + `adversarial_scale_72b_suite.py` into a standalone Kaggle 2xT4 GPU script."""
import base64
import io
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted((REPO / "scar_cache").glob("*.py")):
            zf.write(p, arcname=f"scar_cache/{p.name}")
        suite = REPO / "scripts" / "adversarial_scale_72b_suite.py"
        zf.write(suite, arcname="scripts/adversarial_scale_72b_suite.py")

    b64_payload = base64.b64encode(buf.getvalue()).decode("ascii")
    runner_code = f'''#!/usr/bin/env python3
"""
Standalone Kaggle 2xT4 GPU Runner — MoS & SCache O(1) 72.7B Adversarial Red-Team Suite
Mounts `qwen-lm/qwen2.5/transformers/72b-instruct/1` (37 dense BF16 safetensors shards = 145.4 GB)
read-only in `/kaggle/input` and executes the full 8-vector 72.7B audit privately before Zenodo v1.2.
"""
import base64
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

print("[BOOT] Upgrading transformers>=5.0.0, accelerate, safetensors for exact DynamicLayer parity...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "transformers>=5.0.0", "accelerate>=0.34.0", "safetensors>=0.4.3", "huggingface_hub>=0.26.0"
])

WORK_PKG = Path("/tmp/mos_scache_72b_audit")
WORK_PKG.mkdir(parents=True, exist_ok=True)
PAYLOAD_B64 = "{b64_payload}"

with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD_B64)), "r") as zf:
    zf.extractall(WORK_PKG)

out_json = Path("/kaggle/working/adversarial_scale_72b_report.json")
if not out_json.parent.exists():
    out_json = Path("./adversarial_scale_72b_report.json")

cmd = [
    sys.executable,
    str(WORK_PKG / "scripts" / "adversarial_scale_72b_suite.py"),
    "--model-id", "Qwen/Qwen2.5-72B-Instruct",
    "--output-json", str(out_json),
]
env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["PYTHONUNBUFFERED"] = "1"
print("[EXEC]", " ".join(cmd))
subprocess.check_call(cmd, env=env)
print("[DONE] Saved verified 72.7B JSON telemetry to:", out_json)
'''
    target = HERE / "run_72b_redteam.py"
    target.write_text(runner_code, encoding="utf-8")
    print(f"Built {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
