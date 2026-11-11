#!/usr/bin/env python3
"""
One-command publisher for the official Hugging Face Gradio Space:
  HF_TOKEN=hf_xxx python huggingface_space/deploy_huggingface_space.py --repo-id <your-hf-user>/mos-scache-o1-lab
"""
import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def sync_local_scar_cache_bundle():
    dst = HERE / "scar_cache"
    dst.mkdir(parents=True, exist_ok=True)
    for py in sorted((REPO_ROOT / "scar_cache").glob("*.py")):
        shutil.copyfile(py, dst / py.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/mos-scache-o1-lab")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    sync_local_scar_cache_bundle()

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(HERE),
        repo_id=args.repo_id,
        repo_type="space",
        ignore_patterns=["__pycache__/*", "*.pyc", "deploy_huggingface_space.py"],
        commit_message="Deploy Mixture of Scars & SCache O(1) Red-Team Lab v1.0.0 (DOI: 10.5281/zenodo.24143839)",
    )
    print(f"✓ Live Space deployed: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
