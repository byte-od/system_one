#!/usr/bin/env python3
"""
High-Speed Parallel Downloader & Manifest Merger for the 3-Part Canonical `Qwen2.5-72B-Instruct` (`MoS v2`)
Kaggle Exporter Kernels (`part1`, `part2`, `part3`) -> `models/qwen25_72b_mos_v2/`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.export_canonical_72b_mos import merge_partial_manifests

DEST_DIR = ROOT_DIR / "models" / "qwen25_72b_mos_v2"
KERNEL_SLUGS = [
    "qwen2-5-72b-canonical-mos-v2-part-1",
    "qwen2-5-72b-canonical-mos-v2-part-2",
    "qwen2-5-72b-canonical-mos-v2-part-3",
]


def _download_one(item: Tuple[str, str]) -> Tuple[str, int]:
    file_name, url = item
    base_name = os.path.basename(file_name)
    target = DEST_DIR / base_name
    part = DEST_DIR / f"{base_name}.part"

    t0 = time.time()
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    expected_len = int(resp.headers.get("content-length", 0))
    if target.exists() and expected_len > 0 and target.stat().st_size == expected_len:
        resp.close()
        print(f"  ✓ Exists: {base_name} ({expected_len / (1024**2):.1f} MiB)")
        return base_name, expected_len

    written = 0
    with open(part, "wb") as f:
        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                written += len(chunk)
    part.rename(target)
    dt = max(time.time() - t0, 0.1)
    print(f"  ✓ Downloaded: {base_name} ({written / (1024**2):.1f} MiB in {dt:.1f}s | {(written / dt) / (1024**2):.1f} MiB/s)")
    return base_name, written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parts", default="1,2,3", help="Comma-separated exporter parts to pull (e.g. '1' or '1,2,3')")
    p.add_argument("--skip-merge", action="store_true", help="Skip final manifest merge when pulling partial subsets")
    args = p.parse_args()

    part_nums = [int(x.strip()) for x in args.parts.split(",") if x.strip()]
    selected_slugs = [KERNEL_SLUGS[i - 1] for i in part_nums]

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    dedup_items: Dict[str, Tuple[str, str]] = {}
    with api.build_kaggle_client() as kaggle:
        for slug in selected_slugs:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = "thetod"
            req.kernel_slug = slug
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            for it in resp.files or []:
                if it.file_name.startswith("qwen25_72b_mos_v2/"):
                    base_name = os.path.basename(it.file_name)
                    dedup_items[base_name] = (it.file_name, it.url)

    all_items: List[Tuple[str, str]] = list(dedup_items.values())
    print(f"📥 Downloading {len(all_items)} unique canonical MoS v2 artifacts from {selected_slugs} -> {DEST_DIR}")
    t0 = time.time()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_download_one, it): it[0] for it in all_items}
        for fut in as_completed(futs):
            _, sz = fut.result()
            total_bytes += sz

    if args.skip_merge:
        print(f"✅ Partial download complete ({total_bytes / (1024**3):.2f} GiB in {(time.time()-t0)/60:.1f} min).")
        return

    merged = merge_partial_manifests(DEST_DIR)
    print("🔍 Verifying SHA-256 checksums across all shards...")
    for fe in merged["files"]:
        fp = DEST_DIR / str(fe["name"])
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            while b := f.read(8 * 1024 * 1024):
                h.update(b)
        if h.hexdigest() != fe["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch on {fp.name}: {h.hexdigest()} != {fe['sha256']}")
    print(f"🏆 ALL {len(merged['files'])} SHARDS VERIFIED ({merged['total_gib']} GiB in {(time.time()-t0)/60:.1f} min)!")


if __name__ == "__main__":
    main()
