"""End-to-end unit tests for Canonical Generative MoS v2 Exporter, Local Streaming Engine, SCache O(1), and Resumable HTTP Downloader."""
from __future__ import annotations

import functools
import http.server
import socketserver
import threading
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from scar_cache.cli_72b import CanonicalMoSEngine, download_canonical_checkpoint
from scripts.export_canonical_72b_mos import export_canonical_mos, merge_partial_manifests


def _create_tiny_qwen2_checkpoint(ckpt_dir: Path) -> Qwen2ForCausalLM:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok.save_pretrained(ckpt_dir)

    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    cfg.save_pretrained(ckpt_dir)
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(cfg).eval()
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(ckpt_dir / "model.safetensors"))
    return model


def test_canonical_mos_v2_multipart_export_streaming_and_http_download(tmp_path: Path) -> None:
    src_dir = tmp_path / "tiny_dense"
    mos_dir = tmp_path / "tiny_mos_v2"
    dl_dir = tmp_path / "tiny_mos_downloaded"

    dense_model = _create_tiny_qwen2_checkpoint(src_dir)

    # 1. Multi-part export (Group 0 + Group 1) + manifest merge
    export_canonical_mos(
        model_id=str(src_dir),
        out_dir=mos_dir,
        layers_per_group=2,
        mlp_rank=32,
        attn_rank=32,
        block_size=64,
        start_group=0,
        end_group=1,
        include_embed=True,
        include_head=False,
    )
    export_canonical_mos(
        model_id=str(src_dir),
        out_dir=mos_dir,
        layers_per_group=2,
        mlp_rank=32,
        attn_rank=32,
        block_size=64,
        start_group=1,
        end_group=2,
        include_embed=False,
        include_head=True,
    )
    merged = merge_partial_manifests(mos_dir)
    assert len(merged["files"]) == 4  # embed + 2 groups + lm_head

    # 2. Local HTTP download + SHA-256 verification
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(mos_dir))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            download_canonical_checkpoint(dl_dir, f"http://127.0.0.1:{port}")
            # Re-run to verify idempotent SHA-256 fast-skip path
            download_canonical_checkpoint(dl_dir, f"http://127.0.0.1:{port}")
        finally:
            httpd.shutdown()

    # 3. Streaming execution + unloaded group verification + KL fidelity + O(1) SCache
    eng = CanonicalMoSEngine(dl_dir, device="cpu", streaming=True)
    ids = torch.randint(10, 500, (1, 96), dtype=torch.long)

    with torch.no_grad():
        lg_dense = dense_model(ids).logits[:, -1, :].float()
    lg_mos = eng.forward(ids, scache=None)[:, -1, :].float()

    # Groups must be unloaded immediately after streaming forward
    assert all(not g.is_loaded for g in eng.groups)

    kl = float(F.kl_div(F.log_softmax(lg_mos, dim=-1), F.softmax(lg_dense, dim=-1), reduction="batchmean").item())
    assert kl < 0.01, f"Expected tight KL parity, got {kl}"

    # Verify strict 0-byte growth of SCache O(1) across horizons > window_size
    mem_sizes = []
    for seq_len in (32, 64, 96):
        sc = eng.make_scache(window_size=16, num_slots=8, num_sinks=4)
        for s0 in range(0, seq_len, 16):
            eng.forward(ids[:, s0 : s0 + 16], scache=sc)
        mem_sizes.append(sc.total_memory_bytes)

    assert max(mem_sizes) - min(mem_sizes) == 0, f"Expected 0 B SCache drift, got {mem_sizes}"
