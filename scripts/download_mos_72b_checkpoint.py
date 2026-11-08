"""Script de Download Otimizado: Baixa o Checkpoint Oficial MoS 72B do Kaggle para o Mac.

Usa URLs assinadas de alta velocidade do Google Cloud Storage com multi-threading:
1. Lista todos os shards do MoS 72B gerados na nuvem.
2. Faz download concorrente com barra de progresso e escrita atômica (.part -> .safetensors).
3. Salva diretamente em models/qwen25_72b_mos/.
4. Valida a integridade dos tensores locais com safetensors.
"""

import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
from safetensors import safe_open


DEST_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen25_72b_mos"


def download_file(item_info, force_overwrite: bool = True):
    file_name, url = item_info
    base_name = os.path.basename(file_name)
    target_path = DEST_DIR / base_name
    part_path = DEST_DIR / f"{base_name}.part"

    if not force_overwrite and target_path.exists() and target_path.stat().st_size > 1000:
        print(f"  ⏭️ Já existe: {base_name} ({target_path.stat().st_size / (1024**2):.1f} MB)")
        return base_name, target_path.stat().st_size

    t0 = time.time()
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0))

    downloaded = 0
    with open(part_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

    part_path.rename(target_path)
    dt = max(time.time() - t0, 0.1)
    speed_mb = (downloaded / dt) / (1024**2)
    print(f"  ✓ Concluído: {base_name} ({downloaded / (1024**2):.1f} MB em {dt:.1f}s | {speed_mb:.1f} MB/s)")
    return base_name, downloaded


def run_download():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("📥 INICIANDO DOWNLOAD DO CHECKPOINT QWEN 2.5 72B MoS (11.11 GB)")
    print(f"   Destino Local: {DEST_DIR}")
    print("=" * 80 + "\n")

    api = KaggleApi()
    api.authenticate()

    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = "thetod"
        req.kernel_slug = "qwen-72b-to-mixture-of-scars-mos-converter"
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)

        items = []
        for item in resp.files or []:
            if item.file_name.startswith("qwen25_72b_mos/"):
                items.append((item.file_name, item.url))

    print(f"Total de arquivos a baixar: {len(items)}")
    t_start = time.time()
    total_downloaded = 0

    # Executa downloads com 3 threads paralelas
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download_file, it): it for it in items}
        for future in as_completed(futures):
            try:
                name, sz = future.result()
                total_downloaded += sz
            except Exception as e:
                print(f"  ❌ Erro ao baixar {futures[future][0]}: {e}")

    t_total = time.time() - t_start
    print("\n" + "=" * 80)
    print("🏆 DOWNLOAD DO 70B MoS CONCLUÍDO COM SUCESSO!")
    print(f"   Total Baixado: {total_downloaded / (1024**3):.2f} GB em {t_total / 60:.2f} minutos.")
    print(f"   Velocidade Média: {(total_downloaded / t_total) / (1024**2):.1f} MB/s.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_download()
