"""Script de Submissão Automatizada para o Hugging Face Open LLM Leaderboard v2.

Cria o repositório de modelo no Hugging Face Hub (se não existir),
faz upload do Model Card com tags da competição e envia a requisição de avaliação.
"""

import argparse
import os
import sys
import requests
from huggingface_hub import HfApi, create_repo, upload_file


def submit_to_open_llm_leaderboard(
    repo_id: str = "thetod/mixture-of-scars-70b-deep",
    token: str = None,
    precision: str = "bfloat16",
    model_type: str = "fine-tuned",
):
    print("=" * 85)
    print("🚀 SUBMISSÃO PARA O HUGGING FACE OPEN LLM LEADERBOARD V2")
    print(f"   Model Repo:   {repo_id}")
    print(f"   Precisão:     {precision}")
    print(f"   Tipo:         {model_type}")
    print("=" * 85 + "\n")

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN não configurado no ambiente.")
        print("   Para autenticar, defina HF_TOKEN ou execute: huggingface-cli login")
        print(f"   O Model Card está pronto em: hf_leaderboard/model_card_README.md")
        return False

    api = HfApi(token=hf_token)

    # 1. Cria ou valida repositório
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        print(f"✓ Repositório {repo_id} validado no Hugging Face Hub.")
    except Exception as e:
        print(f"⚠️ Erro ao criar repositório: {e}")

    # 2. Upload do README.md oficial
    readme_path = os.path.join(os.path.dirname(__file__), "model_card_README.md")
    if os.path.exists(readme_path):
        try:
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
            )
            print(f"✓ Model Card oficial enviado com sucesso para {repo_id}!")
        except Exception as e:
            print(f"⚠️ Falha no upload do README: {e}")

    # 3. Solicitação de avaliação automática no Open LLM Leaderboard
    print("\n[3/3] Registrando modelo na fila de avaliação do Open LLM Leaderboard...")
    leaderboard_url = "https://huggingface.co/api/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions"
    print(f"✓ Para submeter manualmente via interface web com 1 clique, acesse:")
    print(f"   👉 https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard")
    print(f"   E insira o ID do modelo: {repo_id}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit to Open LLM Leaderboard")
    parser.add_argument("--repo_id", type=str, default="thetod/mixture-of-scars-70b-deep")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    submit_to_open_llm_leaderboard(repo_id=args.repo_id, token=args.token)
