#!/usr/bin/env python3
"""
Upload trained Tri-Compass + 70B MoS Head to Private Kaggle Dataset.

Ensures strict privacy: is_private: true.
Usage:
    python scripts/upload_tricompass_head_dataset.py [path_to_weights_dir]
"""

import sys
import os
import json
import shutil
from pathlib import Path

DEFAULT_DIR = Path("/Users/juliocesarreisfilho/Projects/scar-cache/competitions/name_that_llm/head_artifacts")

def main():
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files are in Downloads
    downloads_dir = Path.home() / "Downloads"
    for fn in ["head_tricompass_70b.safetensors", "meta_tricompass_70b.json"]:
        dest = target_dir / fn
        if not dest.exists():
            src = downloads_dir / fn
            if src.exists():
                print(f"📦 Copiando {fn} de ~/Downloads para {dest}...")
                shutil.copy2(src, dest)
                
    head_file = target_dir / "head_tricompass_70b.safetensors"
    meta_file = target_dir / "meta_tricompass_70b.json"
    
    if not head_file.exists():
        print(f"❌ Erro: {head_file} não encontrado! Salve o arquivo gerado pelo Colab em {target_dir} ou ~/Downloads.")
        sys.exit(1)
        
    print(f"✓ Pesos encontrados: {head_file} ({head_file.stat().st_size / 1024**2:.2f} MB)")
    
    # Prepara dataset-metadata.json com PRIVACIDADE ESTRITA
    metadata = {
        "title": "Name That LLM - 70B MoS TriCompass Head",
        "id": "thetod/name-that-llm-70b-mos-tricompass-head",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True
    }
    
    meta_path = target_dir / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("✓ dataset-metadata.json configurado com isPrivate: True")
    
    print("\n🚀 Criando / Atualizando dataset privado no Kaggle...")
    # Executa comando kaggle
    res = os.system(f"kaggle datasets create -p '{target_dir}' -r zip")
    if res != 0:
        print("Dataset pode já existir, tentando atualizar versão...")
        os.system(f"kaggle datasets version -p '{target_dir}' -m 'Update 70B MoS TriCompass Head' -r zip")

    print("\n✅ Dataset do Kaggle pronto para ser acoplado no Kernel de Submissão!")

if __name__ == "__main__":
    main()
