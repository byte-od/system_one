import json
from pathlib import Path

script_dir = Path("/Users/juliocesarreisfilho/Projects/scar-cache/competitions/name_that_llm/titans/titan4_multidepth_scar_experts")
py_code = (script_dir / "train_two_stage_mos_scar_experts_colab.py").read_text()

nb = {
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 👑 System One — Milestone 2 & 3 (Colab A100): Two-Stage MoS Residual Scar Experts (`Qwen3-Embedding-8B` 8-Depth)\n",
                "Este notebook executa o treinamento fim-a-fim em **2 Estágios** dos nossos **Especialistas de Cicatriz Residuais MoS** (`rank=16`, zero LoRAs de terceiros) acoplados à extração simultânea de **8 Profundidades (`[4, 8, 12, 16, 20, 24, 30, 36]`)** e **Janelas Deslizantes Sem Truncamento (`max_len=512, stride=384`)**:\n",
                "1. **Estágio 1:** Adaptação geral no corpus limpo desduplicado (`pool_v4.parquet` + `passages.csv` proxy-matched, sem sobreposição com `train.csv`).\n",
                "2. **Estágio 2:** Calibração fina no registro exato do host (`train.csv`, 5-Fold `StratifiedGroupKFold` por `prompt_id`), exportando os 5 estados `.safetensors` e o backbone NF4 MoS diretamente para o Kaggle."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q -U bitsandbytes accelerate safetensors transformers kaggle pyarrow scikit-learn\n",
                "import torch\n",
                "print('GPU:', torch.cuda.get_device_name(0), '| BF16:', torch.cuda.is_bf16_supported())\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Etapa 1: Autenticação Kaggle & Download dos Corpora Oficiais e Estendidos Limpos\n",
                "Faça upload do seu `kaggle.json` (ou defina no Colab Secrets) para baixar `train.csv`, `pool_v4.parquet` e `passages.csv` e publicar automaticamente os artefatos ao final."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, pathlib\n",
                "from google.colab import files\n",
                "kaggle_dir = pathlib.Path.home() / '.kaggle'\n",
                "kaggle_dir.mkdir(exist_ok=True)\n",
                "if not (kaggle_dir / 'kaggle.json').exists():\n",
                "    uploaded = files.upload()\n",
                "    for k, v in uploaded.items():\n",
                "        (kaggle_dir / 'kaggle.json').write_bytes(v)\n",
                "os.chmod(kaggle_dir / 'kaggle.json', 0o600)\n",
                "\n",
                "!mkdir -p /content/ntl_data\n",
                "!kaggle competitions download -c name-that-llm -f train.csv -p /content/ntl_data -q && unzip -o -q /content/ntl_data/train.csv.zip -d /content/ntl_data || true\n",
                "!kaggle datasets download -d nishantdahal/ntl-handwriting-blend -f pool_v4.parquet -p /content/ntl_data --unzip -q\n",
                "!kaggle datasets download -d prvsiyan/prvsiyans-extended-name-that-llm-dataset -f passages.csv -p /content/ntl_data --unzip -q\n",
                "!ls -lh /content/ntl_data\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Etapa 2: Treinamento 2-Estágios dos Especialistas de Cicatriz MoS + 8-Depth Readout + `SCache O(1)`\n",
                "Grava `train_two_stage_mos_scar_experts_colab.py` e executa o pipeline completo em BF16/NF4 no A100 (~45–55 min)."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                f"Path('/content/train_two_stage_mos_scar_experts_colab.py').write_text({repr(py_code)})\n",
                "!python3 /content/train_two_stage_mos_scar_experts_colab.py \\\n",
                "  --model-id Qwen/Qwen3-Embedding-8B \\\n",
                "  --train-csv /content/ntl_data/train.csv \\\n",
                "  --pool-v4 /content/ntl_data/pool_v4.parquet \\\n",
                "  --passages-csv /content/ntl_data/passages.csv \\\n",
                "  --out-dir /content/titan4_qwen3_emb_8b_mos_artifacts \\\n",
                "  --stage1-max-rows 36000 --stage1-epochs 2 --stage2-epochs 4 --batch-size 16 --rank 16\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Etapa 3: Publicação Automática dos Datasets Privados no Kaggle (`Backbone NF4` + `5-Fold Scar Experts`)\n",
                "Publica `thetod/name-that-llm-titan4-qwen3emb-mos-backbone` e `thetod/name-that-llm-titan4-qwen3emb-mos-scars` prontos para acoplamento imediato no kernel `thetod/mixture-of-scars-70b-mos-scache-name-that-llm`."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json, shutil\n",
                "from pathlib import Path\n",
                "\n",
                "art = Path('/content/titan4_qwen3_emb_8b_mos_artifacts')\n",
                "bb_dir = art / 'backbone_nf4_mos'\n",
                "(bb_dir / 'dataset-metadata.json').write_text(json.dumps({\n",
                "    'title': 'Name That LLM Titan 4 Qwen3 Emb MoS Backbone',\n",
                "    'id': 'thetod/name-that-llm-titan4-qwen3emb-mos-backbone',\n",
                "    'licenses': [{'name': 'CC0-1.0'}]\n",
                "}, indent=2))\n",
                "\n",
                "scars_dir = Path('/content/titan4_scars_upload')\n",
                "scars_dir.mkdir(exist_ok=True)\n",
                "for f in art.glob('*.safetensors'): shutil.copy2(f, scars_dir / f.name)\n",
                "for f in art.glob('*.json'): shutil.copy2(f, scars_dir / f.name)\n",
                "for f in art.glob('*.npy'): shutil.copy2(f, scars_dir / f.name)\n",
                "(scars_dir / 'dataset-metadata.json').write_text(json.dumps({\n",
                "    'title': 'Name That LLM Titan 4 Qwen3 Emb MoS Scars',\n",
                "    'id': 'thetod/name-that-llm-titan4-qwen3emb-mos-scars',\n",
                "    'licenses': [{'name': 'CC0-1.0'}]\n",
                "}, indent=2))\n",
                "\n",
                "!kaggle datasets create -p /content/titan4_qwen3_emb_8b_mos_artifacts/backbone_nf4_mos --dir-mode zip || kaggle datasets version -p /content/titan4_qwen3_emb_8b_mos_artifacts/backbone_nf4_mos -m 'Update Titan 4 NF4 MoS Backbone' --dir-mode zip\n",
                "!kaggle datasets create -p /content/titan4_scars_upload --dir-mode zip || kaggle datasets version -p /content/titan4_scars_upload -m 'Update Titan 4 5-Fold MoS Scar Experts' --dir-mode zip\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Conclusão\n",
                "Após concluir o upload acima, basta anexar `thetod/name-that-llm-titan4-qwen3emb-mos-backbone` e `thetod/name-that-llm-titan4-qwen3emb-mos-scars` ao nosso kernel System One Apex — o motor `extract_mos_nf4_streaming` já detecta e funde automaticamente as 8 profundidades e os tensores `.scar_A`/`.scar_B` em `< 2.5 GiB` de VRAM!"
            ],
        },
    ],
}

nb_path = script_dir / "Colab_A100_Titan4_TwoStage_MoS_ScarExperts.ipynb"
nb_path.write_text(json.dumps(nb, indent=1))
print("✓ Created:", nb_path)
