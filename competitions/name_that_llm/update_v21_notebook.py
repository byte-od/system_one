import json
from pathlib import Path

nb_path = Path("/Users/juliocesarreisfilho/Projects/scar-cache/competitions/name_that_llm/name_that_llm_submission.ipynb")
with open(nb_path) as f:
    nb = json.load(f)

# 1. Update Markdown header
nb["cells"][0]["source"] = [
    "# Mixture of Scars (MoS) Multi-Titan Championship Submission (V21)\n",
    "\n",
    "> **Championship Pipeline V21: Titan 2 (Yi-34B L30 SOTA 40.50% OOF) + LinearSVC + DeBERTa + Titan 1**\n",
    "- **Titan 2 (Yi-34B MoS L30)**: Truncated at Layer 30 (7,196D Tri-Compass vector, 40.50% OOF SOTA accuracy).\n",
    "- **Titan 1 (Qwen 72B MoS L40)**: INT4 streaming of layers 0..39 + 28D Tri-Compass Head.\n",
    "- **LinearSVC (4-View N-Grams)**: Solid 0.622 base anchor.\n",
    "- **Continuous Soft Sinkhorn**: 300 iterations uniform probability calibration.\n"
]

# 2. Update Cell 1 with offline bitsandbytes installer
c1_lines = nb["cells"][1]["source"]
c1_str = "".join(c1_lines)
if "bitsandbytes" not in c1_str:
    new_c1 = []
    for line in c1_lines:
        new_c1.append(line)
        if "INPUT_DIR = \"/kaggle/input\"" in line:
            new_c1.append("\n# Install offline bitsandbytes if available\n")
            new_c1.append("bnb_wheels = glob.glob(f\"{INPUT_DIR}/**/bitsandbytes*.whl\", recursive=True)\n")
            new_c1.append("if bnb_wheels:\n")
            new_c1.append("    import subprocess\n")
            new_c1.append("    print(f\"Installing offline bitsandbytes from {bnb_wheels[0]}...\", flush=True)\n")
            new_c1.append("    subprocess.run([\"pip\", \"install\", \"-q\", bnb_wheels[0]], check=False)\n")
    nb["cells"][1]["source"] = new_c1

# 3. Create Titan 2 Cell
titan2_code = [
    "# ==============================================================================\n",
    "# Titan 2: Yi-34B MoS Golden-Slice Layer 30 (40.50% OOF SOTA) + Tri-Compass Head\n",
    "# ==============================================================================\n",
    "W_TITAN2 = 0.50\n",
    "W_TITAN2_TWIN = 0.15\n",
    "\n",
    "try:\n",
    "    head_yi_file = find(\"head_tricompass_yi34b_l30.safetensors\")\n",
    "    print(f\"Loading Titan 2 Yi-34B Head from {head_yi_file}...\", flush=True)\n",
    "    head_yi = load_file(head_yi_file)\n",
    "\n",
    "    try:\n",
    "        yi_backbone_file = find(\"model-00001-of-00002.safetensors\")\n",
    "    except Exception:\n",
    "        yi_backbone_file = find(\"model.safetensors.index.json\")\n",
    "    yi_backbone_dir = os.path.dirname(yi_backbone_file)\n",
    "    print(f\"Loading Titan 2 Yi-34B 30-Layer Backbone from {yi_backbone_dir}...\", flush=True)\n",
    "\n",
    "    from transformers import AutoModel, AutoTokenizer\n",
    "    tok_yi = AutoTokenizer.from_pretrained(yi_backbone_dir)\n",
    "    if tok_yi.pad_token is None:\n",
    "        tok_yi.pad_token = tok_yi.eos_token\n",
    "\n",
    "    model_yi = AutoModel.from_pretrained(\n",
    "        yi_backbone_dir,\n",
    "        device_map=\"auto\",\n",
    "        low_cpu_mem_usage=True\n",
    "    ).eval()\n",
    "\n",
    "    # Stylometrics for Titan 2 (Compass 1)\n",
    "    style_raw = extract_tricompass_style_features(test.text.tolist())\n",
    "    style_normed_yi = (style_raw - head_yi[\"style_scaler.mean\"].float().numpy()) / head_yi[\"style_scaler.scale\"].float().numpy()\n",
    "\n",
    "    # Backbone Feature Extraction (Compass 3)\n",
    "    N_test = len(test)\n",
    "    batch_size = 8\n",
    "    all_reps = []\n",
    "    with torch.no_grad():\n",
    "        for i in range(0, N_test, batch_size):\n",
    "            b_texts = test.text.tolist()[i : i + batch_size]\n",
    "            inp = tok_yi(b_texts, padding=True, truncation=True, max_length=512, return_tensors=\"pt\")\n",
    "            out = model_yi(**{k: v.to(model_yi.device) for k, v in inp.items()})\n",
    "            mask = inp.attention_mask.unsqueeze(-1).float().to(out.last_hidden_state.device)\n",
    "            pooled = (out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)\n",
    "            rms = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)\n",
    "            all_reps.append((pooled * rms).cpu().numpy())\n",
    "    Z_yi = np.concatenate(all_reps, axis=0)\n",
    "    del model_yi\n",
    "    gc.collect()\n",
    "    torch.cuda.empty_cache()\n",
    "    print(\"Titan 2 Yi-34B features extracted successfully!\", flush=True)\n",
    "\n",
    "    # Plain Head Vote\n",
    "    X_plain_raw = np.hstack([Z_yi, style_normed_yi])\n",
    "    X_plain = (X_plain_raw - head_yi[\"scaler_plain.mean\"].float().numpy()) / head_yi[\"scaler_plain.scale\"].float().numpy()\n",
    "    logits_plain = X_plain @ head_yi[\"plain.weight\"].float().numpy().T + head_yi[\"plain.bias\"].float().numpy()\n",
    "    vote(\"titan2_yi34b_l30_plain\", softmax(logits_plain), W_TITAN2)\n",
    "\n",
    "    # Centered Twin Vote\n",
    "    if twin_rows.any():\n",
    "        Zc = Z_yi.copy()\n",
    "        Sc = style_normed_yi.copy()\n",
    "        for g in np.unique(groups[twin_rows]):\n",
    "            m = (groups == g)\n",
    "            if m.sum() > 1:\n",
    "                Zc[m] -= Zc[m].mean(0)\n",
    "                Sc[m] -= Sc[m].mean(0)\n",
    "        X_c_raw = np.hstack([Zc, Sc])\n",
    "        X_c = (X_c_raw - head_yi[\"scaler_centered.mean\"].float().numpy()) / head_yi[\"scaler_centered.scale\"].float().numpy()\n",
    "        logits_c = X_c[twin_rows] @ head_yi[\"centered.weight\"].float().numpy().T + head_yi[\"centered.bias\"].float().numpy()\n",
    "        vote(\"titan2_yi34b_l30_twin\", softmax(logits_c), W_TITAN2_TWIN, rows=twin_rows)\n",
    "    print(\"Titan 2 Yi-34B inference votes recorded!\", flush=True)\n",
    "\n",
    "except Exception as e:\n",
    "    print(f\"Warning: Titan 2 evaluation bypassed: {e}\", flush=True)\n"
]

# Check if Titan 2 cell is already present
has_titan2 = any("titan2_yi34b_l30" in "".join(c.get("source", [])) for c in nb["cells"])
if not has_titan2:
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": titan2_code
    }
    # Insert right before the last cell (Sinkhorn cell)
    nb["cells"].insert(len(nb["cells"]) - 1, new_cell)
    print("Added Titan 2 cell!")
else:
    print("Titan 2 cell already present, updating...")
    for c in nb["cells"]:
        if "titan2_yi34b_l30" in "".join(c.get("source", [])):
            c["source"] = titan2_code
            break

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully written!")
