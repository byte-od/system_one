"""Módulo de exportação ONNX e sincronização automática com o Web HUD e Portfólio.

Garante que sempre que um checkpoint for persistido (via HUD local, CLI ou scripts),
os pesos neurais sejam automaticamente convertidos para ONNX e sincronizados:
  1. No runtime web local do system_one (web/models/)
  2. No portfólio web oficial (../portfolio/public/models/ e ../portfolio/dist/models/)
"""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from system1_engine.core.agent import UniversalS1Agent


class S1OnnxExportWrapper(nn.Module):
    """Wrapper para exportação simbólica ONNX pura sem classes dinâmicas de distribuição."""

    def __init__(self, agent: UniversalS1Agent) -> None:
        super().__init__()
        self.front_end = agent.front_end
        self.trunk = agent.trunk
        self.policy_head = agent.policy_head
        self.value_head = agent.value_head
        self.is_discrete = agent.is_discrete

    def forward(
        self,
        obs: torch.Tensor,
        delta_obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hx: torch.Tensor,
    ):
        z_in = self.front_end(
            obs=obs,
            delta_obs=delta_obs,
            prev_action=prev_action,
            prev_reward=prev_reward,
        )
        h, next_hx = self.trunk(z_in, hx=hx)
        if self.is_discrete:
            logits = self.policy_head.linear(h).squeeze(1)
        else:
            logits = self.policy_head.mu_net(h).squeeze(1)
        val = self.value_head(h).squeeze(1)
        return logits, val, next_hx


def resolve_web_model_filename(env_id: Optional[str] = None, checkpoint_path: Optional[str] = None) -> str:
    """Mapeia o ambiente ou caminho de checkpoint para o nome canônico usado no HUD Web."""
    import re
    combined = f"{env_id or ''} {checkpoint_path or ''}".lower()
    if "rubiks_atomic" in combined or "rubikscube-v0" in combined:
        return "s1_rubiks_atomic.onnx"
    if "rubik" in combined:
        return "s1_rubiks_macro.onnx"
    if "cartpole" in combined:
        return "s1_cartpole.onnx"
    if env_id:
        clean = "".join(c if c.isalnum() else "_" for c in env_id.lower()).strip("_")
        return f"s1_{clean}.onnx"
    if checkpoint_path:
        base = os.path.basename(checkpoint_path)
        name_only = os.path.splitext(base)[0]
        # Remove sufixos como _trained, _trained_v2, _v2 para obter o nome canônico do modelo ONNX
        clean_name = re.sub(r'_trained(?:_v\d+)?$', '', name_only, flags=re.IGNORECASE)
        if not clean_name.startswith("s1_"):
            clean_name = f"s1_{clean_name}"
        return f"{clean_name}.onnx"
    return "s1_model.onnx"


def auto_sync_web_model(
    agent: UniversalS1Agent,
    checkpoint_path: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Exporta o agente em memória para ONNX e copia para os diretórios web e portfólio.

    Returns:
        Caminho do arquivo exportado ou None se o ambiente for incompatível.
    """
    if getattr(agent, "is_visual", False):
        # Modelos convolucionais pesados (ex: ViZDoom/CarRacing) não são exportados automaticamente para web
        return None

    orig_device = next(agent.parameters()).device
    orig_training = agent.training

    try:
        env_id = (extra_info or {}).get("env_id", "")
        model_filename = resolve_web_model_filename(env_id, checkpoint_path)

        # 1. Encontra os diretórios de destino
        project_root = Path(__file__).resolve().parent.parent.parent
        web_models_dir = project_root / "web" / "models"
        web_models_dir.mkdir(parents=True, exist_ok=True)
        primary_onnx_path = web_models_dir / model_filename

        # 2. Prepara o wrapper para exportação
        agent_cpu = agent.cpu()
        agent_cpu.eval()
        export_model = S1OnnxExportWrapper(agent_cpu)
        export_model.eval()

        obs_space = getattr(agent, "obs_space", None)
        if obs_space is not None and hasattr(obs_space, "shape"):
            obs_dim = int(np.prod(obs_space.shape))
        else:
            obs_dim = getattr(agent.front_end, "obs_dim", 4)

        dummy_obs = torch.zeros(1, 1, obs_dim, dtype=torch.float32)
        dummy_delta = torch.zeros(1, 1, obs_dim, dtype=torch.float32)
        if agent.is_discrete:
            dummy_act = torch.zeros(1, 1, dtype=torch.long)
        else:
            act_dim = getattr(agent.policy_head, "act_dim", 1)
        dummy_rew = torch.zeros(1, 1, 1, dtype=torch.float32)
        trunk_hdim = getattr(agent.trunk, "hidden_dim", 256)
        dummy_hx = torch.zeros(1, 1, trunk_hdim, dtype=torch.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                export_model,
                (dummy_obs, dummy_delta, dummy_act, dummy_rew, dummy_hx),
                str(primary_onnx_path),
                input_names=["obs", "delta_obs", "prev_action", "prev_reward", "hx"],
                output_names=["logits", "value", "next_hx"],
                opset_version=17,
                dynamo=False,
            )

        synced_targets: List[str] = [str(primary_onnx_path)]

        # 3. Sincronização automática com o portfólio (se existir como projeto irmão)
        portfolio_candidates = [
            project_root.parent / "portfolio" / "public" / "models",
            project_root.parent / "portfolio" / "dist" / "models",
        ]

        for p_dir in portfolio_candidates:
            if p_dir.parent.exists():
                p_dir.mkdir(parents=True, exist_ok=True)
                dest = p_dir / model_filename
                shutil.copy2(primary_onnx_path, dest)
                synced_targets.append(str(dest))

        # 4. Atualiza o manifest.json unificado de modelos web
        generate_web_manifest(web_models_dir)

        file_size_mb = primary_onnx_path.stat().st_size / (1024 * 1024)
        print(f"\n🌐 [Auto-Sync Web] Pesos ONNX sincronizados com sucesso ({file_size_mb:.2f} MB):")
        for t in synced_targets:
            print(f"   ✓ {t}")

        return str(primary_onnx_path)
    except Exception as e:
        print(f"⚠️ [Auto-Sync Web] Falha na auto-exportação ONNX (não-bloqueante): {e}")
        return None
    finally:
        agent.to(orig_device)
        if orig_training:
            agent.train()
        else:
            agent.eval()


def find_latest_trained_checkpoint(env_id: str, project_root: Optional[Path] = None) -> Optional[str]:
    """Localiza o checkpoint .pt mais recente e de maior versão treinado para um determinado ambiente.

    Convenção do projeto: checkpoints neurais contêm 'trained' no nome (ex: s1_<env>_trained_v<N>.pt).
    Prioriza arquivos contendo 'trained', ordenando pela versão numérica final (_vN) e timestamp mtime.
    """
    import glob
    import re

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    all_pts = [
        Path(p)
        for p in (glob.glob(str(project_root / "*.pt")) + glob.glob(str(project_root / "**" / "*.pt"), recursive=True))
    ]
    # Filtra pastas virtuais, git e caches
    pts = [p for p in all_pts if not any(part.startswith(".") or part in (".venv", "build", "__pycache__") for part in p.parts)]

    env_clean = env_id.lower().replace("-", "_").replace(" ", "_")

    def env_filter(name: str) -> bool:
        n = name.lower()
        if "rubikscube-v0" in env_id or "rubiks_atomic" in env_id:
            return "rubik" in n and "macro" not in n
        if "rubik" in env_id:
            return "rubik" in n and "macro" in n
        if "cartpole" in env_clean:
            return "cartpole" in n
        if "acrobot" in env_clean:
            return "acrobot" in n
        if "mountaincar" in env_clean:
            return "mountaincar" in n
        if "lunarlander" in env_clean:
            return "lunarlander" in n
        if "ant_v5" in env_clean:
            return "ant_v5" in n
        if "ant_v4" in env_clean:
            return "ant_v4" in n
        if "bipedalwalker" in env_clean:
            return "bipedalwalker" in n
        if "carracing" in env_clean:
            return "carracing" in n
        if "frozenlake" in env_clean:
            return "frozenlake" in n
        if "pendulum" in env_clean:
            return "pendulum" in n
        if "vizdoom" in env_clean:
            return "vizdoom" in n
        return env_clean in n

    matching = [p for p in pts if env_filter(p.name)]
    if not matching:
        return None

    def sort_key(p: Path):
        name = p.stem.lower()
        is_aura = 1 if ("aura" in name or "cerberus" in name) else 0
        has_trained = 1 if ("trained" in name or is_aura) else 0
        m = re.search(r"_v(\d+)$", name)
        ver = int(m.group(1)) if m else (1 if has_trained else 0)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (is_aura, has_trained, ver, mtime)

    matching.sort(key=sort_key, reverse=True)
    return str(matching[0])


def generate_web_manifest(
    web_models_dir: Optional[Path] = None,
    ckpt_sources: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Varre os modelos ONNX disponíveis e constrói o manifesto dinâmico manifest.json."""
    import json
    import re
    import time

    project_root = Path(__file__).resolve().parent.parent.parent
    if web_models_dir is None:
        web_models_dir = project_root / "web" / "models"
    web_models_dir.mkdir(parents=True, exist_ok=True)

    metadata_catalog = {
        "s1_rubiks_atomic.onnx": {
            "key": "rubiks_atomic",
            "name": "⚡ Cubo Mágico 3x3 (Atômico 1.6M)",
            "env_id": "RubiksCube-v0",
            "type": "rubiks",
            "mode": "atomic",
            "obs_dim": 324,
            "act_dim": 12,
            "discrete": True,
            "description": "Tronco recorrente de 1.6M parâmetros com Fast GRU + Slow GRU + Residual Highway. 12 giros atômicos, latência sub-milissegundo.",
            "default": True,
        },
        "s1_rubiks_macro.onnx": {
            "key": "rubiks_macro",
            "name": "🧩 Cubo Mágico 3x3 (Macro / CFOP)",
            "env_id": "RubiksCubeMacro-v0",
            "type": "rubiks",
            "mode": "macro",
            "obs_dim": 324,
            "act_dim": 12,
            "discrete": True,
            "description": "Hierarchical RL com algoritmos CFOP (Sune, Sexy Move, T-Perm, Insert Edge).",
        },
        "s1_cartpole.onnx": {
            "key": "cartpole",
            "name": "⚖️ CartPole-v1 (Controle Clássico)",
            "env_id": "CartPole-v1",
            "type": "classic_control",
            "mode": "vector",
            "obs_dim": 4,
            "act_dim": 2,
            "discrete": True,
            "description": "Equilíbrio invertido amortizado em sub-milissegundo com reflexos ultra-estáveis.",
        },
        "s1_acrobot_v1.onnx": {
            "key": "acrobot",
            "name": "🤸 Acrobot-v1 (Braço Duplo)",
            "env_id": "Acrobot-v1",
            "type": "classic_control",
            "mode": "vector",
            "obs_dim": 6,
            "act_dim": 3,
            "discrete": True,
            "description": "Sistema caótico não-linear de dois elos sob controle reflexivo amortizado.",
        },
        "s1_mountaincar_v0.onnx": {
            "key": "mountaincar",
            "name": "⛰️ MountainCar-v0 (Controle Clássico)",
            "env_id": "MountainCar-v0",
            "type": "classic_control",
            "mode": "vector",
            "obs_dim": 2,
            "act_dim": 3,
            "discrete": True,
            "description": "Acúmulo de energia potencial gravitacional via oscilação de momento.",
        },
        "s1_lunarlander_v3.onnx": {
            "key": "lunarlander",
            "name": "🚀 LunarLander-v3 (Pouso Lunar Box2D)",
            "env_id": "LunarLander-v3",
            "type": "box2d",
            "mode": "vector",
            "obs_dim": 8,
            "act_dim": 4,
            "discrete": True,
            "description": "Controle vetorial de empuxo e atitude em gravidade simulada.",
        },
        "s1_ant_v5.onnx": {
            "key": "ant_v5",
            "name": "🐜 Ant-v5 (Robótica Contínua MuJoCo)",
            "env_id": "Ant-v5",
            "type": "mujoco",
            "mode": "continuous",
            "obs_dim": 27,
            "act_dim": 8,
            "discrete": False,
            "description": "Locomoção quadrupedal contínua em alta dimensão com física MuJoCo.",
        },
        "s1_ant_v4.onnx": {
            "key": "ant_v4",
            "name": "🐜 Ant-v4 (Robótica Contínua MuJoCo)",
            "env_id": "Ant-v4",
            "type": "mujoco",
            "mode": "continuous",
            "obs_dim": 27,
            "act_dim": 8,
            "discrete": False,
            "description": "Locomoção quadrupedal contínua em alta dimensão com física MuJoCo v4.",
        },
    }

    models_dict = {}
    for onnx_path in sorted(web_models_dir.glob("*.onnx")):
        filename = onnx_path.name
        stat = onnx_path.stat()
        size_mb = round(stat.st_size / (1024 * 1024), 2)
        meta = dict(metadata_catalog.get(filename, {
            "key": onnx_path.stem,
            "name": onnx_path.stem.replace("_", " ").title(),
            "env_id": onnx_path.stem,
            "type": "custom",
            "mode": "vector",
            "discrete": True,
            "description": "Modelo ONNX exportado do System 1 Engine."
        }))
        meta["file"] = filename
        meta["size_mb"] = size_mb
        meta["size_bytes"] = stat.st_size
        meta["mtime"] = stat.st_mtime
        meta["mtime_str"] = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(stat.st_mtime))

        source_ckpt = (ckpt_sources or {}).get(filename)
        if not source_ckpt:
            target_env = meta.get("env_id", "")
            found = find_latest_trained_checkpoint(target_env, project_root)
            if found:
                source_ckpt = os.path.basename(found)
            else:
                clean_env_id = target_env.lower().replace("-", "_")
                source_ckpt = f"s1_{clean_env_id}_trained.pt"
        if source_ckpt:
            meta["checkpoint_source"] = source_ckpt
            m_ver = re.search(r"_v(\d+)$", Path(source_ckpt).stem)
            meta["checkpoint_version"] = f"v{m_ver.group(1)}" if m_ver else "v1"

        key = meta.get("key", onnx_path.stem)
        models_dict[key] = meta

    manifest_data = {
        "version": "1.0",
        "updated_at": time.time(),
        "updated_at_str": time.strftime("%d/%m/%Y %H:%M:%S"),
        "models": models_dict,
    }

    manifest_path = web_models_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # Replica para os diretórios do portfólio caso existam
    portfolio_candidates = [
        project_root.parent / "portfolio" / "public" / "models" / "manifest.json",
        project_root.parent / "portfolio" / "dist" / "models" / "manifest.json",
    ]
    for p_path in portfolio_candidates:
        if p_path.parent.exists():
            try:
                shutil.copy2(manifest_path, p_path)
            except Exception:
                pass

    return manifest_data


def export_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    env_id: Optional[str] = None,
    sync_portfolio: bool = True,
) -> Optional[str]:
    """Converte qualquer arquivo de checkpoint .pt existente no projeto para formato ONNX."""
    import gymnasium as gym
    from system1_engine.env.wrapper import UniversalS1Wrapper
    from system1_engine.env.adapters.rubiks import RubiksCubeMacroEnv, RubiksCubeEnv

    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint não encontrado: {checkpoint_path}")
        return None

    # Inspeciona o checkpoint para inferir env_id se não fornecido
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    info = ckpt.get("extra_info", {}) if isinstance(ckpt, dict) else {}
    target_env = env_id or info.get("env_id") or os.path.basename(checkpoint_path)

    # Cria o ambiente apropriado
    env_lower = target_env.lower().replace("_", "-")
    try:
        if "atomic" in env_lower or "rubikscube-v0" in env_lower:
            raw_env = RubiksCubeEnv()
            resolved_env_id = "RubiksCube-v0"
        elif "rubik" in env_lower:
            raw_env = RubiksCubeMacroEnv()
            resolved_env_id = "RubiksCubeMacro-v0"
        else:
            raw_env = gym.make(target_env)
            resolved_env_id = target_env
    except Exception:
        # Fallback para ambiente com base no nome do checkpoint
        if "cartpole" in checkpoint_path.lower():
            raw_env = gym.make("CartPole-v1")
            resolved_env_id = "CartPole-v1"
        elif "acrobot" in checkpoint_path.lower():
            raw_env = gym.make("Acrobot-v1")
            resolved_env_id = "Acrobot-v1"
        elif "mountaincar" in checkpoint_path.lower():
            raw_env = gym.make("MountainCar-v0")
            resolved_env_id = "MountainCar-v0"
        elif "lunarlander" in checkpoint_path.lower():
            raw_env = gym.make("LunarLander-v3")
            resolved_env_id = "LunarLander-v3"
        elif "ant" in checkpoint_path.lower():
            raw_env = gym.make("Ant-v5")
            resolved_env_id = "Ant-v5"
        else:
            print(f"⚠️ Não foi possível instanciar ambiente para {checkpoint_path}")
            return None

    wrapped_env = UniversalS1Wrapper(raw_env)
    from system1_engine.transfer.manager import KnowledgeTransferManager
    trunk_type = KnowledgeTransferManager.detect_trunk_type(checkpoint_path)
    agent = UniversalS1Agent(
        obs_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        trunk_type=trunk_type,
    )

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    agent.load_state_dict(state_dict, strict=False)

    return auto_sync_web_model(agent, checkpoint_path=checkpoint_path, extra_info={"env_id": resolved_env_id})


def sync_all_web_models(sync_portfolio: bool = True) -> List[str]:
    """Varre todos os checkpoints treinados do repositório e sincroniza para a pasta web/models/."""
    project_root = Path(__file__).resolve().parent.parent.parent

    # Ambientes alvo do catálogo web
    target_envs = [
        "RubiksCube-v0",
        "RubiksCubeMacro-v0",
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCar-v0",
        "LunarLander-v3",
        "Ant-v5",
        "Ant-v4",
    ]

    # Mapeia checkpoints prioritários de fallback para cada ambiente
    priority_map = {
        "RubiksCube-v0": ["s1_rubikscube_v0_trained.pt"],
        "RubiksCubeMacro-v0": ["s1_rubikscubemacro_v0_trained_v3.pt", "s1_rubiks_macro_trained.pt"],
        "CartPole-v1": ["s1_cartpole_trained.pt", "s1_cartpole.pt"],
        "Acrobot-v1": ["s1_acrobot_v1_trained.pt", "s1_acrobot_v1_checkpoint.pt", "s1_acrobot.pt"],
        "MountainCar-v0": ["s1_mountaincar_v0_trained.pt"],
        "LunarLander-v3": ["s1_lunarlander_v3_trained.pt"],
        "Ant-v5": ["s1_ant_v5_trained_v3.pt", "s1_ant_v5_trained_v2.pt", "s1_ant_v5_trained.pt"],
        "Ant-v4": ["s1_ant_v4_trained.pt"],
    }

    synced = []
    ckpt_source_map = {}
    print("\n" + "=" * 70)
    print("🚀 SINCRONIZANDO TODOS OS MODELOS E PESOS PARA O WEB HUD")
    print("=" * 70)

    for env_id in target_envs:
        chosen_ckpt = find_latest_trained_checkpoint(env_id, project_root)
        if not chosen_ckpt:
            for cand in priority_map.get(env_id, []):
                cand_path = project_root / cand
                if cand_path.exists():
                    chosen_ckpt = str(cand_path)
                    break

        if chosen_ckpt:
            res = export_checkpoint_to_onnx(chosen_ckpt, env_id=env_id, sync_portfolio=sync_portfolio)
            if res:
                synced.append(res)
                onnx_name = os.path.basename(res)
                ckpt_source_map[onnx_name] = os.path.basename(chosen_ckpt)

    # Gera o manifesto consolidado com metadados da origem do checkpoint
    manifest = generate_web_manifest(ckpt_sources=ckpt_source_map)
    print(f"\n✅ Sincronização concluída: {len(synced)} modelos ONNX ativos no manifesto web.")
    print("=" * 70 + "\n")
    return synced


if __name__ == "__main__":
    sync_all_web_models()
