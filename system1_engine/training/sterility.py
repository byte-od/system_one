"""Memória Latente Estéril Auto-Corretiva (Self-Healing Latent Tabu Memory).

Rastreia hipóteses de ação não produtivas associadas ao estado cognitivo latente (motivo da decisão)
do agente, aplicando resistência suave sobre os logits de ações estéreis para forçar a exploração
direcionada rumo a ações virgens, com reabilitação imediata caso a ação venha a ser produtiva.

Otimizado para altíssimo throughput (10.000+ FPS) sem sincronização bloqueante de GPU/MPS.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


class LatentSterilityMemory:
    """Memória episódica de esterilidade latente baseada em similaridade de cosseno.

    Associa vetores latentes h (saída do Trunk de 256-D) a pesos de esterilidade por ação.
    Opera como um sistema de crédito negativo auto-corretivo:
    - Incrementa resistência quando uma ação falha sob determinado motivo cognitivo.
    - Zera a resistência e reabilita a ação imediatamente se ela produzir avanço de Bellman.
    """

    def __init__(
        self,
        capacity: int = 1024,
        latent_dim: int = 256,
        num_actions: int = 12,
        sim_threshold: float = 0.85,
        penalty_scale: float = 2.0,
        step_increment: float = 1.0,
        max_weight: float = 10.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.capacity = max(16, int(capacity))
        self.latent_dim = max(1, int(latent_dim))
        self.num_actions = max(1, int(num_actions))
        self.sim_threshold = float(np.clip(sim_threshold, 0.50, 0.99))
        self.penalty_scale = float(max(0.1, penalty_scale))
        self.step_increment = float(max(0.1, step_increment))
        self.max_weight = float(max(1.0, max_weight))
        
        # Mantém a memória latente no CPU por padrão para evitar stalls de sincronização do Metal (MPS/CUDA)
        self.device = torch.device("cpu") if device is None or device.type in ("mps", "cuda") else device

        # Protótipos latentes L2-normalizados: [K, latent_dim]
        self.prototypes = torch.zeros(
            (self.capacity, self.latent_dim), dtype=torch.float32, device=self.device
        )
        # Matriz de pesos de esterilidade por ação: [K, num_actions]
        self.weights = torch.zeros(
            (self.capacity, self.num_actions), dtype=torch.float32, device=self.device
        )
        # Máscara de ativação dos slots do buffer circular
        self.is_active = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)

        self.cursor: int = 0
        self.total_recorded: int = 0
        self.total_rehabilitated: int = 0

    def to(self, device: torch.device) -> "LatentSterilityMemory":
        """Configura preferência de dispositivo (mantém CPU para MPS/CUDA para evitar stalls de GPU)."""
        if device.type not in ("mps", "cuda"):
            self.device = device
            self.prototypes = self.prototypes.to(device)
            self.weights = self.weights.to(device)
            self.is_active = self.is_active.to(device)
        return self

    def clear(self) -> None:
        """Limpa toda a memória de esterilidade."""
        self.prototypes.zero_()
        self.weights.zero_()
        self.is_active.zero_()
        self.cursor = 0
        self.total_recorded = 0
        self.total_rehabilitated = 0

    def __len__(self) -> int:
        """Retorna o número de protótipos cognitivos ativos na memória."""
        return int(self.is_active.sum().item())

    def query_resistance(self, latent: torch.Tensor) -> torch.Tensor:
        """Calcula a penalidade de resistência sobre os logits das ações para o estado latente h.

        Args:
            latent: Vetor latente com formato [latent_dim], [B, latent_dim] ou [B, 1, latent_dim].

        Returns:
            Tensor de penalidade com mesmo batch shape: [B, num_actions] ou [B, 1, num_actions].
        """
        orig_shape = latent.shape
        orig_device = latent.device

        # Se a memória estiver vazia, retorna resistência nula
        if not self.is_active.any():
            out_shape = (*orig_shape[:-1], self.num_actions)
            return torch.zeros(out_shape, dtype=torch.float32, device=orig_device)

        # Achata dimensões de batch para [B, latent_dim] no CPU sem sincronizações bloqueantes
        flat_latent = latent.reshape(-1, self.latent_dim).detach()
        if flat_latent.device != self.device:
            flat_latent = flat_latent.to(self.device, non_blocking=True)
        flat_latent = flat_latent.float()

        # L2-Normalização do estado de consulta
        h_norm = F.normalize(flat_latent, p=2, dim=-1)

        active_indices = torch.nonzero(self.is_active).squeeze(-1)
        active_protos = self.prototypes[active_indices]  # [K_act, latent_dim]
        active_weights = self.weights[active_indices]    # [K_act, num_actions]

        # Produto escalar para similaridade de cosseno em batch: [B, K_act]
        sims = torch.matmul(h_norm, active_protos.T)

        # Gate suave de similaridade: apenas protótipos acima do limiar participam
        gated_sims = torch.relu(sims - self.sim_threshold) / (1.0 - self.sim_threshold + 1e-6)

        # Combinação linear da resistência por ação: [B, num_actions]
        raw_resistance = self.penalty_scale * torch.matmul(gated_sims, active_weights)

        # Resistência Relativa de Contraste (Princípio Generalista):
        # Desconta a resistência mínima do estado para incentivar ações virgens
        # sem bloquear globalmente todas as ações caso todas tenham histórico estéril.
        min_r = raw_resistance.min(dim=-1, keepdim=True).values
        resistance = (raw_resistance - min_r).clamp(min=0.0, max=2.5)

        # Restaura formato de batch original e devolve no dispositivo de origem
        out_shape = (*orig_shape[:-1], self.num_actions)
        res_shaped = resistance.reshape(out_shape)
        if res_shaped.device != orig_device:
            res_shaped = res_shaped.to(orig_device, non_blocking=True)
        return res_shaped

    def record_batch(
        self,
        latents: torch.Tensor,
        actions: Union[Sequence[int], np.ndarray, torch.Tensor],
        is_productive: Union[Sequence[bool], np.ndarray, torch.Tensor],
    ) -> bool:
        """Registra eficientemente um lote completo de transições vetorizadas (B ambientes simultâneos).

        Calcula similaridades em batch [B x K] e executa atualizações sem stalls de pipeline.
        """
        latents_cpu = latents.reshape(-1, self.latent_dim).detach()
        if latents_cpu.device != self.device:
            latents_cpu = latents_cpu.to(self.device, non_blocking=True)
        latents_cpu = latents_cpu.float()

        B = latents_cpu.shape[0]
        h_norm = F.normalize(latents_cpu, p=2, dim=-1)

        if isinstance(actions, torch.Tensor):
            acts = actions.view(-1).cpu().tolist()
        elif isinstance(actions, np.ndarray):
            acts = actions.flatten().tolist()
        else:
            acts = list(actions)

        if isinstance(is_productive, torch.Tensor):
            prods = is_productive.view(-1).cpu().tolist()
        elif isinstance(is_productive, np.ndarray):
            prods = is_productive.flatten().tolist()
        else:
            prods = list(is_productive)

        has_active = bool(self.is_active.any().item())
        if has_active:
            active_indices = torch.nonzero(self.is_active).squeeze(-1)
            active_protos = self.prototypes[active_indices]
            sims = torch.matmul(h_norm, active_protos.T)  # [B, K_act]
            best_sims, best_locals = torch.max(sims, dim=1)
            best_sims_arr = best_sims.numpy()
            best_locals_arr = best_locals.numpy()
            act_idx_arr = active_indices.numpy()
        else:
            best_sims_arr = np.zeros(B, dtype=np.float32)
            best_locals_arr = np.zeros(B, dtype=np.int64)
            act_idx_arr = np.array([], dtype=np.int64)

        updated_any = False
        for b in range(B):
            act_idx = int(acts[b])
            prod = bool(prods[b])

            if act_idx < 0 or act_idx >= self.num_actions:
                continue

            if has_active and best_sims_arr[b] >= self.sim_threshold:
                g_idx = int(act_idx_arr[best_locals_arr[b]])
                if prod:
                    # 1. Reabilitação Imediata
                    if self.weights[g_idx, act_idx] > 0.0:
                        self.weights[g_idx, act_idx] = 0.0
                        self.total_rehabilitated += 1
                        updated_any = True
                    if (self.weights[g_idx] <= 0.0).all():
                        self.is_active[g_idx] = False
                else:
                    # 2. Reforço de Esterilidade
                    curr_w = float(self.weights[g_idx, act_idx].item())
                    self.weights[g_idx, act_idx] = min(curr_w + self.step_increment, self.max_weight)
                    self.total_recorded += 1
                    updated_any = True
            else:
                if not prod:
                    # 3. Criação de novo protótipo estéril
                    idx = self.cursor
                    self.prototypes[idx] = h_norm[b]
                    self.weights[idx].zero_()
                    self.weights[idx, act_idx] = self.step_increment
                    self.is_active[idx] = True
                    self.cursor = (self.cursor + 1) % self.capacity
                    self.total_recorded += 1
                    updated_any = True

        return updated_any

    def record_transition(
        self,
        latent: torch.Tensor,
        action: Union[int, np.ndarray, torch.Tensor],
        is_productive: bool,
    ) -> bool:
        """Interface individual (compatível com chamadas isoladas de inferência/testes)."""
        if latent.dim() > 1 and latent.shape[0] > 1:
            B = latent.shape[0]
            acts = action if isinstance(action, (list, tuple, np.ndarray, torch.Tensor)) else [action] * B
            prods = [is_productive] * B if not isinstance(is_productive, (list, tuple, np.ndarray)) else is_productive
            return self.record_batch(latent, acts, prods)
        
        flat_latent = latent.reshape(1, self.latent_dim)
        acts = [int(action.item() if isinstance(action, torch.Tensor) else action)]
        prods = [bool(is_productive)]
        return self.record_batch(flat_latent, acts, prods)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas detalhadas para monitoramento no HUD e telemetria."""
        num_act = int(self.is_active.sum().item())
        if num_act > 0:
            active_weights = self.weights[self.is_active]
            pos_weights = active_weights[active_weights > 0.0]
            mean_w = float(pos_weights.mean().item()) if pos_weights.numel() > 0 else 0.0
            max_w = float(pos_weights.max().item()) if pos_weights.numel() > 0 else 0.0
        else:
            mean_w = 0.0
            max_w = 0.0

        return {
            "num_active_prototypes": num_act,
            "total_recorded": self.total_recorded,
            "total_rehabilitated": self.total_rehabilitated,
            "mean_sterility_weight": mean_w,
            "max_sterility_weight": max_w,
            "capacity": self.capacity,
        }
