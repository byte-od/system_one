"""Memória Latente Estéril Auto-Corretiva (Self-Healing Latent Tabu Memory).

Rastreia hipóteses de ação não produtivas associadas ao estado cognitivo latente (motivo da decisão)
do agente, aplicando resistência suave sobre os logits de ações estéreis para forçar a exploração
direcionada rumo a ações virgens, com reabilitação imediata caso a ação venha a ser produtiva.
"""

from typing import Any, Dict, Optional, Tuple, Union
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
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.capacity = max(16, int(capacity))
        self.latent_dim = max(1, int(latent_dim))
        self.num_actions = max(1, int(num_actions))
        self.sim_threshold = float(np.clip(sim_threshold, 0.50, 0.99))
        self.penalty_scale = float(max(0.1, penalty_scale))
        self.step_increment = float(max(0.1, step_increment))
        self.max_weight = float(max(1.0, max_weight))
        self.device = device

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
        """Move os tensores internos para o dispositivo especificado."""
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

        # Achata dimensões de batch para [B, latent_dim]
        flat_latent = latent.reshape(-1, self.latent_dim).to(self.device).float()
        B = flat_latent.shape[0]

        # L2-Normalização do estado de consulta
        h_norm = F.normalize(flat_latent, p=2, dim=-1)

        active_indices = torch.nonzero(self.is_active).squeeze(-1)
        active_protos = self.prototypes[active_indices]  # [K_act, latent_dim]
        active_weights = self.weights[active_indices]    # [K_act, num_actions]

        # Produto escalar para similaridade de cosseno em batch: [B, K_act]
        sims = torch.matmul(h_norm, active_protos.T)

        # Gate suave de similaridade: apenas protótipos acima do limiar participam
        # Escalonado linearmente de 0 à 1 no intervalo [sim_threshold, 1.0]
        gated_sims = torch.relu(sims - self.sim_threshold) / (1.0 - self.sim_threshold + 1e-6)

        # Combinação linear da resistência por ação: [B, num_actions]
        resistance = self.penalty_scale * torch.matmul(gated_sims, active_weights)

        # Restaura formato de batch original
        out_shape = (*orig_shape[:-1], self.num_actions)
        return resistance.reshape(out_shape).to(orig_device)

    def record_transition(
        self,
        latent: torch.Tensor,
        action: Union[int, np.ndarray, torch.Tensor],
        is_productive: bool,
    ) -> bool:
        """Registra o desfecho de uma ação tomada sob o estado de atenção/motivo latente h.

        Se produtivo:
            Reabilita a ação zerando seu peso de esterilidade e desativa o protótipo se todos zerados.
        Se estéril:
            Incrementa o peso de esterilidade do protótipo mais similar ou cria um novo slot.

        Returns:
            True se uma atualização de protótipo ocorreu, False caso contrário.
        """
        # Trata lotes vetorizados
        if latent.dim() > 1 and latent.shape[0] > 1:
            B = latent.shape[0]
            acts = action if isinstance(action, (list, tuple, np.ndarray, torch.Tensor)) else [action] * B
            updated_any = False
            for b in range(B):
                up = self.record_transition(latent[b], acts[b], is_productive)
                if up:
                    updated_any = True
            return updated_any

        # Processamento de transição individual
        h_flat = latent.reshape(-1).to(self.device).float()
        if h_flat.shape[0] != self.latent_dim:
            return False

        h_norm = F.normalize(h_flat, p=2, dim=-1)

        if isinstance(action, torch.Tensor):
            act_idx = int(action.squeeze().item())
        elif isinstance(action, np.ndarray):
            act_idx = int(action.squeeze())
        else:
            act_idx = int(action)

        if act_idx < 0 or act_idx >= self.num_actions:
            return False

        # Verifica se já existe um protótipo ativo com motivo cognitivo similar
        if self.is_active.any():
            active_indices = torch.nonzero(self.is_active).squeeze(-1)
            active_protos = self.prototypes[active_indices]
            sims = torch.matmul(h_norm.unsqueeze(0), active_protos.T).squeeze(0)  # [K_act]
            best_sim, best_local_idx = torch.max(sims, dim=0)

            if best_sim.item() >= self.sim_threshold:
                best_global_idx = int(active_indices[best_local_idx].item())

                if is_productive:
                    # 1. REABILITAÇÃO IMEDIATA (Cura de Falsos Negativos):
                    if self.weights[best_global_idx, act_idx] > 0.0:
                        self.weights[best_global_idx, act_idx] = 0.0
                        self.total_rehabilitated += 1

                    # Se todas as ações deste protótipo estiverem com peso zero, desativa o slot
                    if (self.weights[best_global_idx] <= 0.0).all():
                        self.is_active[best_global_idx] = False

                    return True
                else:
                    # 2. REFORÇO DE ESTERILIDADE:
                    curr_w = float(self.weights[best_global_idx, act_idx].item())
                    new_w = min(curr_w + self.step_increment, self.max_weight)
                    self.weights[best_global_idx, act_idx] = new_w
                    self.total_recorded += 1
                    return True

        # Se não há protótipo similar ativo existente:
        if is_productive:
            # Ação produtiva inédita: nada a penalizar
            return False

        # Cria novo protótipo estéril na posição do cursor circular
        idx = self.cursor
        self.prototypes[idx] = h_norm
        self.weights[idx].zero_()
        self.weights[idx, act_idx] = self.step_increment
        self.is_active[idx] = True

        self.cursor = (self.cursor + 1) % self.capacity
        self.total_recorded += 1
        return True

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
