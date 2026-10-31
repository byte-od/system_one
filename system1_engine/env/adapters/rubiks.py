"""Módulo de Simulação e Ambientes de Aprendizado por Reforço para Cubo Mágico (Rubik's Cube 3x3).

Implementa:
1. `RubiksCubeCore`: Modelo matemático 3D com permutações pré-calculadas ultrarrápidas (< 1 µs/passo).
2. `RubiksCubeEnv`: Ambiente Gymnasium para movimentos atômicos (12 giros elementares).
3. `RubiksCubeMacroEnv`: Ambiente Gymnasium para macro-ações e algoritmos pré-definidos (Hierarchical RL).
4. Renderização visual 2D estilizada (Net/Desdobramento) compatível com o streaming MJPEG do HUD.
"""

from __future__ import annotations

import io
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

try:
    import system1_core
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False



# =====================================================================
# MATEMÁTICA E PERMUTAÇÕES 3D DO CUBO MÁGICO
# =====================================================================

ROT_F = np.array([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]], dtype=np.int32)
ROT_B = np.array([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]], dtype=np.int32)
ROT_R = np.array([[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]], dtype=np.int32)
ROT_L = np.array([[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]], dtype=np.int32)
ROT_U = np.array([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]], dtype=np.int32)
ROT_D = np.array([[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]], dtype=np.int32)

BASE_MOVES = {
    "F": (ROT_F, lambda p: p[2] == 1),
    "F_prime": (ROT_F.T, lambda p: p[2] == 1),
    "B": (ROT_B, lambda p: p[2] == -1),
    "B_prime": (ROT_B.T, lambda p: p[2] == -1),
    "R": (ROT_R, lambda p: p[0] == 1),
    "R_prime": (ROT_R.T, lambda p: p[0] == 1),
    "L": (ROT_L, lambda p: p[0] == -1),
    "L_prime": (ROT_L.T, lambda p: p[0] == -1),
    "U": (ROT_U, lambda p: p[1] == 1),
    "U_prime": (ROT_U.T, lambda p: p[1] == 1),
    "D": (ROT_D, lambda p: p[1] == -1),
    "D_prime": (ROT_D.T, lambda p: p[1] == -1),
    # Rotações do cubo inteiro
    "Y": (ROT_U, lambda p: True),
    "Y_prime": (ROT_U.T, lambda p: True),
}


def _build_initial_stickers() -> List[Dict[str, Any]]:
    stickers = []
    # Face 0: U (y=1, norm=(0,1,0))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([c - 1, 1, 1 - r]), "norm": np.array([0, 1, 0]), "color": 0})
    # Face 1: D (y=-1, norm=(0,-1,0))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([c - 1, -1, r - 1]), "norm": np.array([0, -1, 0]), "color": 1})
    # Face 2: F (z=1, norm=(0,0,1))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([c - 1, 1 - r, 1]), "norm": np.array([0, 0, 1]), "color": 2})
    # Face 3: B (z=-1, norm=(0,0,-1))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([1 - c, 1 - r, -1]), "norm": np.array([0, 0, -1]), "color": 3})
    # Face 4: R (x=1, norm=(1,0,0))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([1, 1 - r, 1 - c]), "norm": np.array([1, 0, 0]), "color": 4})
    # Face 5: L (x=-1, norm=(-1,0,0))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([-1, 1 - r, c - 1]), "norm": np.array([-1, 0, 0]), "color": 5})
    return stickers


def _get_slot_index(pos: np.ndarray, norm: np.ndarray) -> int:
    x, y, z = pos[0], pos[1], pos[2]
    nx, ny, nz = norm[0], norm[1], norm[2]
    if ny == 1:
        face, r, c = 0, 1 - z, x + 1
    elif ny == -1:
        face, r, c = 1, z + 1, x + 1
    elif nz == 1:
        face, r, c = 2, 1 - y, x + 1
    elif nz == -1:
        face, r, c = 3, 1 - y, 1 - x
    elif nx == 1:
        face, r, c = 4, 1 - y, 1 - z
    elif nx == -1:
        face, r, c = 5, 1 - y, z + 1
    else:
        raise ValueError(f"Normal inválida: {norm}")
    return int(face * 9 + (r * 3 + c))


def _precompute_permutations() -> Dict[str, np.ndarray]:
    perms = {}
    for m, (rot_mat, cond) in BASE_MOVES.items():
        st = _build_initial_stickers()
        for i, s in enumerate(st):
            s["color"] = _get_slot_index(s["pos"], s["norm"])
        new_st = []
        for s in st:
            pos, norm, col = s["pos"], s["norm"], s["color"]
            if cond(pos):
                new_st.append({"pos": rot_mat @ pos, "norm": rot_mat @ norm, "color": col})
            else:
                new_st.append({"pos": pos.copy(), "norm": norm.copy(), "color": col})
        perm = np.zeros(54, dtype=np.int32)
        for s in new_st:
            dest_slot = _get_slot_index(s["pos"], s["norm"])
            src_slot = s["color"]
            perm[dest_slot] = src_slot
        perms[m] = perm
    return perms


PERMUTATIONS = _precompute_permutations()


# =====================================================================
# CORE DO CUBO MÁGICO
# =====================================================================

class RubiksCubeCore:
    """Simulador vetorial de alta velocidade para o Cubo Mágico 3x3x3."""

    # 12 Movimentos Atômicos
    ATOMIC_MOVES = [
        "U", "U_prime", "D", "D_prime",
        "F", "F_prime", "B", "B_prime",
        "R", "R_prime", "L", "L_prime",
    ]

    # 12 Macro-Ações Pré-Definidas Simétricas (Pares Inversos Exatos)
    MACRO_ACTIONS = {
        "SEXY_MOVE_R": ["R", "U", "R_prime", "U_prime"],
        "SEXY_MOVE_R_PRIME": ["U", "R", "U_prime", "R_prime"],
        "SEXY_MOVE_L": ["L_prime", "U_prime", "L", "U"],
        "SEXY_MOVE_L_PRIME": ["U_prime", "L_prime", "U", "L"],
        "SUNE": ["R", "U", "R_prime", "U", "R", "U", "U", "R_prime"],
        "ANTI_SUNE": ["R", "U", "U", "R_prime", "U_prime", "R", "U_prime", "R_prime"],
        "YELLOW_CROSS": ["F", "R", "U", "R_prime", "U_prime", "F_prime"],
        "YELLOW_CROSS_PRIME": ["F", "U", "R", "U_prime", "R_prime", "F_prime"],
        "ROTATE_Y": ["Y"],
        "ROTATE_Y_PRIME": ["Y_prime"],
        "U_TURN": ["U"],
        "U_PRIME_TURN": ["U_prime"],
    }
    MACRO_NAMES = list(MACRO_ACTIONS.keys())

    # Paleta de cores oficial (Dark UI / Cyber Style)
    PALETTE = {
        0: (245, 245, 250),  # U: Branco
        1: (250, 204, 21),   # D: Amarelo
        2: (34, 197, 94),    # F: Verde
        3: (59, 130, 246),   # B: Azul
        4: (239, 68, 68),    # R: Vermelho
        5: (249, 115, 22),   # L: Laranja
    }

    # Coordenadas do desdobramento 2D (Net) no viewport (480x360)
    # Layout em Cruz:
    #       [ U ]
    # [ L ] [ F ] [ R ] [ B ]
    #       [ D ]
    FACE_LAYOUT = {
        0: (164, 46),   # U
        5: (88, 122),   # L
        2: (164, 122),  # F
        4: (240, 122),  # R
        3: (316, 122),  # B
        1: (164, 198),  # D
    }

    INVERSE_ATOMIC = {
        "U": "U_prime", "U_prime": "U",
        "D": "D_prime", "D_prime": "D",
        "F": "F_prime", "F_prime": "F",
        "B": "B_prime", "B_prime": "B",
        "R": "R_prime", "R_prime": "R",
        "L": "L_prime", "L_prime": "L",
    }
    INVERSE_MACROS = {
        "SEXY_MOVE_R": "SEXY_MOVE_R_PRIME",
        "SEXY_MOVE_R_PRIME": "SEXY_MOVE_R",
        "SEXY_MOVE_L": "SEXY_MOVE_L_PRIME",
        "SEXY_MOVE_L_PRIME": "SEXY_MOVE_L",
        "SUNE": "ANTI_SUNE",
        "ANTI_SUNE": "SUNE",
        "YELLOW_CROSS": "YELLOW_CROSS_PRIME",
        "YELLOW_CROSS_PRIME": "YELLOW_CROSS",
        "ROTATE_Y": "ROTATE_Y_PRIME",
        "ROTATE_Y_PRIME": "ROTATE_Y",
        "U_TURN": "U_PRIME_TURN",
        "U_PRIME_TURN": "U_TURN",
    }
    # Macro-ações que efetivamente desalinham e embaralham peças (exclui rotações globais do cubo Y e Y')
    SCRAMBLE_MACRO_NAMES = [
        "SEXY_MOVE_R", "SEXY_MOVE_R_PRIME",
        "SEXY_MOVE_L", "SEXY_MOVE_L_PRIME",
        "SUNE", "ANTI_SUNE",
        "YELLOW_CROSS", "YELLOW_CROSS_PRIME",
        "U_TURN", "U_PRIME_TURN",
    ]

    def __init__(self) -> None:
        self._rust = system1_core.RubiksCubeCore() if HAS_RUST_CORE else None
        if self._rust is not None:
            self.state = np.array(self._rust.get_state(), dtype=np.int32)
        else:
            self.state = np.repeat(np.arange(6, dtype=np.int32), 9)

    def reset(self) -> None:
        """Restaura o cubo para o estado perfeitamente resolvido."""
        if self._rust is not None:
            self._rust.reset()
            self.state = np.array(self._rust.get_state(), dtype=np.int32)
        else:
            self.state = np.repeat(np.arange(6, dtype=np.int32), 9)

    def apply_atomic(self, move_name: str) -> None:
        """Executa um movimento atômico via permutação de arrays."""
        if self._rust is not None:
            self._rust.apply_atomic(move_name)
            self.state = np.array(self._rust.get_state(), dtype=np.int32)
        elif move_name in PERMUTATIONS:
            self.state = self.state[PERMUTATIONS[move_name]]

    def apply_macro(self, macro_name: str) -> None:
        """Executa uma macro-ação (sequência atômica)."""
        if self._rust is not None:
            self._rust.apply_macro(macro_name)
            self.state = np.array(self._rust.get_state(), dtype=np.int32)
        elif macro_name in self.MACRO_ACTIONS:
            for m in self.MACRO_ACTIONS[macro_name]:
                self.apply_atomic(m)

    def scramble(self, depth: int = 4, use_macros: bool = False, rng: Optional[np.random.Generator] = None) -> List[str]:
        """Embaralha o cubo a partir do estado resolvido com uma profundidade aberta sem limite, garantindo não-cancelamento."""
        target_depth = max(1, int(depth))
        if self._rust is not None:
            seed = int(rng.integers(0, 2**63 - 1)) if rng is not None else None
            moves = list(self._rust.scramble(target_depth, use_macros, seed))
            self.state = np.array(self._rust.get_state(), dtype=np.int32)
            return moves

        self.reset()
        if rng is None:
            rng = np.random.default_rng()
        applied = []
        last_act = None
        attempts = 0
        max_attempts = max(100, target_depth * 10)

        if use_macros:
            while (len(applied) < target_depth or self.is_solved()) and attempts < max_attempts:
                attempts += 1
                choices = [m for m in self.SCRAMBLE_MACRO_NAMES if m != self.INVERSE_MACROS.get(last_act)]
                act = str(rng.choice(choices))
                self.apply_macro(act)
                applied.append(act)
                last_act = act
        else:
            while (len(applied) < target_depth or self.is_solved()) and attempts < max_attempts:
                attempts += 1
                choices = [m for m in self.ATOMIC_MOVES if m != self.INVERSE_ATOMIC.get(last_act)]
                act = str(rng.choice(choices))
                self.apply_atomic(act)
                applied.append(act)
                last_act = act
        return applied

    def get_aligned_count(self) -> int:
        """Retorna o número de facetas que combinam com a cor do centro da sua respectiva face (6 a 54)."""
        if self._rust is not None:
            return int(self._rust.get_aligned_count())
        count = 0
        for f in range(6):
            center_color = self.state[f * 9 + 4]
            count += int(np.sum(self.state[f * 9 : f * 9 + 9] == center_color))
        return count

    def get_score(self) -> float:
        """Retorna a métrica de resolução normalizada em [0.0, 1.0]."""
        if self._rust is not None:
            return float(self._rust.get_score())
        aligned = self.get_aligned_count()
        # 6 centros sempre coincidem, sobram 48 facetas
        return float(max(0.0, min(1.0, (aligned - 6) / 48.0)))

    def is_solved(self) -> bool:
        """Verifica se o cubo está 100% resolvido."""
        if self._rust is not None:
            return bool(self._rust.is_solved())
        return self.get_aligned_count() == 54

    def get_one_hot(self) -> np.ndarray:
        """Retorna a observação vetorial one-hot: shape (324,) em [0.0, 1.0]."""
        if self._rust is not None:
            return np.array(self._rust.get_one_hot(), dtype=np.float32)
        one_hot = np.zeros((54, 6), dtype=np.float32)
        one_hot[np.arange(54), self.state] = 1.0
        return one_hot.flatten()

    def render_net(
        self,
        width: int = 480,
        height: int = 320,
        last_action: str = "INITIAL",
        steps: int = 0,
    ) -> np.ndarray:
        """Gera imagem RGB (H, W, 3) com estética cibernética dark mode para o HUD."""
        img = Image.new("RGB", (width, height), color=(11, 15, 25))
        draw = ImageDraw.Draw(img)

        # Moldura estética
        draw.rectangle([6, 6, width - 7, height - 7], outline=(31, 41, 55), width=2)
        draw.rectangle([10, 10, width - 11, height - 11], outline=(17, 24, 39), width=1)

        # Cabeçalho
        aligned = self.get_aligned_count()
        score = self.get_score() * 100.0
        draw.text((20, 16), "⚡ SYSTEM 1 HUD | RUBIK'S CUBE", fill=(56, 189, 248))
        status_txt = f"Passos: {steps} | Alinhamento: {aligned}/54 ({score:.1f}%)"
        draw.text((20, 30), status_txt, fill=(148, 163, 184))

        # Desenho das 6 faces no formato de Cruz
        for f_idx, (fx, fy) in self.FACE_LAYOUT.items():
            for r in range(3):
                for c in range(3):
                    sticker_idx = f_idx * 9 + (r * 3 + c)
                    col_id = int(self.state[sticker_idx])
                    color = self.PALETTE.get(col_id, (255, 255, 255))
                    x1 = fx + c * 24
                    y1 = fy + r * 24
                    x2 = x1 + 22
                    y2 = y1 + 22
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline=(15, 20, 30), width=1)

        # Rodapé com ação recente
        action_color = (74, 222, 128) if self.is_solved() else (226, 232, 240)
        footer_label = "STATUS: RESOLVIDO 🏆" if self.is_solved() else f"AÇÃO ATIVA: {last_action}"
        draw.text((20, height - 28), footer_label, fill=action_color)

        return np.array(img, dtype=np.uint8)

    def render_3d(
        self,
        width: int = 480,
        height: int = 320,
        last_action: str = "INITIAL",
        steps: int = 0,
    ) -> np.ndarray:
        """Renderiza o Cubo Mágico em projeção Isométrica 3D fotorrealista com visão dupla."""
        img = Image.new("RGB", (width, height), color=(7, 9, 14))
        draw = ImageDraw.Draw(img)

        # Moldura estética Cyber-Dark
        draw.rectangle([6, 6, width - 7, height - 7], outline=(30, 41, 59), width=2)
        draw.rectangle([10, 10, width - 11, height - 11], outline=(17, 24, 39), width=1)

        cx = 160
        cy = 175
        scale = 32.0
        cos30 = math.cos(math.radians(30))
        sin30 = math.sin(math.radians(30))

        def project(x, y, z):
            sx = cx + scale * (x * cos30 - z * cos30)
            sy = cy + scale * (x * sin30 + z * sin30 - y)
            return (sx, sy)

        # Sombra sob o cubo principal
        shadow_pts = [
            project(-1.6, -1.6, 1.6),
            project(1.6, -1.6, 1.6),
            project(1.6, -1.6, -1.6),
            project(-1.6, -1.6, -1.6),
        ]
        draw.polygon([(p[0], p[1] + 14) for p in shadow_pts], fill=(12, 16, 24))

        # ================= CUBO PRINCIPAL (Faces: U=0, F=2, R=4) =================
        # Face U (Cima, Face 0, y = 1.5)
        for r in range(3):
            for c in range(3):
                x_min = -1.5 + c * 1.0 + 0.08
                x_max = x_min + 0.84
                z_min = -1.5 + r * 1.0 + 0.08
                z_max = z_min + 0.84
                y_val = 1.5

                p_box = [
                    project(-1.5 + c, 1.5, -1.5 + r),
                    project(-1.5 + c + 1, 1.5, -1.5 + r),
                    project(-1.5 + c + 1, 1.5, -1.5 + r + 1),
                    project(-1.5 + c, 1.5, -1.5 + r + 1),
                ]
                draw.polygon(p_box, fill=(15, 18, 26), outline=(25, 30, 42))

                pts = [
                    project(x_min, y_val, z_min),
                    project(x_max, y_val, z_min),
                    project(x_max, y_val, z_max),
                    project(x_min, y_val, z_max),
                ]
                st_idx = 0 * 9 + (r * 3 + c)
                col_id = int(self.state[st_idx])
                col = self.PALETTE.get(col_id, (255, 255, 255))
                shaded = tuple(min(255, int(v * 1.02)) for v in col)
                draw.polygon(pts, fill=shaded, outline=(10, 12, 18))

        # Face F (Frente, Face 2, z = 1.5)
        for r in range(3):
            for c in range(3):
                x_min = -1.5 + c * 1.0 + 0.08
                x_max = x_min + 0.84
                y_max = 1.5 - r * 1.0 - 0.08
                y_min = y_max - 0.84
                z_val = 1.5

                p_box = [
                    project(-1.5 + c, 1.5 - r, 1.5),
                    project(-1.5 + c + 1, 1.5 - r, 1.5),
                    project(-1.5 + c + 1, 1.5 - r - 1, 1.5),
                    project(-1.5 + c, 1.5 - r - 1, 1.5),
                ]
                draw.polygon(p_box, fill=(15, 18, 26), outline=(25, 30, 42))

                pts = [
                    project(x_min, y_max, z_val),
                    project(x_max, y_max, z_val),
                    project(x_max, y_min, z_val),
                    project(x_min, y_min, z_val),
                ]
                st_idx = 2 * 9 + (r * 3 + c)
                col_id = int(self.state[st_idx])
                col = self.PALETTE.get(col_id, (255, 255, 255))
                shaded = tuple(int(v * 0.92) for v in col)
                draw.polygon(pts, fill=shaded, outline=(10, 12, 18))

        # Face R (Direita, Face 4, x = 1.5)
        for r in range(3):
            for c in range(3):
                z_max = 1.5 - c * 1.0 - 0.08
                z_min = z_max - 0.84
                y_max = 1.5 - r * 1.0 - 0.08
                y_min = y_max - 0.84
                x_val = 1.5

                p_box = [
                    project(1.5, 1.5 - r, 1.5 - c),
                    project(1.5, 1.5 - r, 1.5 - c - 1),
                    project(1.5, 1.5 - r - 1, 1.5 - c - 1),
                    project(1.5, 1.5 - r - 1, 1.5 - c),
                ]
                draw.polygon(p_box, fill=(15, 18, 26), outline=(25, 30, 42))

                pts = [
                    project(x_val, y_max, z_max),
                    project(x_val, y_max, z_min),
                    project(x_val, y_min, z_min),
                    project(x_val, y_min, z_max),
                ]
                st_idx = 4 * 9 + (r * 3 + c)
                col_id = int(self.state[st_idx])
                col = self.PALETTE.get(col_id, (255, 255, 255))
                shaded = tuple(int(v * 0.78) for v in col)
                draw.polygon(pts, fill=shaded, outline=(10, 12, 18))

        # Rótulos do cubo principal
        draw.text((cx - 15, cy - 85), "U (Cima)", fill=(200, 210, 225))
        draw.text((cx - 90, cy + 45), "F (Frente)", fill=(34, 197, 94))
        draw.text((cx + 55, cy + 45), "R (Direita)", fill=(239, 68, 68))

        # ================= INSET 3D REVERSO (Faces: D=1, L=5, B=3) =================
        rx_box = width - 170
        ry_box = 48
        rw_box = 150
        rh_box = 184
        draw.rectangle([rx_box, ry_box, rx_box + rw_box, ry_box + rh_box], fill=(11, 15, 23), outline=(30, 41, 59))
        draw.text((rx_box + 12, ry_box + 8), "REVERSE 3D (D, L, B)", fill=(148, 163, 184))

        rcx = rx_box + rw_box // 2
        rcy = ry_box + rh_box // 2 + 12
        rscale = 17.0

        def rproject(x, y, z):
            sx = rcx + rscale * (x * cos30 - z * cos30)
            sy = rcy + rscale * (x * sin30 + z * sin30 - y)
            return (sx, sy)

        # Face D (Base, Face 1) no topo do inset reverso
        for r in range(3):
            for c in range(3):
                x_min = -1.5 + c * 1.0 + 0.08
                x_max = x_min + 0.84
                z_min = -1.5 + r * 1.0 + 0.08
                z_max = z_min + 0.84
                y_val = 1.5

                pts = [
                    rproject(x_min, y_val, z_min),
                    rproject(x_max, y_val, z_min),
                    rproject(x_max, y_val, z_max),
                    rproject(x_min, y_val, z_max),
                ]
                st_idx = 1 * 9 + (r * 3 + c)
                col_id = int(self.state[st_idx])
                col = self.PALETTE.get(col_id, (255, 255, 255))
                shaded = tuple(min(255, int(v * 1.02)) for v in col)
                draw.polygon(pts, fill=shaded, outline=(10, 12, 18))

        # Face L (Esquerda, Face 5) na frente-esquerda do inset
        for r in range(3):
            for c in range(3):
                x_min = -1.5 + c * 1.0 + 0.08
                x_max = x_min + 0.84
                y_max = 1.5 - r * 1.0 - 0.08
                y_min = y_max - 0.84
                z_val = 1.5

                pts = [
                    rproject(x_min, y_max, z_val),
                    rproject(x_max, y_max, z_val),
                    rproject(x_max, y_min, z_val),
                    rproject(x_min, y_min, z_val),
                ]
                st_idx = 5 * 9 + (r * 3 + c)
                col_id = int(self.state[st_idx])
                col = self.PALETTE.get(col_id, (255, 255, 255))
                shaded = tuple(int(v * 0.92) for v in col)
                draw.polygon(pts, fill=shaded, outline=(10, 12, 18))

        # Face B (Atrás, Face 3) na frente-direita do inset
        for r in range(3):
            for c in range(3):
                z_max = 1.5 - c * 1.0 - 0.08
                z_min = z_max - 0.84
                y_max = 1.5 - r * 1.0 - 0.08
                y_min = y_max - 0.84
                x_val = 1.5

                pts = [
                    rproject(x_val, y_max, z_max),
                    rproject(x_val, y_max, z_min),
                    rproject(x_val, y_min, z_min),
                    rproject(x_val, y_min, z_max),
                ]
                st_idx = 3 * 9 + (r * 3 + c)
                col_id = int(self.state[st_idx])
                col = self.PALETTE.get(col_id, (255, 255, 255))
                shaded = tuple(int(v * 0.78) for v in col)
                draw.polygon(pts, fill=shaded, outline=(10, 12, 18))

        draw.text((rcx - 12, rcy - 50), "D (Base)", fill=(250, 204, 21))
        draw.text((rcx - 55, rcy + 25), "L (Esq)", fill=(249, 115, 22))
        draw.text((rcx + 30, rcy + 25), "B (Tras)", fill=(59, 130, 246))

        # Cabeçalho e Rodapé HUD
        aligned = self.get_aligned_count()
        score = self.get_score() * 100.0
        draw.text((20, 16), "⚡ SYSTEM 1 HUD | 3D VIEW (ISOMETRIC)", fill=(56, 189, 248))
        draw.text((20, 30), f"Passos: {steps} | Alinhamento: {aligned}/54 ({score:.1f}%)", fill=(148, 163, 184))

        action_color = (74, 222, 128) if self.is_solved() else (226, 232, 240)
        footer_label = "STATUS: RESOLVIDO 🏆" if self.is_solved() else f"AÇÃO ATIVA: {last_action}"
        draw.text((20, height - 28), footer_label, fill=action_color)

        return np.array(img, dtype=np.uint8)

    def render(
        self,
        view_mode: str = "3d",
        width: int = 480,
        height: int = 320,
        last_action: str = "INITIAL",
        steps: int = 0,
    ) -> np.ndarray:
        """Renderiza o cubo no modo especificado ('3d' ou '2d')."""
        if str(view_mode).lower() in ("2d", "net", "flat"):
            return self.render_net(width, height, last_action, steps)
        return self.render_3d(width, height, last_action, steps)


# =====================================================================
# AMBIENTES GYMNASIUM
# =====================================================================

class RubiksCubeEnv(gym.Env):
    """Ambiente Gymnasium para Cubo Mágico com Ações Atômicas (12 giros simples)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        scramble_depth: int = 3,
        max_steps: int = 40,
        curriculum: bool = False,
        min_depth: int = 1,
        max_depth: Optional[int] = None,
        target_success_rate: float = 0.90,
        elastic_curriculum: bool = True,
        curriculum_window: int = 64,
        mixed_replay_rate: float = 0.15,
        dynamic_frontier: bool = True,
        proximal_decay: float = 0.8,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.scramble_depth = max(1, int(scramble_depth))
        self.max_steps = max_steps
        self.curriculum = curriculum
        self.min_depth = max(1, int(min_depth))
        self.max_depth = max_depth
        self.target_success_rate = target_success_rate
        self.elastic_curriculum = bool(elastic_curriculum)
        self.curriculum_window = curriculum_window
        self.mixed_replay_rate = max(0.0, min(1.0, float(mixed_replay_rate)))
        self.dynamic_frontier = bool(dynamic_frontier)
        self.proximal_decay = float(proximal_decay)
        self._last_p_frontier = 1.0 - self.mixed_replay_rate
        self.current_depth = min_depth if curriculum else self.scramble_depth
        self._episode_depth = self.current_depth
        self.recent_successes: deque[float] = deque(maxlen=curriculum_window)
        self.frontier_successes: deque[float] = deque(maxlen=curriculum_window)
        self.curriculum_promotions: int = 0
        self.core = RubiksCubeCore()

        # 12 Ações Atômicas: U, U', D, D', F, F', B, B', R, R', L, L'
        self.action_space = gym.spaces.Discrete(len(self.core.ATOMIC_MOVES))
        # 54 facetas x 6 cores = 324 floats em [0, 1]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(324,), dtype=np.float32)

        self.view_mode = "3d"
        self._steps = 0
        self._last_action_name = "NONE"
        self._prev_score = 0.0
        self.rng = np.random.default_rng()

    def get_target_success_rate(self, depth: Optional[int] = None) -> float:
        """Retorna a meta de taxa de sucesso necessária para promoção curricular.

        Se elastic_curriculum estiver ativo, aplica uma curva decrescente adaptada à
        complexidade combinatória do espaço de estados (evitando o platô de erro composto
        p^d em profundidades elevadas):
          - D1..D4 (até 10k estados): 85%
          - D5..D7 (até 8M estados): 70%
          - D8..D10 (até 700M estados): 50%
          - D11..D14: 40%
          - D15+: 35%
        """
        d = depth if depth is not None else self.current_depth
        if not getattr(self, "elastic_curriculum", True):
            return self.target_success_rate
        if d <= 4:
            return min(self.target_success_rate, 0.85)
        elif d <= 6:
            return min(self.target_success_rate, 0.65)
        elif d <= 10:
            return min(self.target_success_rate, 0.50)
        elif d <= 14:
            return min(self.target_success_rate, 0.40)
        else:
            return min(self.target_success_rate, 0.35)

    @property
    def inverse_action_map(self) -> Dict[int, int]:
        return {i: i ^ 1 for i in range(12)}

    @property
    def curriculum_samples(self) -> int:
        f = self.frontier_successes
        if len(f) > 0:
            return len(f)
        return len(self.recent_successes)

    @property
    def curriculum_success_rate(self) -> float:
        f = self.frontier_successes
        if len(f) > 0:
            effective_window = max(len(f), 16) if (self.current_depth >= 5) else len(f)
            return float(sum(f) / effective_window)
        r = self.recent_successes
        if len(r) > 0:
            effective_window = max(len(r), 16) if (self.current_depth >= 5) else len(r)
            return float(sum(r) / effective_window)
        return 0.0

    def set_view_mode(self, mode: str) -> None:
        self.view_mode = str(mode).lower()

    def set_curriculum_depth(self, depth: int) -> int:
        """Define a profundidade curricular atual (Warm-Start), preservando min_depth."""
        clamped = max(self.min_depth, int(depth))
        if self.max_depth is not None:
            clamped = min(clamped, int(self.max_depth))
        self.current_depth = clamped
        self._episode_depth = clamped
        self.curriculum_promotions = max(0, clamped - self.min_depth)
        self.recent_successes.clear()
        self.frontier_successes.clear()
        return self.current_depth

    def sample_curriculum_depth(self) -> int:
        """Amostra a profundidade de embaralhamento com Focalização Dinâmica na Fronteira Proximal.

        Modula adaptativamente a probabilidade de visitar a fronteira ativa (current_depth) vs.
        a vizinhança proximal (current_depth - 1, current_depth - 2) com base na taxa real
        de sucesso observada na fronteira.
        """
        if self.current_depth <= self.min_depth or self.mixed_replay_rate <= 0.0:
            self._last_p_frontier = 1.0
            return self.current_depth

        if not getattr(self, "dynamic_frontier", True):
            # Fallback para mixed replay estático legado
            self._last_p_frontier = 1.0 - self.mixed_replay_rate
            if self.rng.random() >= self.mixed_replay_rate:
                return self.current_depth
            low_depth = max(self.min_depth, self.current_depth - 2)
            return int(self.rng.integers(low_depth, self.current_depth))

        # --- Focalização Dinâmica na Fronteira Proximal ---
        # 1. Taxa de sucesso real observada na fronteira ativa
        success_rate = self.curriculum_success_rate

        # 2. Modulação Dinâmica da Probabilidade da Fronteira (P_frontier)
        # Se o sucesso for baixo (<20%), recua para treinar no degrau preparatório D-1.
        # Conforme sobe, concentra até 90% na fronteira para disparar a promoção.
        p_base = 1.0 - self.mixed_replay_rate
        p_min = max(0.40, p_base - 0.25)
        p_max = min(0.92, p_base + 0.15)

        s_factor = 1.0 / (1.0 + np.exp(-10.0 * (success_rate - 0.35)))
        p_frontier = float(p_min + (p_max - p_min) * s_factor)
        self._last_p_frontier = p_frontier

        if self.rng.random() < p_frontier:
            return self.current_depth

        # 3. Distribuição na Vizinhança Proximal (d < current_depth)
        cur_d = self.current_depth
        proximal_candidates = [d for d in range(self.min_depth, cur_d)]
        if not proximal_candidates:
            return cur_d
        if len(proximal_candidates) == 1:
            return proximal_candidates[0]

        # Separação entre vizinhança proximal (D-1, D-2) e âncoras distais
        proximal_zone = [d for d in proximal_candidates if d >= cur_d - 2]
        anchor_zone = [d for d in proximal_candidates if d < cur_d - 2]

        if anchor_zone and self.rng.random() < 0.12:
            return int(self.rng.choice(anchor_zone))

        # Amostragem com decaimento exponencial na zona proximal (D-1 tem o maior peso)
        decay = getattr(self, "proximal_decay", 0.8)
        weights = np.array([np.exp(-decay * (cur_d - d)) for d in proximal_zone], dtype=np.float64)
        weights /= weights.sum()
        return int(self.rng.choice(proximal_zone, p=weights))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options and "scramble_depth" in options:
            depth = max(1, int(options["scramble_depth"]))
        elif self.curriculum:
            eval_window = self.frontier_successes if len(self.frontier_successes) >= self.curriculum_window else self.recent_successes
            if len(eval_window) >= self.curriculum_window:
                success_rate = float(np.mean(eval_window))
                can_promote = self.max_depth is None or self.current_depth < self.max_depth
                eff_target = self.get_target_success_rate()
                if success_rate >= eff_target and can_promote:
                    old_d = self.current_depth
                    self.current_depth += 1
                    self.curriculum_promotions += 1
                    self.recent_successes.clear()
                    self.frontier_successes.clear()
                    max_d_str = str(self.max_depth) if self.max_depth is not None else "∞"
                    print(
                        f"\n🚀 [CURRICULUM ATÔMICO] Sucesso {success_rate*100:.1f}% >= {eff_target*100:.0f}%! "
                        f"Profundidade promovida: {old_d} -> {self.current_depth}/{max_d_str}"
                    )
            # Focalização Dinâmica na Fronteira Proximal
            depth = self.sample_curriculum_depth()
        else:
            depth = self.scramble_depth

        self._episode_depth = depth

        if options and "view_mode" in options:
            self.set_view_mode(options["view_mode"])
        self._last_scramble_moves = self.core.scramble(depth=depth, use_macros=False, rng=self.rng)
        self._steps = 0
        self._last_action_name = "RESET"
        self._prev_score = self.core.get_score()

        obs = self.core.get_one_hot()
        info = {
            "aligned_stickers": self.core.get_aligned_count(),
            "score": self._prev_score,
            "is_solved": self.core.is_solved(),
            "current_depth": self.current_depth if self.curriculum else depth,
            "scramble_depth": depth,
            "max_depth": self.max_depth,
            "curriculum": self.curriculum,
            "curriculum_promotions": self.curriculum_promotions,
            "curriculum_p_frontier": getattr(self, "_last_p_frontier", 1.0),
            "dynamic_frontier": getattr(self, "dynamic_frontier", True),
            "target_success_rate": self.get_target_success_rate(),
            "curriculum_success_rate": self.curriculum_success_rate,
            "curriculum_samples": self.curriculum_samples,
        }
        return obs, info

    def simulate_step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Simula a execução de 1 passo atômico e desfaz imediatamente via ação inversa (zero alocação).
        Retorna a observação resultante (324,), recompensa estimada, e se foi resolvido."""
        act_idx = int(action)
        action_name = self.core.ATOMIC_MOVES[act_idx]
        inv_name = self.core.INVERSE_ATOMIC[action_name]

        self.core.apply_atomic(action_name)
        is_solved = self.core.is_solved()
        cur_score = self.core.get_score()
        delta_score = cur_score - self._prev_score
        rew = (delta_score * 5.0) - 0.02 + (10.0 if is_solved else 0.0)
        obs = self.core.get_one_hot()

        # Desfaz imediatamente restaurando o estado original exato
        self.core.apply_atomic(inv_name)

        return obs, rew, is_solved, {"is_solved": is_solved}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._steps += 1
        act_idx = int(action)
        action_name = self.core.ATOMIC_MOVES[act_idx]
        self._last_action_name = action_name

        self.core.apply_atomic(action_name)
        cur_score = self.core.get_score()
        delta_score = cur_score - self._prev_score
        self._prev_score = cur_score

        # Recompensa Densa:
        # 1. Delta de progresso (avanço em direção ao alinhamento)
        # 2. Penalidade de tempo (-0.02) para incentivar trajetórias curtas
        # 3. Grande bônus ao resolver (+10.0)
        reward = (delta_score * 5.0) - 0.02
        is_solved = self.core.is_solved()
        if is_solved:
            reward += 10.0

        terminated = is_solved
        eff_max_steps = min(self.max_steps, max(15, getattr(self, "_episode_depth", self.current_depth) * 5)) if self.curriculum else self.max_steps
        truncated = self._steps >= eff_max_steps

        if (terminated or truncated) and self.curriculum:
            self.recent_successes.append(1.0 if is_solved else 0.0)
            if getattr(self, "_episode_depth", self.current_depth) == self.current_depth:
                self.frontier_successes.append(1.0 if is_solved else 0.0)

        obs = self.core.get_one_hot()
        info = {
            "aligned_stickers": self.core.get_aligned_count(),
            "score": cur_score,
            "is_solved": is_solved,
            "action_name": action_name,
            "current_depth": self.current_depth if self.curriculum else self.scramble_depth,
            "scramble_depth": getattr(self, "_episode_depth", self.current_depth),
            "max_depth": self.max_depth,
            "curriculum": self.curriculum,
            "curriculum_promotions": self.curriculum_promotions,
            "target_success_rate": self.get_target_success_rate(),
            "curriculum_success_rate": self.curriculum_success_rate,
            "curriculum_samples": self.curriculum_samples,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self.core.render(
                view_mode=self.view_mode,
                last_action=self._last_action_name,
                steps=self._steps,
            )
        return None

    def clone(self) -> "RubiksCubeEnv":
        """Cria uma cópia idêntica e independente do ambiente no estado exato atual."""
        new_env = RubiksCubeEnv(
            render_mode=self.render_mode,
            scramble_depth=self.scramble_depth,
            max_steps=self.max_steps,
            curriculum=self.curriculum,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            target_success_rate=self.target_success_rate,
            elastic_curriculum=self.elastic_curriculum,
            curriculum_window=self.curriculum_window,
            mixed_replay_rate=self.mixed_replay_rate,
            dynamic_frontier=self.dynamic_frontier,
            proximal_decay=self.proximal_decay,
        )
        new_env.reset()
        new_env.core.reset()
        for m in getattr(self, "_last_scramble_moves", []):
            new_env.core.apply_atomic(m)
        new_env.core.state = self.core.state.copy()
        new_env._steps = self._steps
        new_env._last_action_name = self._last_action_name
        new_env._prev_score = self._prev_score
        new_env._last_scramble_moves = list(getattr(self, "_last_scramble_moves", []))
        new_env._episode_depth = getattr(self, "_episode_depth", self.current_depth)
        new_env.frontier_successes = deque(self.frontier_successes, maxlen=self.curriculum_window)
        return new_env


class RubiksCubeMacroEnv(gym.Env):
    """Ambiente Gymnasium para Cubo Mágico com Macro-Ações e Algoritmos Pré-Definidos.
    
    Ações (Discrete 12 - Pares Inversos Exatos):
    0: SEXY_MOVE_R        (R U R' U')
    1: SEXY_MOVE_R_PRIME  (U R U' R')
    2: SEXY_MOVE_L        (L' U' L U)
    3: SEXY_MOVE_L_PRIME  (U' L' U L)
    4: SUNE               (R U R' U R U2 R')
    5: ANTI_SUNE          (R U2 R' U' R U' R')
    6: YELLOW_CROSS       (F R U R' U' F')
    7: YELLOW_CROSS_PRIME (F U R U' R' F')
    8: ROTATE_Y           (Giro do cubo todo no eixo Y +90°)
    9: ROTATE_Y_PRIME     (Giro do cubo todo no eixo Y -90°)
    10: U_TURN            (U - Giro superior +90°)
    11: U_PRIME_TURN      (U' - Giro superior -90°)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        scramble_depth: int = 2,
        max_steps: int = 25,
        curriculum: bool = False,
        min_depth: int = 1,
        max_depth: Optional[int] = None,
        target_success_rate: float = 0.90,
        elastic_curriculum: bool = True,
        curriculum_window: int = 64,
        mixed_replay_rate: float = 0.15,
        dynamic_frontier: bool = True,
        proximal_decay: float = 0.8,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.scramble_depth = max(1, int(scramble_depth))
        self.max_steps = max_steps
        self.curriculum = curriculum
        self.min_depth = max(1, int(min_depth))
        self.max_depth = max_depth
        self.target_success_rate = target_success_rate
        self.elastic_curriculum = bool(elastic_curriculum)
        self.curriculum_window = curriculum_window
        self.mixed_replay_rate = max(0.0, min(1.0, float(mixed_replay_rate)))
        self.dynamic_frontier = bool(dynamic_frontier)
        self.proximal_decay = float(proximal_decay)
        self._last_p_frontier = 1.0 - self.mixed_replay_rate
        self.current_depth = min_depth if curriculum else self.scramble_depth
        self._episode_depth = self.current_depth
        self.recent_successes: deque[float] = deque(maxlen=curriculum_window)
        self.frontier_successes: deque[float] = deque(maxlen=curriculum_window)
        self.curriculum_promotions: int = 0
        self.core = RubiksCubeCore()

        # 12 Macro-Ações
        self.action_space = gym.spaces.Discrete(len(self.core.MACRO_NAMES))
        # 54 facetas x 6 cores = 324 floats em [0, 1]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(324,), dtype=np.float32)

        self.view_mode = "3d"
        self._steps = 0
        self._last_action_name = "NONE"
        self._prev_score = 0.0
        self.rng = np.random.default_rng()

    def get_target_success_rate(self, depth: Optional[int] = None) -> float:
        """Retorna a meta de taxa de sucesso necessária para promoção curricular.

        Se elastic_curriculum estiver ativo, aplica uma curva decrescente adaptada à
        complexidade combinatória do espaço de estados:
          - D1..D4: 85%
          - D5..D7: 70%
          - D8..D10: 50%
          - D11..D14: 40%
          - D15+: 35%
        """
        d = depth if depth is not None else self.current_depth
        if not getattr(self, "elastic_curriculum", True):
            return self.target_success_rate
        if d <= 4:
            return min(self.target_success_rate, 0.85)
        elif d <= 6:
            return min(self.target_success_rate, 0.65)
        elif d <= 10:
            return min(self.target_success_rate, 0.50)
        elif d <= 14:
            return min(self.target_success_rate, 0.40)
        else:
            return min(self.target_success_rate, 0.35)

    @property
    def inverse_action_map(self) -> Dict[int, int]:
        return {i: i ^ 1 for i in range(12)}

    @property
    def curriculum_samples(self) -> int:
        f = self.frontier_successes
        if len(f) > 0:
            return len(f)
        return len(self.recent_successes)

    @property
    def curriculum_success_rate(self) -> float:
        f = self.frontier_successes
        if len(f) > 0:
            effective_window = max(len(f), 16) if (self.current_depth >= 5) else len(f)
            return float(sum(f) / effective_window)
        r = self.recent_successes
        if len(r) > 0:
            effective_window = max(len(r), 16) if (self.current_depth >= 5) else len(r)
            return float(sum(r) / effective_window)
        return 0.0

    def set_view_mode(self, mode: str) -> None:
        self.view_mode = str(mode).lower()

    def set_curriculum_depth(self, depth: int) -> int:
        """Define a profundidade curricular atual (Warm-Start), preservando min_depth."""
        clamped = max(self.min_depth, int(depth))
        if self.max_depth is not None:
            clamped = min(clamped, int(self.max_depth))
        self.current_depth = clamped
        self._episode_depth = clamped
        self.curriculum_promotions = max(0, clamped - self.min_depth)
        self.recent_successes.clear()
        self.frontier_successes.clear()
        return self.current_depth

    def sample_curriculum_depth(self) -> int:
        """Amostra a profundidade de embaralhamento com Focalização Dinâmica na Fronteira Proximal.

        Modula adaptativamente a probabilidade de visitar a fronteira ativa (current_depth) vs.
        a vizinhança proximal (current_depth - 1, current_depth - 2) com base na taxa real
        de sucesso observada na fronteira.
        """
        if self.current_depth <= self.min_depth or self.mixed_replay_rate <= 0.0:
            self._last_p_frontier = 1.0
            return self.current_depth

        if not getattr(self, "dynamic_frontier", True):
            # Fallback para mixed replay estático legado
            self._last_p_frontier = 1.0 - self.mixed_replay_rate
            if self.rng.random() >= self.mixed_replay_rate:
                return self.current_depth
            low_depth = max(self.min_depth, self.current_depth - 2)
            return int(self.rng.integers(low_depth, self.current_depth))

        # --- Focalização Dinâmica na Fronteira Proximal ---
        # 1. Taxa de sucesso real observada na fronteira ativa
        success_rate = self.curriculum_success_rate

        # 2. Modulação Dinâmica da Probabilidade da Fronteira (P_frontier)
        p_base = 1.0 - self.mixed_replay_rate
        p_min = max(0.40, p_base - 0.25)
        p_max = min(0.92, p_base + 0.15)

        s_factor = 1.0 / (1.0 + np.exp(-10.0 * (success_rate - 0.35)))
        p_frontier = float(p_min + (p_max - p_min) * s_factor)
        self._last_p_frontier = p_frontier

        if self.rng.random() < p_frontier:
            return self.current_depth

        # 3. Distribuição na Vizinhança Proximal (d < current_depth)
        cur_d = self.current_depth
        proximal_candidates = [d for d in range(self.min_depth, cur_d)]
        if not proximal_candidates:
            return cur_d
        if len(proximal_candidates) == 1:
            return proximal_candidates[0]

        # Separação entre vizinhança proximal (D-1, D-2) e âncoras distais
        proximal_zone = [d for d in proximal_candidates if d >= cur_d - 2]
        anchor_zone = [d for d in proximal_candidates if d < cur_d - 2]

        if anchor_zone and self.rng.random() < 0.12:
            return int(self.rng.choice(anchor_zone))

        # Amostragem com decaimento exponencial na zona proximal (D-1 tem o maior peso)
        decay = getattr(self, "proximal_decay", 0.8)
        weights = np.array([np.exp(-decay * (cur_d - d)) for d in proximal_zone], dtype=np.float64)
        weights /= weights.sum()
        return int(self.rng.choice(proximal_zone, p=weights))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options and "scramble_depth" in options:
            depth = max(1, int(options["scramble_depth"]))
        elif self.curriculum:
            eval_window = self.frontier_successes if len(self.frontier_successes) >= self.curriculum_window else self.recent_successes
            if len(eval_window) >= self.curriculum_window:
                success_rate = float(np.mean(eval_window))
                can_promote = self.max_depth is None or self.current_depth < self.max_depth
                eff_target = self.get_target_success_rate()
                if success_rate >= eff_target and can_promote:
                    old_d = self.current_depth
                    self.current_depth += 1
                    self.curriculum_promotions += 1
                    self.recent_successes.clear()
                    self.frontier_successes.clear()
                    max_d_str = str(self.max_depth) if self.max_depth is not None else "∞"
                    print(
                        f"\n🚀 [CURRICULUM MACRO] Sucesso {success_rate*100:.1f}% >= {eff_target*100:.0f}%! "
                        f"Profundidade promovida: {old_d} -> {self.current_depth}/{max_d_str}"
                    )
            # Focalização Dinâmica na Fronteira Proximal
            depth = self.sample_curriculum_depth()
        else:
            depth = self.scramble_depth

        self._episode_depth = depth

        if options and "view_mode" in options:
            self.set_view_mode(options["view_mode"])
        self._last_scramble_moves = self.core.scramble(depth=depth, use_macros=True, rng=self.rng)
        self._steps = 0
        self._last_action_name = "RESET"
        self._prev_score = self.core.get_score()

        obs = self.core.get_one_hot()
        info = {
            "aligned_stickers": self.core.get_aligned_count(),
            "score": self._prev_score,
            "is_solved": self.core.is_solved(),
            "current_depth": self.current_depth if self.curriculum else depth,
            "scramble_depth": depth,
            "max_depth": self.max_depth,
            "curriculum": self.curriculum,
            "curriculum_promotions": self.curriculum_promotions,
            "curriculum_p_frontier": getattr(self, "_last_p_frontier", 1.0),
            "dynamic_frontier": getattr(self, "dynamic_frontier", True),
            "target_success_rate": self.get_target_success_rate(),
            "curriculum_success_rate": self.curriculum_success_rate,
            "curriculum_samples": self.curriculum_samples,
        }
        return obs, info

    def simulate_step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Simula a execução de 1 macro-ação e desfaz imediatamente via macro inversa."""
        act_idx = int(action)
        macro_name = self.core.MACRO_NAMES[act_idx]
        inv_name = self.core.INVERSE_MACROS[macro_name]

        self.core.apply_macro(macro_name)
        is_solved = self.core.is_solved()
        cur_score = self.core.get_score()
        delta_score = cur_score - self._prev_score
        rew = (delta_score * 8.0) - 0.02 + (15.0 if is_solved else 0.0)
        obs = self.core.get_one_hot()

        self.core.apply_macro(inv_name)

        return obs, rew, is_solved, {"is_solved": is_solved}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._steps += 1
        act_idx = int(action)
        macro_name = self.core.MACRO_NAMES[act_idx]
        self._last_action_name = macro_name

        self.core.apply_macro(macro_name)
        cur_score = self.core.get_score()
        delta_score = cur_score - self._prev_score
        self._prev_score = cur_score

        # Recompensa com sub-metas e bônus de resolução
        reward = (delta_score * 8.0) - 0.02
        is_solved = self.core.is_solved()
        if is_solved:
            reward += 15.0

        terminated = is_solved
        eff_max_steps = min(self.max_steps, max(15, getattr(self, "_episode_depth", self.current_depth) * 5)) if self.curriculum else self.max_steps
        truncated = self._steps >= eff_max_steps

        if (terminated or truncated) and self.curriculum:
            self.recent_successes.append(1.0 if is_solved else 0.0)
            if getattr(self, "_episode_depth", self.current_depth) == self.current_depth:
                self.frontier_successes.append(1.0 if is_solved else 0.0)

        obs = self.core.get_one_hot()
        info = {
            "aligned_stickers": self.core.get_aligned_count(),
            "score": cur_score,
            "is_solved": is_solved,
            "action_name": macro_name,
            "current_depth": self.current_depth if self.curriculum else self.scramble_depth,
            "scramble_depth": getattr(self, "_episode_depth", self.current_depth),
            "max_depth": self.max_depth,
            "curriculum": self.curriculum,
            "curriculum_promotions": self.curriculum_promotions,
            "target_success_rate": self.get_target_success_rate(),
            "curriculum_success_rate": self.curriculum_success_rate,
            "curriculum_samples": self.curriculum_samples,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self.core.render(
                view_mode=self.view_mode,
                last_action=self._last_action_name,
                steps=self._steps,
            )
        return None

    def clone(self) -> "RubiksCubeMacroEnv":
        """Cria uma cópia idêntica e independente do ambiente no estado exato atual."""
        new_env = RubiksCubeMacroEnv(
            render_mode=self.render_mode,
            scramble_depth=self.scramble_depth,
            max_steps=self.max_steps,
            curriculum=self.curriculum,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            target_success_rate=self.target_success_rate,
            elastic_curriculum=self.elastic_curriculum,
            curriculum_window=self.curriculum_window,
            mixed_replay_rate=self.mixed_replay_rate,
            dynamic_frontier=self.dynamic_frontier,
            proximal_decay=self.proximal_decay,
        )
        new_env.reset()
        new_env.core.reset()
        for m in getattr(self, "_last_scramble_moves", []):
            new_env.core.apply_macro(m)
        new_env.core.state = self.core.state.copy()
        new_env._steps = self._steps
        new_env._last_action_name = self._last_action_name
        new_env._prev_score = self._prev_score
        new_env._last_scramble_moves = list(getattr(self, "_last_scramble_moves", []))
        new_env._episode_depth = getattr(self, "_episode_depth", self.current_depth)
        new_env.frontier_successes = deque(self.frontier_successes, maxlen=self.curriculum_window)
        return new_env


class InfoList(list):
    """Lista de dicionários de informações que suporta acesso transparente tanto como lista quanto dicionário.

    Garante agregação transparente para o batch vetorizado e propagação correta
    de flags de eventos globais (como 'curriculum_promoted').
    """

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "curriculum_promoted":
                return any(isinstance(x, dict) and x.get("curriculum_promoted", False) for x in self)
            return self[0][key]
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str):
            return any(isinstance(x, dict) and key in x for x in self)
        return super().__contains__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "curriculum_promoted":
            if any(isinstance(x, dict) and x.get("curriculum_promoted", False) for x in self):
                return True
        if len(self) > 0 and isinstance(self[0], dict):
            return self[0].get(key, default)
        return default


class VectorizedRubiksEnv(gym.Env):
    """Ambiente vetorizado nativo para múltiplos cubos mágicos em paralelo.

    Permite simular N cubos simultaneamente em memória (usando o Core Rust),
    sincronizando o progresso curricular de toda a frota e viabilizando forward pass
    em lote (batch) na GPU para throughput massivo.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        num_envs: int = 64,
        is_macro: bool = False,
        render_mode: Optional[str] = "rgb_array",
        **env_kwargs,
    ) -> None:
        super().__init__()
        self.num_envs = max(1, int(num_envs))
        self.is_macro = is_macro
        self.render_mode = render_mode
        env_cls = RubiksCubeMacroEnv if is_macro else RubiksCubeEnv
        self.envs: List[RubiksCubeEnv] = [
            env_cls(render_mode=render_mode, **env_kwargs) for _ in range(self.num_envs)
        ]
        if len(self.envs) > 1:
            shared_successes = self.envs[0].recent_successes
            shared_frontier = getattr(self.envs[0], "frontier_successes", None)
            for env in self.envs[1:]:
                env.recent_successes = shared_successes
                if shared_frontier is not None:
                    env.frontier_successes = shared_frontier

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.view_mode = "3d"

    @property
    def unwrapped(self):
        return self

    @property
    def current_depth(self) -> int:
        return self.envs[0].current_depth

    @property
    def curriculum(self) -> bool:
        return self.envs[0].curriculum

    @property
    def max_depth(self) -> Optional[int]:
        return self.envs[0].max_depth

    @property
    def min_depth(self) -> int:
        return self.envs[0].min_depth

    @property
    def scramble_depth(self) -> int:
        return self.envs[0].scramble_depth

    @property
    def target_success_rate(self) -> float:
        return self.envs[0].get_target_success_rate()

    def get_target_success_rate(self, depth: Optional[int] = None) -> float:
        return self.envs[0].get_target_success_rate(depth)

    @property
    def curriculum_promotions(self) -> int:
        return self.envs[0].curriculum_promotions

    @property
    def curriculum_window(self) -> int:
        return self.envs[0].curriculum_window

    @property
    def recent_successes(self) -> deque[float]:
        return self.envs[0].recent_successes

    @property
    def frontier_successes(self) -> deque[float]:
        return self.envs[0].frontier_successes

    @property
    def curriculum_samples(self) -> int:
        f = self.frontier_successes
        if len(f) > 0:
            return len(f)
        return len(self.recent_successes)

    @property
    def curriculum_success_rate(self) -> float:
        f = self.frontier_successes
        if len(f) > 0:
            effective_window = max(len(f), 16) if (self.envs[0].current_depth >= 5) else len(f)
            return float(sum(f) / effective_window)
        r = self.recent_successes
        if len(r) > 0:
            effective_window = max(len(r), 16) if (self.envs[0].current_depth >= 5) else len(r)
            return float(sum(r) / effective_window)
        return 0.0

    @property
    def action_descriptions(self) -> List[str]:
        return self.envs[0].action_descriptions

    @property
    def inverse_action_map(self) -> Dict[int, int]:
        return self.envs[0].inverse_action_map

    @property
    def mixed_replay_rate(self) -> float:
        return self.envs[0].mixed_replay_rate

    @mixed_replay_rate.setter
    def mixed_replay_rate(self, val: float) -> None:
        v = max(0.0, min(1.0, float(val)))
        for env in self.envs:
            env.mixed_replay_rate = v

    @property
    def dynamic_frontier(self) -> bool:
        return getattr(self.envs[0], "dynamic_frontier", True)

    @dynamic_frontier.setter
    def dynamic_frontier(self, val: bool) -> None:
        v = bool(val)
        for env in self.envs:
            env.dynamic_frontier = v

    @property
    def proximal_decay(self) -> float:
        return getattr(self.envs[0], "proximal_decay", 0.8)

    @proximal_decay.setter
    def proximal_decay(self, val: float) -> None:
        v = float(val)
        for env in self.envs:
            env.proximal_decay = v

    def set_view_mode(self, mode: str) -> None:
        self.view_mode = str(mode).lower()
        for env in self.envs:
            env.set_view_mode(mode)

    def set_curriculum_depth(self, depth: int) -> int:
        """Define e sincroniza a profundidade curricular atual em todos os sub-ambientes."""
        target_d = self.envs[0].set_curriculum_depth(depth)
        for env in self.envs[1:]:
            env.set_curriculum_depth(target_d)
        return target_d

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, InfoList]:
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            env_seed = (seed + i) if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            obs_list.append(obs)
            info_list.append(info)

        shared_depth = max(e.current_depth for e in self.envs)
        shared_promos = max(e.curriculum_promotions for e in self.envs)
        global_rate = self.curriculum_success_rate

        for env in self.envs:
            if env.curriculum:
                env.current_depth = shared_depth
                env.curriculum_promotions = shared_promos

        mean_p_frontier = float(np.mean([inf.get("curriculum_p_frontier", 1.0) for inf in info_list]))
        cur_samples = self.curriculum_samples
        for inf in info_list:
            inf["current_depth"] = shared_depth
            inf["curriculum_promotions"] = shared_promos
            inf["curriculum_success_rate"] = global_rate
            inf["curriculum_samples"] = cur_samples
            inf["dynamic_frontier"] = self.dynamic_frontier
            inf["curriculum_p_frontier"] = mean_p_frontier

        return np.stack(obs_list, axis=0), InfoList(info_list)

    def simulate_step(self, action: int, env_idx: int = 0) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self.envs[env_idx].simulate_step(action)

    def step(
        self, actions: Any, active_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, InfoList]:
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()
        act_arr = np.asarray(actions).flatten()

        next_obs_list = []
        rewards = np.empty(self.num_envs, dtype=np.float32)
        terminateds = np.empty(self.num_envs, dtype=bool)
        truncateds = np.empty(self.num_envs, dtype=bool)
        info_list = []

        shared_depth = self.envs[0].current_depth
        shared_promos = self.envs[0].curriculum_promotions
        any_promoted = False

        for i, env in enumerate(self.envs):
            if active_mask is not None and not active_mask[i]:
                # Preserva o estado do ambiente inativo sem executar rotações residuais
                next_obs_list.append(env.core.get_one_hot())
                rewards[i] = 0.0
                terminateds[i] = False
                truncateds[i] = False
                info_list.append({
                    "aligned_stickers": env.core.get_aligned_count(),
                    "score": env._prev_score,
                    "is_solved": env.core.is_solved(),
                    "current_depth": env.current_depth,
                    "scramble_depth": getattr(env, "_episode_depth", env.current_depth),
                })
                continue

            act_i = int(act_arr[i])
            obs, rew, term, trunc, info = env.step(act_i)
            if term or trunc:
                obs, reset_info = env.reset()
                info["terminal_observation"] = obs
                info["reset_info"] = reset_info
                if env.curriculum and env.current_depth > shared_depth:
                    shared_depth = env.current_depth
                    shared_promos = max(shared_promos + 1, env.curriculum_promotions)
                    any_promoted = True

            next_obs_list.append(obs)
            rewards[i] = rew
            terminateds[i] = term
            truncateds[i] = trunc
            info_list.append(info)

        if any_promoted:
            for other in self.envs:
                other.current_depth = shared_depth
                other.curriculum_promotions = shared_promos
                other.recent_successes.clear()
                other.frontier_successes.clear()
            eff_target = self.get_target_success_rate()
            for inf in info_list:
                inf["current_depth"] = shared_depth
                inf["curriculum_promotions"] = shared_promos
                inf["curriculum_promoted"] = True
                inf["curriculum_success_rate"] = 0.0
                inf["curriculum_samples"] = 0
                inf["target_success_rate"] = eff_target
        else:
            global_rate = self.curriculum_success_rate
            cur_samples = self.curriculum_samples
            eff_target = self.get_target_success_rate()
            for inf in info_list:
                inf["current_depth"] = shared_depth
                inf["curriculum_promotions"] = shared_promos
                inf["curriculum_success_rate"] = global_rate
                inf["curriculum_samples"] = cur_samples
                inf["target_success_rate"] = eff_target

        return (
            np.stack(next_obs_list, axis=0),
            rewards,
            terminateds,
            truncateds,
            InfoList(info_list),
        )

    def render(self) -> Optional[np.ndarray]:
        return self.envs[0].render()


# =====================================================================
# REGISTRO NO GYMNASIUM
# =====================================================================

def register_rubiks_environments() -> None:
    """Registra os ambientes no Gymnasium se ainda não estiverem presentes."""
    try:
        if "RubiksCube-v0" not in gym.envs.registry:
            gym.register(
                id="RubiksCube-v0",
                entry_point="system1_engine.env.adapters.rubiks:RubiksCubeEnv",
                max_episode_steps=50,
            )
        if "RubiksCubeMacro-v0" not in gym.envs.registry:
            gym.register(
                id="RubiksCubeMacro-v0",
                entry_point="system1_engine.env.adapters.rubiks:RubiksCubeMacroEnv",
                max_episode_steps=30,
            )
    except Exception:
        pass


# Executa o registro imediatamente ao importar o módulo
register_rubiks_environments()
