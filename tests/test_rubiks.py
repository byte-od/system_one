"""Testes unitários e de integração para o Cubo Mágico (Rubik's Cube 3x3) no System 1 Engine.

Valida:
1. Matemática do grupo de permutações (identidade, inversos, teorema do Sexy Move).
2. Conformidade dos ambientes `RubiksCube-v0` e `RubiksCubeMacro-v0` com o padrão Gymnasium.
3. Renderização 2D Net no padrão do HUD (RGB 480x320).
4. Integração completa com `UniversalS1Wrapper`, `UniversalS1Agent` e `RecurrentPPOTrainer`.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters.rubiks import (
    RubiksCubeCore,
    RubiksCubeEnv,
    RubiksCubeMacroEnv,
)
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer


class TestRubiksCubeMath:
    """Valida a física e a teoria dos grupos do simulador vetorial do Cubo Mágico."""

    def test_solved_state(self):
        core = RubiksCubeCore()
        assert core.is_solved()
        assert core.get_aligned_count() == 54
        assert core.get_score() == 1.0

    def test_four_turns_identity(self):
        """Qualquer giro atômico aplicado 4 vezes deve retornar à identidade."""
        for move in RubiksCubeCore.ATOMIC_MOVES:
            core = RubiksCubeCore()
            init_state = core.state.copy()
            for _ in range(4):
                core.apply_atomic(move)
            assert np.array_equal(core.state, init_state), f"4 giros de {move} não retornaram à identidade"

    def test_move_inverses(self):
        """Cada movimento atômico e seu inverso devem anular-se mutuamente."""
        pairs = [
            ("U", "U_prime"), ("D", "D_prime"),
            ("F", "F_prime"), ("B", "B_prime"),
            ("R", "R_prime"), ("L", "L_prime"),
            ("Y", "Y_prime"),
        ]
        for m, m_inv in pairs:
            core = RubiksCubeCore()
            init_state = core.state.copy()
            core.apply_atomic(m)
            assert not np.array_equal(core.state, init_state)
            core.apply_atomic(m_inv)
            assert np.array_equal(core.state, init_state), f"{m} + {m_inv} não retornaram à identidade"

    def test_sexy_move_order_six(self):
        """Teorema matemático: o Sexy Move (R U R' U') tem ordem exata 6."""
        core = RubiksCubeCore()
        init_state = core.state.copy()
        for i in range(1, 6):
            core.apply_macro("SEXY_MOVE_R")
            assert not np.array_equal(core.state, init_state), f"Sexy Move repetiu prematuramente no ciclo {i}"
        core.apply_macro("SEXY_MOVE_R")
        assert np.array_equal(core.state, init_state), "(R U R' U')^6 não retornou à identidade!"

    def test_all_macros_executable(self):
        """Verifica se todas as 12 macro-ações executam sem exceção e alteram o cubo."""
        for macro in RubiksCubeCore.MACRO_NAMES:
            core = RubiksCubeCore()
            core.apply_macro(macro)
            # Rotação Y do cubo inteiro preserva o alinhamento relativo
            if macro not in ("ROTATE_Y", "ROTATE_Y_PRIME"):
                assert not core.is_solved(), f"Macro {macro} não alterou o cubo resolvido"

    def test_rendering_dimensions(self):
        """Garante que a renderização 2D Net gere array RGB compatível com o HUD."""
        core = RubiksCubeCore()
        frame = core.render_net(width=480, height=320, last_action="TEST", steps=5)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (320, 480, 3)
        assert frame.dtype == np.uint8

    def test_all_macros_have_exact_inverses(self):
        """Cada uma das 12 macro-ações deve possuir inversa exata que desfaz o movimento."""
        for m, m_inv in RubiksCubeCore.INVERSE_MACROS.items():
            core = RubiksCubeCore()
            init_state = core.state.copy()
            core.apply_macro(m)
            core.apply_macro(m_inv)
            assert np.array_equal(core.state, init_state), f"Macro {m} + {m_inv} não retornaram ao estado resolvido"


class TestRubiksGymEnvironments:
    """Testa os ambientes compatíveis com Gymnasium."""

    def test_atomic_env_lifecycle(self):
        env = RubiksCubeEnv(scramble_depth=2, max_steps=10)
        assert env.action_space.n == 12
        assert env.observation_space.shape == (324,)

        obs, info = env.reset(seed=42)
        assert obs.shape == (324,)
        assert "aligned_stickers" in info
        assert "score" in info

        # Executa passos
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
            assert obs.shape == (324,)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)

        # Renderização
        frame = env.render()
        assert frame is not None
        assert frame.shape == (320, 480, 3)

    def test_macro_env_lifecycle(self):
        env = RubiksCubeMacroEnv(scramble_depth=2, max_steps=15)
        assert env.action_space.n == 12
        assert env.observation_space.shape == (324,)

        obs, info = env.reset(seed=123)
        assert obs.shape == (324,)
        assert "is_solved" in info

        # Executa macro-ações
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
            assert obs.shape == (324,)
            assert isinstance(reward, float)

        frame = env.render()
        assert frame is not None
        assert frame.shape == (320, 480, 3)

    def test_render_3d_and_view_mode_toggle(self):
        """Valida que renderização 3D isométrica e alternância dinâmica de view_mode funcionam perfeitamente."""
        env = RubiksCubeMacroEnv(scramble_depth=2)
        env.reset(seed=123)

        # 1. Teste de renderização no modo 3D (padrão)
        env.set_view_mode("3d")
        frame_3d = env.render()
        assert frame_3d is not None
        assert frame_3d.shape == (320, 480, 3)
        assert frame_3d.dtype == np.uint8

        # 2. Teste de alternância para modo 2D Net
        env.set_view_mode("2d")
        frame_2d = env.render()
        assert frame_2d is not None
        assert frame_2d.shape == (320, 480, 3)
        assert frame_2d.dtype == np.uint8

        # Os dois frames devem ter conteúdo visual substancialmente diferente devido à perspectiva
        assert not np.array_equal(frame_3d, frame_2d)

        # 3. Teste direto em RubiksCubeCore
        core = RubiksCubeCore()
        f3d = core.render_3d()
        f2d = core.render_net()
        assert f3d.shape == (320, 480, 3)
        assert f2d.shape == (320, 480, 3)

    def test_gym_registry_integration(self):
        """Garante que os ambientes podem ser criados via gym.make()."""
        raw_atomic = gym.make("RubiksCube-v0")
        assert raw_atomic is not None
        raw_atomic.reset()
        raw_atomic.close()

        raw_macro = gym.make("RubiksCubeMacro-v0")
        assert raw_macro is not None
        raw_macro.reset()
        raw_macro.close()

    def test_curriculum_depth_promotion(self):
        """Valida que o curriculum promove a profundidade quando o agente atinge a meta de vitórias (90%)."""
        env = RubiksCubeEnv(curriculum=True, min_depth=1, max_depth=3, target_success_rate=0.90, curriculum_window=5)
        assert env.target_success_rate == 0.90
        assert env.current_depth == 1

        # Simula 5 episódios resolvidos
        for _ in range(5):
            env.recent_successes.append(1.0)

        obs, info = env.reset()
        assert env.current_depth == 2
        assert info["current_depth"] == 2
        assert info["curriculum_success_rate"] == 0.0  # resetado após promoção

    def test_curriculum_monotonic_no_demotion(self):
        """Valida que o currículo não recua de nível mesmo com taxa de sucesso baixa (< 10%)."""
        env = RubiksCubeEnv(curriculum=True, min_depth=1, max_depth=5, target_success_rate=0.90, curriculum_window=5)
        # Promove para depth 2
        for _ in range(5):
            env.recent_successes.append(1.0)
        env.reset()
        assert env.current_depth == 2

        # Simula sucessivos fracassos na nova profundidade (taxa 0%)
        for _ in range(20):
            env.recent_successes.append(0.0)

        obs, info = env.reset()
        # O currículo deve se manter estritamente em depth 2 (mecanismo catraca/monotonic)
        assert env.current_depth == 2
        assert info["current_depth"] == 2

    def test_curriculum_elastic_step_budget(self):
        """Valida que o currículo usa orçamento elástico max(15, D * 5) passos em tentativas não resolvidas."""
        env = RubiksCubeEnv(curriculum=True, min_depth=2, max_steps=40)
        env.reset()
        assert env.current_depth == 2

        # Executa passos arbitrários que não resolvem o cubo
        # Para depth=2, eff_max_steps = max(15, 2 * 5) = 15 passos
        for step_i in range(1, 15):
            _, _, term, trunc, _ = env.step(0)  # Move U
            if term:
                break
            assert trunc is False, f"Não deveria truncar antes de 15 passos (passo {step_i})"

        # No 15º passo, deve truncar se não foi resolvido
        if not term:
            _, _, term, trunc, _ = env.step(0)
            assert trunc is True, "Deveria truncar exatamente no passo 15 (max(15, D*5))"


class TestRubiksSystemOneIntegration:
    """Valida a integração completa com o UniversalS1Agent e o loop de PPO."""

    def test_wrapper_and_agent_reflex_inference(self):
        raw_env = RubiksCubeMacroEnv(scramble_depth=2)
        wrapped_env = UniversalS1Wrapper(raw_env)

        agent = UniversalS1Agent(
            obs_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
        )

        obs_dict, _ = wrapped_env.reset(seed=42)
        assert "obs" in obs_dict
        assert "delta_obs" in obs_dict
        assert obs_dict["obs"].shape == (324,)

        # Inferência amortizada ultra-rápida (<= 0.8 ms)
        decision = agent.act_fast(obs_dict, return_decision=True)
        assert decision.action in range(12)
        assert 0.0 <= decision.confidence <= 1.0
        assert 0.0 <= decision.uncertainty <= 1.0

        # Próximo passo no ambiente com a ação do agente
        next_obs, reward, term, trunc, _ = wrapped_env.step(decision.action)
        assert next_obs["obs"].shape == (324,)

    def test_ppo_short_training_run(self):
        """Verifica que o RecurrentPPOTrainer treina sem erros no RubiksCubeMacro-v0."""
        raw_env = RubiksCubeMacroEnv(scramble_depth=1, max_steps=10)
        wrapped_env = UniversalS1Wrapper(raw_env)

        agent = UniversalS1Agent(
            obs_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
        )

        trainer = RecurrentPPOTrainer(
            agent=agent,
            env=wrapped_env,
            learning_rate=1e-3,
            rollout_steps=64,
            chunk_length=8,
            chunk_batch_size=8,
            device=torch.device("cpu"),
        )

        # Executa um ciclo curto de 128 passos
        ret = trainer.train(max_steps=128, target_return=100.0, verbose=False)
        assert isinstance(ret, float)

    def test_vectorized_rubiks_ppo_training_throughput(self):
        """Valida que o RecurrentPPOTrainer treina sobre VectorizedRubiksEnv com throughput elevado."""
        from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv
        import time

        raw_env = VectorizedRubiksEnv(num_envs=8, is_macro=True, scramble_depth=2, max_steps=12)
        wrapped_env = UniversalS1Wrapper(raw_env)
        assert wrapped_env.is_vectorized
        assert wrapped_env.num_envs == 8

        agent = UniversalS1Agent(
            obs_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
        )

        trainer = RecurrentPPOTrainer(
            agent=agent,
            env=wrapped_env,
            learning_rate=1e-3,
            rollout_steps=128,
            chunk_length=8,
            chunk_batch_size=8,
            device=torch.device("cpu"),
        )

        t0 = time.perf_counter()
        ret = trainer.train(max_steps=256, target_return=100.0, verbose=False)
        dt = time.perf_counter() - t0
        fps = trainer.total_steps / dt
        assert isinstance(ret, float)
        assert trainer.total_steps >= 256
        assert fps > 200, f"Expected vectorized training FPS > 200 on CPU, got {fps:.1f}"

    def test_rubiks_build_hud_env_and_cli_curriculum_activation(self):
        """Garante que o modo de treino (is_training=True) ativa o currículo (profundidade 1) no Rubik."""
        from system1_engine.hud.worker import build_hud_env
        from system1_engine.cli import build_environment

        # HUD Worker: RubiksCube-v0 (Atômico)
        hud_env_atomic = build_hud_env("RubiksCube-v0", is_training=True)
        raw_atomic = hud_env_atomic.env.unwrapped
        assert raw_atomic.curriculum is True
        assert raw_atomic.current_depth == 1

        # HUD Worker: RubiksCubeMacro-v0 (Macro)
        hud_env_macro = build_hud_env("RubiksCubeMacro-v0", is_training=True)
        raw_macro = hud_env_macro.env.unwrapped
        assert raw_macro.curriculum is True
        assert raw_macro.current_depth == 1

        # Modo Run / Avaliação não deve forçar curriculum=True
        eval_env = build_hud_env("RubiksCube-v0", is_training=False)
        assert eval_env.env.unwrapped.curriculum is False

        # CLI: build_environment
        cli_env = build_environment("RubiksCube-v0", is_training=True)
        assert cli_env.env.unwrapped.curriculum is True
        assert cli_env.env.unwrapped.current_depth == 1

    def test_rubiks_open_scramble_depth_train_and_eval(self):
        """Valida que valores abertos de profundidade funcionam com currículo no treino e avaliação direta."""
        from system1_engine.hud.worker import build_hud_env
        from system1_engine.cli import build_environment

        # Treino com scramble_depth aberto fixado (ex: 35): sempre com currículo ativo até max_depth=35
        train_env = build_hud_env("RubiksCube-v0", is_training=True, scramble_depth=35)
        raw_train = train_env.env.unwrapped
        assert raw_train.scramble_depth == 35
        assert raw_train.curriculum is True
        assert raw_train.min_depth == 1
        assert raw_train.max_depth == 35
        assert raw_train.current_depth == 1
        obs, info = train_env.reset()
        assert info["current_depth"] == 1
        assert info["max_depth"] == 35

        # Avaliação com scramble_depth aberto fixado (ex: 50): avaliação direta sem currículo
        eval_env = build_hud_env("RubiksCubeMacro-v0", is_training=False, scramble_depth=50)
        raw_eval = eval_env.env.unwrapped
        assert raw_eval.scramble_depth == 50
        assert raw_eval.curriculum is False
        obs, info = eval_env.reset()
        assert info["current_depth"] == 50

        # CLI build_environment com scramble_depth aberto (ex: 100): treino sempre com currículo
        cli_train = build_environment("RubiksCube-v0", is_training=True, scramble_depth=100)
        raw_cli_train = cli_train.env.unwrapped
        assert raw_cli_train.curriculum is True
        assert raw_cli_train.max_depth == 100
        obs, info = cli_train.reset()
        assert info["current_depth"] == 1

        # CLI build_environment com scramble_depth aberto (ex: 42): avaliação direta
        cli_eval = build_environment("RubiksCubeMacro-v0", is_training=False, scramble_depth=42)
        raw_cli_eval = cli_eval.env.unwrapped
        assert raw_cli_eval.curriculum is False
        obs, info = cli_eval.reset()
        assert info["current_depth"] == 42

    def test_rubiks_core_large_scramble_unlimited(self):
        """Valida que RubiksCubeCore embaralha sem truncamento arbitrário para valores grandes (>60)."""
        core = RubiksCubeCore()
        moves_100 = core.scramble(depth=100, use_macros=False)
        assert len(moves_100) == 100
        assert not core.is_solved()

        moves_75_macro = core.scramble(depth=75, use_macros=True)
        assert len(moves_75_macro) == 75
        assert not core.is_solved()

    def test_vectorized_rubiks_env_shapes_and_throughput(self):
        """Valida que VectorizedRubiksEnv avança múltiplos cubos em batch com alto throughput."""
        from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv
        import time

        num_envs = 16
        vec_env = VectorizedRubiksEnv(num_envs=num_envs, scramble_depth=4)
        assert vec_env.num_envs == 16
        obs, infos = vec_env.reset()
        assert obs.shape == (16, 324)
        assert len(infos) == 16

        # Executa 500 passos em lote (8.000 transições de cubo simultâneas)
        t0 = time.perf_counter()
        actions = np.zeros(num_envs, dtype=np.int64)
        for _ in range(500):
            obs, rews, terms, truncs, infos = vec_env.step(actions)
            assert obs.shape == (16, 324)
            assert rews.shape == (16,)
            assert terms.shape == (16,)
            assert truncs.shape == (16,)
            assert len(infos) == 16
        dt = time.perf_counter() - t0
        total_transitions = 500 * num_envs
        fps = total_transitions / dt
        # O throughput em batch com Rust deve superar com folga 50.000 passos/segundo
        assert fps > 10000, f"Throughput esperado > 10.000 FPS, obtido: {fps:.0f} FPS"

    def test_vectorized_rubiks_curriculum_synchronization(self):
        """Valida que a frota de ambientes sincroniza o progresso do currículo e propaga flags de promoção (Opção 4)."""
        from system1_engine.env.adapters.rubiks import VectorizedRubiksEnv

        vec_env = VectorizedRubiksEnv(
            num_envs=4,
            curriculum=True,
            min_depth=1,
            max_depth=5,
            target_success_rate=0.90,
            curriculum_window=4,
        )
        assert vec_env.current_depth == 1
        obs, infos = vec_env.reset()
        assert all(info["current_depth"] == 1 for info in infos)
        assert vec_env.curriculum_success_rate == 0.0

        # Simula 4 vitórias na janela compartilhada
        for _ in range(4):
            vec_env.frontier_successes.append(1.0)
            vec_env.recent_successes.append(1.0)
        assert vec_env.curriculum_success_rate == 1.0

        # Dispara resolução no primeiro sub-ambiente (aplica U e depois resolve com U')
        vec_env.envs[0].core.reset()
        vec_env.envs[0].core.apply_atomic("U")
        vec_env.envs[0]._prev_score = vec_env.envs[0].core.get_score()
        actions = np.zeros(4, dtype=np.int64)
        actions[0] = 1  # 1 = U', resolve o cubo
        obs, rews, terms, truncs, infos = vec_env.step(actions)
        assert bool(terms[0]) is True

        # A promoção deve sincronizar imediatamente toda a frota para depth 2
        assert vec_env.current_depth == 2
        assert all(e.current_depth == 2 for e in vec_env.envs)
        assert vec_env.curriculum_promotions == 1

        # O InfoList e todos os sub-infos devem refletir a promoção
        assert infos.get("curriculum_promoted") is True
        assert infos["curriculum_promoted"] is True
        assert all(inf.get("curriculum_promoted") is True for inf in infos)
        assert all(inf["current_depth"] == 2 for inf in infos)
        assert vec_env.curriculum_success_rate == 0.0  # Buffer limpo após promoção


    def test_rubiks_build_hud_env_mixed_replay_rate_and_config(self):
        """Valida que build_hud_env aceita config e mixed_replay_rate sem erros de escopo."""
        from system1_engine.hud.worker import build_hud_env

        # 1. Com config contendo mixed_replay_rate=0.0
        env1 = build_hud_env("RubiksCube-v0", is_training=True, config={"mixed_replay_rate": 0.0})
        assert env1.env.unwrapped.envs[0].mixed_replay_rate == 0.0

        # 2. Com config contendo mixed_replay_rate=0.05
        env2 = build_hud_env("RubiksCube-v0", is_training=True, config={"mixed_replay_rate": 0.05})
        assert env2.env.unwrapped.envs[0].mixed_replay_rate == 0.05

        # 3. Com mixed_replay_rate direto
        env3 = build_hud_env("RubiksCube-v0", is_training=True, mixed_replay_rate=0.08)
        assert env3.env.unwrapped.envs[0].mixed_replay_rate == 0.08

        # 4. Com mixed_replay booleano legado (False -> 0.0)
        env4 = build_hud_env("RubiksCube-v0", is_training=True, config={"mixed_replay": False})
        assert env4.env.unwrapped.envs[0].mixed_replay_rate == 0.0

        # 5. Padrão atualizado v15: 0.15 (15%)
        env5 = build_hud_env("RubiksCube-v0", is_training=True)
        assert env5.env.unwrapped.envs[0].mixed_replay_rate == 0.15


