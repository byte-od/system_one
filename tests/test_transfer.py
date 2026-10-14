import os
import gymnasium as gym
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def test_transfer_without_catastrophic_forgetting(tmp_path):
    """Verify cross-scenario transfer: weights loaded from CartPole to Acrobot remain frozen and identical."""
    # 1. Train or save a checkpoint on CartPole
    ckpt_path = str(tmp_path / "s1_cartpole.pt")
    if os.path.exists("s1_cartpole.pt"):
        # Use existing verified checkpoint
        ckpt_path = "s1_cartpole.pt"
    else:
        # Create a fresh agent and save state
        cart_env = UniversalS1Wrapper(gym.make("CartPole-v1"))
        cart_agent = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space)
        KnowledgeTransferManager.save_checkpoint(cart_agent, ckpt_path, auto_sync_web=False)

    # 2. Initialize agent in Acrobot-v1 (obs_dim=6, act_dim=3)
    acrobot_raw = gym.make("Acrobot-v1")
    acrobot_env = UniversalS1Wrapper(acrobot_raw)

    acrobot_agent = UniversalS1Agent(
        obs_space=acrobot_env.env.observation_space,
        action_space=acrobot_env.action_space,
    )

    # Confirm observation shapes differ
    assert acrobot_env.env.observation_space.shape[0] == 6

    # 3. Load transferable trunk weights with freeze_trunk=True
    loaded_keys = KnowledgeTransferManager.load_transferable_weights(
        agent=acrobot_agent,
        checkpoint_path=ckpt_path,
        freeze_trunk=True,
    )
    assert len(loaded_keys) > 0, "No trunk weights loaded"

    # 4. Verify all trunk parameters have requires_grad == False and record initial values
    initial_trunk_weights = {}
    for name, param in acrobot_agent.trunk.named_parameters():
        assert not param.requires_grad, f"Trunk param {name} requires_grad must be False"
        initial_trunk_weights[name] = param.detach().clone()

    # Verify non-trunk parameters DO require grad
    for name, param in acrobot_agent.front_end.named_parameters():
        assert param.requires_grad, f"Front-end param {name} should be trainable"
    for name, param in acrobot_agent.policy_head.named_parameters():
        assert param.requires_grad, f"Policy head param {name} should be trainable"

    # 5. Train 5 epochs on Acrobot-v1
    trainer = RecurrentPPOTrainer(
        agent=acrobot_agent,
        env=acrobot_env,
        learning_rate=1e-3,
        rollout_steps=1024,
        chunk_length=16,
        chunk_batch_size=16,
        n_epochs=5,
    )

    obs_dict, _ = acrobot_env.reset()
    obs_dict, hx, ep_start, _ = trainer.collect_rollouts(obs_dict, None, True)
    trainer.train_epoch()

    # 6. CRITICAL ACCEPTANCE CHECK:
    # Verify all trunk parameters remain strictly requires_grad == False and bitwise identical
    for name, param in acrobot_agent.trunk.named_parameters():
        assert not param.requires_grad, f"Trunk param {name} requires_grad became True after training"
        diff = torch.max(torch.abs(param - initial_trunk_weights[name])).item()
        assert diff == 0.0, f"Trunk param {name} changed numerically by {diff}!"


def test_checkpoint_inspection_and_evaluation_compatibility(tmp_path):
    """Ensure inspect_checkpoint reads metadata and validate_evaluation_compatibility detects dimension mismatches."""
    # 1. Save an Acrobot checkpoint
    acro_env = UniversalS1Wrapper(gym.make("Acrobot-v1"))
    acro_agent = UniversalS1Agent(acro_env.env.observation_space, acro_env.action_space)
    acro_path = str(tmp_path / "s1_acrobot.pt")
    KnowledgeTransferManager.save_checkpoint(
        acro_agent,
        acro_path,
        extra_info={"env_id": "Acrobot-v1", "final_return": -85.0, "steps": 5000, "curriculum_depth": 3, "curriculum_max_depth": 5},
        auto_sync_web=False,
    )

    # 2. Inspect checkpoint
    meta = KnowledgeTransferManager.inspect_checkpoint(acro_path)
    assert meta["env_id"] == "Acrobot-v1"
    assert meta["obs_dim"] == 6
    assert meta["act_dim"] == 3
    assert meta["final_return"] == -85.0
    assert meta["steps"] == 5000
    assert meta["curriculum_depth"] == 3
    assert meta["depth"] == 3
    assert meta["curriculum_max_depth"] == 5

    # 3. Create a CartPole agent (obs=4, act=2)
    cart_env = UniversalS1Wrapper(gym.make("CartPole-v1"))
    cart_agent = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space)

    # 4. Check compatibility: evaluating an Acrobot checkpoint on CartPole must return an informative error message
    checkpoint = torch.load(acro_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    err = KnowledgeTransferManager.validate_evaluation_compatibility(cart_agent, state_dict, "CartPole-v1", acro_path)
    assert err is not None
    assert "ERRO DE COMPATIBILIDADE DE CHECKPOINT" in err
    assert "front_end.state_encoder.net.0.weight" in err
    assert "Acrobot-v1" in err

    # 5. But evaluating it on an Acrobot agent must return None (compatible!)
    err_compat = KnowledgeTransferManager.validate_evaluation_compatibility(acro_agent, state_dict, "Acrobot-v1", acro_path)
    assert err_compat is None


def test_load_for_training_warm_start(tmp_path):
    """Ensure load_for_training restores full model (warm-start) when architecture matches, preserving policy head."""
    cart_env = UniversalS1Wrapper(gym.make("CartPole-v1"))
    agent_source = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space)

    ckpt_path = str(tmp_path / "cart_source.pt")
    KnowledgeTransferManager.save_checkpoint(agent_source, ckpt_path, auto_sync_web=False)

    # Create target agent with identical specs
    agent_target = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space)

    # Verify initial weights differ before loading
    initial_phead_diff = torch.max(
        torch.abs(agent_target.policy_head.linear.weight - agent_source.policy_head.linear.weight)
    ).item()
    assert initial_phead_diff > 0.0

    # Load for training with default settings
    res = KnowledgeTransferManager.load_for_training(agent_target, ckpt_path)
    assert res["mode"] == "warm_start"
    assert res["trunk_frozen"] is False
    assert res["total_params"] == len(agent_target.state_dict())

    # Verify policy head was loaded identically (knowledge preserved!)
    phead_diff = torch.max(
        torch.abs(agent_target.policy_head.linear.weight - agent_source.policy_head.linear.weight)
    ).item()
    assert phead_diff == 0.0

    # Verify all parameters remain trainable for fine-tuning
    for param in agent_target.parameters():
        assert param.requires_grad


def test_load_for_training_cross_domain_fallback(tmp_path):
    """Ensure load_for_training falls back to trunk-only transfer when shapes mismatch."""
    cart_env = UniversalS1Wrapper(gym.make("CartPole-v1"))
    cart_agent = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space)
    cart_path = str(tmp_path / "cart.pt")
    KnowledgeTransferManager.save_checkpoint(cart_agent, cart_path, auto_sync_web=False)

    acro_env = UniversalS1Wrapper(gym.make("Acrobot-v1"))
    acro_agent = UniversalS1Agent(acro_env.env.observation_space, acro_env.action_space)

    # Acrobot has different obs_dim (6 vs 4) and act_dim (3 vs 2)
    res = KnowledgeTransferManager.load_for_training(acro_agent, cart_path)
    assert res["mode"] == "cross_domain"
    assert res["trunk_frozen"] is True

    # Verify trunk is frozen and heads are trainable
    for param in acro_agent.trunk.parameters():
        assert not param.requires_grad
    for param in acro_agent.policy_head.parameters():
        assert param.requires_grad
    for param in acro_agent.front_end.parameters():
        assert param.requires_grad


def test_load_for_training_force_trunk_only(tmp_path):
    """Ensure force_trunk_only=True bypasses warm-start and loads only trunk even if shapes match."""
    cart_env = UniversalS1Wrapper(gym.make("CartPole-v1"))
    cart_agent = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space)
    cart_path = str(tmp_path / "cart_force.pt")
    KnowledgeTransferManager.save_checkpoint(cart_agent, cart_path, auto_sync_web=False)

    agent_target = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space)
    target_phead_before = agent_target.policy_head.linear.weight.detach().clone()

    res = KnowledgeTransferManager.load_for_training(agent_target, cart_path, force_trunk_only=True, freeze_trunk=True)
    assert res["mode"] == "cross_domain"
    assert res["trunk_frozen"] is True

    # Policy head must NOT have been overwritten by source checkpoint
    diff = torch.max(torch.abs(agent_target.policy_head.linear.weight - target_phead_before)).item()
    assert diff == 0.0


def test_inspect_checkpoint_both_returns_and_successful_depth(tmp_path):
    """Verifica se inspect_checkpoint extrai pico médio, retorno final e aponta para a profundidade de sucesso."""
    cart_env = UniversalS1Wrapper(gym.make("CartPole-v1"))
    agent = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space)
    ckpt_path = str(tmp_path / "s1_test_curriculum.pt")

    extra = {
        "env_id": "RubiksCube-v0",
        "best_mean_return": 8.5,
        "final_return": 2.1,
        "successful_depth": 6,
        "curriculum_depth": 7,
        "curriculum_max_depth": 10,
        "depth": 6,
        "steps": 50000,
    }
    KnowledgeTransferManager.save_checkpoint(agent, ckpt_path, extra_info=extra, auto_sync_web=False)

    meta = KnowledgeTransferManager.inspect_checkpoint(ckpt_path)
    assert meta["best_mean_return"] == 8.5
    assert meta["final_return"] == 2.1
    assert meta["successful_depth"] == 6
    assert meta["curriculum_depth"] == 7
    assert meta["curriculum_max_depth"] == 10
    # depth deve apontar para o nível com sucesso
    assert meta["depth"] == 6


def test_load_for_training_net2net_expansion(tmp_path):
    """Ensure Net2Net architectural expansion from classic trunk (256D) to LeanResidualTrunk (512D) preserves decisions."""
    cart_env = UniversalS1Wrapper(gym.make("CartPole-v1"))
    source_agent = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space, trunk_type="classic")

    ckpt_path = str(tmp_path / "classic_source.pt")
    KnowledgeTransferManager.save_checkpoint(source_agent, ckpt_path, auto_sync_web=False)

    # Create target agent with LeanResidualTrunk
    target_agent = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space, trunk_type="lean")
    assert target_agent.trunk.hidden_dim == 512

    # Load for training with Net2Net automatic expansion
    res = KnowledgeTransferManager.load_for_training(target_agent, ckpt_path)
    assert res["mode"] == "net2net_expansion"
    assert len(res["loaded_keys"]) > 0

    # Verify input_norm was transferred
    assert torch.allclose(target_agent.trunk.input_norm.weight, source_agent.trunk.input_norm.weight)
    # Verify fast_gru was transferred from classic gru
    assert torch.allclose(target_agent.trunk.fast_gru.weight_ih_l0, source_agent.trunk.gru.weight_ih_l0)
    # Verify policy head first 256 dims match
    assert torch.allclose(target_agent.policy_head.linear.weight[:, :256], source_agent.policy_head.linear.weight)
    assert torch.allclose(target_agent.policy_head.linear.weight[:, 256:], torch.zeros_like(target_agent.policy_head.linear.weight[:, 256:]))



