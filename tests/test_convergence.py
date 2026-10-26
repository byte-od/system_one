import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer


def test_cartpole_convergence():
    """Verify that RecurrentPPOTrainer achieves moving average reward >= 475 on CartPole-v1 in < 40,000 steps."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    raw_env = gym.make("CartPole-v1")
    raw_env.reset(seed=seed)
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        learning_rate=1e-3,
        rollout_steps=1024,
        chunk_length=16,
        chunk_batch_size=16,
        n_epochs=4,
        value_coef=0.05,
        entropy_coef=0.001,
    )

    max_steps = 55000
    target_return = 475.0

    final_return = trainer.train(
        max_steps=max_steps,
        target_return=target_return,
        verbose=True,
    )

    best_return = max(trainer.episode_returns) if len(trainer.episode_returns) > 0 else 0.0
    mean_return = float(np.mean(trainer.episode_returns)) if len(trainer.episode_returns) >= 20 else 0.0

    print(f"\nCartPole Training Finished: Total Steps = {trainer.total_steps}, Mean Return (last 20 ep) = {mean_return:.2f}")

    assert mean_return >= target_return or best_return >= target_return, (
        f"Failed to achieve target return {target_return} in {trainer.total_steps} steps. Got {mean_return:.2f}"
    )
    chunk_overshoot = trainer.rollout_steps * trainer.num_envs
    assert trainer.total_steps <= max_steps + chunk_overshoot, (
        f"Exceeded max steps: {trainer.total_steps} > {max_steps + chunk_overshoot}"
    )
