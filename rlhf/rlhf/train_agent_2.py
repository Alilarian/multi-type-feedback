"""Module for training an RL agent."""

import argparse
import os
import sys
import typing
from os import path
from pathlib import Path
import wandb
import numpy as np
from wandb.integration.sb3 import WandbCallback

import numpy
import gymnasium as gym
import pytorch_lightning as pl
import torch
from imitation.rewards.reward_function import RewardFn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed

# register custom envs
import ale_py
import minigrid
import highway_env

from rlhf.datatypes import FeedbackType
from rlhf.networks import LightningNetwork, LightningCnnNetwork, calculate_pairwise_loss, calculate_single_reward_loss

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, StoreDict
from rlhf.utils import TrainingUtils

class CustomReward(RewardFn):
    """Custom reward based on fine-tuned reward model."""

    def __init__(
        self,
        reward_model_cls: typing.Union[LightningNetwork, LightningCnnNetwork] = None,
        reward_model_path: list[str] = [],
        vec_env_norm_fn: typing.Callable = None,
        device: str = "cuda",
    ):
        """Initialize custom reward."""
        super().__init__()
        self.device = device

        self.reward_model = reward_model_cls.load_from_checkpoint(
            reward_model_path,
            map_location=device
        )

        self.rewards = []
        self.expert_rewards = []
        self.counter = 0
        self.n_discrete_actions = 5 # hard-code for highway-env for now

    def _one_hot_encode_batch(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert nested batch of discrete actions to one-hot encoded format.
        
        Args:
            actions: Tensor of shape (1, batch_size) containing discrete action indices
            
        Returns:
            one_hot_actions: Tensor of shape (1, batch_size, n_discrete_actions)
        """
        outer_batch, inner_batch = actions.shape
        one_hot = torch.zeros((outer_batch, inner_batch, self.n_discrete_actions), device=self.device)
        actions = actions.long().unsqueeze(-1)  # Add dimension for scatter
        return one_hot.scatter_(2, actions, 1)
    
    def __call__(
        self,
        state: numpy.ndarray,
        actions: numpy.ndarray,
        next_state: numpy.ndarray,
        _done: numpy.ndarray,
    ) -> list:
        """Return reward given the current state."""
        
        state = torch.as_tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)        
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float).unsqueeze(0)

        if len(actions.shape) < 3:
            actions = self._one_hot_encode_batch(actions)
        
        with torch.no_grad():
            if self.reward_model.ensemble_count > 1:
                state = state.expand(self.reward_model.ensemble_count, *state.shape[1:])
                actions = actions.expand(self.reward_model.ensemble_count, *actions.shape[1:])
            
            rewards = self.reward_model(
                torch.as_tensor(state, device=self.device, dtype=torch.float),
                torch.as_tensor(actions, device=self.device, dtype=torch.float)
            )
            # Reshape rewards to always have 3 dimensions: (ensemble_count, batch_size, 1)
            rewards = rewards.view(self.reward_model.ensemble_count, -1, 1)
            # Take mean across ensemble dimension (dim=0)
            mean_rewards = torch.mean(rewards, dim=0).squeeze(-1)
            
            return mean_rewards.cpu().numpy()

def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument("--feedback-type", type=str, default="evaluative", help="Type of feedback")
    args = parser.parse_args()

    TrainingUtils.set_seeds(args.seed)
    _, model_id = TrainingUtils.get_model_ids(args)
    
    script_path = Path(__file__).parents[1].resolve()
    reward_model_path = (os.path.join(script_path, "reward_models_lul", f"{model_id}.ckpt") 
                        if args.feedback_type != "baseline" else None)

    TrainingUtils.setup_wandb_logging(f"RL_{model_id}", args)

    architecture_cls = (LightningCnnNetwork if "ALE/" in args.environment or "procgen" in args.environment 
                       else LightningNetwork)

    exp_manager = ExperimentManager(
        args,
        args.algorithm,
        args.environment,
        os.path.join("agents", f"RL_{model_id}"),
        tensorboard_log=f"runs/RL_{model_id}",
        seed=args.seed,
        log_interval=-1,
        reward_function=CustomReward(
            reward_model_cls=architecture_cls,
            reward_model_path=reward_model_path,
            device=TrainingUtils.get_device(),
        ) if args.feedback_type != "baseline" else None,
        use_wandb_callback=True,
    )

    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)


if __name__ == "__main__":
    main()
