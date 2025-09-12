import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import genesis as gs
import datetime

from panda_env import PandaEnv

# Define the models (deterministic models) for the TD3 agent using mixins
# and programming with two approaches (torch functional and torch.nn.Sequential class).
# - Actor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy

def get_cfgs():
    env_cfg = {
        "num_obs": 14,
        "num_actions": 6,
        "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "episode_length_s": 3.0,
        "ctrl_dt": 0.01,
        "box_size": [0.04, 0.04, 0.06],
        "box_collision": False,
        "box_fixed": True,
        "visualize_camera": False,
    }
    reward_scales = {
        "keypoints": 1.0,
        "table_contact": -1.0,  # Negative scale for penalty
    }
    # panda robot specific
    robot_cfg = {
        "ee_link_name": "hand",
        "gripper_link_names": ["left_finger", "right_finger"],
        "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "default_gripper_dof": [0.04, 0.04],
        "ik_method": "dls_ik",
    }
    return env_cfg, reward_scales, robot_cfg

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", type=str, default="grasp")
parser.add_argument("-v", "--vis", action="store_true", default=False)
parser.add_argument("-B", "--num_envs", type=int, default=2048)
parser.add_argument("--max_timesteps", type=int, default=500)
parser.add_argument("--memory_size", type=int, default=250)
args = parser.parse_args()

gs.init(logging_level="warning", precision="32")

env_cfg, reward_scales, robot_cfg = get_cfgs()

if args.vis:
    env_cfg["visualize_target"] = True

env = PandaEnv(
    num_envs=args.num_envs,
    env_cfg=env_cfg,
    reward_cfg=reward_scales,
    robot_cfg=robot_cfg,
    show_viewer=args.vis,
)
env=wrap_env(env,wrapper='genesis')
# print("num_envs is:",env.num_envs)
# print("obs_space is",env.observation_space)
# print("act_space is",env.action_space)
device=gs.device
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # print("num_obs:",self.num_observations)
        # print("num_act:",self.num_actions)
        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=14, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 14  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  # 96 * 4096 / 98304
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "min_lr": 1e-5}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = True
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 336
cfg["experiment"]["checkpoint_interval"] = 100
cfg["experiment"]["directory"] = "runs/torch/Isaac-Lift-Franka-v0"

# print(env.observation_space)
agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": args.max_timesteps, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()