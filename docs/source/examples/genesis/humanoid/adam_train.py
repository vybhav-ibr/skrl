import argparse
import argparse

import torch
import torch.nn as nn

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

from adam_env import AdamEnv

def get_cfgs():
    env_cfg = {
        "num_actions": 25,
        # joint/link names
        "default_dof_properties": {
            "hipPitch_Left" : [0.0, 120, 3.0],
            "hipPitch_Right" : [0.0, 120, 3.0],
            "waistRoll" : [0.0, 480, 0.24],
            "hipRoll_Left" : [0.0, 120, 3.0],
            "hipRoll_Right" : [0.0, 120, 3.0],
            "waistPitch" : [0.0, 480, 0.24],
            "hipYaw_Left" : [0.0, 120, 3.0],
            "hipYaw_Right" : [0.0, 120, 3.0],
            "waistYaw" : [0.0, 480, 0.24],
            "kneePitch_Left" : [0.0, 240, 6.0],
            "kneePitch_Right" : [0.0, 240, 6.0],
            "shoulderPitch_Left" : [0.0, 108, 2.4],
            "shoulderPitch_Right" : [0.0, 108, 2.4],
            "anklePitch_Left" : [0.0, 24, 0.24],
            "anklePitch_Right" : [0.0, 24, 0.24],
            "shoulderRoll_Left" : [0.0, 72, 1.2],
            "shoulderRoll_Right" : [0.0, 72, 1.2],
            "ankleRoll_Left" : [0.0, 24, 0.12],
            "ankleRoll_Right" : [0.0, 24, 0.12],
            "shoulderYaw_Left" : [0.0, 24, 0.6],
            "shoulderYaw_Right" : [0.0, 24, 0.6],
            "elbow_Left" : [0.0, 72, 1.2],
            "elbow_Right" : [0.0, 72, 1.2],
            "wristYaw_Left" : [0.0, 4.8, 0.12],
            "wristYaw_Right" : [0.0, 4.8, 0.12]
        },
        # termination
        "termination_criteria_roll": 25,  # degree
        "termination_criteria_pitch": 25,
        "termination_criteria_base_height": 0.35,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.95],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 50.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 87,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.75,
        "feet_height_target": 0.25,
        "jump_upward_velocity": 1.2,  
        "jump_reward_steps": 50,
        "locomotion_max_contact_force":10.0,
        "close_feet_threshold":0.15,
        "close_knees_threshold":0.15,
        # "reward_scales": {
        #     "tracking_lin_vel": 1.0,
        #     "tracking_ang_vel": 0.2,
        #     "lin_vel_z": -1.0,
        #     "base_height": -50.0,
        #     "action_rate": -0.005,
        #     "similar_to_default": -0.1,
        #     # "jump": 4.0,
        #     "jump_height_tracking": 0.5,
        #     "jump_height_achievement": 10,
        #     "jump_speed" : 1.0,
        #     "jump_landing": 0.08,
        # },
        "reward_scales":{
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "penalty_lin_vel_z": -2.0,
            "penalty_ang_vel_xy": -1.0,
            "penalty_ang_vel_xy_torso": -1.0,
            "penalty_feet_contact_forces": -0.01,
            "feet_air_time":1.0,
            "penalty_in_the_air":-100.0,
            "penalty_stumble":-10.0,
            "penalty_feet_ori":-1.0,
            "base_height":-10.0,
            "feet_heading_alignment":-0.3,
            "feet_ori":1.0,
            "penalty_feet_slippage":-1.0,
            "penalty_feet_height":-2.5,
            "penalty_close_feet_xy":-10.0,
            "penalty_close_knees_xy":-2.5
        }
    }
    command_cfg = {
        "num_commands": 5,  # [lin_vel_x, lin_vel_y, ang_vel, height, jump]
        "lin_vel_x_range": [-5.0, 5.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-2.5, 2.5],
        # "lin_vel_x_range": [0.0, 0.0],
        # "lin_vel_y_range": [0.0, 0.0],
        # "ang_vel_range": [0.0, 0.0],
        "height_range": [0.73, 0.78],
        "jump_range": [0.5, 1.5],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
parser.add_argument("-v", "--vis", action="store_true", default=False)
parser.add_argument("-B", "--num_envs", type=int, default=1024)
parser.add_argument("--max_timesteps", type=int, default=50000)
parser.add_argument("--memory_size", type=int, default=2500)
args = parser.parse_args()

gs.init(logging_level="warning")

env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()


env = AdamEnv(
    num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=args.vis
)

env=wrap_env(env,wrapper='genesis')
device=gs.device
set_seed()  # e.g. `set_seed(42)` for fixed seed

# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
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
memory = RandomMemory(memory_size=45, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 45  # memory_size
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
cfg["experiment"]["write_interval"] = 60
cfg["experiment"]["checkpoint_interval"] = 100
cfg["experiment"]["directory"] = "runs/torch/Go2"

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

# from skrl.utils.huggingface import download_model_from_huggingface

# # download the trained agent's checkpoint from Hugging Face Hub and load it
# path = "/home/vybhav/gs_gym_wrapper_reference/runs/torch/Go2/25-08-24_15-56-19-148207_PPO/checkpoints/agent_30000.pt"
# agent.load(path)

# # # start evaluation
# trainer.eval()
