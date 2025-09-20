import argparse
# Import the skrl components to build the RL system
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import gymnasium

# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
from scipy import signal
import genesis as gs
from animal_piper_env import APWEnv
import numpy as np
# from ultralytics import YOLO
# from torchvision.transforms import Normalize

from skrl.utils.spaces.torch import compute_space_size
from typing import List, Optional, Tuple, Union

def get_cfgs():
    env_cfg = {
        "num_actions": 20,
        # joint/link names
        "default_dof_properties": {  # [rad]
            "LF_HAA": [0.0, 100, 5.0],
            "RF_HAA": [0.0, 100, 5.0],
            "LH_HAA": [0.0, 100, 5.0],
            "RH_HAA": [0.0, 100, 5.0],

            "LF_HFE": [0.4, 100, 5.0],
            "RF_HFE": [0.4, 100, 5.0],
            "LH_HFE": [-0.4, 100, 5.0],
            "RH_HFE": [-0.4, 100, 5.0],

            "LF_KFE": [-0.75, 100, 5.0],
            "RF_KFE": [-0.75, 100, 5.0],
            "LH_KFE": [0.75, 100, 5.0],
            "RH_KFE": [0.75, 100, 5.0],

            "joint1": [0.0,80,5.0],
            "joint2": [0.0,80,5.0],
            "joint3": [0.0,80,5.0],
            "joint4": [0.0,40,5.0],
            "joint5": [0.0,10,1.5],
            "joint6": [0.0,10,1.5],
            "joint7": [0.0,40.0,5],
            "joint8": [0.0,40.0,5],
            
        },
        "arm_pick_pos": {
            "joint1": [0.0],
            "joint2": [0.87],
            "joint3": [-0.87],
            "joint4": [0.0],
            "joint5": [0.0],
            "joint6": [0.0],
            "joint7": [0.0],
            "joint8": [0.0],
        },
        "links_to_keep":["LF_FOOT","RF_FOOT","LH_FOOT","RH_FOOT",
                         "depth_camera_front_lower_camera","depth_camera_rear_lower_camera"],
        # termination
        "termination_criteria_roll": 25,  # degree
        "termination_criteria_pitch": 25,
        "termination_criteria_base_height": 0.35,
        "contact_exclusion_pairs": #link,entity
            [["LH_FOOT","plane"],
             ["RH_FOOT","plane"],
             ["LF_FOOT","plane"],
             ["RH_FOOT","plane"]],
        # base pose
        "base_init_pos": [0.0, 0.0, 0.67],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 75.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 105,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            
            "eef_dof_pos":1.0,
            "eef_dof_quat":1.0,
            "grasp_status":1.0,
            "basket_contact":1.0,
        },
    }
    reward_cfg = {
        "base_height_target": 0.53,
        "eef_pos_object_threshold":0.25,
        "reward_scales": {
            # "survival":0.5,
            # "home":10.0,
            # "pick_eef_pos_object":1.0,
            # "pick_grasp_object":1.0,
            # "place_ungrasp_object":1.0,
            # "place_object_pos_basket":1.0,
            "goto":10.0,
            "high_joint_force":-0.005,
            "action_rate":-0.01,
            "base_height":-10.0,
            "goal_proximity":1.0
        }
    }
    command_cfg = {
        "num_commands": 11,
        "home":[True,False],
        
        "pick":[True,False],
        # "deliver":[True,False],
        
        "goto":[True,False],
        "goto_pos": [[0,5],[-0.5,0.5],[0.65,0.35]],
        "goto_quat": [[-1,1],[-1,1],[-1,1],[-1,1]],
        
        "place":[True,False],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", type=str, default="animal-piper-small-goto")
parser.add_argument("-B", "--num_envs", type=int, default=5)
parser.add_argument("--vis",action="store_true")
parser.add_argument("--max_iterations", type=int, default=50000)
args = parser.parse_args()

gs.init(logging_level="debug",precision="32")
env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

env = APWEnv(
    num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=args.vis
)


for i in range(10000):
    action=torch.randn(args.num_envs, 20)
    env.step(action)