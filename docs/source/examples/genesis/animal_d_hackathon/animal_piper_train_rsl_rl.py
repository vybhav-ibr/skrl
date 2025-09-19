import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from animal_piper_env import APWEnv


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 20,
        # joint/link names
        "default_dof_properties": {  # [rad]
            "LF_HAA": [0.0, 100, 5.0],
            "RF_HAA": [0.0, 100, 5.0],
            "LH_HAA": [0.0, 100, 5.0],
            "RH_HAA": [0.0, 100, 5.0],

            "LF_HFE": [0.8, 100, 5.0],
            "RF_HFE": [0.8, 100, 5.0],
            "LH_HFE": [-0.8, 100, 5.0],
            "RH_HFE": [-0.8, 100, 5.0],

            "LF_KFE": [-1.5, 100, 5.0],
            "RF_KFE": [-1.5, 100, 5.0],
            "LH_KFE": [1.5, 100, 5.0],
            "RH_KFE": [1.5, 100, 5.0],

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
        "base_init_pos": [0.0, 0.0, 0.55],
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
        "base_height_target": 0.55,
        "eef_pos_object_threshold":0.25,
        "reward_scales": {
            "survival":0.5,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="animal-rsl-rl-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = APWEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,show_viewer=args.vis
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
