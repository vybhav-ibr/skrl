import argparse
import torch
import genesis as gs
from drone_env import HoverEnv

def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # termination
        "termination_if_roll_greater_than": 180,  # degree
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 3.0,
        "termination_if_y_greater_than": 3.0,
        "termination_if_z_greater_than": 2.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "num_gates": 9,
        "grid_size": 10,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 791,  # Updated: 16*16*3 (768) + 17 other obs
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-4,
            "yaw": 0.01,
            "angular": -2e-4,
            "crash": -10.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [1.0, 1.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", type=str, default="drone-visual-check")
parser.add_argument("-B", "--num_envs", type=int, default=1)
parser.add_argument("--vis", action="store_true", help="Visualize the environment")
parser.add_argument("--max_iterations", type=int, default=50000)
args = parser.parse_args()

gs.init(logging_level="debug", precision="32")
env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

# Force visualization if requested, although HoverEnv uses show_viewer arg
env = HoverEnv(
    num_envs=args.num_envs,
    env_cfg=env_cfg,
    obs_cfg=obs_cfg,
    reward_cfg=reward_cfg,
    command_cfg=command_cfg,
    show_viewer=args.vis
)

print(f"Starting simulation with {args.num_envs} environments...")
for i in range(args.max_iterations):
    # Sample random actions: [num_envs, num_actions]
    # num_actions is 4 for the drone (thrust + 3 torques usually, or similar)
    action = torch.randn(args.num_envs, env_cfg["num_actions"], device=gs.device)
    
    # Step the environment
    obs, rew, reset, extras = env.step(action)
    
    if i % 100 == 0:
        print(f"Step {i}: Reward mean = {rew.mean().item():.4f}")
