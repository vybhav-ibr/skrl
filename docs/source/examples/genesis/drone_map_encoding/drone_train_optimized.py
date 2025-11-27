import argparse
# Import the skrl components to build the RL system
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import gymnasium

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy import signal
import genesis as gs
from drone_env import HoverEnv
from my_random_memory import MyRandomMemory
import numpy as np

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
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 785,  # Updated: 16*16*3 (768) + 17 other obs
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
parser.add_argument("-e", "--exp_name", type=str, default="drone-goto")
parser.add_argument("-B", "--num_envs", type=int, default=5)
parser.add_argument("--vis",action="store_true")
parser.add_argument("--max_iterations", type=int, default=50000)
parser.add_argument("--low_vram", action="store_true", help="Use low VRAM configuration")
args = parser.parse_args()

gs.init(logging_level="info",precision="32")
env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

env = HoverEnv(
    num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=args.vis
)
# env.step()
env=wrap_env(env,wrapper='genesis')
device=gs.device
set_seed()  # e.g. `set_seed(42)` for fixed seed
# print("single obs shape is:",env._env.obs_buf.shape)

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2, reduction="sum",
                 map_shape=(16, 16, 3), proprio_dim=13,  # Updated to 16x16
                 map_feat_dim=32, attn_heads=4, low_vram=False):

        Model.__init__(self,observation_space, action_space, device)

        # Init mixins
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # Adjust sizes for low VRAM mode
        if low_vram:
            map_feat_dim = max(16, map_feat_dim // 2)
            attn_heads = max(2, attn_heads // 2)
            cnn_channels = [8, 16]
            mlp_sizes = [128, 64, 32]
        else:
            cnn_channels = [16, 32]
            mlp_sizes = [256, 128, 64]

        self.L, self.W, self.C = map_shape
        self.map_feat_dim = map_feat_dim
        self.flat_mlp_input_dim = map_feat_dim + proprio_dim
        self.low_vram = low_vram

        # ------------------- CNN Encoder for Map -------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(self.C, cnn_channels[0], kernel_size=3, padding=1), nn.ELU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1), nn.ELU(),
            nn.Conv2d(cnn_channels[1], map_feat_dim, kernel_size=3, padding=1), nn.ELU()
        )

        # Separate height extraction layer
        self.height_layer = nn.Conv2d(self.C, 1, kernel_size=1)

        # ------------------- Proprioception Processing -------------------
        self.proprio_linear = nn.Linear(proprio_dim, map_feat_dim)

        # ------------------- Attention -------------------
        self.map_enc_linear = nn.Linear(map_feat_dim + 1, map_feat_dim)  # +1 from height
        self.attn = nn.MultiheadAttention(embed_dim=map_feat_dim, num_heads=attn_heads, batch_first=True)

        # ------------------- MLP Trunk -------------------
        layers = []
        in_dim = self.flat_mlp_input_dim
        for out_dim in mlp_sizes:
            layers.extend([nn.Linear(in_dim, out_dim), nn.ELU()])
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # ------------------- Output Heads -------------------
        self.mean_layer = nn.Linear(mlp_sizes[-1], self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))
        self.value_layer = nn.Linear(mlp_sizes[-1], 1)

        self._shared_output = None  # Cache for shared encoder

    def _forward_cnn_impl(self, map_scans):
        """
        Internal CNN implementation for gradient checkpointing.
        """
        B = map_scans.shape[0]
        x = map_scans.permute(0, 3, 1, 2)  # (B, C, L, W)

        cnn_feats = self.cnn(x)  # (B, map_feat_dim, L, W)
        height = self.height_layer(x)  # (B, 1, L, W)

        combined = torch.cat([cnn_feats, height], dim=1)  # (B, map_feat_dim + 1, L, W)
        flat_feats = combined.view(B, combined.shape[1], -1).permute(0, 2, 1)  # (B, L*W, map_feat_dim + 1)
        return flat_feats

    def forward_cnn(self, map_scans):
        """
        Encode map scans using CNN and height.
        Uses gradient checkpointing in low VRAM mode.
        Input: (B, L, W, C)
        Output: (B, L*W, map_feat_dim + 1)
        """
        if self.low_vram and self.training:
            # Use gradient checkpointing to trade compute for memory
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_cnn_impl, map_scans, use_reentrant=False)
        else:
            return self._forward_cnn_impl(map_scans)

    def compute(self, inputs, role):
        """
        Compute either policy or value output.
        Args:
            inputs["states"]: dict of obs tensors
            role: "policy" or "value"
        """
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)

        # ------------------- Extract Observations -------------------
        map_scans = space["front_depth"]  # (B, L, W, C)

        proprio = torch.cat([
            space["base_ang_vel"],    # (B, 3)
            space["base_lin_vel"],    # (B, 3)
            space["base_quat"],       # (B, 4)
            space["base_rel_pos"],        # (B, 3)
        ], dim=-1)  # → (B, proprio_dim)

        # ------------------- Encode Map -------------------
        map_features = self.forward_cnn(map_scans)  # (B, L*W, map_feat_dim + 1)
        map_encoded = self.map_enc_linear(map_features)  # (B, L*W, map_feat_dim)

        # ------------------- Encode Proprio -------------------
        proprio_encoded = self.proprio_linear(proprio)  # (B, map_feat_dim)

        # ------------------- Multi-Head Attention -------------------
        # Query = proprio (context vector)
        # Key, Value = map (spatial features)
        # Set need_weights=False to prevent allocation of attention weights entirely
        attn_out, _ = self.attn(
            query=proprio_encoded.unsqueeze(1),  # (B, 1, d)
            key=map_encoded,                     # (B, L*W, d)
            value=map_encoded,                   # (B, L*W, d)
            need_weights=False  # ✅ Don't allocate attention weights
        )
        attn_out = attn_out.squeeze(1)  # (B, d)

        # ------------------- MLP Trunk -------------------
        # Concatenate proprioception and attended map encoding
        mlp_input = torch.cat([attn_out, proprio], dim=-1)  # (B, map_feat_dim + proprio_dim)
        shared_out = self.mlp(mlp_input)  # (B, final_mlp_size)

        # ------------------- Heads -------------------
        if role == "policy":
            # Use no_grad for cache to ensure no gradient tracking
            with torch.no_grad():
                self._shared_output = shared_out.clone()
            return self.mean_layer(shared_out), self.log_std_parameter, {}

        elif role == "value":
            value_input = self._shared_output if self._shared_output is not None else shared_out
            self._shared_output = None  # clear cache
            return self.value_layer(value_input), {}

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)  

memory = MyRandomMemory(
    memory_size=16,  # Reduced from 1000 to save VRAM (~67KB per env)
    num_envs=env.num_envs,
    obs_space=env.observation_space,
    exclude_keys=["front_depth"],
    dummy_fillers={"front_depth": 16*16*3},  # Updated to 16x16x3
)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device, low_vram=args.low_vram)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 8  # memory_size
cfg["learning_epochs"] = 3 if args.low_vram else 5  # Reduced for low VRAM
cfg["mini_batches"] = 8 if args.low_vram else 4  # Increased for low VRAM
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 0.5 if args.low_vram else 1.0  # Reduced for stability
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 60
cfg["experiment"]["checkpoint_interval"] = 100
cfg["experiment"]["directory"] = "runs/torch/Genesis-Goto-Anymal-C-v0"


agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": args.max_iterations, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# Print memory stats before training
if torch.cuda.is_available():
    print(f"\n{'='*60}")
    print(f"VRAM Usage Before Training:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.3f} GB")
    print(f"  Low VRAM Mode: {args.low_vram}")
    print(f"  Num Envs: {args.num_envs}")
    print(f"{'='*60}\n")

# # start training
trainer.train()

# # download the trained agent's checkpoint from Hugging Face Hub and load it
path = "/home/vybhav/gs_gym_wrapper_reference/skrl/runs/torch/Genesis-Goto-Anymal-C-v0/25-09-15_14-58-17-605290_PPO/checkpoints/agent_5100.pt"
# agent.load(path)

# # # # start evaluation
# trainer.eval()
