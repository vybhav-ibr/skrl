import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import genesis as gs
from drone_env import HoverEnv
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # termination
        "termination_if_roll_greater_than": 180,
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_close_to_ceiling": 5.0,
        "termination_if_x_greater_than": 75.0,
        "termination_if_y_greater_than": 75.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.5],
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
        # 3+3+3+3+3+3+256+4 = 278
        "num_obs": 278,
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
            "smooth": -5e-4,
            "yaw": 0.01,
            "angular": -2e-4,
            "crash": -10.0,
        },
    }
    command_cfg = {
        "num_poses": 3,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


# ---------------------------------------------------------------------------
# Network — faithfully implements Figure 8B from He et al. 2025
# ---------------------------------------------------------------------------

class Shared(GaussianMixin, DeterministicMixin, Model):
    """
    Attention-based map encoding controller (He et al., ETH Zurich, 2025).

    Architecture adapted from Figure 8B (He et al., ETH 2025) for a scalar
    distance sensor returning (B, L, W) — one depth value per pixel, no xyz.

    Map path:
      map_scans (B, L, W) — scalar depth per pixel
        → unsqueeze → (B, 1, L, W)
        → CNN: two layers, kernel=5, zero-padding to preserve L×W
            layer1: Conv2d(1,  16, k=5, pad=2)
            layer2: Conv2d(16,  d, k=5, pad=2)
        → output: (B, d, L, W) → reshape to (B, L*W, d)
        → point-wise local features: (B, L*W, d)   ← Keys and Values for MHA
        (no xyz concat since sensor returns only scalar distance)

    Proprioception path:
      proprio (B, d_obs)
        → Linear(d_obs, d)
        → proprio embedding (B, d)                 ← Query for MHA (n=1)

    MHA:
      Q = proprio embedding  (B, 1, d)
      K = V = local features  (B, L*W, d)
      output: map encoding    (B, 1, d) → squeeze → (B, d)

    Policy MLP:
      concat(map_encoding, proprio) → (B, d + d_obs)
        → MLP → (B, 64)
        → mean_layer → actions
        → value_layer → V(s)

    proprio_dim (d_obs) = ang_vel(3) + lin_vel(3) + quat(4)
                        + rel_pos(3) + rel_pos_1(3) + rel_pos_2(3)
                        + last_actions(4) = 23
    d = 64  (MHA embed dim, same as paper)
    h = 4   (MHA heads; paper uses 16 but that requires d divisible by h,
             and with smaller d we use 4 — change to 16 if you increase d)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        # map scan dimensions
        L=16, W=16,          # spatial grid size
        d=64,                # MHA embed dim (paper uses 64)
        attn_heads=4,        # MHA heads (paper uses 16; scale with d)
        proprio_dim=22,      # proprioception dimension (see docstring)
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.L = L
        self.W = W
        self.d = d
        self.proprio_dim = proprio_dim

        # ------------------------------------------------------------------
        # CNN — input is scalar depth (1 channel), output d channels.
        # Preserves L×W spatial dims throughout (zero-pad, no downsampling).
        # Paper uses kernel=5, pad=2 for both layers.
        # No xyz concat since the sensor returns scalar distance only.
        # ------------------------------------------------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(1,   16, kernel_size=5, padding=2), nn.ELU(),
            nn.Conv2d(16,   d, kernel_size=5, padding=2), nn.ELU(),
        )
        # Output: (B, d, L, W) — reshaped directly to (B, L*W, d) point-wise features

        # ------------------------------------------------------------------
        # Proprioception → query embedding
        # ------------------------------------------------------------------
        self.proprio_linear = nn.Linear(proprio_dim, d)

        # ------------------------------------------------------------------
        # Multi-Head Attention
        # Q = proprio embedding  (n=1 query)
        # K = V = point-wise local features (L*W tokens)
        # ------------------------------------------------------------------
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=attn_heads,
            batch_first=True,
        )

        # ------------------------------------------------------------------
        # MLP trunk  (input = map_encoding(d) + proprio(proprio_dim))
        # ------------------------------------------------------------------
        mlp_input_dim = d + proprio_dim  # 64 + 23 = 87
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )

        # ------------------------------------------------------------------
        # Output heads
        # ------------------------------------------------------------------
        self.mean_layer  = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))
        self.value_layer = nn.Linear(64, 1)

        # Cache shared trunk output so policy and value share one forward pass.
        # Never detached — gradients from both policy and value losses flow back.
        self._shared_output = None

    # ------------------------------------------------------------------
    # Map encoding (CNN + concat xyz + MHA)
    # ------------------------------------------------------------------

    def _encode_map(self, map_scans: torch.Tensor, proprio_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_scans:   (B, L, W) — scalar depth per pixel
            proprio_emb: (B, d)    — proprioception embedding (the MHA query)
        Returns:
            map_encoding: (B, d)
        """
        B = map_scans.shape[0]

        # CNN on scalar depth values
        depth_chw = map_scans.unsqueeze(1)               # (B, 1, L, W)
        cnn_out   = self.cnn(depth_chw)                  # (B, d, L, W)

        # Reshape to token sequence — these become Keys and Values for MHA
        local_features = cnn_out.permute(0, 2, 3, 1).reshape(B, self.L * self.W, self.d)  # (B, L*W, d)

        # Multi-Head Attention: proprio query attends over spatial depth tokens
        query = proprio_emb.unsqueeze(1)                 # (B, 1, d)
        attn_out, _ = self.attn(
            query=query,
            key=local_features,
            value=local_features,
            need_weights=False,
        )
        return attn_out.squeeze(1)                       # (B, d)

    # ------------------------------------------------------------------
    # compute() — called by skrl for both policy and value roles
    # ------------------------------------------------------------------

    def compute(self, inputs, role):
        states = inputs["states"]
        space  = self.tensor_to_space(states, self.observation_space)

        # Extract observations
        map_scans = space["front_depth"]     # (B, 16, 16) — scalar depth per pixel

        proprio = torch.cat([
            space["base_ang_vel"],           # (B, 3)
            space["base_lin_vel"],           # (B, 3)
            space["base_euler"],              # (B, 3)
            space["base_rel_pos"],           # (B, 3)  — current waypoint
            space["base_rel_pos_1"],         # (B, 3)  — next waypoint
            space["base_rel_pos_2"],         # (B, 3)  — waypoint after next
            space["taken_actions"],          # (B, 4)
        ], dim=-1)                           # (B, 23)

        # Proprioception embedding — this is the MHA query
        proprio_emb = self.proprio_linear(proprio)   # (B, d)

        # Map encoding via CNN + MHA
        map_enc = self._encode_map(map_scans, proprio_emb)  # (B, d)

        # MLP trunk: concat map encoding with raw proprioception (matches paper Fig 8B)
        mlp_input  = torch.cat([map_enc, proprio], dim=-1)  # (B, d + proprio_dim)
        shared_out = self.mlp(mlp_input)                     # (B, 64)

        if role == "policy":
            # Cache for value head reuse — NOT detached so value-loss gradients
            # flow back through the full encoder+MLP stack.
            self._shared_output = shared_out
            return self.mean_layer(shared_out), self.log_std_parameter, {}

        elif role == "value":
            # Reuse the cached output from the preceding policy call.
            # Falls back to the freshly computed shared_out if called standalone
            # (e.g. during the very first critic warmup step).
            value_input = self._shared_output if self._shared_output is not None else shared_out
            self._shared_output = None  # clear after use
            return self.value_layer(value_input), {}

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", type=str, default="drone-goto")
parser.add_argument("-B", "--num_envs", type=int, default=5)
parser.add_argument("--vis", action="store_true")
parser.add_argument("--max_iterations", type=int, default=50000)
args = parser.parse_args()

gs.init(logging_level="info", precision="32")
env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

env = HoverEnv(
    num_envs=args.num_envs,
    env_cfg=env_cfg,
    obs_cfg=obs_cfg,
    reward_cfg=reward_cfg,
    command_cfg=command_cfg,
    show_viewer=args.vis,
)
env   = wrap_env(env, wrapper="genesis")
device = gs.device
set_seed()

models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"]  = models["policy"]  # shared instance — same encoder, same MLP trunk

memory = RandomMemory(
    memory_size=8,         # must match cfg["rollouts"]
    num_envs=env.num_envs,
    device=device,
)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"]          = 8
cfg["learning_epochs"]   = 5
cfg["mini_batches"]      = 4
cfg["discount_factor"]   = 0.99
cfg["lambda"]            = 0.95
cfg["learning_rate"]     = 1e-4
cfg["learning_rate_scheduler"]        = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
cfg["random_timesteps"]  = 0
cfg["learning_starts"]   = 0
cfg["grad_norm_clip"]    = 1.0
cfg["ratio_clip"]        = 0.2
cfg["value_clip"]        = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"]    = 0.0
cfg["value_loss_scale"]      = 1.0
cfg["kl_threshold"]          = 0
cfg["rewards_shaper"]        = None
cfg["time_limit_bootstrap"]  = False
cfg["experiment"]["write_interval"]      = 60
cfg["experiment"]["checkpoint_interval"] = 100
cfg["experiment"]["directory"] = "runs/torch/Genesis-Goto-Drone"

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

cfg_trainer = {"timesteps": args.max_iterations, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

trainer.train()

# Evaluation (uncomment to run)
# path = "runs/torch/Genesis-Goto-Drone/.../checkpoints/agent_XXXXX.pt"
# agent.load(path)
# trainer.eval()