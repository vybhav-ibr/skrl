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

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy import signal
import genesis as gs
from animal_piper_env import APEnv
import numpy as np
from ultralytics import YOLO
from torchvision.transforms import Normalize

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
            "LH_HFE": [1.0, 100, 5.0],
            "RH_HFE": [1.0, 100, 5.0],

            "LF_KFE": [-1.5, 100, 5.0],
            "RF_KFE": [-1.5, 100, 5.0],
            "LH_KFE": [-1.5, 100, 5.0],
            "RH_KFE": [-1.5, 100, 5.0],

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
        "termination_criteria_roll": 10,  # degree
        "termination_criteria_pitch": 10,
        "termination_criteria_base_height": 0.35,
        "contact_exclusion_pairs":
            [["LH_FOOT","plane"],
             ["RH_FOOT","plane"],
             ["LF_FOOT","plane"],
             ["RH_FOOT","plane"]],
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
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
        "base_height_target": 0.50,
        "eef_pos_object_threshold":0.5,
        "reward_scales": {
            "survival":100.0,
            "home":10.0,
            "pick_eef_pos_object":1.0,
            "pick_grasp_object":1.0,
            "place_ungrasp_object":1.0,
            "place_object_pos_basket":1.0,
            "goto":50,
            "high_joint_force":-0.005,
            "action_rate":-0.1,
            "base_height":-100.0,
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
parser.add_argument("-e", "--exp_name", type=str, default="animal-piper-small")
parser.add_argument("-B", "--num_envs", type=int, default=10)
parser.add_argument("--vis",action="store_true")
parser.add_argument("--max_iterations", type=int, default=101)
args = parser.parse_args()

gs.init(logging_level="warning",precision="32")
env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

env = APEnv(
    num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=args.vis
)
# env.step()
env=wrap_env(env,wrapper='genesis')
device=gs.device
set_seed()  # e.g. `set_seed(42)` for fixed seed
# print("single obs shape is:",env._env.obs_buf.shape)

import torch.nn as nn
from ultralytics import YOLO

class FrozenYOLO(nn.Module):
    """Safe wrapper around Ultralytics YOLO for inference-only use (prevents training behavior)."""

    def __init__(self, model_path):
        super().__init__()
        self.model = YOLO(model_path)
        self.model.fuse()
        self.model.eval()  # Ensure eval mode

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        # Allow usage with torch.no_grad by default
        with torch.no_grad():
            return self.model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with torch.no_grad():
            return self.model.predict(*args, **kwargs)

    def train(self, mode=True):
        # Prevent .train() from triggering Ultralytics training logic
        return self

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

class YOLOForegroundExtractor(nn.Module):
    """YOLOv8-based batched foreground extractor using instance segmentation"""

    def __init__(self, model_path="yolo11s-seg.pt"):
        super().__init__()

        self.model = FrozenYOLO(model_path)  # ← use the wrapper here
        self.model.eval()

        self.imagenet_mean = (0.485, 0.456, 0.406)
        self.imagenet_std = (0.229, 0.224, 0.225)
        self.normalizer = Normalize(mean=self.imagenet_mean, std=self.imagenet_std)

    def forward(self, rgb_images):
        B, _, H, W = rgb_images.shape
        device = rgb_images.device
        # print("rgb_images shape in fg extractor:",rgb_images.shape)
        # if rgb_images.max() > 1.5:
        #     rgb_images = rgb_images / 255.0
        rgb_images = self.normalizer(rgb_images)

        foreground_masks = []

        for i in range(B):
            img_np = rgb_images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
            result = self.model.predict(source=img_np, imgsz=max(H, W), verbose=False)[0]

            if result.masks is not None:
                mask_tensor = result.masks.data  # [N_instances, h, w]
                combined_mask = mask_tensor.any(dim=0).float()
            else:
                combined_mask = torch.zeros((result.orig_shape[0], result.orig_shape[1]), dtype=torch.float32)

            # Resize back to original image shape if necessary
            if combined_mask.shape != (H, W):
                combined_mask = TF.resize(combined_mask.unsqueeze(0), size=[H, W], antialias=True)[0]

            foreground_masks.append(combined_mask)

        return torch.stack(foreground_masks, dim=0).to(device)
    
class MaskedDepthFeatureExtractor(nn.Module):
    """Depth feature extractor that operates on foreground-masked depth images"""
    
    def __init__(self, output_dim=64, dropout_rate=0.2, mask_threshold=0.5):
        super().__init__()
        
        self.mask_threshold = mask_threshold
        
        # Depth processing backbone
        self.depth_backbone = nn.Sequential(
            # First block - larger receptive field for depth understanding
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),  # Lower dropout in early layers
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Fourth block - focus on high-level features
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Adaptive pooling for consistent size regardless of input resolution
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Feature compression and processing
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def apply_foreground_mask(self, depth_images, foreground_masks):
        """
        Apply foreground mask to depth images, setting background to zero
        
        Args:
            depth_images: [B, 1, H, W] depth images
            foreground_masks: [B, H, W] foreground probability masks
        Returns:
            masked_depth: [B, 1, H, W] foreground-masked depth images
        """
        # Threshold the probability masks to get binary masks
        binary_masks = (foreground_masks > self.mask_threshold).float()
        
        # Expand mask to match depth image dimensions
        binary_masks = binary_masks.unsqueeze(1)  # [B, 1, H, W]
        
        # Apply mask (background becomes 0, foreground preserves depth values)
        masked_depth = depth_images * binary_masks
        
        return masked_depth
        
    def forward(self, depth_images, foreground_masks):
        """
        Args:
            depth_images: [B, 1, H, W] depth images
            foreground_masks: [B, H, W] foreground probability masks
        Returns:
            features: [B, output_dim] masked depth features
        """
        # Apply foreground masking
        masked_depth = self.apply_foreground_mask(depth_images, foreground_masks)
        
        # Extract features from masked depth
        depth_features = self.depth_backbone(masked_depth)
        features = self.feature_head(depth_features)
        
        return features

GLOBAL_YOLO = YOLOForegroundExtractor(model_path="yolo11n-seg.pt")
class Shared(GaussianMixin, DeterministicMixin, Model):
    """Enhanced actor with RGB-guided depth processing"""
    
    def __init__(self, observation_space, action_space, device, dropout_rate=0.2,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        DeterministicMixin.__init__(self, clip_actions)
        # self.fg_extractor = YOLOForegroundExtractor(model_path="yolo11n-seg.pt")
        
        # Masked depth feature extractor (trainable)
        self.depth_extractor = MaskedDepthFeatureExtractor(
            output_dim=64, 
            dropout_rate=dropout_rate
        )
        
        # Calculate other state dimensions
        # ang_vel(3) + commands(3) + dof_vel(12) + dof_diff(12) + proj_gravity(3) = 33
        other_state_dim = 85
        total_input_dim = 64 + other_state_dim  # depth features + other states
        # total_input_dim=169
        # Action prediction network
        self.action_net = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            
            # nn.Linear(64, self.num_actions),
            # nn.Tanh()
        )
        self.mean_layer= nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.value_layer = nn.Linear(64, 1)
    
    def act(self,inputs,role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
        
    def compute(self, inputs, role):
        # global GLOBAL_YOLO
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)
        
        # Extract RGB images (front_cloud is assumed to be RGB)
        shapes_dict={}
        for key,value in space.items():
            shapes_dict[key]=value.shape
        # print(shapes_dict)
        rgb_images = space['gripper_img'].permute(0, 3, 1, 2).float()  # [B, 3, H, W]
        rgb_shape=rgb_images.shape
        # Get depth images - check if available in observation space
        depth_images = space['gripper_depth'].permute(0, 3, 1, 2).float()  # [B, 1, H, W]

        # Extract foreground segmentation masks
        # foreground_masks = rgb_images[:][:][:][0]  # [B, H, W]
        foreground_masks = GLOBAL_YOLO(rgb_images)  # [B, H, W]        
        # Extract features from foreground-masked depth
        depth_features = self.depth_extractor(depth_images, foreground_masks)  # [B, 64]
        
        # Prepare other state inputs
        other_states = torch.cat([
            space["commands"].view(states.shape[0], -1), 
            space["dof_diff"].view(states.shape[0], -1),
            space["dof_vel"].view(states.shape[0], -1),
            space["object_pos"].view(states.shape[0], -1),
            space["object_quat"].view(states.shape[0], -1),
            space["robot_base_pos"].view(states.shape[0], -1),
            space["robot_base_quat"].view(states.shape[0], -1),
            space["taken_actions"].view(states.shape[0], -1),
        ], dim=-1).float()
        # Assuming states.shape[0] is the batch size (n_envs or number of environments)

        # Combine masked depth features with other states
        # print("other_states size",other_states.shape)
        combined_input = torch.cat([depth_features, other_states], dim=-1)
        
        if role=="policy":
            # Predict actions
            # print("combined_input shape:",combined_input.shape)
            self._shared_output = self.action_net(combined_input)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role =="value":
            shared_output = self.action_net(combined_input) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}
            
memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  # 24 * 4096 / 24576
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
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
cfg["experiment"]["checkpoint_interval"] = 600
cfg["experiment"]["directory"] = "runs/torch/Isaac-Velocity-Anymal-C-v0"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 12000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
