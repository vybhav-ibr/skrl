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

import gymnasium
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
        "base_init_pos": [0.0, 0.0, 0.52],
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
parser.add_argument("-B", "--num_envs", type=int, default=5)
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
torch.autograd.set_detect_anomaly(True)
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
            nn.ReLU(),
            nn.Dropout2d(dropout_rate * 0.5),  # Lower dropout in early layers
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Fourth block - focus on high-level features
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Adaptive pooling for consistent size regardless of input resolution
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Feature compression and processing
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, output_dim),
            nn.ReLU()
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
        # print("depth_images_shape",depth_images.shape)
        # print("fg_mask_shape",foreground_masks.shape)
        # print("masked_depth_shape",masked_depth.shape)
        depth_features = self.depth_backbone(masked_depth)
        # print("depth_features_shape",depth_features.shape)
        features = self.feature_head(depth_features)
        
        return features

class DepthSequenceEncoder(nn.Module):
    def __init__(self, img_size=(64, 64), embedding_dim=128, rnn_hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # CNN encoder for a single depth image
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> [B, 16, 32, 32]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> [B, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> [B, 64, 8, 8]
            nn.ReLU(),
            nn.Flatten(),                                           # -> [B, 64*8*8]
            nn.Linear(64 * 8 * 8, embedding_dim),
            nn.ReLU()
        )

        # GRU to process sequence of fused embeddings
        self.gru = nn.GRU(input_size=embedding_dim * 2,
                          hidden_size=rnn_hidden_dim,
                          batch_first=True)

        # Internal hidden state (not registered as parameter)
        self.hidden_state = None

    def reset(self, batch_size=1, device=None):
        """
        Reset the GRU hidden state. Call at the start of an episode or sequence.
        """
        self.hidden_state = None  # Let GRU handle initialization automatically

    def step(self, front_depth: torch.Tensor, back_depth: torch.Tensor) -> torch.Tensor:
        """
        Process one timestep and return the current temporal embedding.
        
        Args:
            front_depth: [B, 64, 64] front depth image
            back_depth: [B, 64, 64] back depth image
        
        Returns:
            current_embedding: [B, rnn_hidden_dim] embedding for this timestep
        """
        B = front_depth.size(0)

        # # Add channel dim: [B, 1, 64, 64]
        # front = front_depth.unsqueeze(1)
        # back = back_depth.unsqueeze(1)

        # Encode images
        front_feat = self.depth_cnn(front_depth)  # [B, embedding_dim]
        back_feat = self.depth_cnn(back_depth)    # [B, embedding_dim]

        # Fuse views: [B, embedding_dim * 2]
        fused = torch.cat([front_feat, back_feat], dim=-1).unsqueeze(1)  # [B, 1, fused_dim]
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()
        output, self.hidden_state = self.gru(fused, self.hidden_state)  # output: [B, 1, rnn_hidden_dim]
        return output.squeeze(1)  # [B, rnn_hidden_dim]
    
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
        self.depth_mask_extractor = MaskedDepthFeatureExtractor(
            output_dim=64, 
            dropout_rate=dropout_rate
        )
        self.depth_encoder=DepthSequenceEncoder()
        
        # Calculate other state dimensions
        # ang_vel(3) + commands(3) + dof_vel(12) + dof_diff(12) + proj_gravity(3) = 33
        other_state_dim = 85
        total_input_dim = 64 + other_state_dim  +256# depth features + other states
        # total_input_dim=169
        # Action prediction network
        
        self.action_net = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            
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
        # print("%"*10,"\n",shapes_dict,"\n","%"*10)
        rgb_images = space['gripper_img'].permute(0, 3, 1, 2).float()  # [B, 3, H, W]
        # rgb_shape=rgb_images.shape
        # Get depth images - check if available in observation space
        depth_images = space['gripper_depth'].permute(0, 3, 1, 2).float()  # [B, 1, H, W]

        # Extract foreground segmentation masks
        # foreground_masks = rgb_images[:][:][:][0]  # [B, H, W]
        foreground_masks = GLOBAL_YOLO(rgb_images)  # [B, H, W]        
        # Extract features from foreground-masked depth
        arm_depth_features = self.depth_mask_extractor(depth_images, foreground_masks)  # [B, 64]
        
        # print("front_depth_shape_is",space["front_depth"].shape)
        # exit(0)
        depth_features=self.depth_encoder.step(space["front_depth"].permute(0, 3, 1, 2).float(),space["back_depth"].permute(0, 3, 1, 2).float())
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
        combined_input = torch.cat([arm_depth_features,depth_features, other_states], dim=-1)
        
        if role=="policy":
            # Predict actions
            self._shared_output = self.action_net(combined_input)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role =="value":
            shared_output = self.action_net(combined_input) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}
# print("num_envs",env.num_envs)
# exit(0) 
def extract_simplified_state(flattened_obs: torch.Tensor, obs_space=env.observation_space) -> torch.Tensor:
    num_envs = flattened_obs.shape[0]
    start = 0
    chunks = []

    # Keys to exclude (image/depth)
    exclude_keys = {"back_depth", "front_depth", "gripper_depth", "gripper_img"}

    for key, value in obs_space.items():
        # Flatten per-environment feature size (exclude batch dim if it exists)
        shape = value.shape
        if shape[0] == 1:
            flat_dim = int(torch.tensor(shape[1:]).prod().item())  # remove batch dim
        else:
            flat_dim = int(torch.tensor(shape).prod().item())  # no batch dim present

        end = start + flat_dim

        if key not in exclude_keys:
            chunks.append(flattened_obs[:, start:end])

        start = end

    return torch.cat(chunks, dim=-1)   

def simplify_gym_space(original_space, exclude_keys: list):
    """
    Return a new Dict space excluding the specified keys.

    Args:
        original_space (spaces.Dict): The original Gym Dict space.
        exclude_keys (list): List of keys to exclude.

    Returns:
        spaces.Dict: A new Dict space with the specified keys removed.
    """
    return gymnasium.Space({
        key: space for key, space in original_space.spaces.items()
        if key not in exclude_keys
    })

def expand_obs_tensor(small_obs: torch.Tensor) -> torch.Tensor:
    """
    Expands a [B, 85] observation tensor to [B, 1081429] by adding dummy image and depth data.

    Args:
        small_obs (torch.Tensor): Input tensor of shape [B, 85], containing only non-image features.

    Returns:
        torch.Tensor: Expanded tensor of shape [B, 1081429] with dummy image/depth values added.
    """
    B = small_obs.size(0)

    # Sizes of each non-image feature (adds up to 85)
    non_image_feature_sizes = {
        "commands": 11,
        "dof_diff": 20,
        "dof_vel": 20,
        "object_pos": 3,
        "object_quat": 4,
        "robot_base_pos": 3,
        "robot_base_quat": 4,
        "taken_actions": 20,
    }

    # Image/depth fields with their flattened sizes
    image_fields_sizes = {
        "back_depth": 64 * 64 * 1,
        "front_depth": 64 * 64 * 1,
        "gripper_depth": 512 * 512 * 1,
        "gripper_img": 512 * 512 * 3,
    }

    # Split the input tensor into components
    components = []
    start = 0
    for size in non_image_feature_sizes.values():
        end = start + size
        components.append(small_obs[:, start:end])
        start = end

    # Create dummy tensors for image/depth inputs
    image_components = [
        torch.zeros(B, size, dtype=small_obs.dtype, device=small_obs.device)
        for size in image_fields_sizes.values()
    ]

    # Order of final concatenation (matches obs_space)
    # back_depth, commands, dof_diff, dof_vel, front_depth, gripper_depth, gripper_img,
    # object_pos, object_quat, robot_base_pos, robot_base_quat, taken_actions
    full_obs = torch.cat([
        image_components[0],  # back_depth
        components[0],        # commands
        components[1],        # dof_diff
        components[2],        # dof_vel
        image_components[1],  # front_depth
        image_components[2],  # gripper_depth
        image_components[3],  # gripper_img
        components[3],        # object_pos
        components[4],        # object_quat
        components[5],        # robot_base_pos
        components[6],        # robot_base_quat
        components[7],        # taken_actions
    ], dim=1)

    return full_obs

class MyRandomMemory(RandomMemory):
    def __init__(self, memory_size, num_envs = 1, device = None, export = False, export_format = "pt", export_directory = "", replacement=True):
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory, replacement)    
    
    def create_tensor(
        self,
        name: str,
        size: Union[int, Tuple[int], gymnasium.Space],
        dtype: Optional[torch.dtype] = None,
        keep_dimensions: bool = False,
    ) -> bool:
        """Create a new internal tensor in memory

        The tensor will have a 3-components shape (memory size, number of environments, size).
        The internal representation will use _tensor_<name> as the name of the class property

        :param name: Tensor name (the name has to follow the python PEP 8 style)
        :type name: str
        :param size: Number of elements in the last dimension (effective data size).
                     The product of the elements will be computed for sequences or gymnasium spaces
        :type size: int, tuple or list of integers or gymnasium space
        :param dtype: Data type (torch.dtype) (default: ``None``).
                      If None, the global default torch data type will be used
        :type dtype: torch.dtype or None, optional
        :param keep_dimensions: Whether or not to keep the dimensions defined through the size parameter (default: ``False``)
        :type keep_dimensions: bool, optional

        :raises ValueError: The tensor name exists already but the size or dtype are different

        :return: True if the tensor was created, otherwise False
        :rtype: bool
        """
        # compute data size
        if not keep_dimensions:
            size = compute_space_size(size, occupied_size=True)
        # check dtype and size if the tensor exists
        if name in self.tensors:
            tensor = self.tensors[name]
            if tensor.size(-1) != size:
                raise ValueError(f"Size of tensor {name} ({size}) doesn't match the existing one ({tensor.size(-1)})")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(f"Dtype of tensor {name} ({dtype}) doesn't match the existing one ({tensor.dtype})")
            return False
        # define tensor shape
        if name in ["states","next_states"]:
            size=85
        tensor_shape = (
            (self.memory_size, self.num_envs, *size) if keep_dimensions else (self.memory_size, self.num_envs, size)
        )
        view_shape = (-1, *size) if keep_dimensions else (-1, size)
        # create tensor (_tensor_<name>) and add it to the internal storage
        setattr(self, f"_tensor_{name}", torch.zeros(tensor_shape, device=self.device, dtype=dtype))
        # update internal variables
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].view(*view_shape)
        self.tensors_keep_dimensions[name] = keep_dimensions
        # fill the tensors (float tensors) with NaN
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.fill_(float("nan"))
        return True
    
    def add_samples(self, **tensors):
        """Record samples in memory

        Samples should be a tensor with 2-components shape (number of environments, data size).
        All tensors must be of the same shape

        According to the number of environments, the following classification is made:

        - one environment:
          Store a single sample (tensors with one dimension) and increment the environment index (second index) by one

        - number of environments less than num_envs:
          Store the samples and increment the environment index (second index) by the number of the environments

        - number of environments equals num_envs:
          Store the samples and increment the memory index (first index) by one

        :param tensors: Sampled data as key-value arguments where the keys are the names of the tensors to be modified.
                        Non-existing tensors will be skipped
        :type tensors: dict

        :raises ValueError: No tensors were provided or the tensors have incompatible shapes
        """
        if not tensors:
            raise ValueError(
                "No samples to be recorded in memory. Pass samples as key-value arguments (where key is the tensor name)"
            )

        # dimensions and shapes of the tensors (assume all tensors have the dimensions of the first tensor)
        # print(tensors.keys())
        # for key, value in tensors.items():
        #     print(key,value.shape)
        tmp = tensors.get("states", tensors[next(iter(tensors))])  # ask for states first
        dim, shape = tmp.ndim, tmp.shape

        # multi environment (number of environments equals num_envs)
        if dim > 1 and shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    if name in ["states","next_states"]:
                        tensor=extract_simplified_state(tensor)
                        tensor_shape=self.tensors[name][self.memory_index].shape
                        self.tensors[name][self.memory_index].copy_(tensor)
                    else:
                        self.tensors[name][self.memory_index].copy_(tensor)
            self.memory_index += 1
        # multi environment (number of environments less than num_envs)
        elif dim > 1 and shape[0] < self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index : self.env_index + tensor.shape[0]].copy_(
                        tensor
                    )
            self.env_index += tensor.shape[0]
        # single environment - multi sample (number of environments greater than num_envs (num_envs = 1))
        elif dim > 1 and self.num_envs == 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    num_samples = min(shape[0], self.memory_size - self.memory_index)
                    remaining_samples = shape[0] - num_samples
                    # copy the first n samples
                    self.tensors[name][self.memory_index : self.memory_index + num_samples].copy_(
                        tensor[:num_samples].unsqueeze(dim=1)
                    )
                    self.memory_index += num_samples
                    # storage remaining samples
                    if remaining_samples > 0:
                        self.tensors[name][:remaining_samples].copy_(tensor[num_samples:].unsqueeze(dim=1))
                        self.memory_index = remaining_samples
        # single environment
        elif dim == 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index].copy_(tensor)
            self.env_index += 1
        else:
            raise ValueError(f"Expected shape (number of environments = {self.num_envs}, data size), got {shape}")

        # update indexes and flags
        if self.env_index >= self.num_envs:
            self.env_index = 0
            self.memory_index += 1
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True

            # export tensors to file
            if self.export:
                self.save(directory=self.export_directory, format=self.export_format)
    
    def sample_all(
        self, names: Tuple[str], mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample all data from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from memory.
                 The sampled tensors will have the following shape: (memory size * number of environments, data size)
        :rtype: list of torch.Tensor list
        """
        # sequential order
        if sequence_length > 1:
            if mini_batches > 1:
                batches = np.array_split(self.all_sequence_indexes, mini_batches)
                return [[self.tensors_view[name][batch] for name in names] for batch in batches]
            return [[self.tensors_view[name][self.all_sequence_indexes] for name in names]]

        # default order
        if mini_batches > 1:
            batch_size = (self.memory_size * self.num_envs) // mini_batches
            batches = [(batch_size * i, batch_size * (i + 1)) for i in range(mini_batches)]
            return_list=[]
            for batch in batches:
                single_batch=[]
                for name in names:
                    if name in ["states","next_states"]:
                        # print("batch_shape",self.tensors_view[name][batch[0]:batch[1]].shape)
                        # expanded_tensor=torch.zeros((self.tensors_view[name][batch[0]:batch[1]].shape[0],1081429))
                        # print("exp_tens",expanded_tensor.shape)
                        single_batch.append(expand_obs_tensor(self.tensors_view[name][batch[0]:batch[1]]))
                    else:
                        single_batch.append(self.tensors_view[name][batch[0]:batch[1]])
                return_list.append(single_batch)
            return return_list
            # return [[self.tensors_view[name][batch[0] : batch[1]] for name in names] for batch in batches]
        return [[self.tensors_view[name] for name in names]]

           
memory = MyRandomMemory(memory_size=8, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 8  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 8  # 24 * 4096 / 24576
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
