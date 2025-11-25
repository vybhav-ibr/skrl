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
from ackerman_env import AKEnv
from my_random_memory import MyRandomMemory
import numpy as np


def get_cfgs():
    env_cfg = {
        "num_actions": 6,
        # joint/link names
        "default_dof_properties": {  # [rad]
            "rear_left_wheel_joint": [0.0,100.0,10.0],
            "rear_right_wheel_joint": [0.0,100.0,10.0],
            "left_wheel_steering_joint": [0.0,100.0,10.0],
            "right_wheel_steering_joint": [0.0,100.0,10.0],
            "front_left_wheel_joint": [0.0,100.0,10.0],
            "front_right_wheel_joint": [0.0,100.0,10.0],
        },
        "links_to_keep":["camera_link"],

        # base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 125.0,
        # "resampling_time_s": 4.0,
        "action_scale_pos": 0.125,
        "action_scale_vel": 5.0,
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
        # "base_height_target": 0.45,
        # "eef_pos_object_threshold":0.25,
        "reward_scales": {
            "goto":2.5,
        }
    }
    command_cfg = {
        "num_commands": 3,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", type=str, default="animal-piper-small-goto")
parser.add_argument("-B", "--num_envs", type=int, default=5)
parser.add_argument("--vis",action="store_true")
parser.add_argument("--max_iterations", type=int, default=50000)
args = parser.parse_args()

gs.init(logging_level="info",precision="32")
env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

env = AKEnv(
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
                 map_shape=(64, 64, 3), proprio_dim=15,
                 map_feat_dim=32, attn_heads=4):

        Model.__init__(self,observation_space, action_space, device)

        # Init mixins
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.L, self.W, self.C = map_shape
        self.map_feat_dim = map_feat_dim
        self.flat_mlp_input_dim = map_feat_dim + proprio_dim

        # ------------------- CNN Encoder for Map -------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(self.C, 16, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv2d(32, map_feat_dim, kernel_size=3, padding=1), nn.ELU()
        )

        # Separate height extraction layer
        self.height_layer = nn.Conv2d(self.C, 1, kernel_size=1)

        # ------------------- Proprioception Processing -------------------
        self.proprio_linear = nn.Linear(proprio_dim, map_feat_dim)

        # ------------------- Attention -------------------
        self.map_enc_linear = nn.Linear(map_feat_dim + 1, map_feat_dim)  # +1 from height
        self.attn = nn.MultiheadAttention(embed_dim=map_feat_dim, num_heads=attn_heads, batch_first=True)

        # ------------------- MLP Trunk -------------------
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_mlp_input_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU()
        )

        # ------------------- Output Heads -------------------
        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))
        self.value_layer = nn.Linear(64, 1)

        self._shared_output = None  # Cache for shared encoder

    def forward_cnn(self, map_scans):
        """
        Encode map scans using CNN and height.
        Input: (B, L, W, C)
        Output: (B, L*W, map_feat_dim + 1)
        """
        B = map_scans.shape[0]
        x = map_scans.permute(0, 3, 1, 2)  # (B, C, L, W)

        cnn_feats = self.cnn(x)  # (B, map_feat_dim, L, W)
        height = self.height_layer(x)  # (B, 1, L, W)

        combined = torch.cat([cnn_feats, height], dim=1)  # (B, map_feat_dim + 1, L, W)
        flat_feats = combined.view(B, combined.shape[1], -1).permute(0, 2, 1)  # (B, L*W, map_feat_dim + 1)
        return flat_feats

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
            space["base_pos"],        # (B, 3)
            space["dof_pos"],         # (B, N)
            space["dof_vel"],         # (B, N)
        ], dim=-1)  # → (B, proprio_dim)

        # ------------------- Encode Map -------------------
        map_features = self.forward_cnn(map_scans)  # (B, L*W, map_feat_dim + 1)
        map_encoded = self.map_enc_linear(map_features)  # (B, L*W, map_feat_dim)

        # ------------------- Encode Proprio -------------------
        proprio_encoded = self.proprio_linear(proprio)  # (B, map_feat_dim)

        # ------------------- Multi-Head Attention -------------------
        # Query = proprio (context vector)
        # Key, Value = map (spatial features)
        attn_out, _ = self.attn(
            query=proprio_encoded.unsqueeze(1),  # (B, 1, d)
            key=map_encoded,                     # (B, L*W, d)
            value=map_encoded                    # (B, L*W, d)
        )
        attn_out = attn_out.squeeze(1)  # (B, d)

        # ------------------- MLP Trunk -------------------
        # Concatenate proprioception and attended map encoding
        mlp_input = torch.cat([attn_out, proprio], dim=-1)  # (B, map_feat_dim + proprio_dim)
        shared_out = self.mlp(mlp_input)  # (B, 64)

        # ------------------- Heads -------------------
        if role == "policy":
            self._shared_output = shared_out  # cache for value
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

# def extract_simplified_state(flattened_obs: torch.Tensor, obs_space=env.observation_space) -> torch.Tensor:
#     num_envs = flattened_obs.shape[0]
#     start = 0
#     chunks = []

#     # Keys to exclude (image/depth)
#     exclude_keys = {"front_depth"}

#     for key, value in obs_space.items():
#         # Flatten per-environment feature size (exclude batch dim if it exists)
#         shape = value.shape
#         if shape[0] == 1:
#             flat_dim = int(torch.tensor(shape[1:]).prod().item())  # remove batch dim
#         else:
#             flat_dim = int(torch.tensor(shape).prod().item())  # no batch dim present

#         end = start + flat_dim

#         if key not in exclude_keys:
#             chunks.append(flattened_obs[:, start:end])

#         start = end

#     return torch.cat(chunks, dim=-1)   

# def simplify_gym_space(original_space, exclude_keys: list):
#     """
#     Return a new Dict space excluding the specified keys.

#     Args:
#         original_space (spaces.Dict): The original Gym Dict space.
#         exclude_keys (list): List of keys to exclude.

#     Returns:
#         spaces.Dict: A new Dict space with the specified keys removed.
#     """
#     return gymnasium.Space({
#         key: space for key, space in original_space.spaces.items()
#         if key not in exclude_keys
#     })

# def expand_obs_tensor(small_obs: torch.Tensor) -> torch.Tensor:
#     """
#     Expands a [B, 85] observation tensor to [B, 1081429] by adding dummy image and depth data.

#     Args:
#         small_obs (torch.Tensor): Input tensor of shape [B, 85], containing only non-image features.

#     Returns:
#         torch.Tensor: Expanded tensor of shape [B, 1081429] with dummy image/depth values added.
#     """
#     B = small_obs.size(0)

#     # Sizes of each non-image feature (adds up to 85)
#     non_image_feature_sizes = {
#         # "commands": 11,
#         "base_pos":3,
#         "base_ang_vel":3,
#         "base_lin_vel":3,
#         "dof_pos": 2,
#         "dof_vel": 4,
#         "taken_actions": 6,
#     }

#     # Image/depth fields with their flattened sizes
#     image_fields_sizes = {
#         "front_depth": 64 * 64 * 3,
#     }

#     # Split the input tensor into components
#     components = []
#     start = 0
#     for size in non_image_feature_sizes.values():
#         end = start + size
#         components.append(small_obs[:, start:end])
#         start = end

#     # Create dummy tensors for image/depth inputs
#     image_components = [
#         torch.zeros(B, size, dtype=small_obs.dtype, device=small_obs.device)
#         for size in image_fields_sizes.values()
#     ]

#     # Order of final concatenation (matches obs_space)
#     # back_depth, commands, dof_diff, dof_vel, front_depth, gripper_depth, gripper_img,
#     # object_pos, object_quat, robot_base_pos, robot_base_quat, taken_actions
#     full_obs = torch.cat([
#         # image_components[0],  # back_depth
#         # components[0],        # commands
#         components[0],        # dof_diff
#         components[1],        # dof_vel
#         components[2],        # dof_diff
#         components[3],        # dof_vel
#         components[4],
#         image_components[0],  # front_depth
#         # image_components[2],  # gripper_depth
#         # image_components[3],  # gripper_img
#         # components[3],        # object_pos
#         # components[4],        # object_quat
#         # components[5],        # robot_base_pos
#         # components[6],        # robot_base_quat
#         components[5],        # taken_actions
#     ], dim=1)
#     # print("full_obs_shape_in_expand_obs_tensor()",full_obs.shape)
#     # exit(0)
#     return full_obs

# class MyRandomMemory(RandomMemory):
#     def __init__(self, memory_size, num_envs = 1, device = None, export = False, export_format = "pt", export_directory = "", replacement=True):
#         super().__init__(memory_size, num_envs, device, export, export_format, export_directory, replacement)    
    
#     def create_tensor(
#         self,
#         name: str,
#         size: Union[int, Tuple[int], gymnasium.Space],
#         dtype: Optional[torch.dtype] = None,
#         keep_dimensions: bool = False,
#     ) -> bool:
#         """Create a new internal tensor in memory

#         The tensor will have a 3-components shape (memory size, number of environments, size).
#         The internal representation will use _tensor_<name> as the name of the class property

#         :param name: Tensor name (the name has to follow the python PEP 8 style)
#         :type name: str
#         :param size: Number of elements in the last dimension (effective data size).
#                      The product of the elements will be computed for sequences or gymnasium spaces
#         :type size: int, tuple or list of integers or gymnasium space
#         :param dtype: Data type (torch.dtype) (default: ``None``).
#                       If None, the global default torch data type will be used
#         :type dtype: torch.dtype or None, optional
#         :param keep_dimensions: Whether or not to keep the dimensions defined through the size parameter (default: ``False``)
#         :type keep_dimensions: bool, optional

#         :raises ValueError: The tensor name exists already but the size or dtype are different

#         :return: True if the tensor was created, otherwise False
#         :rtype: bool
#         """
#         # compute data size
#         if not keep_dimensions:
#             size = compute_space_size(size, occupied_size=True)
#         # check dtype and size if the tensor exists
#         if name in self.tensors:
#             tensor = self.tensors[name]
#             if tensor.size(-1) != size:
#                 raise ValueError(f"Size of tensor {name} ({size}) doesn't match the existing one ({tensor.size(-1)})")
#             if dtype is not None and tensor.dtype != dtype:
#                 raise ValueError(f"Dtype of tensor {name} ({dtype}) doesn't match the existing one ({tensor.dtype})")
#             return False
#         # define tensor shape
#         if name in ["states","next_states"]:
#             size=21
#         tensor_shape = (
#             (self.memory_size, self.num_envs, *size) if keep_dimensions else (self.memory_size, self.num_envs, size)
#         )
#         view_shape = (-1, *size) if keep_dimensions else (-1, size)
#         # create tensor (_tensor_<name>) and add it to the internal storage
#         setattr(self, f"_tensor_{name}", torch.zeros(tensor_shape, device=self.device, dtype=dtype))
#         # update internal variables
#         self.tensors[name] = getattr(self, f"_tensor_{name}")
#         self.tensors_view[name] = self.tensors[name].view(*view_shape)
#         self.tensors_keep_dimensions[name] = keep_dimensions
#         # fill the tensors (float tensors) with NaN
#         for tensor in self.tensors.values():
#             if torch.is_floating_point(tensor):
#                 tensor.fill_(float("nan"))
#         return True
    
#     def add_samples(self, **tensors):
#         """Record samples in memory

#         Samples should be a tensor with 2-components shape (number of environments, data size).
#         All tensors must be of the same shape

#         According to the number of environments, the following classification is made:

#         - one environment:
#           Store a single sample (tensors with one dimension) and increment the environment index (second index) by one

#         - number of environments less than num_envs:
#           Store the samples and increment the environment index (second index) by the number of the environments

#         - number of environments equals num_envs:
#           Store the samples and increment the memory index (first index) by one

#         :param tensors: Sampled data as key-value arguments where the keys are the names of the tensors to be modified.
#                         Non-existing tensors will be skipped
#         :type tensors: dict

#         :raises ValueError: No tensors were provided or the tensors have incompatible shapes
#         """
#         if not tensors:
#             raise ValueError(
#                 "No samples to be recorded in memory. Pass samples as key-value arguments (where key is the tensor name)"
#             )

#         # dimensions and shapes of the tensors (assume all tensors have the dimensions of the first tensor)
#         # print(tensors.keys())
#         # for key, value in tensors.items():
#         #     print(key,value.shape)
#         tmp = tensors.get("states", tensors[next(iter(tensors))])  # ask for states first
#         dim, shape = tmp.ndim, tmp.shape

#         # multi environment (number of environments equals num_envs)
#         if dim > 1 and shape[0] == self.num_envs:
#             for name, tensor in tensors.items():
#                 if name in self.tensors:
#                     if name in ["states","next_states"]:
#                         tensor=extract_simplified_state(tensor)
#                         # print("simplified_state",tensor.shape)
#                         tensor_shape=self.tensors[name][self.memory_index].shape
#                         # print(tensor_shape)
#                         self.tensors[name][self.memory_index].copy_(tensor)
#                     else:
#                         self.tensors[name][self.memory_index].copy_(tensor)
#             self.memory_index += 1
#         # multi environment (number of environments less than num_envs)
#         elif dim > 1 and shape[0] < self.num_envs:
#             for name, tensor in tensors.items():
#                 if name in self.tensors:
#                     self.tensors[name][self.memory_index, self.env_index : self.env_index + tensor.shape[0]].copy_(
#                         tensor
#                     )
#             self.env_index += tensor.shape[0]
#         # single environment - multi sample (number of environments greater than num_envs (num_envs = 1))
#         elif dim > 1 and self.num_envs == 1:
#             for name, tensor in tensors.items():
#                 if name in self.tensors:
#                     num_samples = min(shape[0], self.memory_size - self.memory_index)
#                     remaining_samples = shape[0] - num_samples
#                     # copy the first n samples
#                     self.tensors[name][self.memory_index : self.memory_index + num_samples].copy_(
#                         tensor[:num_samples].unsqueeze(dim=1)
#                     )
#                     self.memory_index += num_samples
#                     # storage remaining samples
#                     if remaining_samples > 0:
#                         self.tensors[name][:remaining_samples].copy_(tensor[num_samples:].unsqueeze(dim=1))
#                         self.memory_index = remaining_samples
#         # single environment
#         elif dim == 1:
#             for name, tensor in tensors.items():
#                 if name in self.tensors:
#                     self.tensors[name][self.memory_index, self.env_index].copy_(tensor)
#             self.env_index += 1
#         else:
#             raise ValueError(f"Expected shape (number of environments = {self.num_envs}, data size), got {shape}")

#         # update indexes and flags
#         if self.env_index >= self.num_envs:
#             self.env_index = 0
#             self.memory_index += 1
#         if self.memory_index >= self.memory_size:
#             self.memory_index = 0
#             self.filled = True

#             # export tensors to file
#             if self.export:
#                 self.save(directory=self.export_directory, format=self.export_format)
    
#     def sample_all(
#         self, names: Tuple[str], mini_batches: int = 1, sequence_length: int = 1
#     ) -> List[List[torch.Tensor]]:
#         """Sample all data from memory

#         :param names: Tensors names from which to obtain the samples
#         :type names: tuple or list of strings
#         :param mini_batches: Number of mini-batches to sample (default: ``1``)
#         :type mini_batches: int, optional
#         :param sequence_length: Length of each sequence (default: ``1``)
#         :type sequence_length: int, optional

#         :return: Sampled data from memory.
#                  The sampled tensors will have the following shape: (memory size * number of environments, data size)
#         :rtype: list of torch.Tensor list
#         """
#         # sequential order
#         if sequence_length > 1:
#             if mini_batches > 1:
#                 batches = np.array_split(self.all_sequence_indexes, mini_batches)
#                 return [[self.tensors_view[name][batch] for name in names] for batch in batches]
#             return [[self.tensors_view[name][self.all_sequence_indexes] for name in names]]

#         # default order
#         if mini_batches > 1:
#             batch_size = (self.memory_size * self.num_envs) // mini_batches
#             batches = [(batch_size * i, batch_size * (i + 1)) for i in range(mini_batches)]
#             return_list=[]
#             for batch in batches:
#                 single_batch=[]
#                 for name in names:
#                     if name in ["states","next_states"]:
#                         # print("batch_shape",self.tensors_view[name][batch[0]:batch[1]].shape)
#                         # expanded_tensor=torch.zeros((self.tensors_view[name][batch[0]:batch[1]].shape[0],1081429))
#                         # print("exp_tens",expanded_tensor.shape)
#                         single_batch.append(expand_obs_tensor(self.tensors_view[name][batch[0]:batch[1]]))
#                     else:
#                         single_batch.append(self.tensors_view[name][batch[0]:batch[1]])
#                 return_list.append(single_batch)
#             return return_list
#             # return [[self.tensors_view[name][batch[0] : batch[1]] for name in names] for batch in batches]
#         return [[self.tensors_view[name] for name in names]]
           
# memory = MyRandomMemory(memory_size=8, num_envs=env.num_envs, device=device)
memory = MyRandomMemory(
    memory_size=1000,
    num_envs=env.num_envs,
    obs_space=env.observation_space,
    exclude_keys=["front_depth"],
    dummy_fillers={"front_depth": 64*64*3},
)

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
cfg["mini_batches"] = 4  # 24 * 4096 / 24576
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveLR
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

# # start training
trainer.train()

# # download the trained agent's checkpoint from Hugging Face Hub and load it
path = "/home/vybhav/gs_gym_wrapper_reference/skrl/runs/torch/Genesis-Goto-Anymal-C-v0/25-09-15_14-58-17-605290_PPO/checkpoints/agent_5100.pt"
# agent.load(path)

# # # # start evaluation
# trainer.eval()
