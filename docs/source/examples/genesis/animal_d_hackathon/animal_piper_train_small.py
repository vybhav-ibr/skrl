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

import torch.nn.functional as F
import torchvision.transforms.functional as TF
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
# env.step()
env=wrap_env(env,wrapper='genesis')
device=gs.device
set_seed()  # e.g. `set_seed(42)` for fixed seed
# print("single obs shape is:",env._env.obs_buf.shape)

# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(85, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 128),
                                 nn.ELU())

        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)
        
        # Extract RGB images (front_cloud is assumed to be RGB)
        shapes_dict={}
        for key,value in space.items():
            shapes_dict[key]=value.shape
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
        # combined_input = torch.cat([depth_features, other_states], dim=-1)
        
        if role == "policy":
            self._shared_output = self.net(other_states)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(other_states) if self._shared_output is None else self._shared_output
            self._shared_output = None
            
            return self.value_layer(shared_output), {}   

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
        "back_depth": 128 * 128 * 1,
        "front_depth": 128 * 128 * 1,
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
