import argparse
# Import the skrl components to build the RL system
import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed

import genesis as gs
from go2_env_scan import Go2EnvScan

def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
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
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", type=str, default="go2-scan-walking")
parser.add_argument("-B", "--num_envs", type=int, default=5)
parser.add_argument("-v", "--vis", action="store_true")
parser.add_argument("--max_iterations", type=int, default=101)
args = parser.parse_args()

gs.init(logging_level="warning",precision="32")
env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

env = Go2EnvScan(
    num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,show_viewer=args.vis
)
# env.step()
env=wrap_env(env,wrapper='genesis')
device=gs.device
set_seed()  # e.g. `set_seed(42)` for fixed seed
# print("single obs shape is:",env._env.obs_buf.shape)

class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        print("obs_space:",observation_space)
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
      
        self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=3),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=2, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                nn.Linear(160000, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 8),
                                                nn.Tanh())

        self.net = nn.Sequential(nn.Linear(41, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)
        # print("image space thinhy:",space['front_cloud'].permute(0,3,1,2).shape)
        features = self.features_extractor(space['front_cloud'].permute(0,3,1,2).to(torch.float32))

        mean_actions = torch.tanh(self.net(torch.cat([features,
                                                      space["ang_vel"].view(states.shape[0], -1).to(torch.float32),
                                                      space["commands"].view(states.shape[0], -1).to(torch.float32),
                                                      space["dof_vel"].view(states.shape[0], -1).to(torch.float32),
                                                      space["dof_diff"].view(states.shape[0], -1).to(torch.float32),
                                                      space["proj_gravity"].view(states.shape[0], -1).to(torch.float32),
                                                      ], dim=-1)))

        # print(mean_actions.shape)
        return mean_actions, self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=3),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=2, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                nn.Linear(160000, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 8),
                                                nn.Tanh())

        self.net = nn.Sequential(nn.Linear(41 + self.num_actions, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        states = inputs["states"]

        space = self.tensor_to_space(states, self.observation_space)
        features = self.features_extractor(torch.tensor(space['front_cloud'],dtype=torch.float32).permute(0,3,1,2))

        return self.net(torch.cat([features,
                                    space["ang_vel"].view(states.shape[0], -1),
                                    space["commands"].view(states.shape[0], -1),
                                    space["dof_vel"].view(states.shape[0], -1),
                                    space["dof_diff"].view(states.shape[0], -1),
                                    space["proj_gravity"].view(states.shape[0], -1),
                                    ], dim=-1))


# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=28, num_envs=env.num_envs, device=device, replacement=False)

# Instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#spaces-and-models
models_sac = {}
models_sac["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
models_sac["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_sac.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["gradient_steps"] = 1
cfg_sac["batch_size"] = 256
cfg_sac["random_timesteps"] = 0
cfg_sac["learning_starts"] = 10000
cfg_sac["learn_entropy"] = True
# logging to TensorBoard and write checkpoints each 1000 and 5000 timesteps respectively
cfg_sac["experiment"]["write_interval"] = 1000
cfg_sac["experiment"]["checkpoint_interval"] = 5000

# print("SAC env obs Configuration:",env.observation_space)
agent_sac = SAC(models=models_sac,
                memory=memory,
                cfg=cfg_sac,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
print(type(agent_sac))


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_sac)

# start training
trainer.train()
