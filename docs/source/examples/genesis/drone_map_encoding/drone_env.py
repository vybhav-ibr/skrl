import random
import torch
import math
import copy
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from collections import OrderedDict
import numpy as np

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def sample_positions_and_orientations(grid_size, num_samples, min_distance=2.5):
    """
    Sample positions and orientations with minimum distance constraint.
    
    Args:
        grid_size: Size of the grid
        num_samples: Number of positions to sample
        min_distance: Minimum distance between any two positions
    """
    def random_z_rotation():
        """Generate a random z-axis rotation (roll) between 0 and 360 degrees."""
        return random.uniform(0, 360)
    
    def check_distance(new_pos, existing_positions, min_dist):
        """Check if new position is at least min_dist away from all existing positions."""
        for pos in existing_positions:
            # Calculate 2D distance (ignoring z since all gates are at ground level)
            dist = math.sqrt((new_pos[0] - pos[0])**2 + (new_pos[1] - pos[1])**2)
            if dist < min_dist:
                return False
        return True

    positions = []
    orientations = []
    max_attempts = 1000  # Prevent infinite loops
    
    while len(positions) < num_samples:
        attempts = 0
        valid_position_found = False
        
        while not valid_position_found and attempts < max_attempts:
            # Sample random x, y position within the grid
            x = random.uniform(-grid_size//2, grid_size//2)
            y = random.uniform(-grid_size//2, grid_size//2)
            z = 0  # Gates are on the ground
            
            new_pos = (x, y, z)
            
            # Check if this position is valid (far enough from existing positions)
            if check_distance(new_pos, positions, min_distance):
                valid_position_found = True
                
                # Sample random z-axis rotation
                roll = random_z_rotation()
                pitch, yaw = 0, 0
                
                # Store the position and orientation
                positions.append(new_pos)
                orientations.append((pitch, yaw, roll))
            
            attempts += 1
        
        if not valid_position_found:
            # If we couldn't find a valid position after max_attempts, 
            # reduce min_distance slightly and try again
            min_distance *= 0.9
            print(f"Warning: Reduced min_distance to {min_distance:.2f} to fit all gates")

    return positions, orientations

class HoverEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            show_FPS=False
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane(collision=True))
        self.gate_positions,self.gate_orientations=sample_positions_and_orientations(grid_size=env_cfg["grid_size"], num_samples=env_cfg["num_gates"])
        print(self.gate_orientations[0])
        for gate_idx in range(env_cfg["num_gates"]):
            self.scene.add_entity(gs.morphs.Mesh(file="/home/vybhav/Music/drone_gate.stl",
                                                 pos=self.gate_positions[gate_idx],
                                                 euler=self.gate_orientations[gate_idx],
                                                 convexify=False,collision=True, scale=0.25),
                         vis_mode="collision")

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
                                         
        sensor_kwargs = dict(
            entity_idx=self.drone.idx,
            pos_offset=(0.05, 0.0, 0.05),
            euler_offset=(0.0, 0.0, 0.0),
            min_range=0.1,
            max_range=2.5,
            return_world_frame=False,
            draw_debug=False,
        )

        self.lidar = self.scene.add_sensor(gs.sensors.DepthCamera(pattern=gs.sensors.DepthCameraPattern(
            res=(16,16)), **sensor_kwargs))  # Reduced from (64,64) to save memory
        # build scene
        self.scene.build(n_envs=num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        
        self.command_buffer = torch.zeros((self.num_envs, 3, 3), device=gs.device, dtype=gs.tc_float)
        self.obs_space = {
            "base_ang_vel": self.base_ang_vel[0],
            "base_lin_vel": self.base_lin_vel[0],
            "base_quat": self.base_quat[0],
            "base_rel_pos": self.base_pos[0],
            "base_rel_pos_1": self.base_pos[0], # Placeholder
            "base_rel_pos_2": self.base_pos[0], # Placeholder
            "front_depth": torch.zeros((16, 16, 3)),
            "taken_actions": self.actions[0],
        }
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        self.target = None

        # for i in range(2500):
        #     self.scene.step()
        # exit(0)
        
    def get_dummy_observations(self):
        # for key,value in self.obs_buf.items():
        #     print("obs_buffer_entry:",key,value.shape)
        return OrderedDict(self.obs_space)
    
    def get_dummy_actions(self):
        return np.zeros([1, self.num_actions])

    def _resample_command_buffer(self, envs_idx):
        if len(envs_idx) == 0:
            return
        
        # Get all possible positions
        positions = torch.tensor(self.gate_positions, device=self.device, dtype=gs.tc_float)
        
        # Sample 3 random indices for each env
        indices = torch.randint(0, len(self.gate_positions), (len(envs_idx), 3), device=self.device)
        
        # Assign to buffer
        self.command_buffer[envs_idx] = positions[indices]

    def _update_command_buffer(self, envs_idx):
        if len(envs_idx) == 0:
            return
            
        # Shift targets: 0 <- 1, 1 <- 2
        self.command_buffer[envs_idx, 0] = self.command_buffer[envs_idx, 1]
        self.command_buffer[envs_idx, 1] = self.command_buffer[envs_idx, 2]
        
        # Sample new target for slot 2
        positions = torch.tensor(self.gate_positions, device=self.device, dtype=gs.tc_float)
        indices = torch.randint(0, len(self.gate_positions), (len(envs_idx),), device=self.device)
        self.command_buffer[envs_idx, 2] = positions[indices]
        
        # Update current command
        self.commands[envs_idx] = self.command_buffer[envs_idx, 0]

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx] = self.command_buffer[envs_idx, 0]
        
    def _at_target(self):
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # 14468 is hover rpm
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)
        # update target pos
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # resample commands
        envs_idx = self._at_target()
        self._update_command_buffer(envs_idx)

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        # print("pcd shape:",self.lidar.read()[0].shape)
        # print("pcd shape:",self.lidar.read()[0].view(self.num_envs, -1).shape)
        # exit(0)
        # Calculate relative positions for future targets
        rel_pos_1 = self.command_buffer[:, 1] - self.base_pos
        rel_pos_2 = self.command_buffer[:, 2] - self.base_pos

        self.obs_buf = torch.cat(
            [
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                self.base_quat,
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                torch.clip(rel_pos_1 * self.obs_scales["rel_pos"], -1, 1),
                torch.clip(rel_pos_2 * self.obs_scales["rel_pos"], -1, 1),
                self.lidar.read()[0].view(self.num_envs, -1),  # Detach sensor data to prevent gradient accumulation
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_command_buffer(envs_idx)
        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
