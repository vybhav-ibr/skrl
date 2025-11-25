import torch
import math
import genesis as gs
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.05  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.num_envs//2,self.num_envs//2+4))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        self.ball=self.scene.add_entity(
            gs.morphs.Sphere(
                radius=(0.05),
                fixed=True,
                collision=False ,
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(225/255, 165/225, 0.0),
                ),
            ),
            visualize_contact=False
        )
        # build
        self.scene.build(n_envs=num_envs,env_spacing=(0,0))

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _random_pos_near_base(self, envs_idx, scale=1.0, min_distance=0.5):
        """
        Samples new end-effector positions near the robot base, ensuring the sampled position is 
        at least `min_distance` meters away from the robot's position in the x and y directions.

        Args:
            envs_idx (torch.Tensor or list): Indices of environments to sample positions for.
            scale (float): Scaling factor for offset from base position.
            min_distance (float): Minimum distance in meters that the sampled position must be from the robot's position in the x and y directions.

        Returns:
            torch.Tensor: Sampled positions of shape (k, 3)
        """
        base_pos = self.robot.get_pos(envs_idx=envs_idx)  # shape (k, 3)
        k = base_pos.shape[0]

        # Get the ranges for x, y, z from the configuration
        x_range, y_range, z_range = self.command_cfg["eef_pos"][0], self.command_cfg["eef_pos"][1], self.command_cfg["eef_pos"][2]
        
        # Scale the random offsets based on the given axis ranges and global scale
        ranges = torch.tensor([
            [x_range[0], x_range[1]],
            [y_range[0], y_range[1]],
            [z_range[0], z_range[1]]
        ], device=base_pos.device)

        # Compute half-ranges (for symmetric scaling around 0)
        half_ranges = (ranges[:, 1] - ranges[:, 0]) / 2.0  # (3,)
        
        # Apply scaling to random offsets in [-1, 1]
        random_offsets = torch.rand(k, 3) * 2 - 1  # uniform in [-1, 1]
        scaled_offsets = random_offsets * half_ranges * scale  # (k, 3)

        # Calculate the distance in x and y directions for each sample
        xy_dist = torch.sqrt(scaled_offsets[:, 0]**2 + scaled_offsets[:, 1]**2)

        # Scale the x and y offsets if their distance is less than the min_distance
        scaling_factor = torch.maximum(torch.ones(k, device=base_pos.device), min_distance / xy_dist)
        scaled_offsets[:, 0] *= scaling_factor
        scaled_offsets[:, 1] *= scaling_factor

        # Sampled position = base position + offset
        sampled_pos = base_pos + scaled_offsets

        # Clamp final position within [base + min_range, base + max_range]
        min_bounds = base_pos + ranges[:, 0]
        max_bounds = base_pos + ranges[:, 1]

        sampled_pos = torch.max(sampled_pos, min_bounds)
        sampled_pos = torch.min(sampled_pos, max_bounds)

        return sampled_pos

    # def _random_pos_near_base(self, envs_idx, scale=1.0, min_distance=0.5):
    #     """
    #     Samples new end-effector positions near the robot base, ensuring the sampled position is 
    #     at least `min_distance` meters away from the robot's position in the x and y directions.
    #     The distance is controlled by a Gaussian-like distribution, where the mean of the distribution
    #     is influenced by the scale.

    #     Args:
    #         envs_idx (torch.Tensor or list): Indices of environments to sample positions for.
    #         scale (float): Determines how far from the center (0.5m) the sampled point will be (0 = near 0.5m, 1 = near 2.5m).
    #         min_distance (float): Minimum distance in meters that the sampled position must be from the robot's position in the x and y directions.

    #     Returns:
    #         torch.Tensor: Sampled positions of shape (k, 3)
    #     """
    #     base_pos = self.robot.get_pos(envs_idx=envs_idx)  # shape (k, 3)
    #     k = base_pos.shape[0]

    #     # Get the ranges for x, y, z from the configuration
    #     x_range, y_range, z_range = self.command_cfg["eef_pos"]
    #     ranges = torch.tensor([x_range, y_range, z_range], device=base_pos.device)  # (3, 2)

    #     sampled_offsets = torch.zeros(k, 3, device=base_pos.device)

    #     for axis in range(3):
    #         axis_min, axis_max = ranges[axis]
            
    #         # For x and y axes, we want to sample between min_distance and 2.5m
    #         if axis < 2:  # x and y directions
    #             # Sample a random distance between 0.5 and 2.5 meters
    #             random_distance = torch.rand(k, device=base_pos.device) * (axis_max - min_distance) + min_distance

    #             # Apply Gaussian-like bias for the upper bound using scale
    #             mean_distance = min_distance + scale * (axis_max - min_distance)  # Control the bias
    #             stddev = 0.2 * (axis_max - min_distance)  # Standard deviation for "spread"
                
    #             # Apply a Gaussian distribution for the random distance, biased towards the upper bound
    #             random_distance = torch.normal(mean_distance, stddev)  # mean and stddev from scale

    #             # Clip the values to ensure they are within bounds (between 0.5 and 2.5 meters)
    #             random_distance = torch.clamp(random_distance, min=min_distance, max=axis_max)

    #             # Randomly choose a direction (positive or negative) for each sample
    #             direction = 2 * torch.randint(0, 2, (k,), device=base_pos.device).float() - 1  # -1 or 1

    #             # Apply the direction to the offset
    #             sampled_offsets[:, axis] = random_distance * direction

    #         else:  # Z direction (no restriction other than the defined range)
    #             # Sample a random value in the range of z axis
    #             z_range_min, z_range_max = z_range
    #             sampled_offsets[:, axis] = torch.rand(k, device=base_pos.device) * (z_range_max - z_range_min) + z_range_min

    #     # Calculate the final sampled positions
    #     sampled_pos = base_pos + sampled_offsets

    #     # Clamp final position within the bounds [base + min_range, base + max_range]
    #     min_bounds = base_pos + ranges[:, 0]
    #     max_bounds = base_pos + ranges[:, 1]
    #     sampled_pos = torch.max(sampled_pos, min_bounds)
    #     sampled_pos = torch.min(sampled_pos, max_bounds)

    #     return sampled_pos
    
    def _resample_commands(self, envs_idx):
        sample_scale=0.5
        self.commands[envs_idx, 0:3] = self._random_pos_near_base(envs_idx=envs_idx,scale=sample_scale)
        self.commands[envs_idx,2]=self.reward_cfg["base_height_target"]

        self.ball.set_pos(self.commands[envs_idx, 0:3],envs_idx=envs_idx)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.rel_pos = self.commands - self.base_pos
        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

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
        # print(torch.sum(torch.square(self.commands), dim=1) - torch.sum(torch.square(self.base_pos), dim=1))
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                # self.commands * self.commands_scale,  # 3
                # self.base_pos,
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras
    
    def get_dummy_observations(self):
        return np.zeros([1, self.num_obs])
    
    def get_dummy_actions(self):
        return np.zeros([1, self.num_actions])

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.robot.get_pos(envs_idx=envs_idx)
        self.base_pos[envs_idx,2]=self.base_init_pos[2]
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_pos(self):
        pos_error=torch.norm(self.robot.get_pos()[:,:2] - self.commands[:,0:2], dim=-1)
        pos_reward=1.0 - torch.exp(pos_error)
        # print(pos_error,pos_reward)
        return pos_error
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        # print("summed reward is:",torch.abs(self.dof_pos - self.default_dof_pos).shape)
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
