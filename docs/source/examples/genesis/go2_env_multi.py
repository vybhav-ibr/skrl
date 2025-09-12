import torch
import math
import genesis as gs
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat



def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower



class Go2EnvMulti:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device


        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
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
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.num_envs//2,self.num_envs//2+2))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )


        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))


        # add robot
        self.robot1_base_init_pos = torch.tensor(self.env_cfg["robot1_base_init_pos"], device=gs.device)
        self.robot1_base_init_quat = torch.tensor(self.env_cfg["robot1_base_init_quat"], device=gs.device)
        
        self.robot2_base_init_pos = torch.tensor(self.env_cfg["robot2_base_init_pos"], device=gs.device)
        self.robot2_base_init_quat = torch.tensor(self.env_cfg["robot2_base_init_quat"], device=gs.device)
        
        self.robot1_inv_base_init_quat = inv_quat(self.robot1_base_init_quat)
        self.robot2_inv_base_init_quat = inv_quat(self.robot2_base_init_quat)
        
        self.robot1 = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.robot1_base_init_pos.cpu().numpy(),
                quat=self.robot1_base_init_quat.cpu().numpy(),
            ),
        )
        self.robot2 = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.robot2_base_init_pos.cpu().numpy(),
                quat=self.robot2_base_init_quat.cpu().numpy(),
            ),
        )

        # Create robot list and stacked attributes
        self.robots = [self.robot1, self.robot2]
        self.base_init_pos = torch.stack([self.robot1_base_init_pos, self.robot2_base_init_pos], dim=0)
        self.base_init_quat = torch.stack([self.robot1_base_init_quat, self.robot2_base_init_quat], dim=0)
        self.inv_base_init_quat = torch.stack([self.robot1_inv_base_init_quat, self.robot2_inv_base_init_quat], dim=0)


        # build
        self.scene.build(n_envs=num_envs,env_spacing=(1,1))


        # names to indices
        self.motors_dof_idx = [self.robot1.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]


        # PD control parameters
        for robot in self.robots:
            robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
            robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)


        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((2, self.num_envs), device=gs.device, dtype=gs.tc_float)


        # initialize buffers with (num_robots, num_envs, ...) ordering
        self.base_lin_vel = torch.zeros((2, self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((2, self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((2, self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((2, self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((2, self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((2, self.num_envs), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((2, self.num_envs), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((2, self.num_envs), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((2, self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((2, self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((2, self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        
        self.num_agents=2
        self.possible_agents=[f"agent_{i}" for i in range(self.num_agents)]


    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)


    def step(self, actions):
        # actions should be (2, num_envs, num_actions)
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        # Control both robots in all environments
        for robot_idx, robot in enumerate(self.robots):
            exec_actions = self.last_actions[robot_idx] if self.simulate_action_latency else self.actions[robot_idx]
            target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
            robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        
        self.scene.step()
        
        # Update state for both robots
        for robot_idx, robot in enumerate(self.robots):
            self.episode_length_buf[robot_idx] += 1
            
            self.base_pos[robot_idx] = robot.get_pos()
            self.base_quat[robot_idx] = robot.get_quat()
            
            self.base_euler[robot_idx] = quat_to_xyz(
                transform_quat_by_quat(
                    torch.ones_like(self.base_quat[robot_idx]) * self.inv_base_init_quat[robot_idx], 
                    self.base_quat[robot_idx]
                ),
                rpy=True, degrees=True
            )
            
            inv_quat_robot = inv_quat(self.base_quat[robot_idx])
            self.base_lin_vel[robot_idx] = transform_by_quat(robot.get_vel(), inv_quat_robot)
            self.base_ang_vel[robot_idx] = transform_by_quat(robot.get_ang(), inv_quat_robot)
            self.projected_gravity[robot_idx] = transform_by_quat(self.global_gravity, inv_quat_robot)
            
            self.dof_pos[robot_idx] = robot.get_dofs_position(self.motors_dof_idx)
            self.dof_vel[robot_idx] = robot.get_dofs_velocity(self.motors_dof_idx)
        
        # Resample commands (shared between robots in same env)
        envs_idx = (
            (self.episode_length_buf[0] % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False).flatten()
        )
        self._resample_commands(envs_idx)
        
        # Termination - check for both robots
        for robot_idx in range(2):
            self.reset_buf[robot_idx] = self.episode_length_buf[robot_idx] > self.max_episode_length
            self.reset_buf[robot_idx] |= torch.abs(self.base_euler[robot_idx, :, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
            self.reset_buf[robot_idx] |= torch.abs(self.base_euler[robot_idx, :, 0]) > self.env_cfg["termination_if_roll_greater_than"]
            
            time_out_idx = (self.episode_length_buf[robot_idx] > self.max_episode_length).nonzero(as_tuple=False).flatten()
            if robot_idx == 0:
                self.extras.setdefault("time_outs", torch.zeros((2, self.num_envs), device=gs.device, dtype=gs.tc_float))
            self.extras["time_outs"][robot_idx, time_out_idx] = 1.0
            
            # Reset environments for this robot
            reset_envs = self.reset_buf[robot_idx].nonzero(as_tuple=False).flatten()
            self.reset_idx(robot_idx, reset_envs)
        
        # Compute rewards for both robots
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]  # This should return (2, num_envs)
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        # Compute observations for both robots
        for robot_idx in range(2):
            self.obs_buf[robot_idx] = torch.cat([
                self.base_ang_vel[robot_idx] * self.obs_scales["ang_vel"],
                self.projected_gravity[robot_idx],
                self.commands * self.commands_scale,  # Same commands for both robots
                (self.dof_pos[robot_idx] - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel[robot_idx] * self.obs_scales["dof_vel"],
                self.actions[robot_idx],
            ], axis=-1)
        
        self.last_actions = self.actions.clone()
        self.last_dof_vel = self.dof_vel.clone()
        
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras
    
    def get_dummy_observations(self):
        return np.zeros([1, self.num_obs])  # (num_envs, obs_size)
    
    def get_dummy_actions(self):
        return np.zeros([1, self.num_actions])  # (num_envs, action_size)


    def get_privileged_observations(self):
        return None


    def reset_idx(self, robot_idx, envs_idx):
        if len(envs_idx) == 0:
            return

        robot = self.robots[robot_idx]

        self.dof_pos[robot_idx, envs_idx] = self.default_dof_pos
        self.dof_vel[robot_idx, envs_idx] = 0.0

        robot.set_dofs_position(
            position=self.dof_pos[robot_idx, envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        self.base_pos[robot_idx, envs_idx] = self.base_init_pos[robot_idx]
        self.base_quat[robot_idx, envs_idx] = self.base_init_quat[robot_idx].reshape(1, -1)
        robot.set_pos(self.base_pos[robot_idx, envs_idx], zero_velocity=False, envs_idx=envs_idx)
        robot.set_quat(self.base_quat[robot_idx, envs_idx], zero_velocity=False, envs_idx=envs_idx)

        self.base_lin_vel[robot_idx, envs_idx] = 0
        self.base_ang_vel[robot_idx, envs_idx] = 0
        robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[robot_idx, envs_idx] = 0.0
        self.last_dof_vel[robot_idx, envs_idx] = 0.0
        self.episode_length_buf[robot_idx, envs_idx] = 0
        self.reset_buf[robot_idx, envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][f"rew_{key}_agent{robot_idx}"] = (
                torch.mean(self.episode_sums[key][robot_idx, envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][robot_idx, envs_idx] = 0.0

        self._resample_commands(envs_idx)


    def reset(self):
        self.reset_buf[:] = True
        for robot_idx in range(2):
            self.reset_idx(robot_idx, torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None


    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes) - Return (2, num_envs)
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2].unsqueeze(0) - self.base_lin_vel[:, :, :2]
        ), dim=2)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])


    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) - Return (2, num_envs)
        ang_vel_error = torch.square(
            self.commands[:, 2].unsqueeze(0) - self.base_ang_vel[:, :, 2]
        )
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])


    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity - Return (2, num_envs)
        return torch.square(self.base_lin_vel[:, :, 2])


    def _reward_action_rate(self):
        # Penalize changes in actions - Return (2, num_envs)
        return torch.sum(torch.square(self.last_actions - self.actions), dim=2)


    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose - Return (2, num_envs)
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=2)


    def _reward_base_height(self):
        # Penalize base height away from target - Return (2, num_envs)
        return torch.square(self.base_pos[:, :, 2] - self.reward_cfg["base_height_target"])
