import torch
import math
import genesis as gs
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

#
#
#

# def gs_rand_float(command, shape, device):
#     stacked_commands=None
#     for command_element in command:
#         lower,upper=command_element[0],command_element[1]
#         sampled_command=(upper - lower) * torch.rand(size=shape, device=device) + lower
#         if stacked_commands==None:
#             stacked_commands=sampled_command   
#         else:
#             stacked_commands=torch.stack([stacked_commands,sampled_command],dim=0)
#     return stacked_commands

def gs_rand_float(command, shape, device):
    sampled = []
    for command_element in command:
        lower, upper = command_element[0], command_element[1]
        sample = (upper - lower) * torch.rand(size=shape, device=device) + lower
        sampled.append(sample.unsqueeze(1))  # shape: [n, 1]
    return torch.cat(sampled, dim=1)  # shape: [n, 3]

class AA2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # print(env_cfg)
        # exit(0)
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
            vis_options=gs.options.VisOptions(),#rendered_envs_idx=list(range(self.num_envs//2,self.num_envs//2+4))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.ground=self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        
        self.ball=self.scene.add_entity(
            gs.morphs.Sphere(
                radius=(0.02),
                fixed=True,
                collision=True,
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(225/255, 165/225, 0.0),
                ),
            ),
        )
        
        self.ball_2=self.scene.add_entity(
            gs.morphs.Sphere(
                radius=(0.02),
                fixed=True,
                collision=True,
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(0/255, 165/225, 255/255),
                ),
            ),
        )

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="anymal_c/urdf/anymal_c.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
            visualize_contact=True
        )

        # build
        self.scene.build(n_envs=num_envs,env_spacing=(2,2))

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
        # self.commands_scale = torch.tensor(
        #     [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
        #     device=gs.device,
        #     dtype=gs.tc_float,
        # )
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
        
        self.eef_link_idx=self.robot.get_link("wrist_2").idx_local
        self.eef_link_name="wrist_2"
        self.arm_links=["shoulder","upperarm","elbow","forearm","wrist_1","wrist_2"]
        self.table=None
        # for link in self.robot.links:
        #     print(link.name)

    def _resample_commands(self, envs_idx):
        #change to later sample from closed form solutions
        self.commands[envs_idx, 0:3] = gs_rand_float(self.command_cfg["eef_pos"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 3:7] = gs_rand_float(self.command_cfg["eef_quat"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 7:10] = gs_rand_float(self.command_cfg["force"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 10] = gs_rand_float(self.command_cfg["num_timesteps"], (len(envs_idx),), gs.device).squeeze(-1)

    def step(self, actions):
        # print(self.robot.get_dofs_kp())
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

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_criteria_roll"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_criteria_pitch"]
        self.reset_buf |= torch.abs(self.robot.get_pos()[:, 2]) < self.env_cfg["termination_criteria_base_height"]
        
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            
            rew = reward_func() * self.reward_scales[name]
            # print(name,"!!!",rew)
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands, # 3
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
        self.base_pos[envs_idx] = self.base_init_pos
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

        # num_reset = len(envs_idx)
        # random_x = torch.rand(num_reset, device=self.device) * 0.4 + 1.2  # 0.2 ~ 0.6
        # random_y = (torch.rand(num_reset, device=self.device) - 0.5) * 0.5  # -0.25 ~ 0.25
        # random_z = torch.ones(num_reset, device=self.device) * 0.025  +0.85# 0.15 ~ 0.15
        random_pos = self.commands[envs_idx,0:3]
        random_pos2= self.commands[envs_idx,0:3]
        random_pos2[:,0]=-random_pos2[:,0]
        random_pos2[:,1]=-random_pos2[:,1]
        self.ball.set_pos(random_pos, envs_idx=envs_idx)
        self.ball_2.set_pos(random_pos2, envs_idx=envs_idx)
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
    
    # def _reward_survival(self):
    #     return self.episode_length_buf/self.max_episode_length
        
    # def _reward_pos_alignment(self):
    #     eef_pos = self.robot.get_pos(self.eef_link_idx)
    #     eef_quat = self.robot.get_pos(self.eef_link_idx)

    #     target_pos = self.goal_pose[:, :3]
    #     target_quat = self.goal_pose[:, 3:7]

    #     # Position error (L2 norm)
    #     pos_error = torch.norm(eef_pos - target_pos, dim=-1)

    #     # Orientation error: compute angular distance from quaternions
    #     dot_product = torch.sum(eef_quat * target_quat, dim=-1).clamp(-1.0, 1.0)
    #     ang_error = 2 * torch.acos(torch.abs(dot_product))

    #     # Combine errors
    #     reward = torch.exp(-2.0 * pos_error) * torch.exp(-1.0 * ang_error)
    #     return reward

    # def _reward_undesirable_contact(self):
    #     # Get current gripper DOF positions
    #     ground_contacts=self.robot.get_contacts(with_entity=self.ground)
    #     ground_contact_forces=[]
    #     table_contact_forces=[]
    #     print("ground_contact:",ground_contacts["link_a"])
    #     forces=torch.zeros((self.num_envs,1))
    #     if ground_contacts["link_a"].sum()>0:
    #         for it,contact_link_a_idx in enumerate(ground_contacts["link_a"]):
    #             if self.scene.rigid_solver.links[contact_link_a_idx].name in self.arm_links:
    #                 ground_contact_forces.append(sum(ground_contacts["force_a"][it]))
                    
    #         # ground_contacts=ground_contacts["link_a" in self.arm_links]
    #         if self.table is not None:
    #             table_contacts=self.robot.get_contacts(with_entity=self.ground)
                
    #             for it,contact_link_a_idx in enumerate(table_contacts["link_a"]):
    #                 if self.scene.rigid_solver.links[contact_link_a_idx].name in self.arm_links:
    #                     table_contact_forces.append(sum(table_contacts["force_a"][it]))
    #         ground_contact_forces.extend(table_contact_forces)
    #         forces=torch.tensor(ground_contact_forces)
    #     print("forces shape is",forces.shape)
    #     return forces

    # def _reward_target_force_and_contact(self):
    #     ball_contacts=self.robot.get_contacts(with_entity=self.ball)
    #     # ball_contact_forces=[]
    #     contact_link_a_idx=ball_contacts["link_a"][0]
    #     if self.scene.rigid_solver.links[contact_link_a_idx].name in self.dummy_eef_link:
    #         contacts_sum=(sum(ball_contacts["force_a"][0]))
    #     forces=torch.tensor(contacts_sum)
    #     return forces

    # def _reward_high_joint_force(self):
    #     # Penalize high joint torques (forces)
    #     joint_torques = self.robot.get_dofs_force(self.motors_dof_idx)
    #     return torch.sum(torch.square(joint_torques), dim=1)
    
    # def _reward_time_cost(self):
    #     pos_alignment=self._reward_pos_alignment()
    #     time_scaled_reward=pos_alignment*self.episode_length_buf
    #     return time_scaled_reward
    # # def _reward_tracking_lin_vel(self):
    # #     # Tracking of linear velocity commands (xy axes)
    # #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    # #     return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    # # def _reward_tracking_ang_vel(self):
    # #     # Tracking of angular velocity commands (yaw)
    # #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    # #     return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    # # def _reward_lin_vel_z(self):
    # #     # Penalize z axis base linear velocity
    # #     return torch.square(self.base_lin_vel[:, 2])

    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # # def _reward_similar_to_default(self):
    # #     # Penalize joint poses far away from default pose
    # #     return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
    def _reward_survival(self):
        reward = self.episode_length_buf / self.max_episode_length
        # print("_reward_survival:", reward.shape)
        return reward

    def _reward_pos_alignment(self):
        eef_pos = self.robot.get_links_pos(self.eef_link_idx).squeeze(1) 
        eef_quat = self.robot.get_links_quat(self.eef_link_idx).squeeze(1) 

        target_pos = self.commands[:, 0:3]
        target_quat = self.commands[:, 3:7]
        
        # print("eef_pos:", eef_pos.shape)
        # print("target_pos:", target_pos.shape)
        # print("eef_quat:", eef_quat.shape)
        # print("target_quat:", target_quat.shape)

        # Position error (L2 norm)
        pos_error = torch.norm(eef_pos - target_pos, dim=-1)

        # Orientation error: compute angular distance from quaternions
        dot_product = torch.sum(eef_quat * target_quat, dim=-1).clamp(-1.0, 1.0)
        ang_error = 2 * torch.acos(torch.abs(dot_product))

        # Combine errors
        reward = torch.exp(-2.0 * pos_error) * torch.exp(-1.0 * ang_error)
        # print("_reward_pos_alignment:", reward.shape)
        return reward
    
    def _reward_undesirable_contact(self):
        ground_contacts = self.robot.get_contacts(with_entity=self.ground)
        forces = torch.zeros((self.num_envs,), device=gs.device)

        if ground_contacts["valid_mask"].any():
            for env_idx in range(self.num_envs):
                contact_links = ground_contacts["link_a"][env_idx]       # [num_contacts]
                contact_forces = ground_contacts["force_a"][env_idx]     # [num_contacts, 3]
                valid_mask = ground_contacts["valid_mask"][env_idx]      # [num_contacts]

                for contact_idx in range(contact_links.shape[0]):
                    if not valid_mask[contact_idx]:
                        continue  # Skip invalid contacts

                    link_idx = contact_links[contact_idx].item()
                    link_name = self.scene.rigid_solver.links[link_idx].name

                    if link_name in self.arm_links:
                        # Use norm of the contact force vector
                        force_vec = contact_forces[contact_idx]
                        force_magnitude = torch.norm(force_vec)
                        forces[env_idx] += force_magnitude

        return forces

    def _reward_target_force_and_contact(self):
        ball_contacts = self.robot.get_contacts(with_entity=self.ball)
        forces = torch.zeros((self.num_envs,), device=gs.device)

        if ball_contacts["valid_mask"].any():
            for env_idx in range(self.num_envs):
                contact_links = ball_contacts["link_a"][env_idx]       # [num_contacts]
                contact_forces = ball_contacts["force_a"][env_idx]     # [num_contacts, 3]
                valid_mask = ball_contacts["valid_mask"][env_idx]      # [num_contacts]

                for contact_idx in range(contact_links.shape[0]):
                    if not valid_mask[contact_idx]:
                        continue  # Skip invalid contacts

                    link_idx = contact_links[contact_idx].item()
                    link_name = self.scene.rigid_solver.links[link_idx].name

                    if link_name in self.eef_link_name:
                        force_vec = contact_forces[contact_idx]
                        force_magnitude = torch.norm(force_vec)
                        forces[env_idx] += force_magnitude

        return forces

    def _reward_high_joint_force(self):
        joint_torques = self.robot.get_dofs_force(self.motors_dof_idx)
        reward = torch.sum(torch.square(joint_torques), dim=1)
        # print("_reward_high_joint_force:", reward.shape)
        return reward

    def _reward_time_cost(self):
        pos_alignment = self._reward_pos_alignment()
        time_scaled_reward = pos_alignment * self.episode_length_buf
        # print("_reward_time_cost:", time_scaled_reward.shape)
        return time_scaled_reward

    def _reward_action_rate(self):
        reward = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        # print("_reward_action_rate:", reward.shape)
        return reward

    def _reward_base_height(self):
        reward = (self.base_pos[:, 2] - self.reward_cfg["base_height_target"])*100
        # print("_reward_base_height:", reward)
        return reward
