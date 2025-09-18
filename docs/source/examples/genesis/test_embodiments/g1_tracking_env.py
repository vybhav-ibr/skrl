import random
import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def gs_rand_gaussian(mean, min, max, n_std, shape, device):
    mean_tensor = mean.expand(shape).to(device)
    std_tensor = torch.full(shape, (max - min)/ 4.0 * n_std, device=device)
    return torch.clamp(torch.normal(mean_tensor, std_tensor), min, max)

def gs_additive(base, increment):
    return base + increment

def pose_to_matrix(position: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert batched positions and quaternions to 4x4 transformation matrices.

    Args:
        position: (N, 3) tensor of positions
        quaternion: (N, 4) tensor of quaternions in (w, x, y, z) format

    Returns:
        (N, 4, 4) tensor of transformation matrices
    """
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    # Rotation matrix elements
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    # Build rotation matrices
    R = torch.stack([
        torch.stack([1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)], dim=-1),
        torch.stack([2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)], dim=-1),
        torch.stack([2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)], dim=-1)
    ], dim=-2)  # shape: (N, 3, 3)

    N = position.shape[0]
    T = torch.eye(4, device=position.device).unsqueeze(0).repeat(N, 1, 1)  # (N, 4, 4)
    T[:, :3, :3] = R
    T[:, :3, 3] = position

    return T.detach().cpu()

class G1EnvTracking:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda", add_camera = False):
        self.device = torch.device(device)
        self.tracking_tracking_folder=command_cfg["tracking_files"]
        self.link_pos_data=torch.tensor(np.load(f"{self.tracking_tracking_folder}/body_positions.npy"))
        self.link_quat_data=torch.tensor(np.load(f"{self.tracking_tracking_folder}/body_rotations.npy"))
        self.link_lin_vel_data=torch.tensor(np.load(f"{self.tracking_tracking_folder}/body_linear_velocities.npy"))
        self.link_ang_vel_data=torch.tensor(np.load(f"{self.tracking_tracking_folder}/body_angular_velocities.npy"))

        self.dof_pos_data=torch.tensor(np.load(f"{self.tracking_tracking_folder}/dof_positions.npy"))
        self.dof_vel_data=torch.tensor(np.load(f"{self.tracking_tracking_folder}/dof_velocities.npy"))
        self.action_seq_length=self.dof_pos_data.shape[0]
        
        self.num_envs = num_envs
        self.num_obs = 30*3+30*4+30*3+30*3+29+29
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = 30*3+30*4+30*3+30*3+29+29

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        # self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.max_episode_length = self.action_seq_length

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
                camera_pos=(3.5, 0.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(show_world_frame=False),#rendered_envs_idx=list(range(num_envs//2,num_envs//2+4)) ),
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
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="/home/vybhav/quadrupeds_locomotion/robot/g1_29dof.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                # fixed=True
            ),
        )
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)

        # build
        self.scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0))
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_dof_properties"][name][0] for name in self.env_cfg["default_dof_properties"].keys()],
            device=self.device,
            dtype=gs.tc_float,
        )
        
        # PD control parameters
        for dof_name,dof_properties in env_cfg["default_dof_properties"].items():
            joint=self.robot.get_joint(dof_name)
            dof_idx=joint.dofs_idx_local
            self.robot.set_dofs_kp([dof_properties[1]],dof_idx)
            self.robot.set_dofs_kv([dof_properties[2]],dof_idx)
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_tracking_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs*2), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        # self.commands_scale = torch.tensor(
        #     [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["lin_vel"], self.obs_scales["lin_vel"]] ,
        #     device=self.device,
        #     dtype=gs.tc_float,
        # )

        # self.last_dof_vel = torch.zeros_like(self.actions)

        dof_names_list=env_cfg["default_dof_properties"].keys()
        self.dof_ids = [
            self.robot.get_joint(name).dofs_idx_local
            for name in dof_names_list
        ]
        self.dof_pos = self.robot.get_dofs_position(self.dof_ids)
        self.dof_vel = self.robot.get_dofs_velocity(self.dof_ids)

        self.link_pos = self.robot.get_links_pos()
        self.link_quat = self.robot.get_links_quat()
        self.link_lin_vel = self.robot.get_links_vel()
        self.link_ang_vel = self.robot.get_links_ang()
        
        self.extras = dict() 
           
    def _sample_commands(self, envs_idx):
        # print("envs_idx:",envs_idx.shape)
        if envs_idx.shape[0]>0:
            t_idx = self.episode_length_buf[envs_idx]
            pos = self.link_pos_data[t_idx]
            pos += torch.randn_like(pos) * self.command_cfg.get("link_pos_noise_std", 0.01)
            self.commands[envs_idx, 0:90] = pos.view(len(envs_idx), -1)

            # Link quaternions (30 x 4) — skip noise unless you plan to renormalize
            quat = self.link_quat_data[t_idx]
            self.commands[envs_idx, 90:210] = quat.view(len(envs_idx), -1)

            # Link linear velocities (30 x 3)
            linvel = self.link_lin_vel_data[t_idx]
            linvel += torch.randn_like(linvel) * self.command_cfg.get("link_linvel_noise_std", 0.02)
            self.commands[envs_idx, 210:300] = linvel.view(len(envs_idx), -1)

            # Link angular velocities (30 x 3)
            angvel = self.link_ang_vel_data[t_idx]
            angvel += torch.randn_like(angvel) * self.command_cfg.get("link_angvel_noise_std", 0.02)
            self.commands[envs_idx, 300:390] = angvel.view(len(envs_idx), -1)

            # Joint positions (29)
            joint_pos = self.dof_pos_data[t_idx]
            joint_pos += torch.randn_like(joint_pos) * self.command_cfg.get("joint_pos_noise_std", 0.01)
            self.commands[envs_idx, 390:419] = joint_pos

            # Joint velocities (29)
            joint_vel = self.dof_vel_data[t_idx]
            joint_vel += torch.randn_like(joint_vel) * self.command_cfg.get("joint_vel_noise_std", 0.01)
            self.commands[envs_idx, 419:448] = joint_vel

        
    def step(self, actions, is_train=True):
        self.scene.clear_debug_objects()
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        new_values = torch.tensor([0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
                   0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0], device='cuda:0')
        # target_dof_pos[:, -14:] = new_values
        # print(target_dof_pos, self.motor_dofs)
        self.robot.control_dofs_position(target_dof_pos, self.dof_ids)
        self.scene.step()

        # update buffers
        
        self.dof_pos = self.robot.get_dofs_position(self.dof_ids)
        self.dof_vel = self.robot.get_dofs_velocity(self.dof_ids)
    
        self.link_pos = self.robot.get_links_pos()
        self.link_quat = self.robot.get_links_quat()[..., [1, 2, 3, 0]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        self.link_lin_vel = self.robot.get_links_vel()
        self.link_ang_vel = self.robot.get_links_ang()

        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        self.scene.draw_debug_frames(pose_to_matrix(self.robot.get_pos(),self.robot.get_quat()))

        # resample commands, it is a variable that holds the indices of environments that need to be resampled or reset.
        envs_idx = (
            (self.episode_length_buf % int(self.action_seq_length) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if is_train:
            self._sample_commands(envs_idx)
            # # Idxs with probability of 5% to sample random commands
            random_idxs_1 = torch.randperm(self.num_envs)[:int(self.num_envs * 0.05)]
            self._sample_commands(random_idxs_1)

        # print("base euler",self.base_euler[:])
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > np.deg2rad(self.env_cfg["termination_criteria_roll"])
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > np.deg2rad(self.env_cfg["termination_criteria_roll"])
        self.reset_buf |= torch.abs(self.base_euler[:, 2]) > np.deg2rad(self.env_cfg["termination_criteria_pitch"])
        # self.reset_buf |= torch.abs(self.robot.get_pos()[:, 2]) < self.env_cfg["termination_criteria_pitch"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        idx_to_reset = self.reset_buf.nonzero(as_tuple=False).flatten()
        if idx_to_reset.numel() > 0:
            self.reset_idx(idx_to_reset)

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.obs_buf = torch.cat(
            [
                self.link_pos.reshape(self.num_envs, -1),
                self.link_quat.reshape(self.num_envs, -1),
                self.link_lin_vel.reshape(self.num_envs, -1),
                self.link_ang_vel.reshape(self.num_envs, -1),
                self.dof_pos.reshape(self.num_envs, -1),
                self.dof_vel.reshape(self.num_envs, -1),
                self.commands ,  # 5
            ],
            axis=-1,
        )
        
        # Reset jump command
        self.episode_length_buf += 1
        self.episode_length_buf[self.episode_length_buf == self.action_seq_length] = 0

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf
    
    def get_dummy_observations(self):
        return np.zeros([1,self.num_obs*2])
    
    def get_dummy_actions(self):
        return np.zeros([1,self.num_actions])

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
            dofs_idx_local=self.dof_ids,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        # self.base_lin_vel[envs_idx] = 0
        # self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / (self.action_seq_length*self.dt)
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._sample_commands(envs_idx)
        
    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None
    
    ##########################################
    ##########################################
    def wrap_to_pi(self,angles):
        angles %= 2*3.1415
        angles -= 2*3.1415* (angles > 3.1415)
        return angles

    def quat_rotate_inverse(self,q, v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c
    
    def quat_apply(self,a, b):
        shape = b.shape
        a = a.reshape(-1, 4)
        b = b.reshape(-1, 3)
        xyz = a[:, :3]
        t = xyz.cross(b, dim=-1) * 2
        return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)
    
    def find_rigid_body_indice(self, body_name):
        for link in self.robot.links:
            flag = False
            if body_name in link.name:
                return link.idx - self.robot.link_start
    
    def find_joint_indice(self, joint_name):
        for joint in self.robot.joints:
            flag = False
            if joint_name in joint.name:
                return joint.idx - 1
    
    def get_axis_params(value, axis_idx, x_value=0., dtype=np.float64, n_dims=3):
        zs = np.zeros((n_dims,))
        assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
        zs[int(axis_idx)] = 1.
        params = np.where(zs == 1., value, zs)
        params[0] = x_value
        return list(params.astype(dtype))
            
    ########################### PENALTY REWARDS ###########################
    
    def _reward_tracking_link_pos(self):
        # print(self.link_pos.shape,self.commands[:, 0:90].shape)
        error = torch.sum(torch.square(self.commands[:, 0:90] - self.link_pos.flatten(start_dim=1)), dim=1)
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_link_quat(self):
        error = torch.sum(torch.square(self.commands[:, 90:210] - self.link_quat.flatten(start_dim=1)), dim=1)
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_link_lin_vel(self):
        error = torch.sum(torch.square(self.commands[:, 210:300] - self.link_lin_vel.flatten(start_dim=1)), dim=1)
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_link_ang_vel(self):
        error = torch.sum(torch.square(self.commands[:, 300:390] - self.link_ang_vel.flatten(start_dim=1)), dim=1)
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_dof_pos(self):
        error = torch.sum(torch.square(self.commands[:, 390:419] - self.dof_pos.flatten(start_dim=1)), dim=1)
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_dof_vel(self):
        error = torch.sum(torch.square(self.commands[:, 419:448] - self.dof_vel.flatten(start_dim=1)), dim=1)
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])
    
    def _reward_tracking_survival(self):
        return self.episode_length_buf/self.action_seq_length
