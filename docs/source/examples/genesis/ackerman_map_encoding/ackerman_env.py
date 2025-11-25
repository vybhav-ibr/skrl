import torch
import math
import genesis as gs
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, quat_to_R
from collections import OrderedDict
from huggingface_hub import snapshot_download
import time
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
import torch

def quat_to_rotation_matrix(quat):
    w, x, y, z = quat
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)]),
        torch.stack([    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)]),
        torch.stack([    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)])
    ])
    return R

def get_two_positions(pos, quat, distance=0.25):
    """
    pos: tensor of shape (3,)
    quat: tensor of shape (4,) in (w, x, y, z) format
    distance: scalar distance between start and end
    """
    # Get rotation matrix
    R = quat_to_rotation_matrix(quat)
    # Choose a reference direction in local frame — e.g. Z+ (0,0,1)
    local_dir = torch.tensor([0.0, 0.0, 1.0], dtype=pos.dtype, device=pos.device)
    # Rotate it into world frame
    world_dir = R @ local_dir
    # Normalize (just in case numerical drift)
    world_dir = torch.nn.functional.normalize(world_dir, dim=0)
    
    half = distance / 2.0
    start = pos - world_dir * half
    end = pos + world_dir * half
    return start.detach().cpu().numpy(), end.detach().cpu().numpy()


class AKEnv:
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
            vis_options=gs.options.VisOptions(
                # rendered_envs_idx=list(range(self.num_envs//2,self.num_envs//2+4)),
                show_link_frame=True,),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False,
            ),
            show_viewer=show_viewer,
            show_FPS=False
        )

        # add plain
        self.plane=self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="/home/vybhav/gs_gym_wrapper_reference/ackerman.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                links_to_keep=self.env_cfg["links_to_keep"]
            ),
            visualize_contact=True
        )
        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                scale=0.05,
                fixed=False,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.5, 0.5),
                ),
            ),
        )
        # build
        self.scene.build(n_envs=num_envs)#,env_spacing=(2,2), n_envs_per_row=num_envs)

        # names to indices
        self.dof_names=env_cfg["default_dof_properties"].keys()
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.dof_names]
        
        # PD control parameters
        for dof_name,dof_properties in env_cfg["default_dof_properties"].items():
            # print("setting kp for :",dof_name)
            joint=self.robot.get_joint(dof_name)
            dof_idx=joint.dofs_idx_local
            self.robot.set_dofs_kp([dof_properties[1]],dof_idx)
            self.robot.set_dofs_kv([dof_properties[2]],dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums,self.last_episode_sums = dict(), dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_dof_properties"][name][0] for name in self.dof_names],
            device=gs.device,
            dtype=gs.tc_float,
        )
        
        self.robot.set_dofs_position(self.default_dof_pos.repeat(self.num_envs,1),self.motors_dof_idx)
        
        self.obj_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.obj_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        
        self.basket_pos=torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.chosen_object=torch.zeros((self.num_envs), device=gs.device, dtype=gs.tc_int)
        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        # self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        # self.last_rewards = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
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
        self.dof_pos = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_commands = torch.zeros_like(self.commands)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        
        # self.arm_pick_pos=torch.tensor(
        #     [self.env_cfg["arm_pick_pos"][name][0] for name in self.arm_names],
        #     device=gs.device,
        #     dtype=gs.tc_float,
        # )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        
        # self.left_eef_link_idx=self.robot.get_link("L_hand_base").idx_local
        # self.right_eef_link_idx=self.robot.get_link("R_hand_base").idx_local
        # self.right_eef_link_name="R_hand_base"
        # self.left_eef_link_name="L_hand_base"
        # self.arm_links=[f"Link{k}" for k in range(1,9)]
        # for link in self.robot.links:
        #     print(link.name)
        dummy_depth= torch.zeros((64, 64,3))
        # dummy_depth_small= torch.zeros((128, 128,1))
        # dummy_image= torch.zeros((512, 512,3))
        self.obs_space = {
            # "ang_vel":self.base_ang_vel[0] * self.obs_scales["ang_vel"],  # 3
            # "back_depth":dummy_depth_small,
            # "commands":self.commands[0] ,  # 3
            "base_ang_vel":self.base_ang_vel[0],
            "base_lin_vel":self.base_lin_vel[0],
            "base_pos":self.base_pos[0],
            "dof_pos":self.dof_pos[0],  # 12
            "dof_vel":self.dof_vel[0],  # 12
            "front_depth":dummy_depth,
            # "gripper_depth":dummy_depth,
            # "gripper_img":dummy_image,
            # "object_pos":self.obj_pos[0],
            # "object_quat":self.obj_quat[0],
            # "robot_base_pos":self.robot.get_pos()[0],
            # "robot_base_quat":self.robot.get_quat()[0],
            "taken_actions":self.actions[0],  # 12
        }
        # print("base_pos:", self.base_pos.shape)
        # print("base_ang_vel:", self.base_ang_vel.shape)
        # print("base_lin_vel:", self.base_lin_vel.shape)
        # print("dof_pos[:,2:-2]:", self.dof_pos.shape)
        # print("dof_vel[:,-2:]:", self.dof_vel.shape)
        # # print("dummy_depth:", self.dummy_depth.view(self.num_envs, -1).shape)
        # print("actions:", self.actions.shape)

        self.obs_buf= torch.zeros((self.num_envs, 12309), device=gs.device, dtype=gs.tc_float)
        
        self.all_envs_idx=torch.arange(0,self.num_envs,dtype=gs.tc_int)
        # self.eef_pos_object_threshold=reward_cfg["eef_pos_object_threshold"]
        # print("obs_buf_shape at init",self.obs_buf.shape)
        
        
        # self.left_gripper = next((link for link in self.robot.links if link.name == "Link7"), None)
        # self.right_gripper = next((link for link in self.robot.links if link.name == "Link8"), None)
        # # Assuming you have these variables:
        self.scene_entities={"robot":self.robot,
                             "plane":self.plane}
        
        self.max_base_pos_buffer=[0.0,0.0,0.0,0.0,0.0]
        self.running_max_base_pos=0.0
        
        self.dummy_depth= torch.zeros((self.num_envs,64, 64,3))
        # self.dummy_depth_small= torch.zeros((self.num_envs,128, 128,1))
        # self.dummy_image= torch.zeros((self.num_envs,512, 512,3))
        
        # self.scene.draw_debug_line(start=(0,0,0),end=(0,0,0.55),radius=0.25)
        # for _ in range(1000):
        #     self.scene.step()
    
    # def _random_quat_z(self,envs_idx):
    #     num_envs=envs_idx.shape[0]
    #     theta = torch.rand(num_envs) * 2 * torch.pi  # angle in [0, 2π)
    #     half_theta = theta / 2
    #     quat = torch.zeros((num_envs, 4))
    #     quat[:, 2] = torch.sin(half_theta)  # z
    #     quat[:, 3] = torch.cos(half_theta)  # w
    #     return quat
    def _random_quat_z(self, envs_idx):
        num_envs = envs_idx.shape[0]
        theta = torch.rand(num_envs) * 2 * torch.pi  # angle in [0, 2π)
        half_theta = theta / 2
        
        # Initialize quaternion array
        quat = torch.zeros((num_envs, 4))
        
        # Quaternion components for rotation around Z-axis
        quat[:, 0] = torch.cos(half_theta)  # w = cos(θ/2)
        quat[:, 1] = 0  # x = sin(θ/2)
        quat[:, 2] = 0                    # y = 0 (for XY-plane rotation)
        quat[:, 3] = torch.sin(half_theta)                    # z = 0 (for XY-plane rotation)
        
        return quat

    # def _sample_TF_command(self,envs_idx,cond_index=None):
    def _random_pos_near_base(self, envs_idx, scale):
        """
        Samples new positions at a certain (scaled) distance from robot base positions.

        Args:
            envs_idx (torch.Tensor or list): Indices of envs to sample positions for.
            scale (float): Scaling factor for sampling distance.

        Returns:
            torch.Tensor: Sampled positions of shape (k, 3)
        """
        selected_poses = self.robot.get_pos(envs_idx=envs_idx)  # shape (k, 3)
        k = selected_poses.shape[0]

        # Random unit vectors
        random_dirs = torch.randn(k, 3)
        random_dirs = random_dirs / torch.norm(random_dirs, dim=1, keepdim=True)

        offset = random_dirs * 1.0 * scale
        sampled_pos = selected_poses + offset

        # Clamp the Y-coordinate (index 1) to [-0.5, 0.5]
        sampled_pos[:, 1] = torch.clamp(sampled_pos[:, 1], min=-0.5, max=0.5)

        return sampled_pos


    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 2] = 1.0
        self.max_base_pos_buffer.extend(self.robot.get_pos(envs_idx)[:, 0].cpu().tolist())
        self.max_base_pos_buffer.sort(reverse=True)
        self.running_max_base_pos=sum(self.max_base_pos_buffer[:6])/5
        sample_scale=5.0
        self.commands[envs_idx,:] = self._random_pos_near_base(envs_idx=envs_idx,scale=sample_scale)
        
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else actions
        target_dof_vel_front = exec_actions[:,:2] * self.env_cfg["action_scale_vel"]
        target_dof_vel_back = exec_actions[:,-2:] * self.env_cfg["action_scale_vel"]
        # print(target_dof_pos[:,:2], self.motors_dof_idx[:2])
        target_dof_pos_front = exec_actions[:,2:-2] * self.env_cfg["action_scale_pos"]
        # print(list(self.dof_names)[:2],list(self.dof_names)[2:-2],list(self.dof_names)[-2:])
        # print(target_dof_vel_front[0],target_dof_vel_back[0])
        self.robot.set_dofs_velocity(target_dof_vel_front, self.motors_dof_idx[:2])
        self.robot.set_dofs_position(target_dof_pos_front, self.motors_dof_idx[2:-2])
        self.robot.set_dofs_velocity(target_dof_vel_back, self.motors_dof_idx[-2:])
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
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx[2:-2])
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx[:2]+self.motors_dof_idx[-2:])

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        # print("resetting these",self.reset_buf)
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        # self.individual_reward_dict = {}
        # print(self.reward_functions.items())
        for name, reward_func in self.reward_functions.items():
            # print(name,":",self.reward_scales[name],"",reward_func())
            rew = reward_func() * self.reward_scales[name]
            # print(name,"!:!",rew)
            self.rew_buf += rew
            # print("reward_buf",self.rew_buf)
            self.episode_sums[name] += rew
        
        self.obs_buf = torch.cat(
            [
                # self.dummy_depth_small.view(self.num_envs, -1),
                # self.commands, # 3
                self.base_pos,
                self.base_ang_vel,
                self.base_lin_vel,
                self.dof_pos ,
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.dummy_depth.view(self.num_envs, -1),
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_commands[:] = self.commands[:]
        self.last_episode_sums = self.episode_sums

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras
    
    def get_dummy_observations(self):
        return OrderedDict(self.obs_space)
    
    def get_dummy_actions(self):
        return np.zeros([1, self.num_actions])

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = torch.tensor([0,0],dtype=gs.tc_float).repeat(envs_idx.shape[0],1)
        self.dof_vel[envs_idx] = torch.tensor([0,0,0,0],dtype=gs.tc_float).repeat(envs_idx.shape[0],1)
        self.robot.set_dofs_velocity(self.dof_vel[envs_idx,:2], self.motors_dof_idx[:2],envs_idx=envs_idx)
        self.robot.set_dofs_position(self.dof_pos[envs_idx], self.motors_dof_idx[2:-2],zero_velocity=True,envs_idx=envs_idx)
        self.robot.set_dofs_velocity(self.dof_vel[envs_idx,-2:], self.motors_dof_idx[-2:],envs_idx=envs_idx)
        # print("sleeping",envs_idx.tolist())
        
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
        # for key in self.last_episode_sums.keys():
        #     self.last_episode_sums[key][envs_idx] = 0.0
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
        self.target.set_pos(self.commands[envs_idx, 0:3],envs_idx=envs_idx)
        
    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None 
    
    def _reward_goto(self):
        base_pos = self.robot.get_pos().squeeze(1)  # Shape: [num_envs, 3]
        
        # Get target positions from command
        target_pos = self.commands[:, 0:3]          # Shape: [num_envs, 3]
        
        # Compute L2 distance (Euclidean) to goal for all environments
        robot_to_goal_dist = torch.norm(base_pos - target_pos, dim=-1)  # Shape: [num_envs]
        reward=1.0 - torch.tanh(robot_to_goal_dist)
        print("_reward_goto:", reward)
        return reward

    def _reward_high_joint_force(self):
        joint_forces = self.robot.get_dofs_force(self.motors_dof_idx)
        joint_kp= self.robot.get_dofs_kp(self.motors_dof_idx)
        reward = torch.abs(torch.sum(joint_forces/joint_kp, dim=1))
        # print("_reward_high_joint_force:", reward)
        return reward