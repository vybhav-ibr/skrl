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


class APTTEnv:
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
                rendered_envs_idx=list(range(self.num_envs//2,self.num_envs//2+4))),
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
        self.plane=self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="skrl/docs/source/examples/genesis/assets/anymal_d/urdf/anymal_d_tt.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                links_to_keep=self.env_cfg["links_to_keep"]
            ),
            visualize_contact=False,
            # vis_mode="sdf"
        )
        self.ball=self.scene.add_entity(
            gs.morphs.Sphere(
                radius=(0.02),
                fixed=True,
                collision=False ,
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(225/255, 165/225, 0.0),
                ),
            ),
        )
        # build
        self.scene.build(n_envs=num_envs,env_spacing=(2,2), n_envs_per_row=num_envs)

        # names to indices
        self.dof_names=env_cfg["default_dof_properties"].keys()
        self.arm_names=[f'joint{k}' for k in range(1,7)]
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.dof_names]
        self.arm_dof_idx = [self.robot.get_joint(name).dof_start for name in self.arm_names]

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
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.last_commands = torch.zeros_like(self.commands)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        
        self.arm_pick_pos=torch.tensor(
            [self.env_cfg["arm_pick_pos"][name][0] for name in self.arm_names],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        
        self.eef_link_idx=self.robot.get_link("bat").idx_local
        self.eef_link_name="bat"
        self.arm_links=[f"Link{k}" for k in range(1,7)]
        self.arm_links.append(self.eef_link_name)
        # for link in self.robot.links:
        #     print(link.name)
        # dummy_depth= torch.zeros((512, 512,1))
        # dummy_depth_small= torch.zeros((128, 128,1))
        # dummy_image= torch.zeros((512, 512,3))
        self.obs_space = {
            # "ang_vel":self.base_ang_vel[0] * self.obs_scales["ang_vel"],  # 3
            # "back_depth":dummy_depth_small,
            "commands":self.commands[0] ,  # 3
            "dof_diff":(self.dof_pos[0] - self.default_dof_pos[0]) * self.obs_scales["dof_pos"],  # 12
            "dof_vel":self.dof_vel[0] * self.obs_scales["dof_vel"],  # 12
            # "front_depth":dummy_depth_small,
            # "gripper_depth":dummy_depth,
            # "gripper_img":dummy_image,
            "object_pos":self.obj_pos[0],
            "object_quat":self.obj_quat[0],
            "robot_base_pos":self.robot.get_pos()[0],
            "robot_base_quat":self.robot.get_quat()[0],
            "taken_actions":self.actions[0],  # 12
        }
        self.obs_buf= torch.zeros((self.num_envs, 75), device=gs.device, dtype=gs.tc_float)
        
        self.all_envs_idx=torch.arange(0,self.num_envs,dtype=gs.tc_int)
        self.eef_pos_object_threshold=reward_cfg["eef_pos_object_threshold"]
        # print("obs_buf_shape at init",self.obs_buf.shape)
        
        
        self.left_gripper = next((link for link in self.robot.links if link.name == "Link7"), None)
        self.right_gripper = next((link for link in self.robot.links if link.name == "Link8"), None)
        # # Assuming you have these variables:
        self.scene_entities={"robot":self.robot,
                             "plane":self.plane}
        
        self.max_base_pos_buffer=[0.0,0.0,0.0,0.0,0.0]
        self.running_max_base_pos=0.0
    
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
    def _random_pos_near_base(self, envs_idx, scale=1.0):
        """
        Samples new end-effector positions near the robot base, within specified axis-aligned ranges.

        Args:
            envs_idx (torch.Tensor or list): Indices of envs to sample positions for.
            scale (float): Scaling factor for offset from base position.
            x_range (tuple): (min_x, max_x) range relative to base position.
            y_range (tuple): (min_y, max_y) range relative to base position.
            z_range (tuple): (min_z, max_z) range relative to base position.

        Returns:
            torch.Tensor: Sampled positions of shape (k, 3)
        """
        base_pos = self.robot.get_pos(envs_idx=envs_idx)  # shape (k, 3)
        k = base_pos.shape[0]

        # Random offsets in range [-1, 1]
        random_offsets = torch.rand(k, 3) * 2 - 1  # uniform in [-1, 1]
        x_range,y_range,z_range=self.command_cfg["eef_pos"][0],self.command_cfg["eef_pos"][1],self.command_cfg["eef_pos"][2]
        # Scale the random offsets based on the given axis ranges and global scale
        ranges = torch.tensor([
            [x_range[0], x_range[1]],
            [y_range[0], y_range[1]],
            [z_range[0], z_range[1]]
        ], device=base_pos.device)

        # Compute half-ranges (for symmetric scaling around 0)
        half_ranges = (ranges[:, 1] - ranges[:, 0]) / 2.0  # (3,)
        
        # Apply scaling
        scaled_offsets = random_offsets * half_ranges * scale  # (k, 3)

        # Sampled position = base position + offset
        sampled_pos = base_pos + scaled_offsets

        # Clamp final position within [base + min_range, base + max_range]
        min_bounds = base_pos + ranges[:, 0]
        max_bounds = base_pos + ranges[:, 1]

        sampled_pos = torch.max(sampled_pos, min_bounds)
        sampled_pos = torch.min(sampled_pos, max_bounds)

        return sampled_pos

    def _resample_commands(self, envs_idx):

        sample_scale=0.5
        self.commands[envs_idx, 0:3] = self._random_pos_near_base(envs_idx=envs_idx,scale=sample_scale)
        # print(torch.max(self.commands[envs_idx, 2]))
        # self.ball.set_pos(self.commands[envs_idx, 0:3],envs_idx=envs_idx)
        self.commands[envs_idx,3]=10.0


    def _check_collisions(self, entity, env_indices, exclude_collision):
        """
        Returns a tensor of env indices where the robot's contacts are either:
        - All within the allowed collision list
        - Or there are no contacts at all

        Parameters
        ----------
        robot_entity : RigidEntity
            The robot entity to get contacts from.
        env_indices : Tensor[int] (e.g. shape (N,))
            The specific environment indices to check (subset of total envs).
        exclude_collision : dict
            Dict with key "exclude collision" mapping to list of [link_name, entity_name] pairs to allow.

        Returns
        -------
        valid_envs : Tensor[int]
            Tensor of env indices (subset of `env_indices`) where only allowed contacts occurred.
        """
        allowed_pairs = set(tuple(pair) for pair in exclude_collision)

        contact_info = entity.get_contacts(exclude_self_contact=True)
        links = entity.links

        link_a_ids = contact_info['link_a']       # shape: (n_envs, n_contacts)
        link_b_ids = contact_info['link_b']
        valid_mask = contact_info['valid_mask']   # shape: (n_envs, n_contacts)

        valid_envs = []
        # print("contact_info is:",contact_info)
        for env_idx in env_indices:
            all_contacts_allowed = True
            for contact_idx in range(valid_mask.shape[1]):
                if not valid_mask[env_idx, contact_idx]:
                    print(f"continueing:{env_idx}")
                    continue

                link_a = links[link_a_ids[env_idx, contact_idx]]
                link_b = links[link_b_ids[env_idx, contact_idx]]

                link_a_name = link_a.name
                link_b_name = link_b.name
                
                # print('link a ',link_a,link_a.entity.links)
                # entity_a_names=[link.name for link in link_a.entity.links]
                # entity_b_names=[link.name for link in link_b.entity.links]
                # entity_a = link_a.entity.name
                # entity_b = link_b.entity.name
                entity_a_entity=self.scene.rigid_solver.links[link_a_ids[env_idx, contact_idx]].entity
                entity_b_entity=self.scene.rigid_solver.links[link_a_ids[env_idx, contact_idx]].entity
                # Identify the robot's link in the contact
                for key,value in self.scene_entities.items():
                    if entity.uid==entity_a_entity.uid:
                        entity_a_name=key
                    if entity.uid==entity_b_entity.uid:
                        entity_b_name=key
                if link_a in self.robot.links:
                    pair = (link_a_name, entity_b_name)
                elif link_b in self.robot.links:
                    pair = (link_b_name, entity_a_name)
                else:
                    continue  # not involving robot

                if pair not in allowed_pairs:
                    all_contacts_allowed = False
                    break  # no need to check more

            if all_contacts_allowed:
                valid_envs.append(False)
            else:
                valid_envs.append(True)

        return torch.tensor(valid_envs, dtype=torch.bool)
    
    def print_reset_causes(self,envs_idx):
        timeout_reset = self.episode_length_buf > self.max_episode_length
        roll_reset = torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_criteria_roll"]
        pitch_reset = torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_criteria_pitch"]
        height_reset = self.robot.get_pos()[:, 2] < self.env_cfg["termination_criteria_base_height"]

        # Combine them for actual reset
        self.reset_buf = timeout_reset | roll_reset | pitch_reset | height_reset

        # Print reasons for each env (if you have many envs, limit output)
        for i in range(self.reset_buf.shape[0]):
            if self.reset_buf[i]:
                reasons = []
                if timeout_reset[i]: reasons.append("timeout")
                if roll_reset[i]: reasons.append("roll")
                if pitch_reset[i]: reasons.append("pitch")
                if height_reset[i]: reasons.append("height")
                # print(f"Env {i} reset due to: {', '.join(reasons)}")

        # Count how many resets per cause
        print("Reset counts:")
        print(f"  Timeout: {timeout_reset.sum().item()}")
        print(f"  Roll:    {roll_reset.sum().item()}")
        print(f"  Pitch:   {pitch_reset.sum().item()}")
        print(f"  Height:  {height_reset.sum().item()}")

        
    def step(self, actions):
        # time.sleep(5)
        # self._update_obj_pos()
        # self._update_basket_pos()
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
        # envs_idx = (
        #     (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
        #     .nonzero(as_tuple=False)
        #     .reshape((-1,))
        # )
        # self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # print("timed_out",self.reset_buf)
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_criteria_roll"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_criteria_pitch"]
        self.reset_buf |= torch.abs(self.robot.get_pos()[:, 2]) < self.env_cfg["termination_criteria_base_height"]
        # self.reset_buf |= torch.abs(self._check_collisions(self.robot,np.arange(0,self.num_envs),self.env_cfg["contact_exclusion_pairs"]))
        
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
        
        # print("rew_buf",self.rew_buf)
        # for key,val in self.episode_sums.items():
        #     print(key,":!:",val.shape)
        # exit(0)
        
        # compute observations
        # print("#"*20)
        # print("obj_pos",self.obj_pos.shape)
        # print("obj_quat",self.obj_quat.shape)
        # print("robot_pos",self.robot.get_pos().shape)
        # print("robot_quat",self.robot.get_pos().shape)
        # print("#"*20)
        self.obs_buf = torch.cat(
            [
                # self.dummy_depth_small.view(self.num_envs, -1),
                self.commands, # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                # self.dummy_depth_small.view(self.num_envs, -1),
                # self.dummy_depth.view(self.num_envs, -1),
                # self.dummy_image.view(self.num_envs, -1),
                self.obj_pos,
                self.obj_quat,
                self.robot.get_pos(),
                self.robot.get_quat(),
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_commands[:] = self.commands[:]
        self.last_episode_sums = self.episode_sums

        # print("obs_buf_shape at step",self.obs_buf.shape)
        # print(self.obs_buf)
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras
    
    def get_dummy_observations(self):
        # for key,value in self.obs_buf.items():
        #     print("obs_buffer_entry:",key,value.shape)
        return OrderedDict(self.obs_space)
    
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
        # print("sleeping",envs_idx.tolist())
        # time.sleep(5)
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

        # num_reset = len(envs_idx)
        # random_x = torch.rand(num_reset, device=self.device) * 0.4 + 1.2  # 0.2 ~ 0.6
        # random_y = (torch.rand(num_reset, device=self.device) - 0.5) * 0.5  # -0.25 ~ 0.25
        # random_z = torch.ones(num_reset, device=self.device) * 0.025  +0.85# 0.15 ~ 0.15
        # random_pos = self.commands[envs_idx,0:3]

        # self.ball.set_pos(random_pos, envs_idx=envs_idx)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        # print(f"resampled for {envs_idx.detach().cpu().tolist()}")
        self.ball.set_pos(self.commands[envs_idx, 0:3],envs_idx=envs_idx)
        # for it,env_id in enumerate(envs_idx.cpu().tolist()):
        #     if self.ball[env_id] is not None:
        #         self.scene.clear_debug_object(self.ball[env_id])
        #     offset=self.make_env_offset(env_id)
        #     target_pos=self.commands[envs_idx[it], 4:7][0]-offset
        #     self.ball[env_id]=self.scene.draw_debug_sphere(pos=target_pos,radius=0.075)
        
    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        # print("obs_buf_shape",self.obs_buf.shape)
        return self.obs_buf, None

    def _get_contact_reward(self, entities, entity_indices, desired_names, undesired_names):
        """
        Compute contact cost per environment based on contact forces with desired and undesired entities,
        using a per-environment selection of the source entity.

        Args:
            entities: List of entities to select from.
            entity_indices: Tensor[int] of shape (n_envs,), indexing into `entities` per env.
            desired_names: List[str] of desired contact entity names.
            undesired_names: List[str] of undesired contact entity names.

        Returns:
            Tensor of shape (n_envs,) with contact reward per environment.
        """
        n_envs = self.num_envs
        # scene = self.scene

        # Initialize per-env cost array as a tensor
        cost_per_env = torch.zeros(n_envs, dtype=torch.float32)

        # Select the appropriate entity for each environment
        # selected_entities[i] is the entity for env i
        # print("entity_ind", entity_indices)
        selected_entities = [entities[i] for i in entity_indices.tolist()]

        def get_force_magnitudes(contact_info):
            """
            Compute contact force magnitudes (||force_a||) per contact.
            Returns shape: (n_envs, n_contacts) or (1, n_contacts) if not parallelized.
            """
            forces = contact_info.get("force_a", None)
            if forces is None:
                return None

            if 'valid_mask' in contact_info:
                # Multiply by valid_mask to zero out invalid contacts
                mask = contact_info['valid_mask']  # shape: (n_envs, n_contacts)
                return torch.norm(forces * mask.unsqueeze(-1), dim=-1)  # (n_envs, n_contacts)
            else:
                # Non-parallelized version: single env
                return torch.norm(forces, dim=-1, keepdim=False)[None]  # (1, n_contacts)

        def accumulate(entities, sign):
            """
            For each environment and each named target entity:
            - Get contacts between the selected entity and the target
            - Compute force magnitudes
            - Accumulate into cost_per_env with the given sign (+1 or -1)
            """
            nonlocal cost_per_env

            for target_entity in entities:
                # Per-env contact force magnitude sums
                env_force_sums = torch.zeros(n_envs, dtype=torch.float32)

                for env_idx in range(n_envs):
                    selected_entity = selected_entities[env_idx]

                    # Get contact info between selected entity and the target
                    contact_info = selected_entity.get_contacts(with_entity=target_entity)

                    # Get per-contact force magnitudes
                    force_mags = get_force_magnitudes(contact_info)

                    if force_mags is not None:
                        # force_mags: shape (1, n_contacts) or (n_envs, n_contacts)
                        # For single env, extract the row (first index)
                        env_force_sums[env_idx] += torch.sum(force_mags[0])

                # Accumulate signed contact force magnitudes into cost
                cost_per_env += sign * env_force_sums

        # Accumulate positive cost for desired contacts
        if desired_names is not None and len(desired_names) > 0:
            accumulate(desired_names, +1)

        # Accumulate negative cost for undesired contacts
        if undesired_names is not None and len(undesired_names) > 0:
            accumulate(undesired_names, -1)

        return cost_per_env

    def _reward_survival(self):
        reward = torch.square(self.max_episode_length/(self.episode_length_buf+1))
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
        reward = (self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
        # print("_reward_base_height:", reward)
        return reward
