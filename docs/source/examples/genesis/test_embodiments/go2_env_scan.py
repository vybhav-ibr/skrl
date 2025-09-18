import torch
import math
import genesis as gs
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from collections import OrderedDict

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2EnvScan:
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
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1)),
                                                 show_link_frame=False),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        for i in range(16):
            for j in range(4):
                self.scene.add_entity(gs.morphs.Box(
                pos=(2.5,i-7.5,j+0.45),
                size=(0.25,0.25,0.25),
                fixed=True
            ))
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True,visualization=False))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                links_to_keep=["Head_upper"],
                # fixed=True
            ),
        )

        # build
        link=self.robot.get_link("Head_upper")
        self.cams = [self.scene.add_camera(GUI=True, fov=70,env_idx=i,res=(320,320)) for i in range(num_envs)]
        T=np.array([[  0.00,   0.00,  -1.00,   0.00],
                    [ -1.00,   0.00,   0.00,   0.00],
                    [  0.00,   1.00,   0.00,   0.00],
                    [  0.00,   0.00,   0.00,   1.00]])
        for cam in self.cams:
            cam.attach(link, T)

        self.scene.build(n_envs=num_envs,n_envs_per_row=num_envs,env_spacing=(1,1))
        
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
        # self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)

        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.cam_seq_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
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
        
        ################################
        clouds = [] 

        for icam, cam in enumerate(self.cams):
            cam.move_to_attach()
            cam.start_recording()
            theta = torch.deg2rad(torch.tensor(self.cam_seq_buf[icam]*15 -30))
            print("theta os:",theta)
            c = torch.cos(theta)
            s = torch.sin(theta)

            rot_y = torch.tensor([
                [ c, 0,  s, 0],
                [ 0, 1,  0, 0],
                [-s, 0,  c, 0],
                [ 0, 0,  0, 1]
            ], dtype=torch.float32)
            T=cam.get_transform() @ rot_y
            cam.set_pose(transform=T)
            # print("T matrix",cam.get_transform().detach().cpu().numpy(),cam.get_transform().detach().cpu().numpy().shape)
            # self.scene.draw_debug_frame(cam.get_transform().detach().cpu().numpy()[0])
            single_cloud = cam.render_pointcloud()[0]
            if self.cam_seq_buf[icam] == 4:
                self.cam_seq_buf[icam] = 0
            else:
                self.cam_seq_buf[icam] += 1
            clouds.append(single_cloud)  # just append each (100, 100, 3) array

        cloud = np.stack(clouds, axis=0)
        print("point cloud shape:", cloud.shape)
        self.obs_space = {
                "ang_vel":self.base_ang_vel[0] * self.obs_scales["ang_vel"],  # 3
                "commands":self.commands[0] * self.commands_scale,  # 3
                "dof_diff":(self.dof_pos[0] - self.default_dof_pos[0]) * self.obs_scales["dof_pos"],  # 12
                "dof_vel":self.dof_vel[0] * self.obs_scales["dof_vel"],  # 12
                "front_cloud":cloud[0],
                "proj_gravity":self.projected_gravity[0],  # 3
                "taken_actions":self.actions[0],  # 12
        }
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs+307200), device=gs.device, dtype=gs.tc_float)
        ################################
        self.step_count=0

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        self.step_count+=1
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
        clouds = [] 

        for icam, cam in enumerate(self.cams):
            cam.move_to_attach()
            # if self.step_count%10==0:
            theta = torch.deg2rad(torch.tensor(self.cam_seq_buf[icam]*15 -30))
            c = torch.cos(theta)
            s = torch.sin(theta)
            
            # rot_x = torch.tensor([
            #     [1, 0,  0, 0],
            #     [0, c, -s, 0],
            #     [0, s,  c, 0],
            #     [0, 0,  0, 1]
            # ], dtype=torch.float32)
            
            rot_y = torch.tensor([
                [ c, 0,  s, 0],
                [ 0, 1,  0, 0],
                [-s, 0,  c, 0],
                [ 0, 0,  0, 1]
            ], dtype=torch.float32)

            T=cam.get_transform() @ rot_y
            cam.set_pose(transform=T)
            
            single_cloud = cam.render_pointcloud()[0]
            cam.render(rgb=True)
            if self.cam_seq_buf[icam] == 4:
                self.cam_seq_buf[icam] = 0
            else:
                self.cam_seq_buf[icam] += 1
            clouds.append(single_cloud)  # just append each (100, 100, 3) array
            if self.step_count==100:
                cam.stop_recording()

        cloud = np.stack(clouds, axis=0)
        # print("!!!!!!!!\n!!!!!!!!!!!!\n!!!!!!!!!!cloud shape:",cloud.shape)
        # print(10/0)
        # self.obs_buf = {
        #         "front_cloud":cloud,
        #         "ang_vel":self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
        #         "proj_gravity":self.projected_gravity,  # 3
        #         "commands":self.commands * self.commands_scale,  # 3
        #         "dof_diff":(self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
        #         "dof_vel":self.dof_vel * self.obs_scales["dof_vel"],  # 12
        #         "dof_vel":self.actions,  # 12
        # }
        # print("base_ang_vel:", self.base_ang_vel.shape)  # [num_envs, 3]
        # print("commands:", self.commands.shape)          # [num_envs, num_commands]
        # print("dof_pos - default_dof_pos:", (self.dof_pos - self.default_dof_pos).shape)  # [num_envs, 12]
        # print("dof_vel:", self.dof_vel.shape)            # [num_envs, 12]
        # print("cloud:", torch.tensor(cloud).view(self.num_envs, -1).shape)       # shape depends on `cloud`, probably [num_envs, N] or [N]
        # print("projected_gravity:", self.projected_gravity.shape)  # [num_envs, 3]
        # print("actions:", self.actions.shape)            # [num_envs, 12]

        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                torch.tensor(cloud).view(self.num_envs, -1),
                self.projected_gravity,  # 3
                self.actions,  # 12
            ],
            axis=-1,
        )
        # print("obs buffer shape is",self.obs_buf.shape)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_dummy_observations(self):
        # for key,value in self.obs_buf.items():
        #     print("obs_buffer_entry:",key,value.shape)
        return OrderedDict(self.obs_space)
    
    def get_dummy_actions(self):
        return np.zeros([1,self.num_actions])
    
    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.cam_seq_buf[envs_idx]=0
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

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
