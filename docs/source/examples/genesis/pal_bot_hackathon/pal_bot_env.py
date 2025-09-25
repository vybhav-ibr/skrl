import torch
import math
import numpy as np
import genesis as gs
from collections import OrderedDict
from genesis.utils.geom import inv_quat, transform_quat_by_quat, quat_to_xyz, transform_by_quat
import torch.nn.functional as F
import random
# Palletizing environment with 7-DOF arm, suction toggle, and pallet-change action.
# Reward is conditional on command mode: pick, goto-home, place, change-pallet.
# Top-down pallet image (depth + segmentation) updates only after successful place.

import numpy as np
import torch
import time

class BoxFactory:
    def __init__(self, scene, n_envs, n_boxes, conveyer_bounds_lower, conveyer_bounds_upper, pallet_volume):
        self.n_envs = n_envs
        self.n_boxes = n_boxes
        self.scene=scene
        self.conveyer_bounds = np.array([conveyer_bounds_lower, conveyer_bounds_upper])  # shape (2, 3)
        self.pallet_volume = pallet_volume

        # Fixed box sizes
        self.box_sizes = np.array([
            np.random.uniform(0.2, 0.4, size=3)
            for _ in range(n_boxes)
        ], dtype=np.float32)

        # Entities per box type and initial positions of boxes
        self.entities = []
        self.initial_positions = []  # To store initial positions of the boxes
        for box_id in range(n_boxes):
            # Initial positions of the boxes when added to the scene
            box_size=self.box_sizes[box_id]
            initial_pos = (0, 0, -((box_id)*0.725+box_size[2]/2+0.720))
            ent = scene.add_entity(
                gs.morphs.Box(
                    pos=initial_pos,
                    size=box_size,
                    fixed=False,
                    # collision=False
                    
                ),
                vis_mode="collision"
            )
            self.entities.append(ent)
            self.initial_positions.append(initial_pos)  # Store initial position

        # Active box info per env
        self.active_box_indices = np.full(n_envs, -1, dtype=int)
        self.active_positions = np.zeros((n_envs, 3), dtype=np.float32)
        self.active_sizes = np.zeros((n_envs, 3), dtype=np.float32)
        
        # Additional attributes
        self.suction_on = [False for _ in range(n_envs)]
        self.placed_box_sizes = [[] for _ in range(n_envs)]

    def _get_underground_pos(self, box_id):
        """ Helper function to compute the underground position """
        pos = np.copy(self.initial_positions[box_id])
        # pos[2] = -(box_id + 1)
        return pos

    def appear(self, envs_idx, last_mode_mask):
        envs_idx = envs_idx.cpu().numpy()
        last_mode_mask_np = last_mode_mask.cpu().numpy()

        bounds_min = self.conveyer_bounds[0]
        bounds_max = self.conveyer_bounds[1]
        ranges = bounds_max - bounds_min

        chosen_box_indices = np.random.randint(0, self.n_boxes, size=len(envs_idx))
        positions = np.random.uniform(0, 1, size=(len(envs_idx), 3)) * ranges + bounds_min
        positions[:, 2] = 1.0
        if len(positions) > 0:
            print(positions)
            # exit(0)
        else:
            pass
            # exit(0)
            # fixed height above conveyer
        # time.sleep(2.5)
        print("appears",envs_idx)
        sizes = self.box_sizes[chosen_box_indices]

        for i, env_id in enumerate(envs_idx):

            
            chosen_box_id = chosen_box_indices[i]
            pos = positions[i]

            if last_mode_mask_np[i]:  # last was pick -> restore previous box
                prev_box_id = self.active_box_indices[env_id]
                if prev_box_id != -1:
                    underground_pos = self._get_underground_pos(prev_box_id)
                    self.entities[prev_box_id].set_pos(underground_pos[np.newaxis, :], envs_idx=np.array([env_id]))

                self.active_box_indices[env_id] = -1
                self.active_positions[env_id] = 0.0
                self.active_sizes[env_id] = 0.0

            else:
                # print("materualising object:",env_id,"with last box id",self.active_box_indices[env_id])
                # Place new box
                self.active_box_indices[env_id] = chosen_box_id
                self.active_positions[env_id] = pos
                self.active_sizes[env_id] = self.box_sizes[chosen_box_id]
                pose_to_set=pos[np.newaxis, :]
                # print("setting pos",pose_to_set,"for env",env_id)
                self.entities[chosen_box_id].set_pos(pose_to_set, envs_idx=np.array([env_id]))

        # exit(0)
        return torch.tensor(positions, dtype=gs.tc_float), torch.tensor(sizes, dtype=gs.tc_float)
        
    def suction_switch(self, suction_mask):
        """
        Turn suction on or off based on the input mask and contact check.

        Args:
            suction_mask (torch.BoolTensor): shape (n_envs,), True = attempt to enable suction
        """
        for env_id in range(self.n_envs):
            if suction_mask[env_id]:
                if self._check_contact_and_pos(env_id):  # <-- Check if contact is made and valid
                    if not self.suction_on[env_id]:  # Only add suction if it's not already on
                        self.add_suction_constraint(env_id)  # Add suction constraint
                        self.suction_on[env_id] = True
            else:
                if self.suction_on[env_id]:  # Only remove suction if it's currently on
                    self.remove_suction_constraint(env_id)  # Remove suction constraint
                    self.suction_on[env_id] = False
    
    def add_suction_constraint(self, env_id):
        """
        Add a suction constraint between the end-effector and the object.
        This works by adding a 'weld' or similar constraint to simulate suction.
        
        Args:
            env_id (int): The environment ID where the suction should be applied.
        """
        rigid = self.scene.sim.rigid_solver
        link_cube = np.array([self.entities[self.active_box_indices[env_id]].get_link("box_baselink").idx], dtype=gs.np_int)
        link_ur = np.array([self.ur.get_link("ee_virtual_link").idx], dtype=gs.np_int)
        
        # Assuming add_weld_constraint is the correct function for "suction"
        rigid.add_weld_constraint(link_cube, link_ur, envs_idx=[env_id])


    def remove_suction_constraint(self, env_id):
        """
        Remove the suction constraint between the end-effector and the object.
        
        Args:
            env_id (int): The environment ID where the suction constraint should be removed.
        """
        rigid = self.scene.sim.rigid_solver
        link_cube = np.array([self.entities[self.active_box_indices[env_id]].get_link("box_baselink").idx], dtype=gs.np_int)
        link_ur = np.array([self.ur.get_link("ee_virtual_link").idx], dtype=gs.np_int)
        
        # Assuming remove_weld_constraint is the correct function to remove the suction
        rigid.remove_weld_constraint(link_cube, link_ur, envs_idx=[env_id])

    def update_pos(self):
        """
        Update self.active_positions for all envs by querying entity positions
        """
        for env_id in range(self.n_envs):
            box_id = self.active_box_indices[env_id]
            if box_id == -1:
                continue  # no active box for this env

            # get_pos returns positions for all envs for this box entity
            all_positions = self.entities[box_id].get_pos().cpu().numpy()  # shape (n_envs, 3)
            self.active_positions[env_id] = all_positions[env_id]

    # def reset_pallet(self, envs_idx):
    #     """
    #     envs_idx: torch.BoolTensor of shape (n_envs,), where True means "reset this env"
    #     """
    #     envs_with_reset = torch.arange(self.n_envs)[~envs_idx]  # Extract env indices where mask is True

    #     print(f"reset_pallet called for envs: {envs_idx.tolist()}")

    #     for env_id in envs_with_reset:
    #         env_id = env_id.item()
    #         for box_id in range(self.n_boxes):
    #             underground_pos = self._get_underground_pos(box_id)
    #             self.entities[box_id].set_pos(underground_pos[np.newaxis, :], envs_idx=np.array([env_id]))

    #         self.active_box_indices[env_id] = -1
    #         self.active_positions[env_id] = 0.0
    #         self.active_sizes[env_id] = 0.0
    #         self.placed_box_sizes[env_id] = []
    
    def reset_pallet(self, envs_idx):
        all_envs_idx = torch.arange(self.n_envs)
        envs_with_reset = all_envs_idx[envs_idx]
        
        print(f"resetpallet_called for {envs_idx}")
        
        for reset_id in envs_with_reset:
            env_id = envs_idx[reset_id].item()
            
            for box_id in range(self.n_boxes):
                underground_pos = self._get_underground_pos(box_id)
                self.entities[box_id].set_pos(
                    underground_pos[np.newaxis, :], 
                    envs_idx=np.array([env_id])
                )
            
            self.active_box_indices[env_id] = -1
            self.active_positions[env_id] = 0.0
            self.active_sizes[env_id] = 0.0
            self.placed_box_sizes[env_id] = []


    def record_placed_box(self, envs_idx):
        for env_id in envs_idx:
            env_id = env_id.item()
            box_id = self.active_box_indices[env_id]
            if box_id != -1:
                size = self.box_sizes[box_id]
                self.placed_box_sizes[env_id].append(size)

    def get_pallet_fill_ratio(self):
        fill_ratios = []
        for box_list in self.placed_box_sizes:
            total_volume = sum([np.prod(box_size) for box_size in box_list])
            fill_ratio = total_volume / self.pallet_volume
            fill_ratios.append(min(fill_ratio, 1.0))  # Clamp to 1.0
        return torch.tensor(fill_ratios, dtype=torch.float32)

class PalletizeEnv(BoxFactory):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.device = gs.device
        self.dt = env_cfg.get("dt", 0.02)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # DOFs and actions: 7 arm DOFs + suction + change-pallet
        self.num_arm_dofs = 7
        self.num_actions = 9  # [7 joint targets, suction_toggle, change_pallet]
        self.action_scale = env_cfg.get("action_scale", 1.0)
        self.clip_actions = env_cfg.get("clip_actions", 1.0)

        # Command modes: 0=pick, 1=goto_home, 2=place, 3=change_pallet (action-driven)
        # For pick, commands carry [box_pos(3), box_quat(4), box_size(3)] = 10
        self.num_commands = 1 + 10  # [mode, pick_payload]
        self.commands = torch.zeros((num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.last_commands=torch.zeros_like(self.commands)

        self.last_mode=torch.zeros((num_envs,), device=self.device, dtype=gs.tc_float)
        self.is_first_sample=torch.ones((num_envs,), device=self.device, dtype=torch.bool)
        # Home pose
        self.home_dof_pos = torch.tensor(env_cfg["home_dof_pos"], device=self.device, dtype=gs.tc_float)  # (7,)

        # Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=env_cfg.get("substeps", 2)),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False,
                gravity=(0,0,-9.8)
            ),
            show_viewer=show_viewer,
        )

        # self.plane = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),vis_mode="collision")

        # Robot
        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device, dtype=gs.tc_float)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device, dtype=gs.tc_float)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="skrl/docs/source/examples/genesis/dex_bot/dex_bot.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                fixed=True,
                links_to_keep=env_cfg.get("links_to_keep", []),
            ),
            visualize_contact=False,
        )
        self.conveyer=self.scene.add_entity(
            gs.morphs.Box(
                pos=(-3,0,0.25),
                size=(5,2,0.5),
                fixed=True
            ),
        )
        self.box_rack=self.scene.add_entity(
            gs.morphs.Mesh(
                file="skrl/docs/source/examples/genesis/dex_bot/meshes/obj_rack.obj",
                pos=(0,0,-0.3),
                scale=0.01,
                fixed=True,
                convexify=False,
                decimate=False,
                visualization=True
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        # for i in range(50):
        #     self.scene.add_entity(gs.morphs.Box(
        #     pos=(0,0,-i-2.5),
        #     size=[random.sample(range(10, 20), 1),random.sample(range(10, 20), 1),0.5],
        #     fixed=True
        # ))
        
        # Pallet params and occupancy grid
        self.pallet_origin = torch.tensor(env_cfg["pallet_origin"], device=self.device, dtype=gs.tc_float)  # (3,)
        self.pallet_size = env_cfg["pallet_size"]
        self.pallet_volume=self.pallet_size[0]*self.pallet_size[0]*self.pallet_size[1]*self.pallet_size[2]
        self.box_factory=BoxFactory(self.scene,self.num_envs,
                                    env_cfg["num_boxes_per_env"],env_cfg["conveyer_bounds_lower"],
                                    env_cfg["conveyer_bounds_upper"],pallet_volume=self.pallet_volume)

        # Name mappings and DOF indices (expecting 7 actuated joints names provided)
        self.dof_names = env_cfg["default_dof_properties"].keys()  # list of 7 joint names
        self.motors_dof_idx = [self.robot.get_joint(n).dof_start for n in self.dof_names]

        self.cam_res = tuple(env_cfg.get("cam_res", (512, 512)))
        self.cam_height = float(env_cfg.get("cam_height", 2.0))  # meters above pallet center

        self.top_cams = []
        for i in range(num_envs):
            cam = self.scene.add_camera(
                res=self.cam_res, fov=90, GUI=True, env_idx=i
            )  # per-env camera
            self.top_cams.append(cam)

        # Allocate cached image buffer; two channels: depth and segmentation
        H, W = self.cam_res[1], self.cam_res[0]
        self.top_image = torch.zeros((num_envs, H, W, 2), device=self.device, dtype=gs.tc_float)

        # Update observation sizing to reflect camera-based image
        self.obs_img_flat = H * W * 2
        self.num_obs = 3 + 7 + 7 + 3 + 4 + 3 + self.obs_img_flat  # mode(3) + proprio + eef + box + image
        self.obs_buf = torch.zeros((num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        # print("obs buf size",self.obs_buf.shape)
        # Scene build
        self.scene.build(n_envs=num_envs, env_spacing=tuple(env_cfg.get("env_spacing", (4, 4))), n_envs_per_row=num_envs)
        
        # PD gains
        for jname, props in env_cfg["default_dof_properties"].items():
            joint = self.robot.get_joint(jname)
            dof_idx = joint.dofs_idx_local
            self.robot.set_dofs_kp([props[1]], dof_idx)
            self.robot.set_dofs_kv([props[2]], dof_idx)

        # EEF link (tool)
        self.eef_link_name = env_cfg.get("eef_link_name", None)
        if self.eef_link_name is None:
            raise ValueError("eef_link_name must be provided in env_cfg")
        self.eef_link_idx = self.robot.get_link(self.eef_link_name).idx_local

        #set cam origin, Top-down pose over pallet center for env i
        for cam in self.top_cams:           
            pal_center = self.pallet_origin.cpu().numpy().tolist()
            cam.set_pose(
                pos=(pal_center[0], pal_center[1], self.cam_height),
                lookat=(pal_center[0], pal_center[1], pal_center[2]),
                up=(1.0, 0.0, 0.0),  # roll to look straight down (adjust as needed)
            )
        # Pallet image cache (depth and segmentation), updated only on place success
        self.top_image = torch.zeros((num_envs, H, W, 2), device=self.device, dtype=gs.tc_float)

        # Box state per env
        self.box_size = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)  # (L, W, H)
        self.holding_box = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_int)
        self.suction_on = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_int)

        # Buffers
        self.dof_pos = torch.zeros((num_envs, self.num_arm_dofs), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.actions = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(num_envs, 1)

        self.rew_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_int)
        self.extras = {"observations": {}}

        # Obs configuration
        self.obs_scales = obs_cfg["obs_scales"]

        # Reward configuration
        self.reward_cfg = reward_cfg
        self.reward_scales = reward_cfg["reward_scales"]
        for k in self.reward_scales:
            self.reward_scales[k] *= self.dt

        # Default joint positions
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_dof_properties"][n][0] for n in self.dof_names], device=self.device, dtype=gs.tc_float
        )

        # Initialize robot state
        self.robot.set_dofs_position(self.default_dof_pos.repeat(num_envs, 1), self.motors_dof_idx)

        # Modes: start with pick by default
        self._set_mode(torch.zeros((num_envs,), device=self.device, dtype=gs.tc_int))

        # Home pose target buffer and threshold
        self.home_quat = self.base_init_quat.clone()
        self.home_pos = self.base_init_pos.clone()
        self.home_eef_tol = env_cfg.get("home_eef_tol", 0.02)
        
        # # for _
        # for _ in range(10000):
        #     time.sleep(1)
        #     self.scene.step()
        #     print("acc",self.box_factory.entities[0].get_links_acc()[0])
        #     print("pos",self.box_factory.entities[0].get_links_pos()[0])
        #     # self.scene.visualizer.update()
        # exit(0)

    def _set_mode(self, mode_tensor_int):
        # commands[:,0] stores mode as int; for pick, commands[:,1:11] carries [pos(3), quat(4), size(3)]
        self.commands[:, 0] = mode_tensor_int.float()

    def _get_mode(self):
        return self.commands[:, 0].long()

    # Utility to get one-hot mode
    def _mode_one_hot(self):
        # Modes: 0=pick, 1=goto_home, 2=place; clamp just in case
        mode = self._get_mode().clamp(0, 2)                # (N,)
        return F.one_hot(mode, num_classes=3).float()      # (N, 3)
    
    def _seed_pick_commands(self, envs_idx=None):
        # box_positions/quats/sizes: tensors shaped (k,3)/(k,4)/(k,3)
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self.device)
        else:
            envs_idx = torch.arange(self.num_envs, device=self.device)[envs_idx]
        self.commands[envs_idx, 0] = 0  # pick
        self.commands[envs_idx, 1:4],self.commands[envs_idx, 4:7] = self.box_factory.appear(envs_idx=envs_idx,last_mode_mask=self.last_mode)
        self.box_size[envs_idx] = self.commands[envs_idx, 4:7]

    def _eef_pose(self):
        eef_pos = self.robot.get_links_pos(self.eef_link_idx).squeeze(1)
        eef_quat = self.robot.get_links_quat(self.eef_link_idx).squeeze(1)
        return eef_pos, eef_quat

    def _eef_near(self, pos_a, pos_b, tol):
        return (torch.norm(pos_a - pos_b, dim=-1) <= tol).int()

    def _xy_in_pallet_bounds(self, xy):
        # xy: (N,2) in world frame; bounds are axis-aligned rectangle at pallet_origin on XY plane
        min_xy = self.pallet_origin[:2] - 0.5 * self.pallet_size_xy
        max_xy = self.pallet_origin[:2] + 0.5 * self.pallet_size_xy
        in_x = (xy[:, 0] >= min_xy[0]) & (xy[:, 0] <= max_xy[0])
        in_y = (xy[:, 1] >= min_xy[1]) & (xy[:, 1] <= max_xy[1])
        return (in_x & in_y)

    def _footprint_in_bounds(self, centers_xy, box_sizes):
        # Check if box footprint fits within pallet rectangle
        half = 0.5 * box_sizes[:, :2]
        min_xy = self.pallet_origin[:2] - 0.5 * self.pallet_size_xy
        max_xy = self.pallet_origin[:2] + 0.5 * self.pallet_size_xy
        min_c = centers_xy - half
        max_c = centers_xy + half
        ok_x = (min_c[:, 0] >= min_xy[0]) & (max_c[:, 0] <= max_xy[0])
        ok_y = (min_c[:, 1] >= min_xy[1]) & (max_c[:, 1] <= max_xy[1])
        return ok_x & ok_y

    def _render_top_image(self, env_mask):
        if env_mask.sum() == 0:
            return
        idx = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        # Depth normalize by max_stack_h

        self.top_image[idx, :, :, 0] = self.top_cams[idx].render(rgb=False,depth=True)[1]
        self.top_image[idx, :, :, 1] = self.top_cams[idx].render(rgb=False,segmentation=True)[2]

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # Reset DOFs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(self.dof_pos[envs_idx], self.motors_dof_idx, zero_velocity=True, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Reset buffers/state
        self.holding_box[envs_idx] = 0
        self.suction_on[envs_idx] = 0
        # self.pallet_height_grid[envs_idx] = 0.0
        # self.pallet_occ_grid[envs_idx] = 0.0
        self.top_image[envs_idx] = 0.0

        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # Default to pick mode; boxes must be seeded externally
        self._set_mode(torch.zeros((len(envs_idx),), device=self.device, dtype=gs.tc_int))

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._compute_obs(), None

    def _compute_obs(self):
        # Update core robot states
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        
        eef_pos, eef_quat = self._eef_pose()              # (N, 3)
        img_flat = self.top_image.view(self.num_envs, -1)
        mode_oh = self._mode_one_hot()  # 3-long one-hot for [pick, goto_home, place]
        # print("flat image from compute_obs:",img_flat.shape)
        obs = torch.cat(
            [
                mode_oh,  # 3
                self.dof_pos * self.obs_scales["dof_pos"],  # 7
                self.dof_vel * self.obs_scales["dof_vel"],  # 7
                eef_pos,  # 3
                eef_quat,  # 4
                self.box_size,  # 3
                img_flat,  # H*W*2 (depth, segmentation)
            ],
            dim=-1,
        )
        self.obs_buf[:] = obs
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf

    def _reward_pick(self):
        # Encourage EEF close to box target position in pick mode
        eef_pos, _ = self._eef_pose()
        target_pos = self.commands[:, 1:4]
        dist = torch.norm(eef_pos - target_pos, dim=-1)
        # rew = torch.exp(-self.reward_cfg.get("pick_pos_scale", 50.0) * dist)
        print("reward_pick:",dist)
        return dist

    def _reward_goto_home(self):
        # Penalize deviation from home DOF pose
        diff = torch.norm(self.dof_pos - self.home_dof_pos[None, :], dim=-1)
        return -diff

    def _reward_place(self):
        # EEF must be within pallet bounds and box footprint must fit
        eef_pos, _ = self._eef_pose()
        centers_xy = eef_pos[:, :2]
        in_xy = self._xy_in_pallet_bounds(centers_xy)
        fit = self._footprint_in_bounds(centers_xy, self.box_size)
        ok = in_xy & fit
        # Positive reward for OK, negative for violations with distance penalty
        dist_xy = torch.norm(
            torch.clamp(centers_xy - self.pallet_origin[:2][None, :], min=-1e3, max=1e3), dim=-1
        )
        base = torch.where(ok, torch.ones_like(dist_xy), -torch.exp(-self.reward_cfg.get("place_neg_scale", 2.0) * dist_xy))
        return base

    def _reward_change_pallet(self, change_trigger_mask):
        # Return how full the pallet is and punish accordingly: fill_ratio - 1 in [-1, 0], zero when full
        # Fill ratio computed from occupancy cells
        occ = (self.pallet_occ_grid > 0.5).float()
        fill_ratio = occ.view(self.num_envs, -1).mean(dim=-1)
        rew = (fill_ratio - 1.0)
        return torch.where(change_trigger_mask, rew, torch.zeros_like(rew))
    
    def step(self, actions):
        # Clip and apply action
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        joint_targets = self.actions[:, :self.num_arm_dofs]
        suction_cmd = (self.actions[:, 7] > 0.5).int()
        change_pallet_cmd = (self.actions[:, 8] > 0.5)

        target_dof_pos = joint_targets * self.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        self.box_factory.update_pos()
        self.scene.step()
        self.episode_length_buf += 1

        self.suction_on = suction_cmd

        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        mode = self._get_mode()
        eef_pos, _ = self._eef_pose()

        # Track which envs need reset
        success_resets = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # PICK success = close to target + suction on
        if (mode == 0).any():
            pick_mask = (mode == 0)
            is_first_mask = self.is_first_sample == True
            last_pick_mask = (self.last_mode != 0) | is_first_mask
            self._seed_pick_commands(last_pick_mask)
            target_pos = self.commands[:, 1:4]
            close = self._eef_near(eef_pos, target_pos, self.reward_cfg.get("pick_tol", 0.05))
            attach = (close & (self.suction_on > 0)).int()
            success_idx = torch.nonzero((pick_mask.int() & attach) > 0, as_tuple=False).squeeze(-1)
            if len(success_idx) > 0:
                self.holding_box[success_idx] = 1
                success_resets[success_idx] = True
            self.is_first_sample = False

        # GOTO_HOME success = close to home position while holding box
        if (mode == 1).any():
            goto_mask = (mode == 1)
            diff = torch.norm(self.dof_pos - self.home_dof_pos[None, :], dim=-1)
            arrived = (diff <= self.home_eef_tol).int()
            to_reset = (goto_mask.int() & arrived & (self.holding_box > 0)).int()
            idx = torch.nonzero(to_reset > 0, as_tuple=False).squeeze(-1)
            if len(idx) > 0:
                success_resets[idx] = True

        # PLACE success = in bounds, fits, suction off
        # place_success = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        if (mode == 2).any():
            place_mask = (mode == 2)
            centers_xy = eef_pos[:, :2]
            in_xy = self._xy_in_pallet_bounds(centers_xy)
            fit = self._footprint_in_bounds(centers_xy, self.box_size)
            ready_to_drop = (self.suction_on == 0) & (self.holding_box > 0)
            ok = place_mask & in_xy & fit & ready_to_drop.bool()
            if ok.any():
                self._render_top_image(ok)
                self.holding_box[ok] = 0
                # place_success = ok
                success_resets[ok] = True

        # # CHANGE_PALLET
        # seg = self.top_image[:, :, :, 1]
        # if hasattr(self, "seg_box_labels_mask"):
        #     occ_mask = self.seg_box_labels_mask
        # else:
        #     occ_mask = (seg > 0.5).float()
        # fill_ratio = occ_mask.view(self.num_envs, -1).mean(dim=-1)
        # full_mask = (fill_ratio >= 1.0 - 1e-6) & change_pallet_cmd
        # if full_mask.any():
        #     idx = torch.nonzero(full_mask, as_tuple=False).squeeze(-1)
        #     self.top_image[idx] = 0.0
        print("change_pallet_cmd",change_pallet_cmd)
        # self.box_factory.reset_pallet(change_pallet_cmd)

        # Timeout-based resets
        self.reset_buf = (self.episode_length_buf > self.max_episode_length)
        time_out_idx = self.reset_buf.nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        if len(time_out_idx) > 0:
            self.extras["time_outs"][time_out_idx] = 1.0

        # Combine resets from timeout and success
        self.reset_buf |= success_resets
        reset_idx = self.reset_buf.nonzero(as_tuple=False).reshape((-1,))
        if len(reset_idx) > 0:
            self.reset_idx(reset_idx)

        # Rewards
        self.rew_buf[:] = 0.0
        pick_mask = (mode == 0)
        goto_mask = (mode == 1)
        place_mask = (mode == 2)
        if pick_mask.any():
            r = self._reward_pick()
            self.rew_buf += self.reward_scales.get("pick", 1.0) * r * pick_mask.float()
        if goto_mask.any():
            r = self._reward_goto_home()
            self.rew_buf += self.reward_scales.get("goto_home", 1.0) * r * goto_mask.float()
        if place_mask.any():
            r = self._reward_place()
            self.rew_buf += self.reward_scales.get("place", 1.0) * r * place_mask.float()
        self.rew_buf += self.reward_scales.get("change_pallet", 1.0) * self.box_factory.get_pallet_fill_ratio() * change_pallet_cmd.float()

        if "action_rate" in self.reward_scales:
            self.rew_buf += self.reward_scales["action_rate"] * torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        if "high_joint_force" in self.reward_scales:
            joint_torques = self.robot.get_dofs_force(self.motors_dof_idx)
            self.rew_buf += self.reward_scales["high_joint_force"] * torch.sum(torch.square(joint_torques), dim=1)

        self.last_actions[:] = self.actions[:]
        self.last_commands[:] = self.commands[:]
        self.last_mode[:] = mode

        obs = self._compute_obs()
        return obs, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_dummy_observations(self):
        eef_pos, eef_quat = torch.zeros(3), torch.tensor([0, 0, 0, 1.0], dtype=torch.float32)
        dummy = OrderedDict(
            commands=torch.tensor([1.0, 0.0, 0.0]),      # default to "pick"
            dof_pos=torch.zeros(7),
            dof_vel=torch.zeros(7),
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            box_size=torch.zeros(3),
            pallet_top_image=torch.zeros(self.obs_img_flat),
        )
        return dummy
    
    def get_dummy_actions(self):
        return torch.zeros(self.num_actions)

    def get_privileged_observations(self):
        return None
