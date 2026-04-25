
    
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


# ---------------------------------------------------------------------------
# Gate placement helpers
# ---------------------------------------------------------------------------

def _sample_bimodal_coord(half_grid, mu_frac=0.55, sigma_frac=0.18):
    """
    Sample one coordinate from a 1-D inverted-bimodal:
        p(x) = 0.5 · N(−μ, σ) + 0.5 · N(+μ, σ)   clipped to [−half_grid, half_grid]

    The distribution has a trough at the origin and two peaks at ±μ so that
    gates cluster away from the drone spawn point at (0,0).

    Args:
        half_grid:  half the grid side length
        mu_frac:    peak offset as a fraction of half_grid  (default ≈ 0.55 → 55 % out)
        sigma_frac: spread as a fraction of half_grid       (default ≈ 0.18)
    """
    mu    = half_grid * mu_frac
    sigma = half_grid * sigma_frac
    sign  = random.choice((-1, 1))
    val   = random.gauss(sign * mu, sigma)
    return max(-half_grid, min(half_grid, val))


def sample_positions_and_orientations(grid_size, num_samples, min_distance=2.5):
    """
    Sample gate positions using the inverted-bimodal distribution so that
    gates are spread in the outer region of the arena and away from origin.

    Args:
        grid_size:    arena side length (metres)
        num_samples:  number of gates to place
        min_distance: minimum inter-gate distance (shrinks adaptively if needed)
    """
    def random_z_rotation():
        return random.uniform(0, 360)

    def check_distance(new_pos, existing_positions, min_dist):
        for pos in existing_positions:
            dist = math.sqrt((new_pos[0] - pos[0]) ** 2 + (new_pos[1] - pos[1]) ** 2)
            if dist < min_dist:
                return False
        return True

    half      = grid_size / 2
    positions = []
    orientations = []

    while len(positions) < num_samples:
        max_attempts       = 1000
        valid_position_found = False

        for _ in range(max_attempts):
            x = _sample_bimodal_coord(half)
            y = _sample_bimodal_coord(half)
            new_pos = (x, y, 0.0)

            if check_distance(new_pos, positions, min_distance):
                valid_position_found = True
                positions.append(new_pos)
                orientations.append((0.0, 0.0, random_z_rotation()))
                break

        if not valid_position_found:
            min_distance *= 0.9
            print(f"[gate_spawn] Warning: relaxed min_distance → {min_distance:.2f} m")

    return positions, orientations


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HoverEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs          = num_envs
        self.rendered_env_num  = min(10, self.num_envs)
        self.num_obs           = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions       = env_cfg["num_actions"]
        self.device            = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt               = 0.005  # 100 Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg     = env_cfg
        self.obs_cfg     = obs_cfg
        self.reward_cfg  = reward_cfg

        self.obs_scales    = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # ------------------------------------------------------------------ scene
        self.scene = gs.Scene(
            sim_options   = gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options = gs.options.ViewerOptions(
                max_FPS      = env_cfg["max_visualize_FPS"],
                camera_pos   = (3.0, 0.0, 3.0),
                camera_lookat = (0.0, 0.0, 1.0),
                camera_fov   = 40,
            ),
            vis_options  = gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options = gs.options.RigidOptions(
                dt                = self.dt,
                constraint_solver = gs.constraint_solver.Newton,
                enable_collision  = True,
                enable_joint_limit = True,
            ),
            show_viewer = show_viewer,
            show_FPS    = False,
        )

        # ground plane
        self.scene.add_entity(gs.morphs.Plane(collision=True))

        # gates — placed with inverted-bimodal distribution
        self.gate_positions, self.gate_orientations = sample_positions_and_orientations(
            grid_size   = env_cfg["grid_size"],
            num_samples = env_cfg["num_gates"],
        )
        print(f"[gate_spawn] first gate orientation: {self.gate_orientations[0]}")

        for gate_idx in range(env_cfg["num_gates"]):
            self.scene.add_entity(
                gs.morphs.Mesh(
                    file      = "/home/vybhav/gs_gym_wrapper_reference/skrl/docs/source/examples/genesis/assets/drone_gate.stl",
                    pos       = self.gate_positions[gate_idx],
                    euler     = self.gate_orientations[gate_idx],
                    convexify = False,
                    collision = True,
                    scale     = 0.25,
                    fixed=True,
                ),
                vis_mode = "collision",
            )

        # optional overhead camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res    = (640, 480),
                pos    = (3.5, 0.0, 2.5),
                lookat = (0, 0, 0.5),
                fov    = 30,
                GUI    = True,
            )

        # drone
        self.base_init_pos      = torch.tensor(self.env_cfg["base_init_pos"],  device=gs.device)
        self.base_init_quat     = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # depth camera sensor (forward-facing, body frame)
        sensor_kwargs = dict(
            entity_idx      = self.drone.idx,
            pos_offset      = (0.05, 0.0, 0.05),
            euler_offset    = (0.0, 0.0, 0.0),
            min_range       = 0.25,
            max_range       = 7.5,
            return_world_frame = False,
            draw_debug      = False,
        )
        self.lidar = self.scene.add_sensor(
            gs.sensors.DepthCamera(
                pattern = gs.sensors.DepthCameraPattern(res=(16, 16)),
                **sensor_kwargs,
            )
        )

        # contact sensor — used for collision termination
        self.contact_sensor = self.scene.add_sensor(
            gs.sensors.Contact(entity_idx=self.drone.idx)
        )

        # build
        self.scene.build(n_envs=num_envs)

        # ------------------------------------------------------------------ rewards
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name]  *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name]    = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # ------------------------------------------------------------------ buffers
        self.obs_buf          = torch.zeros((self.num_envs, self.num_obs),    device=gs.device, dtype=gs.tc_float)
        self.rew_buf          = torch.zeros((self.num_envs,),                 device=gs.device, dtype=gs.tc_float)
        self.reset_buf        = torch.ones ((self.num_envs,),                 device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,),              device=gs.device, dtype=gs.tc_int)

        self.actions      = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos      = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_euler     = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat     = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel  = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel  = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        # command_buffer: [current, next, next+1]
        self.command_buffer = torch.zeros((self.num_envs, command_cfg["num_poses"], 3), device=gs.device, dtype=gs.tc_float)

        # obs_space: 3+3+4+3+3+3+256+4 = 279
        self.obs_space = {
            "base_ang_vel":   self.base_ang_vel[0],                     # 3
            "base_lin_vel":   self.base_lin_vel[0],                     # 3
            "base_euler":     self.base_euler[0],                        # 4
            "base_rel_pos":   self.base_pos[0],                         # 3
            "base_rel_pos_1": self.base_pos[0],                         # 3
            "base_rel_pos_2": self.base_pos[0],                         # 3
            "front_depth":    torch.zeros((16, 16), device=gs.device),  # 256
            "taken_actions":  self.actions[0],                          # 4
        }

        self.extras = dict()
        self.extras["observations"] = dict()
        self.target = None

    # ------------------------------------------------------------------
    # Gym helpers
    # ------------------------------------------------------------------

    def get_dummy_observations(self):
        return OrderedDict(self.obs_space)

    def get_dummy_actions(self):
        return np.zeros([1, self.num_actions])

    # ------------------------------------------------------------------
    # Command buffer helpers
    # ------------------------------------------------------------------

    def _sample_nearby_gate(self, anchor_pos_2d, neighborhood_radius=None):
        """
        Given a 2-D anchor position (tensor, shape (2,)), return the index of
        a gate sampled preferentially from those within *neighborhood_radius*.
        Falls back to the full set if no neighbours exist.

        This implements the locality-aware curriculum: after a successful pass
        the next waypoint is drawn from nearby gates so the drone learns
        compact sequential flight first, then generalises to longer hops.
        """
        if neighborhood_radius is None:
            neighborhood_radius = self.env_cfg.get("command_neighborhood_radius", 4.0)

        positions = torch.tensor(self.gate_positions, device=self.device, dtype=gs.tc_float)  # (G, 3)
        dists     = torch.norm(positions[:, :2] - anchor_pos_2d, dim=1)                       # (G,)

        nearby_mask = dists < neighborhood_radius
        if nearby_mask.sum() == 0:
            nearby_mask = torch.ones(len(positions), dtype=torch.bool, device=self.device)

        candidates = nearby_mask.nonzero(as_tuple=False).reshape(-1)
        chosen     = candidates[torch.randint(0, len(candidates), (1,), device=self.device)]
        return chosen.squeeze()

    def _resample_command_buffer(self, envs_idx):
        """
        Full re-initialisation of the 3-slot command buffer.
        Slot 0: random gate (uniform — fresh start after reset)
        Slot 1: nearby gate relative to slot-0
        Slot 2: nearby gate relative to slot-1
        """
        if len(envs_idx) == 0:
            return

        positions = torch.tensor(self.gate_positions, device=self.device, dtype=gs.tc_float)

        # slot 0 — uniform random
        idx0 = torch.randint(0, len(positions), (len(envs_idx),), device=self.device)
        self.command_buffer[envs_idx, 0] = positions[idx0]

        # slot 1 — nearby slot-0
        for i, env_i in enumerate(envs_idx):
            anchor = self.command_buffer[env_i, 0, :2]
            self.command_buffer[env_i, 1] = positions[self._sample_nearby_gate(anchor)]

        # slot 2 — nearby slot-1
        for i, env_i in enumerate(envs_idx):
            anchor = self.command_buffer[env_i, 1, :2]
            self.command_buffer[env_i, 2] = positions[self._sample_nearby_gate(anchor)]

    def _update_command_buffer(self, envs_idx):
        """
        Called when an env reaches its current target.
        Shifts the buffer forward and appends a new locality-aware target.
        """
        if len(envs_idx) == 0:
            return

        # shift: 0 ← 1, 1 ← 2
        self.command_buffer[envs_idx, 0] = self.command_buffer[envs_idx, 1]
        self.command_buffer[envs_idx, 1] = self.command_buffer[envs_idx, 2]

        # append new slot-2 near slot-1 (just promoted)
        positions = torch.tensor(self.gate_positions, device=self.device, dtype=gs.tc_float)
        for env_i in envs_idx:
            anchor = self.command_buffer[env_i, 1, :2]
            self.command_buffer[env_i, 2] = positions[self._sample_nearby_gate(anchor)]

    def _at_target(self):
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions):
        # print("[step] actions: mean {:.3f}, max {:.3f}, min {:.3f}".format(actions.mean(), actions.max(), actions.min()))
        self.actions  = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions  = self.actions

        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        # state buffers
        self.episode_length_buf += 1
        self.last_base_pos[:]    = self.base_pos[:]
        self.base_pos[:]         = self.drone.get_pos()
        self.rel_pos             = self.command_buffer[:, 0] - self.base_pos
        self.last_rel_pos        = self.command_buffer[:, 0] - self.last_base_pos
        self.base_quat[:]        = self.drone.get_quat()
        self.base_euler          = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True, degrees=True,
        )
        inv_base_quat          = inv_quat(self.base_quat)
        self.base_lin_vel[:]   = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:]   = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # advance waypoints on success
        envs_idx = self._at_target()
        self._update_command_buffer(envs_idx)

        # termination
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            | (self.base_pos[:, 2] > self.env_cfg["termination_if_close_to_ceiling"])
            | self.contact_sensor.read().T.squeeze(0)
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # rewards
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf          += rew
            self.episode_sums[name] += rew

        # relative positions for look-ahead waypoints
        rel_pos_1 = self.command_buffer[:, 1] - self.base_pos
        rel_pos_2 = self.command_buffer[:, 2] - self.base_pos

        # log-normalised scalar depth  (N, 16, 16) → (N, 256), range [0, 1]
        depth_raw  = self.lidar.read()[1].detach()
        depth_max  = 7.5
        depth_norm = torch.log1p(depth_raw) / math.log1p(depth_max)

        # obs_buf: 3+3+4+3+3+3+256+4 = 279
        self.obs_buf = torch.cat(
            [
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),   # 3
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),   # 3
                self.base_euler,                                                     # 3
                torch.clip(self.rel_pos   * self.obs_scales["rel_pos"], -1, 1),       # 3
                torch.clip(rel_pos_1      * self.obs_scales["rel_pos"], -1, 1),       # 3
                torch.clip(rel_pos_2      * self.obs_scales["rel_pos"], -1, 1),       # 3
                depth_norm.view(self.num_envs, -1),                                   # 256
                self.last_actions,                                                     # 4
            ],
            axis=-1,
        )

        self.last_actions[:]                        = self.actions[:]
        self.extras["observations"]["critic"]       = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        n = len(envs_idx)

        # ---- domain randomisation: position --------------------------------
        # Small Gaussian jitter around the nominal spawn point so the policy
        # is robust to imperfect placement / takeoff disturbances.
        dr_cfg      = self.env_cfg.get("domain_rand", {})
        pos_noise_std = dr_cfg.get("init_pos_noise_std", 0.05)   # metres
        pos_noise   = torch.randn((n, 3), device=self.device) * pos_noise_std
        pos_noise[:, 2] = pos_noise[:, 2].abs()                  # keep z positive
        init_pos    = self.base_init_pos.unsqueeze(0) + pos_noise

        # ---- domain randomisation: orientation -----------------------------
        # Apply a small random rotation around a random axis so the policy
        # handles slight tilt at takeoff.  Maximum tilt is ~±rot_noise_deg.
        rot_noise_deg = dr_cfg.get("init_rot_noise_deg", 5.0)    # degrees
        rot_noise_rad = math.radians(rot_noise_deg)

        # random unit axis in the tangent space (x,y), z kept small for tilt only
        axis = torch.randn((n, 3), device=self.device)
        axis[:, 2] *= 0.3                                         # bias toward tilt, not spin
        axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-6)

        # random half-angle magnitude in [0, rot_noise_rad]
        half_angle = (torch.rand((n, 1), device=self.device) * rot_noise_rad)

        # quaternion: [cos(θ/2), sin(θ/2)·axis]
        init_quat        = torch.zeros((n, 4), device=self.device)
        init_quat[:, 0]  = (half_angle.squeeze() * 0).cos()      # w = cos(0) = 1 when no noise
        init_quat[:, 0]  = torch.cos(half_angle.squeeze())
        init_quat[:, 1:] = torch.sin(half_angle) * axis
        # renormalise to unit quaternion
        init_quat = init_quat / init_quat.norm(dim=1, keepdim=True)

        # ---- apply --------------------------------------------------------
        self.base_pos[envs_idx]  = init_pos
        self.last_base_pos[envs_idx] = init_pos
        self.rel_pos             = self.command_buffer[:, 0] - self.base_pos
        self.last_rel_pos        = self.command_buffer[:, 0] - self.last_base_pos
        self.base_quat[envs_idx] = init_quat
        self.drone.set_pos(init_pos,  zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(init_quat, zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset episode state
        self.last_actions[envs_idx]   = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]      = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # resample waypoint buffer with locality-aware chaining
        self._resample_command_buffer(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _reward_target(self):
        return (
            torch.sum(torch.square(self.last_rel_pos), dim=1)
            - torch.sum(torch.square(self.rel_pos),    dim=1)
        )

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159
        return torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))

    def _reward_angular(self):
        return torch.norm(self.base_ang_vel / 3.14159, dim=1)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew