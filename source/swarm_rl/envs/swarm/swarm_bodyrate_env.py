# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import omni
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import find_matching_prim_paths
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim.schemas import activate_contact_sensors
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, matrix_from_quat
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.markers import CUBOID_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG
from isaaclab.utils.timer import Timer
import isaacsim.core.utils.prims as prims_utils
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, Gf
from collections import deque
import numpy as np
import textwrap
import random
import math
import time
import os

# from swarm_rl.utils.e2e_drone.bev_utils import depth_to_bev_torch
# from swarm_rl.utils.e2e_drone.controller import Quadrotor, QuadrotorDynamics
from swarm_rl.utils.e2e_drone.map_generator import MapGenerator
# from swarm_rl.utils.e2e_drone.occ_collector import OccCollector
# from swarm_rl.utils.e2e_drone.death_replay_collector import DeathReplayCollector
# from swarm_rl.utils.e2e_drone.wind_gen import WindGustGenerator
# from swarm_rl.utils.e2e_drone.thrust_uncertainty import ThrustUncertaintySimulator
# from swarm_rl.utils.e2e_drone.height_randomizer import HeightRandomizer
from swarm_rl.utils.e2e_drone.asset_paths import get_ui_arrow_usd_path, get_ui_frame_usd_path
from enum import IntEnum, auto
import collections
import itertools

from swarm_rl.utils.controller import bodyrate_control_without_thrust
from swarm_rl.utils.depth_camera_array import DepthCameraArray
from swarm_rl.utils.point_provider import PointProvider

from swarm_rl.utils.e2e_drone.reloadable_raycaster_camera import ReloadableRayCasterCameraCfg

from .swarm_bodyrate_env_cfg import QuadcopterEnvCfg


# [0, 2pi] -> [-pi, pi]
def normallize_angle(angle: torch.Tensor):
    return torch.fmod(angle + math.pi, 2 * math.pi) - math.pi


class QuadcopterEnv(DirectRLEnv):
    """A quadcopter environment adapted to use the reward logic from the training code."""

    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.extras["log"] = dict() # 初始化日志字典


        # TODO: 待测试并加入油门不确定度、风扰动
        # # Initialize wind generator
        # self._wind_gen = WindGustGenerator(
        #     num_envs=self.num_envs,
        #     device=self.device,
        #     dt=self.cfg.sim.dt,
        #     tau=self.cfg.wind_tau,
        #     sigma=self.cfg.wind_sigma
        # )
        # self.wind_acc_log = torch.zeros(self.cfg.decimation, self.num_envs, 3, device=self.device)  # Wind acceleration log for each environment
        # # Initialize thrust uncertainty simulator
        # if self.cfg.enable_thrust_uncertainty:
        #     episode_length_steps = int(self.cfg.episode_length_s * self.cfg.step_freq)
        #     self._thrust_uncertainty = ThrustUncertaintySimulator(
        #         num_envs=self.num_envs,
        #         device=self.device,
        #         dt=self.cfg.sim.dt,
        #         episode_length_steps=episode_length_steps,
        #         initial_effectiveness_range=self.cfg.thrust_uncertainty_initial_range,
        #         degradation_factor_range=self.cfg.thrust_uncertainty_degradation_range,
        #         min_effectiveness=self.cfg.thrust_uncertainty_min_effectiveness,
        #     )
        # else:
        #     self._thrust_uncertainty = None


        # TODO: 待测试并加入高度平滑
        # # Initialize height randomizer
        # if self.cfg.enable_height_randomization:
        #     self._height_randomizer = HeightRandomizer(
        #         num_envs=self.num_envs,
        #         device=self.device,
        #         height_range=self.cfg.height_randomization_range,
        #         dt=self.cfg.sim.dt * self.cfg.decimation,
        #         climb_rate=self.cfg.height_randomization_climb_rate,
        #         waypoint_distance=self.cfg.height_randomization_waypoint_distance,
        #         noise_scale=self.cfg.height_randomization_noise_scale,
        #         debug_save_dir=self.cfg.height_randomization_debug_dir
        #     )
        # else:
        #     self._height_randomizer = None


        # TODO: self._robot_mass, self._robot_inertia 仿真与代码中不一致是为什么（底层控制器会用到）
        # TODO: 考虑使用 lsaaclab 管理类自动实现域随机化
        # Mass randomization setup
        if self.cfg.enable_mass_randomization:
            self._base_mass = self.cfg.robot_mass
            self._base_inertia = torch.tensor(self.cfg.robot_inertia, device=self.device)
            self._mass_range = (
                self._base_mass * (1.0 - self.cfg.mass_randomization_percent),
                self._base_mass * (1.0 + self.cfg.mass_randomization_percent)
            )
            # Initialize with randomized masses
            self._robot_mass = torch.empty(self.num_envs, device=self.device).uniform_(*self._mass_range)
            # Scale inertia with mass ratio (I = I_base * mass_ratio)
            mass_ratio = self._robot_mass / self._base_mass
            self._robot_inertia = self._base_inertia.unsqueeze(0) * mass_ratio.unsqueeze(1)
        else:
            self._robot_mass = torch.full((self.num_envs,), self.cfg.robot_mass, device=self.device)
            self._robot_inertia = torch.tensor(self.cfg.robot_inertia, device=self.device).repeat(self.num_envs, 1)

        self._robot_inertia = torch.diag_embed(self._robot_inertia)  # Convert to inertia tensors


        # Controller gains
        self._controller_kp_bodyrate = torch.tensor(self.cfg.kp_bodyrate, device=self.device)


        # 策略 action
        self._actions       = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device) # [thrust, bodyrate(x, y, z)]
        self._last_actions  = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device) # [thrust, bodyrate(x, y, z)]
        # 低层控制量
        self._thrust_max    = self.cfg.thrust_weight_ratio * self.cfg.robot_mass * 9.81 # 最大推力 (推重比 * M * g) (N)
        self._bodyrate_max  = self.cfg.bodyrate_max                                     # 最大角速度 (rad/s)
        self._thrust_desired    = torch.zeros(self.num_envs, 1, device=self.device) # 策略 action 映射到的期望推力
        self._bodyrate_desired  = torch.zeros(self.num_envs, 3, device=self.device) # 策略 action 映射到的期望角速度 (x, y, z)
        self._forces    = torch.zeros(self.num_envs, 1, 3, device=self.device)  # 控制器计算出的控制力
        self._torques   = torch.zeros(self.num_envs, 1, 3, device=self.device)  # 控制器计算出的控制力矩

        # “上一时刻” 数据
        self._last_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # done 相关标志
        self._numerical_instability = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


        # TODO: 可以抽象为目标管理对象
        # Goal queue system
        self._goal_queue = torch.zeros(self.num_envs, self.cfg.num_goals, 3, device=self.device)  # Queue of goals for each environment
        self._current_goal_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Current goal index for each environment
        self._num_goals_remaining = torch.full((self.num_envs,), self.cfg.num_goals, dtype=torch.long, device=self.device)  # Goals remaining for each environment

        # Current goal (legacy compatibility)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_yaw_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_speed_init = torch.zeros(self.num_envs, 1, device=self.device)
        self._desired_speed = self._desired_speed_init


        # TODO: 刚体名写成可配置参数；进行打印、收紧逻辑以确保映射正确
        # Robot references
        body_id, _ = self._robot.find_bodies("body")
        self._body_id = torch.tensor(body_id, dtype=torch.long, device=self.device)
        contact_ids, _ = self._contact_sensor.find_bodies("body")
        self._undesired_contact_ids = torch.tensor(contact_ids, dtype=torch.long, device=self.device)


        # TODO: 考虑抽象为多地图管理器对象
        # Initialize dual map system
        self._active_map_id = 0  # 0 or 1
        self._env_map_assignments = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # Track which map each env uses
        self._map_data = [None, None]  # Store data for both maps
        self._map_generators = [
            MapGenerator(sim=self.sim, device=self.device, map_origin=(0.0, -self.cfg.scene.env_spacing * self.cfg.map_spacing_factor, 0.0), base_prim="/World/ground/map_1"),
            MapGenerator(sim=self.sim, device=self.device, map_origin=(0.0, +self.cfg.scene.env_spacing * self.cfg.map_spacing_factor, 0.0), base_prim="/World/ground/map_2")
        ]
        self._map_regeneration_in_progress = False
        self._map_generation_timer = self.cfg.map_generation_step_threshold

        # Legacy variables for compatibility
        self.occ_kdtree = None
        self.free_points = np.array([[0, 0, 0]], dtype=np.float32)
        self._closest_points = torch.zeros(self.num_envs, 3, device=self.device)


        # TODO: 考虑抽象为回合评估统计对象
        # Add tracking for episode outcomes and success rate
        self._success_rate_window = self.cfg.success_rate_window_size
        self._episode_outcomes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0=ongoing, 1=success, 2=died
        self._episodes_completed = 0
        self._episodes_succeeded = 0
        self._success_rate = 0.0
        self._episode_outcome_history = collections.deque(maxlen=self._success_rate_window)
        self._termination_reason_history = collections.deque(maxlen=self._success_rate_window)
        self._final_distances = collections.deque(maxlen=self._success_rate_window)




    def _setup_scene(self):
        """Create and clone the environment scene."""

        # 初始化 depth cameras，并注册到 scene 中
        self._depth_cameras = DepthCameraArray(
            self.cfg.depth_cameras,
            device=self.device,
            camera_cfg_cls=ReloadableRayCasterCameraCfg,
        )
        self._depth_cameras.register_to_scene(self.scene)


        # TODO: self._robot_mass, self._robot_inertia 仿真与代码中不一致是为什么（底层控制器会用到）
        # TODO: 考虑使用 lsaaclab 管理类自动实现域随机化
        # Set up the robot
        with Timer("[INFO]: Time taken for Articulation generation (inside gym.make)", "QuadcopterEnv"):
            self._robot = Articulation(self.cfg.robot)
        robot_prims = find_matching_prim_paths("/World/envs/env_.*/Robot")
        # with Timer("[INFO]: Time taken for robot prims setup (inside gym.make)", "QuadcopterEnv"):
            # for prim_path in robot_prims:
            #     prims_utils.set_prim_property(prim_path + "/body", "physics:mass", 0.049)
            #     prims_utils.set_prim_property(prim_path + "/body", "physics:diagonalInertia", (1.3615e-5, 1.3615e-5, 3.257e-5))
            #     prims_utils.set_prim_property(prim_path, "visibility", "invisible")
            #     scale = 1.5
            #     prims_utils.set_prim_property(prim_path, "xformOp:scale", (scale, scale, scale))

        
        # TODO: 考虑抽象为多地图管理器对象
        # Initialize dual map system BEFORE camera setup
        self._active_map_id = 0  # 0 or 1
        self._env_map_assignments = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # Track which map each env uses
        self._map_data = [None, None]  # Store data for both maps
        self._map_generators = [
            MapGenerator(sim=self.sim, device=self.device, map_origin=(0.0, -self.cfg.scene.env_spacing * self.cfg.map_spacing_factor, 0.0), base_prim="/World/ground/map_1"),
            MapGenerator(sim=self.sim, device=self.device, map_origin=(0.0, +self.cfg.scene.env_spacing * self.cfg.map_spacing_factor, 0.0), base_prim="/World/ground/map_2")
        ]
        self._map_regeneration_in_progress = False
        self._map_generation_timer = self.cfg.map_generation_step_threshold
        self._regenerate_terrain()


        # Clone the scene
        self.scene.clone_environments(copy_from_source=False)

        # Add the robot and camera to the scene
        self.scene.articulations["robot"] = self._robot

        # Add lights
        self.cfg.light.func("/World/Light", self.cfg.light)

        # Activate contact sensors
        with Timer("[INFO]: Time taken to activate contact sensors (inside gym.make)", "QuadcopterEnv"):
            activate_contact_sensors("/World", threshold=1.0)
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor


        # TODO: 考虑抽象为多地图管理器对象
        # Counters
        self._map_generation_timer = 0




    # TODO: 考虑抽象为多地图管理器对象
    def _regenerate_terrain(self):
        """Generate new terrain and obstacles using dual map system."""
        self._map_generation_timer += 1
        if self._map_regeneration_in_progress or self._map_generation_timer < self.cfg.map_generation_step_threshold:
            return
        self._map_regeneration_in_progress = True

        # self._update_curriculum()

        try:
            # Determine which map to regenerate (the inactive one)
            inactive_map_id = 1 - self._active_map_id

            # Check if any environments are still using the inactive map
            envs_on_inactive = torch.sum(self._env_map_assignments == inactive_map_id).item()

            if envs_on_inactive > 0:
                print(f"Cannot regenerate inactive map {inactive_map_id}: {envs_on_inactive} environments still using it")
                return

            print(f"Regenerating inactive map {inactive_map_id}")

            # Get the appropriate map generator
            map_generator = self._map_generators[inactive_map_id]

            # Create obstacles environment for inactive map
            env_data = map_generator.create_environment(
                self.cfg.scene,
                num_obstacles=int(self.cfg.scene.env_spacing * self.cfg.scene.env_spacing // 0.25),
                num_floaters=int(self.cfg.scene.env_spacing * self.cfg.scene.env_spacing // 1.0),
                # num_floaters=0,
                min_distance=self.cfg.obstacle_min_distance_init,
                obstacle_size_range=self.cfg.obstacle_size_range,
                obstacle_height_range=(2.4, 2.5),
                floaters_size_range=self.cfg.floater_size_range,
                floaters_height_range=(0.3, 1.7),
            )

            # Store the new map data
            self._map_data[inactive_map_id] = env_data

            # Swap active/inactive maps
            self._active_map_id = inactive_map_id

            # Update current environment references to use new active map
            self.occ_kdtree = env_data["kdtree"]
            self.free_points = env_data["free_points"]
            self._map_generation_timer = 0  # Reset timer after regeneration


            # 深度相机重新加载地图网格
            self._depth_cameras.reload_cameras()


            # # Request mesh reload for all ray-casting cameras after terrain regeneration
            # cameras_to_reload = []
            # # Request mesh reload for all cameras
            # for sensor_name, camera_desc in cameras_to_reload:
            #     if hasattr(self.scene.sensors[sensor_name], 'request_mesh_reload'):
            #         self.scene.sensors[sensor_name].request_mesh_reload()
            #         print(f"Requested mesh reload for {camera_desc}")
            #     else:
            #         print(f"Warning: {camera_desc} does not support mesh reloading")

            self._point_provider = PointProvider(
                self.cfg.point_provider,
                self,
                self.free_points,
            )

            print(f"Map regeneration complete. New active map: {self._active_map_id}")

        finally:
            self._map_regeneration_in_progress = False




    # def _update_curriculum(self):
    #     if not hasattr(self, "_success_rate"):
    #         return
    #     if self._success_rate > 0.9:
    #         span = max(self.cfg.hover_hold_max_s - self.cfg.hover_hold_initial_s, 0.0)
    #         desired_requirement = self.cfg.hover_hold_initial_s + span * self._success_rate
    #         desired_requirement = min(self.cfg.hover_hold_max_s, desired_requirement)
    #         if desired_requirement > self._hover_hold_requirement_s:
    #             step = min(self.cfg.hover_hold_increment_s, desired_requirement - self._hover_hold_requirement_s)
    #             self._hover_hold_requirement_s += step
    #             print(f"CURRICULUM: Raising hover hold to {self._hover_hold_requirement_s:.2f}s (success rate {self._success_rate:.3f})")

    #         if self.cfg.obstacle_min_distance_init > self.cfg.obstacle_min_distance_min:
    #             new_min_dist = self.cfg.obstacle_min_distance_init / 1.1
    #             self.cfg.obstacle_min_distance_init = max(new_min_dist, self.cfg.obstacle_min_distance_min)
    #             print(f"CURRICULUM: Success rate {self._success_rate:.3f} > 0.9, reducing obstacle min distance to {self.cfg.obstacle_min_distance_init:.3f}")
    #         else:
    #             self.cfg.reward_coef_vel_speed_excess_penalty = 0.6
    #             self.cfg.reward_coef_vel_speed_match_reward = 0.2
    #             print(f"CURRICULUM: Success rate {self._success_rate:.3f} > 0.9, setting vel speed excess penalty to {self.cfg.reward_coef_vel_speed_excess_penalty:.3f}")




    # TODO: 考虑抽象为多地图管理器对象
    def _get_current_map_data(self, map_id=None):
        """Get map data for specified map ID or current active map."""
        if map_id is None:
            map_id = self._active_map_id

        if self._map_data[map_id] is not None:
            return self._map_data[map_id]
        else:
            # Fallback to current global data
            return {
                "kdtree": self.occ_kdtree,
                "free_points": self.free_points
            }




    def _pre_physics_step(self, actions: torch.Tensor):
        # self._noise_10_cfg.func(actions, self._noise_10_cfg)
        # TODO: 1、确认返回的 action 是否已经被处理到 [-1, 1]；
        #       2、思考把 action 剪裁放在 rl 库还是这里哪个更好（rl算法优化用 action 分布和剪裁后分布不一致）
        #       3、思考选择 tanh 还是 clip 哪个更好（如果刚才分布不一致问题无法解决的话）
        self._actions.copy_(actions)    # 原地复制 action
        self._actions.tanh_()           # 原地tanh，将 action 平滑限幅到[-1, 1]
        # self._actions.clamp_(-1.0, 1.0)

        # action 映射到低层控制器输入
        self._thrust_desired.copy_(self._actions[:, 0:1]).add_(1.0).mul_(0.5).mul_(self._thrust_max)    # thrust: ((a+1)*0.5*thrust_max)
        self._bodyrate_desired.copy_(self._actions[:, 1:]).mul_(self._bodyrate_max)                     # bodyrate: a * bodyrate_max




    def _apply_action(self):
        """Apply thrust/moment to the quadcopter."""

        # TODO: 优化 action delay 逻辑（加入随机延迟时间等）
        # if not hasattr(self, "_action_delay_buffer"):
        #     # Initialize buffer with zeros for each environment
        #     default_action = torch.zeros((self.num_envs, 4), device=self.device)
        #     default_action[:, 2] = 0.5  # Set yaw rate to 0.5 for all environments
        #     self._action_delay_buffer = collections.deque(
        #         [default_action.clone() for _ in range(self.cfg.action_delay_steps + 1)],
        #         maxlen=self.cfg.action_delay_steps + 1
        #     )
        # self._action_delay_buffer.append(self._actions.clone())
        # delayed_actions = self._action_delay_buffer[0]

        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, 0, 2:3]  = self._thrust_desired
        self._torques[:, 0, :] = bodyrate_control_without_thrust(
            self._robot.data.root_ang_vel_b,
            self._bodyrate_desired,
            self._robot_inertia,
            self._controller_kp_bodyrate
        )

        # TODO: 待测试并加入油门不确定度、风扰动
        # # Apply thrust uncertainty if enabled
        # if self._thrust_uncertainty is not None:
        #     thrust_effectiveness = self._thrust_uncertainty.step()
        #     self._forces[:, 0, 2] *= thrust_effectiveness
        # # Apply wind disturbances
        # wind_acc = self._wind_gen.step()                       # (num_envs,3) m/s²
        # self.wind_acc_log[(self._sim_step_counter - 1) % self.cfg.decimation] = wind_acc
        # wind_force_world = wind_acc * self._robot_mass.unsqueeze(1)  # (num_envs,3) N
        # quat_w = self._robot.data.root_quat_w  # quaternion representing rotation from body to world
        # rot_matrices_w2b = matrix_from_quat(quat_w).transpose(1, 2)  # shape: (num_envs, 3, 3)
        # wind_force_body = torch.bmm(rot_matrices_w2b, wind_force_world.unsqueeze(2)).squeeze(2)
        # self._forces[:, 0, :] += wind_force_body
        # # print(f"original force: {self._forces[0, 0, :]}")
        # # print(f"Wind force: {wind_force_body[0]}")
        # # print(f"Controller compute time: {start.elapsed_time(end)} ms")
        # # print(f"Env[0] - Action: {self._actions[0]}")
        # # print(f"Env[0] - Force: {self._forces[0]}, Torque: {self._torques[0]}")

        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id)




    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Define terminations and timeouts."""

        # -------------------------
        # 计算各类结束条件
        # -------------------------
        # 计算回合超时的 env
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 计算发生碰撞的 env
        net_forces = self._contact_sensor.data.net_forces_w_history  # (N, T, B, 3)
        selected = net_forces[:, :, self._undesired_contact_ids, :]  # (N, T, K, 3)
        max_contact, _ = torch.norm(selected, dim=-1).max(dim=1)     # (N, K)
        self._is_contact = (max_contact > self.cfg.contact_force_threshold).any(dim=1)  # Threshold is important for REAL contact detection

        # -------------------------
        # 计算总结束条件
        # -------------------------
        terminated_numerical = self._numerical_instability
        # TODO: TEMPORARY DISABLE COLLISION TERMINATION
        terminated_collision = self._is_contact
        # terminated_collision = torch.zeros_like(self._is_contact)
        terminated_success   = self._is_success

        terminated = terminated_numerical | terminated_collision | terminated_success

        # -------------------------
        # 记录回合结束原因到日志（本 step 内“将要结束”的 env 统计）
        # -------------------------
        # 构造互斥的结束原因（按优先级：success > collision > numerical > timeout）
        end_success   = terminated_success
        end_collision = terminated_collision & ~end_success
        end_numerical = terminated_numerical & ~(end_success | end_collision)
        end_timeout   = time_out & ~(end_success | end_collision | end_numerical)
        end_total     = end_success | end_collision | end_numerical | end_timeout
        # 计算统计量
        end_total_count = end_total.sum().to(torch.float32)
        zero = torch.zeros_like(end_total_count)
        succ_ratio = torch.where(end_total_count > 0, end_success.float().sum() / end_total_count, zero)
        collision_ratio = torch.where(end_total_count > 0, end_collision.float().sum() / end_total_count, zero)
        numerical_ratio = torch.where(end_total_count > 0, end_numerical.float().sum() / end_total_count, zero)
        timeout_ratio = torch.where(end_total_count > 0, end_timeout.float().sum() / end_total_count, zero)
        # 记录到日志
        self.extras["log"].update({
            # count（数量）
            "End / Total (count)"    : end_total_count,
            "End / Success (count)"  : end_success.sum().to(torch.float32),
            "End / Collision (count)": end_collision.sum().to(torch.float32),
            "End / Numerical (count)": end_numerical.sum().to(torch.float32),
            "End / Timeout (count)"  : end_timeout.sum().to(torch.float32),
            # ratio（比例）
            "End / Success (ratio)"  : succ_ratio,
            "End / Collision (ratio)": collision_ratio,
            "End / Numerical (ratio)": numerical_ratio,
            "End / Timeout (ratio)"  : timeout_ratio,
        })

        return terminated, time_out




    def _get_rewards(self) -> torch.Tensor:
        """
        Calculate the reward for each environment.
        """

        reward_cfg = self.cfg.reward
        robot_data = self._robot.data

        # Current position, orientation, and velocity of the robot
        pos_w = robot_data.root_state_w[:, :3]
        rot_E_w = torch.stack(euler_xyz_from_quat(robot_data.root_state_w[:, 3:7]), dim=1)
        rot_E_w = torch.stack([normallize_angle(rot_E_w[:, 0]), normallize_angle(rot_E_w[:, 1]), normallize_angle(rot_E_w[:, 2])], dim=1)
        vel_b = robot_data.root_lin_vel_b


        goal_completion_mask, _ = self._point_provider.update()
        self._desired_pos_w.copy_(self._point_provider.get_target_points())


        # ----------------------------------------
        # 计算每项奖励和惩罚
        # ----------------------------------------
        # TODO: 重新思考抵达目标点+保持期望速度的奖励机制（比如可参考 “Extreme Parkour” 文章的内积设计）
        # distance to goal center
        distance_to_gap = (pos_w - self._desired_pos_w).norm(dim=1)
        last_distance_to_gap = (self._last_pos_w - self._desired_pos_w).norm(dim=1)
        delta_distance = last_distance_to_gap - distance_to_gap
        distance_reward = torch.clamp(delta_distance / reward_cfg.delta_distance_clamp, min=-1.0, max=1.0)

        # TODO: 重新思考抵达目标点+保持期望速度的奖励机制（比如可参考 “Extreme Parkour” 文章的内积设计）
        # direction penalty [-2, 0]
        # (We define a "forward" direction as +X in world space for illustration.)
        rot_M_w = matrix_from_quat(robot_data.root_quat_w)
        dir_body_w = rot_M_w[:, 0:2, 0] / (rot_M_w[:, 0:2, 0].norm(dim=1, keepdim=True) + 1e-6)  # Forward direction in world frame
        dir_to_goal = (self._desired_pos_w - pos_w)[:, :2]
        dir_to_goal = dir_to_goal / (dir_to_goal.norm(dim=-1, keepdim=True) + 1e-6)
        yaw_direction_penalty = (dir_body_w * dir_to_goal).sum(dim=-1) - 1.0

        # action magnitude penalty [-6, 0]
        # shape is (num_envs, 4) -> thrust + rates
        # Apply different weights to each action dimension for magnitude calculation
        action_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)  # [thrust, bodyrate(x, y, z)] # TODO set action_weights to 0
        weighted_actions = self._actions * action_weights
        action_magnitude_penalty = -weighted_actions.norm(dim=1)

        # action change penalty (difference relative to last actions) [-4, 0]
        diff_actions = self._actions - self._last_actions
        # Apply different weights to each action dimension
        action_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)  # [thrust, bodyrate(x, y, z)]
        weighted_diff_actions = diff_actions * action_weights
        action_change_penalty = -weighted_diff_actions.norm(dim=1)

        # Velocity-related rewards and penalties - now separated into individual components
        speed = vel_b.norm(dim=1)
        distance_to_goal = (pos_w - self._desired_pos_w).norm(dim=1)

        # TODO: 重新思考抵达目标点+保持期望速度的奖励机制（比如可参考 “Extreme Parkour” 文章的内积设计）
        # TODO: 评估全向视野感知下是否需要加入速度方向奖励
        # Velocity direction penalty [-inf, 0]
        # Penalize deviations from forward (+X body axis) scaled by speed, shaped with Huber loss
        safe_speed = torch.clamp(speed, min=1e-6)
        cos_forward = torch.clamp(vel_b[:, 0] / safe_speed, -1.0, 1.0)
        direction_misalignment = speed * (1.0 - cos_forward)
        vel_dir_delta = max(reward_cfg.vel_direction_huber_delta, 1e-6)
        vel_dir_delta_tensor = torch.tensor(vel_dir_delta, device=self.device, dtype=direction_misalignment.dtype)
        quadratic_region = 0.5 * direction_misalignment.square() / vel_dir_delta_tensor
        linear_region = direction_misalignment - 0.5 * vel_dir_delta_tensor
        vel_direction_penalty = -torch.where(direction_misalignment <= vel_dir_delta_tensor, quadratic_region, linear_region)

        # Adjust desired speed based on distance to goal
        # Linearly decrease speed when within distance threshold of goal
        speed_adjust_start = reward_cfg.speed_adjustment_distance  # Start slowing down
        speed_adjust_end = 0.0   # Speed should be zero
        slowdown_factor = torch.clamp((speed_adjust_start - distance_to_goal) / (speed_adjust_start - speed_adjust_end), 0.0, 1.0)
        yaw_direction_penalty = yaw_direction_penalty * (1.0 - slowdown_factor)
        yaw_direction_penalty = torch.where(
            distance_to_goal < self.cfg.hover_yaw_penalty_distance,
            torch.zeros_like(yaw_direction_penalty),
            yaw_direction_penalty,
        )


        # TODO: 可以抽象为目标管理对象
        # Adjust desired speed: original speed when far, 0 when at goal
        desired_speed = self._desired_speed_init.squeeze(-1) * (1.0 - slowdown_factor)
        self._desired_speed = desired_speed.unsqueeze(-1)  # Ensure it's a column vector


        # Speed magnitude penalty [-5, 0]
        # Penalize when speed exceeds desired speed (now using adjusted desired_speed)
        vel_speed_excess_penalty = torch.where(
            speed > desired_speed,
            -torch.clamp(torch.exp((speed - desired_speed) * 5) - 1.0, max=5.0),
            torch.zeros_like(speed)
        )

        # TODO: 重新思考抵达目标点+保持期望速度的奖励机制（比如可参考 “Extreme Parkour” 文章的内积设计）
        # Velocity matching reward [0, 2]
        # Reward for having speed close to desired speed (now using adjusted desired_speed)
        vel_speed_match_reward = torch.exp(-5.0 * torch.abs(speed - desired_speed)) * 2.0

        # z position penalty shaped with Huber loss [-5, 0]
        z_pos = pos_w[:, 2]
        z_err = z_pos - self._desired_pos_w[:, 2]
        delta = max(reward_cfg.z_position_huber_delta, 1e-6)
        delta_tensor = torch.tensor(delta, device=self.device, dtype=z_pos.dtype)
        abs_z_err = torch.abs(z_err)
        quadratic_region = 0.5 * abs_z_err.square() / delta_tensor
        linear_region = abs_z_err - 0.5 * delta_tensor
        z_position_penalty = -torch.where(abs_z_err <= delta_tensor, quadratic_region, linear_region)

        # collision penalty. [-1, 0]
        obstacle_collision_penalty = torch.where(
            self._is_contact,
            torch.ones_like(vel_b[:, 0]),
            torch.zeros_like(vel_b[:, 0]),
        )
        obstacle_collision_penalty = -obstacle_collision_penalty

        # # Perform KD-tree query once to get nearest obstacle distances
        # nearest_obstacle_distances = None
        # if self.occ_kdtree is not None:
        #     d, _ = self.occ_kdtree.query(pos_w.cpu(), workers=-1, distance_upper_bound=4.0)
        #     nearest_obstacle_distances = torch.tensor(d, dtype=pos_w.dtype, device=self.device)

        # # ESDF-based reward
        # esdf_reward = torch.zeros_like(vel_b[:, 0])
        # if nearest_obstacle_distances is not None:
        #     safe_threshold = 0.5
        #     esdf_reward = torch.where(
        #         nearest_obstacle_distances < safe_threshold,
        #         -(torch.exp(5.0 * (safe_threshold - nearest_obstacle_distances)) - 1.0),
        #         torch.zeros_like(nearest_obstacle_distances),
        #     )


        # Succeed reward [0, 1] - only for individual goal completion
        succeed_reward = goal_completion_mask.float()

        # Angular velocity penalty
        max_angular_velocity = reward_cfg.max_angular_velocity_penalty # rad/s
        ang_vel_b = robot_data.root_ang_vel_b.clone() # (num_envs, 3)
        max_ang_vel_penalty = torch.where(
            torch.abs(ang_vel_b) > max_angular_velocity,
            -torch.clamp(torch.exp(torch.abs(torch.abs(ang_vel_b) - max_angular_velocity)) - 1.0, max=10.0),
            torch.zeros_like(ang_vel_b),
        ) # (num_envs, 3) -> (num_envs,)
        max_ang_vel_penalty = torch.sum(max_ang_vel_penalty, dim=1)

        # Angle penalty [-20, 0]
        max_angle = reward_cfg.max_angle_penalty # rad
        max_angle_penalty = torch.where(
            torch.abs(rot_E_w[:, :2]) > max_angle,
            -torch.clamp(torch.exp(torch.abs(torch.abs(rot_E_w[:, :2]) - max_angle)) - 1.0, max=10.0),
            torch.zeros_like(rot_E_w[:, :2]),
        )
        max_angle_penalty = torch.sum(max_angle_penalty, dim=1)

        # TODO: 审查 z_vel 惩罚的设置意图；注意此处坐标系选取是在 body 系下，是否需要换到 world 系下？
        # z velocity penalty [-1, 0]
        z_vel_diff = torch.abs(vel_b[:, 2])
        z_vel_penalty = -torch.clamp(z_vel_diff, max=1.0)

        # TODO: 不置 0 可能会导致拖延完成任务时间
        # Alive reward (before collision) [0, 1]
        alive_reward = torch.logical_not(torch.logical_or(self._is_success, self._is_contact)).float()

        # # TODO: 意图不名，是惩罚吗？
        # lin_vel = torch.sum(torch.square(robot_data.root_lin_vel_b), dim=1)
        # ang_vel = torch.sum(torch.square(robot_data.root_ang_vel_b), dim=1)

        # # TODO: 重新思考抵达目标点+保持期望速度的奖励机制（比如可参考 “Extreme Parkour” 文章的内积设计）
        # distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / reward_cfg.distance_goal_mapping_scale)

        # ----------------------------------------
        # 计算总奖励，并记录各项奖励到日志
        # ----------------------------------------
        reward_specs = [
            ("distance_reward",            distance_reward,            reward_cfg.coef_distance_reward),
            ("yaw_direction_penalty",      yaw_direction_penalty,      reward_cfg.coef_yaw_direction_penalty),
            ("action_magnitude_penalty",   action_magnitude_penalty,   reward_cfg.coef_action_magnitude_penalty),
            ("action_change_penalty",      action_change_penalty,      reward_cfg.coef_action_change_penalty),
            ("vel_direction_penalty",      vel_direction_penalty,      reward_cfg.coef_vel_direction_penalty),
            ("vel_speed_excess_penalty",   vel_speed_excess_penalty,   reward_cfg.coef_vel_speed_excess_penalty),
            ("vel_speed_match_reward",     vel_speed_match_reward,     reward_cfg.coef_vel_speed_match_reward),
            ("z_position_penalty",         z_position_penalty,         reward_cfg.coef_z_position_penalty),
            ("obstacle_collision_penalty", obstacle_collision_penalty, reward_cfg.coef_obstacle_collision_penalty),
            # ("esdf_reward",                esdf_reward,                reward_cfg.coef_esdf_reward),
            ("succeed_reward",             succeed_reward,             reward_cfg.coef_succeed_reward),
            ("max_ang_vel_penalty",        max_ang_vel_penalty,        reward_cfg.coef_max_ang_vel_penalty),
            ("max_angle_penalty",          max_angle_penalty,          reward_cfg.coef_max_angle_penalty),
            ("alive_reward",               alive_reward,               reward_cfg.coef_alive_reward),
            ("z_vel_penalty",              z_vel_penalty,              reward_cfg.coef_z_vel_penalty),
            # ("lin_vel_reward",             lin_vel,                    reward_cfg.coef_lin_vel_reward_scale),
            # ("ang_vel_reward",             ang_vel,                    reward_cfg.coef_ang_vel_reward_scale),
            # ("distance_to_goal_reward",    distance_to_goal_mapped,    reward_cfg.coef_distance_to_goal_reward_scale),
        ]
        # 剔除权重为0的项
        reward_specs = [(n, t, w) for (n, t, w) in reward_specs if w != 0.0]
        # 提取各项奖励名称、原始数值和权重
        names = [n for n, _, _ in reward_specs]
        raw_terms = torch.stack([t for _, t, _ in reward_specs], dim=1)  # (N, K)
        weights = raw_terms.new_tensor([w for _, _, w in reward_specs])  # (K,) 自动对齐 device/dtype
        # 计算加权奖励、总和、均值
        weighted_terms = raw_terms * weights        # (N, K)
        reward_per_env = weighted_terms.sum(dim=1)  # (N,)
        mean_weighted_terms = weighted_terms.mean(dim=0)  # (K,)
        reward_env_mean   = reward_per_env.mean()         # scalar
        # 记录到日志（skrl会自动处理无前缀 log name，加上 Info / 前缀）
        self.extras["log"].update({f"{names[i]}": mean_weighted_terms[i] for i in range(len(names))})
        self.extras["log"]["total"] = reward_env_mean

        # ----------------------------------------
        # 更新 “上一时刻” 数据
        # ----------------------------------------
        self._last_pos_w.copy_(pos_w)
        self._last_actions.copy_(self._actions)

        return reward_per_env




    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environment indexes."""

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # TODO: 考虑抽象为多地图管理器对象
        # Always call regenerate terrain on reset to maintain map data
        self._regenerate_terrain()


        # TODO: 待测试并加入油门不确定度、风扰动
        # # Reset wind generator for the environments being reset
        # self._wind_gen.reset(env_ids)
        # # Reset thrust uncertainty for the environments being reset
        # if self._thrust_uncertainty is not None:
        #     self._thrust_uncertainty.reset(env_ids)


        # Reset height randomizer later after initial positions are set

        # Determine episode outcomes for completed episodes
        success_mask = self._is_success[env_ids]
        died_mask = torch.logical_and(self.reset_terminated[env_ids], ~success_mask)
        timed_out_mask = self.reset_time_outs[env_ids]


        # TODO: 考虑抽象为回合评估统计对象
        # Update episode outcomes and metrics
        # self._update_episode_outcomes_and_metrics(env_ids, success_mask, died_mask, timed_out_mask)


        # Reset environment states
        self._robot.reset(env_ids)
        # Parent method sets done buffers, etc.
        super()._reset_idx(env_ids)

        # Assign reset environments to the active map
        self._env_map_assignments[env_ids] = self._active_map_id


        self._desired_speed_init[env_ids] = torch.zeros_like(self._desired_speed_init[env_ids]).uniform_(*self.cfg.des_vel_range)
        self._desired_speed[env_ids] = self._desired_speed_init[env_ids]

        self._point_provider.resample(env_ids)
        self._desired_pos_w[env_ids] = self._point_provider.get_target_points()[env_ids]


        # TODO: 评估是否需要抽象为初始位置采样对象
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        default_root_state[:, :3] = self._point_provider.get_spawn_points()[env_ids]

        # Apply random yaw rotation to the initial root state
        initial_random_yaw = torch.zeros_like(default_root_state[:, 0]).uniform_(-math.pi, math.pi)
        default_root_state[:, 3] = torch.cos(initial_random_yaw * 0.5)  # w
        default_root_state[:, 6] = torch.sin(initial_random_yaw * 0.5)  # z
        default_root_state[:, 4] = 0.0  # x
        default_root_state[:, 5] = 0.0  # y

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)



        # 重置 “上一时刻” 数据
        self._last_pos_w[env_ids] = default_root_state[:, :3]
        self._last_actions[env_ids] = torch.zeros(4, device=self.device)

        # 重置 done 相关标志
        self._numerical_instability[env_ids] = False
        self._is_contact[env_ids] = False
        self._is_success[env_ids] = False


        # TODO: 待测试并加入高度平滑
        # # Reset height randomizer with initial positions now that they are set
        # if self._height_randomizer is not None:
        #     initial_positions = default_root_state[:, :3]  # [x, y, z] coordinates
        #     self._height_randomizer.reset(env_ids, initial_positions)


        # TODO: 考虑抽象为回合评估统计对象
        # Reset episode outcome tracking for the reset environments
        self._episode_outcomes[env_ids] = 0



    
    def _get_observations(self) -> dict:
        """
        Return the observations for the agent in a dictionary.
        """

        obs_cfg = self.cfg.observations
        robot_data = self._robot.data

        # TODO: 待测试并加入高度平滑
        # # Update height randomization
        # if self._height_randomizer is not None:
        #     self._height_randomizer.step(robot_data.root_state_w[:, :3])
        #     # Apply height randomization to current goal positions
        #     self._desired_pos_w = self._height_randomizer.apply_to_goals(self._desired_pos_w)

        # ----------------------------------------
        # 原始数据读取
        # ----------------------------------------
        # 多相机深度图像
        max_d = self.cfg.depth_cameras.max_distance
        depth_image_list = self._depth_cameras.read_batch()
        # print(depth_image.shape)

        # 深度图像 (TODO: 待加入噪声)
        # # (N, C, H, W) Normalize to [0, 1] and scale
        # image_noised = image_raw = torch.stack(depth_image_list, dim=1) / max_d * obs_cfg.depth_image_scale
        image_noised = image_raw = torch.cat(depth_image_list, dim=2).unsqueeze(1) / max_d * obs_cfg.depth_image_scale
        # print(image_raw.shape)

        # TODO: 待加入多相机类中
        # # Debug: visualize depth images
        # depth_image_cat = torch.cat(depth_image_list, dim=2)
        # depth_image_cat = depth_image_cat[0].squeeze(-1)
        # H, W = depth_image_cat.shape
        # depth_image_cat_u8 = torch.clamp((depth_image_cat / max_d * 255.0), 0, 255).to(torch.uint8)
        # scale = 6
        # new_w = max(1, int(W * scale))
        # new_h = max(1, int(H * scale))
        # img_big = cv2.resize(depth_image_cat_u8.cpu().numpy(), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # win = "Raycast Camera Robot 0 Depth"
        # if not hasattr(self, "_cv_win_inited"):
        #     self._cv_win_inited = set()
        # if win not in self._cv_win_inited:
        #     cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow(win, new_w, new_h)  # 给个初始大窗口
        #     self._cv_win_inited.add(win)
        # cv2.imshow(win, img_big)
        # cv2.waitKey(1)

        
        # TODO: 可参考人家噪声是怎么加的
        # # Get depth image from ReloadableRayCasterCamera
        # depth_image = self._raycast_camera.data.output["distance_to_image_plane"].clone()
        # depth_image = depth_image_orig = depth_image.reshape(self.num_envs, -1)

        # # Apply sensor noise simulation (matching original environment)
        # invalid_rate = random.random() * self.cfg.depth_invalid_rate_max
        # invalid_masks = torch.rand_like(depth_image) < invalid_rate
        # depth_image = torch.where(invalid_masks,
        #                       torch.tensor(self.cfg.camera_max_distance, device=self.device, dtype=depth_image.dtype),
        #                       depth_image)

        # 当前机器人状态量
        pos_w     = robot_data.root_state_w[:, :3]
        quat_w    = robot_data.root_quat_w
        lin_vel_b = robot_data.root_lin_vel_b
        ang_vel_b = robot_data.root_ang_vel_b

        # ----------------------------------------
        # 计算每项观测量
        # ----------------------------------------
        # TODO: 下面obs计算归一化计算可抽象成函数；对于 fp16/bf16 可以提升到 fp32 统一处理

        # 计算旋转矩阵
        rot_matrix_b2w = matrix_from_quat(quat_w)   # Shape: (num_envs, 3, 3)
        rot_matrix = rot_matrix_b2w.flatten(1)      # Shape: (num_envs, 9)

        # 计算目标点特征 (TODO: 待评估这种 goal_feat 定义是否合适)
        pos_to_goal = self._desired_pos_w - pos_w
        # TODO: 评估是否需要对 dir_xy 加入距离过小时的门控衰减
        err_xy = pos_to_goal[:, :2]
        eps = 1e-6
        if err_xy.dtype in (torch.float16, torch.bfloat16):
            eps = 1e-3
        dir_xy = err_xy / torch.linalg.vector_norm(err_xy, dim=1, keepdim=True).clamp_min(eps)
        # TODO: 评估 err_Z 是否需要做尺度处理
        err_z  = pos_to_goal[:, 2]
        goal_feat = torch.cat([dir_xy, err_z.unsqueeze(-1)], dim=1)  # [dir_x, dir_y, err_Z]

        # 计算相对于目标点的位移（限制最大距离，保持方向不变）
        eps = 1e-6
        if pos_to_goal.dtype in (torch.float16, torch.bfloat16):
            eps = 1e-3
        distance_to_goal = torch.linalg.vector_norm(pos_to_goal, dim=1, keepdim=True)
        dir_to_goal = pos_to_goal / distance_to_goal.clamp_min(eps)
        pos_to_goal_norm = dir_to_goal * torch.clamp(distance_to_goal / obs_cfg.max_goal_distance, max=1.0)
        
        # # 计算相对于最近障碍物的位移（来自全局点云的 KD-tree；限制最大距离，保持方向不变）
        # _, indices = self.occ_kdtree.query(pos_w.cpu().numpy(), workers=-1)
        # self._closest_points = torch.tensor(self.occ_kdtree.data[indices], device=self.device, dtype=pos_w.dtype)
        # pos_to_obstacle = self._closest_points - pos_w
        # eps = 1e-6
        # if pos_to_obstacle.dtype in (torch.float16, torch.bfloat16):
        #     eps = 1e-3
        # distance_to_obstacle = torch.linalg.vector_norm(pos_to_obstacle, dim=1, keepdim=True)
        # dir_to_obstacle = pos_to_obstacle / distance_to_obstacle.clamp_min(eps)
        # pos_to_obstacle_norm = dir_to_obstacle * torch.clamp(distance_to_obstacle / obs_cfg.max_obstacle_distance, max=1.0)

        # ----------------------------------------
        # 构造观测向量
        # ----------------------------------------
        # 构造 policy 网络观测
        policy_obs = torch.cat(
            [
                ang_vel_b             * obs_cfg.ang_vel_scale,         # 角速度（body系）[wx, wy, wz]
                rot_matrix            * obs_cfg.rot_mat_scale,         # 9D 旋转矩阵 (TODO: 改为前六维/其他旋转表示?)
                (self._desired_speed) * obs_cfg.desired_speed_scale,   # 期望速度
                goal_feat             * obs_cfg.goal_feat_scale,       # 目标点特征 [dir_x, dir_y, z_err] (TODO: z_err scale 能否与 dir_xy 相同?)
                self._last_actions    * obs_cfg.last_action_scale,     # 上一步 action [thrust, bodyrate(x, y, z)]
            ],
            dim=-1,
        )
        # 构造 critic 网络观测
        critic_obs = torch.cat(
            [
                # 常规观测
                ang_vel_b             * obs_cfg.ang_vel_scale,         # 角速度（body系）[wx, wy, wz]
                rot_matrix            * obs_cfg.rot_mat_scale,         # 9D 旋转矩阵 (TODO: 改为前六维/其他旋转表示?)
                (self._desired_speed) * obs_cfg.desired_speed_scale,   # 期望速度
                goal_feat             * obs_cfg.goal_feat_scale,       # 目标点特征 [dir_x, dir_y, z_err] (TODO: z_err scale 能否与 dir_xy 相同?)
                self._last_actions    * obs_cfg.last_action_scale,     # 上一步 action [thrust, bodyrate(x, y, z)]

                # 特权观测
                lin_vel_b             * obs_cfg.lin_vel_scale,         # 线速度（body系）[vx, vy, vz]
                pos_to_goal_norm      * obs_cfg.pos_to_goal_scale,     # 相对于目标点的位移（最大距离剪裁，方向不变）
                # pos_to_obstacle_norm  * obs_cfg.pos_to_obstacle_scale, # 相对于最近障碍物的位移（最大距离剪裁，方向不变）

                # TODO: 待测试并加入风扰动
                # torch.mean(self.wind_acc_log, dim=0) * 30, # wind disturbance
            ],
            dim=-1,
        )
        # 观测值检查
        policy_obs = self.CHECK_NAN(policy_obs)
        critic_obs = self.CHECK_NAN(critic_obs)
        self.CHECK_state()

        return {
            "policy": {"image": image_noised, "state": policy_obs},
            "critic": {"image": image_raw,    "state": critic_obs},
        }




    def _set_debug_vis_impl(self, debug_vis: bool):
        """Show debug markers if debug_vis is True."""
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
                print("Created goal_pos_visualizer")
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)

            if not hasattr(self, "goal_yaw_visualizer"):
                goal_arrow_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                goal_arrow_cfg.markers["arrow"].usd_path = get_ui_arrow_usd_path()
                goal_arrow_cfg.markers["arrow"].scale = (0.05, 0.05, 0.2)
                # -- goal yaw
                goal_arrow_cfg.prim_path = "/Visuals/Command/goal_yaw"
                self.goal_yaw_visualizer = VisualizationMarkers(goal_arrow_cfg)
                print("Created goal_yaw_visualizer")
            # set their visibility to true
            self.goal_yaw_visualizer.set_visibility(True)

            if not hasattr(self, "current_yaw_visualizer"):
                current_arrow_cfg = FRAME_MARKER_CFG.copy()
                current_arrow_cfg.markers["frame"].usd_path = get_ui_frame_usd_path()
                current_arrow_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

                print("[debug] current_arrow_cfg.markers keys:", list(current_arrow_cfg.markers.keys()))
                if "connecting_line" in current_arrow_cfg.markers:
                    del current_arrow_cfg.markers["connecting_line"]  # Remove connecting line for clarity

                # -- current yaw
                current_arrow_cfg.prim_path = "/Visuals/Command/current_yaw"
                self.current_yaw_visualizer = VisualizationMarkers(current_arrow_cfg)
                print("Created current_yaw_visualizer")
            # set their visibility to true
            self.current_yaw_visualizer.set_visibility(True)

            if not hasattr(self, "closest_points_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/State/closest_points",
                    markers={
                        "closest_point": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                        ),
                    },
                )
                # -- closest points
                self.closest_points_visualizer = VisualizationMarkers(marker_cfg)
                print("Created closest_points_visualizer")
            # set their visibility to true
            self.closest_points_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "goal_yaw_visualizer"):
                self.goal_yaw_visualizer.set_visibility(False)
            if hasattr(self, "current_yaw_visualizer"):
                self.current_yaw_visualizer.set_visibility(False)
            if hasattr(self, "closest_points_visualizer"):
                self.closest_points_visualizer.set_visibility(False)




    def _debug_vis_callback(self, event):
        """Update debug markers with new goal positions."""
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        self.goal_yaw_visualizer.visualize(self._desired_pos_w, self._desired_yaw_quat)
        self.current_yaw_visualizer.visualize(self._robot.data.root_pos_w, self._robot.data.root_quat_w)
        self.closest_points_visualizer.visualize(self._closest_points)




    def CHECK_NAN(self, tensor):
            # 1. 计算 NaN 掩码 (GPU 操作)
            nan_mask = torch.isnan(tensor)

            # 2. 计算每行的 NaN 情况 (GPU 操作)
            row_nan_mask = nan_mask.any(dim=1)
            
            # 3. 直接更新 instability 状态 (全 GPU 操作，无 CPU 同步)
            # 假设 self._numerical_instability 也在 GPU 上
            self._numerical_instability.logical_or_(row_nan_mask)
            
            # 4. 原地修复数值 (GPU 操作)
            tensor.nan_to_num_(nan=0.0)
            
            # 注意：这里去掉了 print，因为 print 必须打断 GPU 流水线。
            # 如果确实需要监控，建议使用 TensorBoard 或 wandb 记录 row_nan_mask.sum()
            
            return tensor




    def CHECK_state(self):
        # Limit
        max_angular_velocity = self.cfg.max_angular_velocity_check # rad/s

        # State
        ang_vel_b = self._robot.data.root_ang_vel_b
        # rot_w = torch.stack(euler_xyz_from_quat(self._robot.data.root_quat_w), dim=1) # (num_envs, 3) roll, pitch, yaw
        # rot_w = torch.stack([normallize_angle(rot_w[:, 0]), normallize_angle(rot_w[:, 1]), normallize_angle(rot_w[:, 2])], dim=1)
        # print(f"Roll: {rot_w[0, 0]}, Pitch: {rot_w[0, 1]}, Yaw: {rot_w[0, 2]}")
        # Check if the state is unstable
        state_is_unstable = torch.any(torch.abs(ang_vel_b) > max_angular_velocity, dim=1)

        self._numerical_instability = torch.logical_or(self._numerical_instability, state_is_unstable)




    class EpisodeOutcome(IntEnum):
        ONGOING = 0
        SUCCESS = 1
        DIED = 2
        TIMEOUT = 3




    # TODO: 考虑抽象为回合评估统计对象
    def _update_episode_outcomes_and_metrics(self, env_ids, success_mask, died_mask, timed_out_mask):
        """Update episode outcomes and calculate metrics for completed episodes."""

        # Find completed episodes
        completed_mask = success_mask | died_mask | timed_out_mask
        if not torch.any(completed_mask) or len(env_ids) == 0: # double check
            return 0, 0

        # Update episode outcomes for reset environments
        self._episode_outcomes[env_ids] = torch.where(
            success_mask,
            torch.tensor(self.EpisodeOutcome.SUCCESS, device=self.device),
            torch.where(
                died_mask,
                torch.tensor(self.EpisodeOutcome.DIED, device=self.device),
                torch.tensor(self.EpisodeOutcome.TIMEOUT, device=self.device)
            )
        )
        self._episode_outcome_history.extend(self._episode_outcomes[env_ids])

        # Record final distances to goal
        final_distances = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids],
            dim=1
        ).cpu().tolist()
        self._final_distances.extend(final_distances)

        # Count termination reasons using optimized single-pass approach
        died_env_ids = env_ids[died_mask]
        if len(died_env_ids) > 0:
            # Get termination conditions (convert to CPU once)
            is_unstable = self._numerical_instability[died_env_ids].cpu().numpy()
            is_collision = self._is_contact[died_env_ids].cpu().numpy()
            pos_z = self._robot.data.root_pos_w[died_env_ids, 2].cpu().numpy()

            # Record termination reasons for all failed episodes (vectorized)
            termination_reasons = [
                {
                    "numerical_instability": bool(is_unstable[i]),
                    "collision": bool(is_collision[i]),
                    "too_low": bool(pos_z[i] < 0.2),
                    "too_high": bool(pos_z[i] > 2.8)
                }
                for i in range(len(died_env_ids))
            ]
            self._termination_reason_history.extend(termination_reasons)

        # Update cumulative episode counters
        total_completed = len(env_ids)
        total_succeeded = torch.sum(success_mask).item()
        self._episodes_completed += total_completed
        self._episodes_succeeded += total_succeeded

        # Log metrics
        self._log_metrics(len(self._episode_outcome_history))

        return total_completed, total_succeeded




    def _log_metrics(self, total_episodes):
        """Calculate statistics from episode history and update logs."""
        if total_episodes == 0:
            return

        termination_counts = {"numerical_instability": 0, "collision": 0, "too_low": 0, "too_high": 0}
        if len(self._termination_reason_history) > 0:
            # Single pass through history with vectorized operations where possible
            for reason_dict in self._termination_reason_history:
                # Unroll the inner loop for better performance
                if reason_dict.get("numerical_instability", False):
                    termination_counts["numerical_instability"] += 1
                if reason_dict.get("collision", False):
                    termination_counts["collision"] += 1
                if reason_dict.get("too_low", False):
                    termination_counts["too_low"] += 1
                if reason_dict.get("too_high", False):
                    termination_counts["too_high"] += 1

        # Calculate episode outcome statistics using vectorized approach
        outcome_array = np.array([item.item() for item in self._episode_outcome_history])
        success_count = int(np.sum(outcome_array == self.EpisodeOutcome.SUCCESS))
        died_count = int(np.sum(outcome_array == self.EpisodeOutcome.DIED))
        timeout_count = int(np.sum(outcome_array == self.EpisodeOutcome.TIMEOUT))

        self._success_rate = success_count / total_episodes
        died_rate = died_count / total_episodes
        avg_final_distance = sum(self._final_distances) / len(self._final_distances) if self._final_distances else 0.0

        # Calculate goal queue statistics
        avg_goals_remaining = self._num_goals_remaining.float().mean().item()
        avg_goal_progress = (self.cfg.num_goals - avg_goals_remaining) / self.cfg.num_goals * 100.0

        self.extras["log"].update({
            # Episode termination statistics (as percentages)
            "Metrics / success_rate": self._success_rate * 100.0,
            "Metrics / died_rate": died_rate * 100.0,
            "Metrics / time_out_rate": timeout_count / total_episodes * 100.0,

            # died reason statistics (as percentages of total episodes)
            "Metrics / Died / numerical_instability": termination_counts["numerical_instability"] * died_rate / total_episodes * 100.0,
            "Metrics / Died / collision": termination_counts["collision"] * died_rate / total_episodes * 100.0,
            "Metrics / Died / too_low": termination_counts["too_low"] * died_rate / total_episodes * 100.0,
            "Metrics / Died / too_high": termination_counts["too_high"] * died_rate / total_episodes * 100.0,

            # Performance tracking
            "Metrics / final_distance_to_goal": avg_final_distance,
            # Goal queue tracking
            "Metrics / avg_goals_remaining": avg_goals_remaining,
            "Metrics / avg_goal_progress_percent": avg_goal_progress,

            # Environment configuration
            "Metrics / obstacle_min_distance": self.cfg.obstacle_min_distance_init,
            # "Metrics / hover_hold_requirement_s": float(self._hover_hold_requirement_s),
        })