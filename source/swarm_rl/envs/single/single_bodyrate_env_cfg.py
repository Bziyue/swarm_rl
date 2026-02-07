# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the single bodyrate quadcopter environment."""

from __future__ import annotations

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg, DomeLightCfg
from isaaclab.utils import configclass

from swarm_rl.utils.depth_camera_array import DepthCameraArrayCfg, DepthCameraItemCfg
from swarm_rl.utils.point_provider import PointProviderCfg

from swarm_rl.utils.quadcopter import DJI_FPV_CFG
from swarm_rl.utils.asset_paths import get_dji_fpv_usd_path


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: DirectRLEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterObsCfg:
    """Configuration for quadcopter observations."""

    # 观测缩放因子
    ang_vel_scale: float         = 30.0     # 角速度
    rot_mat_scale: float         = 127.0    # 旋转矩阵
    desired_speed_scale: float   = 25.0     # 期望速度
    goal_feat_scale: float       = 127.0    # 目标点特征
    last_action_scale: float     = 127.0    # 上一步动作
    depth_image_scale: float     = 127.0    # 深度图像

    lin_vel_scale: float         = 25.0     # 线速度
    pos_to_goal_scale: float     = 127.0    # 目标点相对位置
    pos_to_obstacle_scale: float = 127.0    # 障碍物相对位置

    # 观测裁剪参数
    max_goal_distance: float     = 5.0      # 目标点最大距离
    max_obstacle_distance: float = 1.0      # 障碍物最大距离


@configclass
class QuadcopterRewardCfg:
    """Configuration for quadcopter reward."""
    
    # reward 权重
    coef_distance_reward: float             = 5.0
    coef_yaw_direction_penalty: float       = 0.0
    coef_action_magnitude_penalty: float    = 0.0
    coef_action_change_penalty: float       = 0.1
    coef_vel_direction_penalty: float       = 0.15
    coef_vel_speed_excess_penalty: float    = 0.5
    coef_vel_speed_match_reward: float      = 0.0
    coef_z_position_penalty: float          = 0.25
    coef_obstacle_collision_penalty: float  = 80.0
    coef_esdf_reward: float                 = 0.1
    coef_succeed_reward: float              = 200.0
    coef_max_ang_vel_penalty: float         = 0.0
    coef_max_angle_penalty: float           = 0.0
    coef_alive_reward: float                = 0.0
    coef_z_vel_penalty: float               = 0.0
    # # Position control rewards
    # coef_lin_vel_reward_scale: float = 0
    # coef_ang_vel_reward_scale: float = 0
    # coef_distance_to_goal_reward_scale: float = 0

    # reward 计算参数
    delta_distance_clamp: float         = 3.0 / 15.0   # 接近目标速率奖励的上下界 (3m/s / 15hz)
    # distance_goal_mapping_scale = 0.8   # Scale factor for distance-to-goal mapping
    speed_adjustment_distance: float    = 1.0           # Distance to start speed adjustment
    z_position_huber_delta: float       = 0.3           # Transition point between quadratic and linear z penalty
    vel_direction_huber_delta: float    = 0.2           # Transition point for velocity direction Huber penalty

    max_angular_velocity_penalty: float = 3.14 / 4.0    # rad/s for penalty
    max_angle_penalty: float = 3.14 / 4.0               # rad for angle penalty


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    """Configuration for the single quadcopter environment."""

    # ========================================================================
    # Basic Configuration
    # ========================================================================

    # Spawn mode configuration
    spawn_mode = "free_points"  # Options: "edges", "free_points"

    # Task queue configuration
    num_goals = 1  # Number of goals in the task queue for each robot

    # ========================================================================
    # Environment Timing Configuration
    # ========================================================================

    episode_length_s = 80 * num_goals  # 最大回合长度 (s)
    step_freq    = 15.0     # 策略控制频率 (Hz)
    decimation   = 10       # 物理步/控制步 (Physics sub-steps)

    # ========================================================================
    # Robot Configuration
    # ========================================================================

    # Robot physical parameters
    robot_mass = 1.0  # kg
    robot_inertia = [6.8e-4, 4.8e-4, 8.5e-4]  # kg*m^2 [Ixx, Iyy, Izz]
    thrust_weight_ratio = 4.0   # 推重比 (thrust_max = thrust_weight_ratio * robot_mass * 9.81)
    bodyrate_max = 6.0          # 最大角速度 (rad/s)
    # Mass randomization configuration
    enable_mass_randomization = False
    mass_randomization_percent = 0.2  # +/- percentage of mass randomization
    # Angular stability check
    max_angular_velocity_check = 3.14 * 2.0 * 20.0  # rad/s

    # Range of desired velocities in m/s
    # max_lin_vel = 6.0  # 最大线速度 (m/s)
    des_vel_range = [6.0, 6.0]

    # Controller parameters
    kp_bodyrate = [20, 20, 20]

    # ========================================================================
    # Map Generation Configuration
    # ========================================================================

    # TODO: 考虑抽象为多地图管理器对象
    # Map generation parameters
    map_generation_step_threshold = 3000000000000 * num_goals   # 暂时不更新地图
    obstacle_min_distance_init = 1.5
    obstacle_min_distance_min = 0.4
    map_spacing_factor = 0.8  # Factor for map spacing relative to env_spacing
    obstacle_size_range = (0.4, 0.8)  # Size range for obstacles
    floater_size_range = (0.4, 0.8)  # Size range for floating obstacles
    # goal_sampling_noise_range = 0.15  # Noise range for goal position sampling

    # ========================================================================
    # Curriculum Configuration
    # ========================================================================

    # # Hover stabilization curriculum
    # hover_hold_initial_s = 0.0
    # hover_hold_max_s = 2.0
    # hover_hold_increment_s = 0.1
    # hover_hold_distance_threshold = 0.5
    # hover_hold_speed_threshold = 0.5
    hover_yaw_penalty_distance = 0.2

    point_provider: PointProviderCfg = PointProviderCfg(
        target_hold_speed_threshold = 1.0,
        target_hold_threshold_s = 0.4,
    )

    # ========================================================================
    # Environmental Effects Configuration
    # ========================================================================

    # TODO: 待测试并加入油门不确定度、风扰动
    # # Wind generation parameters
    # wind_tau = 1.0  # Wind correlation time constant
    # wind_sigma = 0.5  # Wind disturbance intensity
    # # Thrust uncertainty configuration
    # enable_thrust_uncertainty = True
    # thrust_uncertainty_initial_range = (0.85, 1.0)  # Initial effectiveness range
    # thrust_uncertainty_degradation_range = (0.10, 0.15)  # Degradation factor range
    # thrust_uncertainty_min_effectiveness = 0.7  # Minimum thrust effectiveness

    # TODO: 待测试并加入高度平滑
    # # Height randomization configuration
    # enable_height_randomization = False
    # height_randomization_range = (0.5, 1.5)  # Height range in meters (min, max)
    # height_randomization_climb_rate = 0.2  # Maximum climb/descent rate in m/s
    # height_randomization_waypoint_distance = 8.0  # Distance between waypoint changes in meters
    # height_randomization_noise_scale = 0.1  # Amplitude of natural pilot variation
    # height_randomization_debug_dir = None

    # ========================================================================
    # Metrics and Logging Configuration
    # ========================================================================

    # Success rate tracking
    success_rate_window_size = 100  # Number of episodes for success rate calculation

    # ========================================================================
    # Data Recording Configuration
    # ========================================================================

    # TODO: 待评估是否需要加入数据收集器
    # TODO: 待评估是否需要加入死亡回放收集器

    # ========================================================================
    # Scene Configuration
    # ========================================================================

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1000,
        env_spacing=16.0,
        replicate_physics=True,
        # clone_in_fabric=True
    )

    # Robot
    robot: ArticulationCfg = DJI_FPV_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=DJI_FPV_CFG.spawn.replace(usd_path=get_dji_fpv_usd_path()),
    )

    # ========================================================================
    # Sensor Configuration
    # ========================================================================

    # Contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        history_length=decimation,
        update_period=0.0,
        track_air_time=False,
        debug_vis=False,
    )
    contact_force_threshold = 0.01  # Minimum contact force for collision detection

    # Depth camera array
    image_width: int = 64
    image_height: int = 32
    camera_num: int = 4
    depth_cameras: DepthCameraArrayCfg = DepthCameraArrayCfg(
        cameras = [
            DepthCameraItemCfg(name="front", pos_BC=( 0.02,  0.0,  0.0), quat_BC=(1.0, 0.0, 0.0, 0.0), ),
            DepthCameraItemCfg(name="right", pos_BC=( 0.0,  -0.02, 0.0), quat_BC=(0.70710678, 0.0, 0.0, -0.70710678), ),
            DepthCameraItemCfg(name="back",  pos_BC=(-0.02,  0.0,  0.0), quat_BC=(0.0, 0.0, 0.0, 1.0), ),
            DepthCameraItemCfg(name="left",  pos_BC=( 0.0,   0.02, 0.0), quat_BC=(0.70710678, 0.0, 0.0, 0.70710678), ),
        ],

        # 广播字段：写一次默认所有相机通用
        resolution=(image_width, image_height),     # (W, H)
        K=[388.963/(640/image_width), 0.0,             317.04/(640/image_width),
           0.0,              388.963/(480/image_height), 241.99/(480/image_height),
           0.0,              0.0,              1.0,],

        prim_path = "/World/envs/env_.*/Robot/body",
        mesh_prim_paths = ["/map_mesh"],
        max_distance = 4.0,
        depth_clipping_behavior = "max",
        data_type = "distance_to_image_plane",
        update_period = 0.0,
        debug_vis = False,

        usd_focal_length=24.0,
        
        normalize = "0_1",      # 或 "none" / "-1_1"
        flatten = False,

        invalid_rate_max = 0.15,
        invalid_sampling = "per_frame",
        invalid_fill_value = "max_distance",
    )

    # ========================================================================
    # Space Configuration
    # ========================================================================

    # action
    action_space = 4  # 1 (thrust) + 3 (bodyrate x, y, z)

    # obs-policy
    observation_space = {
        "image": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(1, image_height, image_width * camera_num), dtype="float32"),
        # 20 = 3 (gyro) + 9 (rot) + 3 (goal) + 1 (speed) + 4 (actions)
        "state": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(20,), dtype="float32"),
    }

    # obs-critic
    state_space = {
        "image": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(1, image_height, image_width * camera_num), dtype="float32"),
        # 29 = 3 (gyro) + 9 (rot) + 3 (goal) + 1 (speed) + 4 (actions) + 3 (vel) + 3 (goal_dir) + 3 (obstacle_pos)
        "state": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(29,), dtype="float32"),
    }

    # ========================================================================
    # Simulation Infrastructure
    # ========================================================================

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / (step_freq * decimation),  # 物理仿真步长 (s)
        render_interval=decimation,         # 渲染与策略控制频率对齐
        physx=sim_utils.PhysxCfg(
            gpu_collision_stack_size=2**28, # 增加碰撞堆栈到 2**28 (256MB) 防止溢出
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        render=sim_utils.RenderCfg(
            enable_dl_denoiser=False,
            dlss_mode=2,
        )
    )

    # Light
    light: DomeLightCfg = DomeLightCfg(
        intensity=2000.0,
        color=(0.75, 0.75, 0.75),
    )

    # Viewer
    viewer: ViewerCfg = ViewerCfg()

    # UI settings
    ui_window_class_type: type = QuadcopterEnvWindow

    # ========================================================================
    # Reward Configuration
    # ========================================================================

    reward : QuadcopterRewardCfg = QuadcopterRewardCfg()

    # ========================================================================
    # Observation Configuration
    # ========================================================================
    
    observations: QuadcopterObsCfg = QuadcopterObsCfg()


    def __post_init__(self):
        super().__post_init__()
        