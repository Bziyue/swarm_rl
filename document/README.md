# FAST-Swarm-RL 项目文档

> **基于 Isaac Lab 的多无人机集群强化学习仿真框架**

---

## 📋 目录

1. [项目概述](#1-项目概述)
2. [项目架构](#2-项目架构)
3. [环境系统详解](#3-环境系统详解)
4. [控制与规划算法](#4-控制与规划算法)
5. [强化学习训练框架](#5-强化学习训练框架)
6. [快速开始指南](#6-快速开始指南)
7. [配置文件说明](#7-配置文件说明)
8. [API 参考](#8-api-参考)
9. [常见问题](#9-常见问题)

---

## 1. 项目概述

### 1.1 简介

FAST-Swarm-RL 是一个基于 NVIDIA Isaac Lab 的多无人机集群强化学习仿真框架。该项目支持：

- **单机控制**：单个无人机的速度/加速度/航点控制
- **集群控制**：多无人机协同导航与避障
- **多种RL算法**：支持 PPO、IPPO、MAPPO 等算法
- **多框架支持**：支持 SKRL、RSL-RL、Stable-Baselines3

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| 🚁 无人机模型 | Crazyflie、DJI FPV |
| 🤖 控制模式 | Velocity、Acceleration、Bodyrate、Waypoint |
| 🎯 集群任务 | Migration、Crossover、Chaotic |
| 🧠 RL算法 | PPO、IPPO、MAPPO |
| 📊 域随机化 | 观测噪声、时延、丢包模拟 |
| 🎮 遥控支持 | 键盘遥控单机飞行 |

### 1.3 支持的训练任务

```
单机环境 (Quadcopter):
├── FAST-Quadcopter-Bodyrate  # 机体角速度控制
├── FAST-Quadcopter-Vel       # 速度控制
├── FAST-Quadcopter-Acc       # 加速度控制
├── FAST-Quadcopter-Waypoint  # 航点跟踪
├── FAST-RGB-Waypoint         # 基于RGB相机的航点跟踪
└── FAST-Depth-Waypoint       # 基于深度相机的航点跟踪

集群环境 (Swarm):
├── FAST-Swarm-Bodyrate       # 集群角速度控制
├── FAST-Swarm-Vel            # 集群速度控制
├── FAST-Swarm-Acc            # 集群加速度控制
├── FAST-Swarm-AJ             # 集群加加速度(Jerk)控制
└── FAST-Swarm-Waypoint       # 集群航点控制
```

---

## 2. 项目架构

### 2.1 目录结构

```
swarm-rl/
├── assets/                    # 3D模型资源
│   ├── crazyfile/            # Crazyflie无人机模型
│   ├── dji_fpv/              # DJI FPV无人机模型
│   ├── flat_plane/           # 地面平面
│   └── black_oak_fall/       # 森林环境
│
├── config/                    # 配置文件
│   ├── agents/               # RL算法配置
│   │   ├── quadcopter_sb3_ppo_cfg.yaml
│   │   ├── swarm_skrl_ppo_cfg.yaml
│   │   ├── swarm_skrl_ippo_cfg.yaml
│   │   ├── swarm_skrl_mappo_cfg.yaml
│   │   └── swarm_rsl_rl_ppo_cfg.py
│   └── *_plotjuggler_cfg.xml # PlotJuggler可视化配置
│
├── envs/                      # 环境实现
│   ├── quadcopter.py         # 无人机配置定义
│   ├── quadcopter_vel_env.py
│   ├── quadcopter_acc_env.py
│   ├── quadcopter_bodyrate_env.py
│   ├── quadcopter_waypoint_env.py
│   ├── quadcopter_pvajyyd_env.py
│   ├── swarm_vel_env.py
│   ├── swarm_acc_env.py
│   ├── swarm_aj_env.py
│   ├── swarm_bodyrate_env.py
│   ├── swarm_waypoint_env.py
│   └── camera_waypoint_env.py
│
├── reinforcement_learning/    # RL训练脚本
│   ├── skrl/
│   │   ├── train.py          # SKRL训练入口
│   │   └── play.py           # SKRL推理入口
│   ├── rsl_rl/
│   │   ├── train.py
│   │   └── play.py
│   └── sb3/
│       ├── train.py
│       └── play.py
│
├── scripts/                   # 实用脚本
│   ├── teleop.py             # 键盘遥控
│   ├── rvo.py                # RVO避障算法
│   ├── lissajous.py          # Lissajous轨迹
│   └── eight.py              # 八字轨迹
│
├── utils/                     # 工具库
│   ├── controller.py         # 无人机控制器
│   ├── minco.py              # MINCO轨迹优化
│   ├── custom_trajs.py       # 自定义轨迹生成
│   └── utils.py              # 辅助函数
│
├── outputs/                   # 训练输出
│   └── skrl/                 # SKRL训练结果
│
└── document/                  # 项目文档
    └── README.md             # 本文件
```

### 2.2 核心依赖

```python
# 仿真平台
isaaclab          # NVIDIA Isaac Lab 仿真框架
isaacsim          # NVIDIA Omniverse Isaac Sim

# 强化学习
skrl              # 推荐，支持 JAX/Torch
rsl_rl            # ETH Zurich RL库
stable-baselines3 # SB3算法库

# 机器人中间件
rclpy             # ROS2 Python客户端

# 其他
torch             # PyTorch深度学习框架
loguru            # 日志库
gymnasium         # RL环境接口标准
```

---

## 3. 环境系统详解

### 3.1 环境层次结构

```
Isaac Lab Base Env
      │
      ├── DirectRLEnv (单机环境基类)
      │         │
      │         ├── QuadcopterVelEnv      # 速度控制环境
      │         ├── QuadcopterAccEnv      # 加速度控制环境
      │         ├── QuadcopterBodyrateEnv # 角速度控制环境
      │         ├── QuadcopterWaypointEnv # 航点跟踪环境
      │         └── CameraWaypointEnv     # 相机感知环境
      │
      └── DirectMARLEnv (多智能体环境基类)
                │
                ├── SwarmVelEnv      # 集群速度控制
                ├── SwarmAccEnv      # 集群加速度控制
                ├── SwarmAJEnv       # 集群Jerk控制
                ├── SwarmBodyrateEnv # 集群角速度控制
                └── SwarmWaypointEnv # 集群航点控制
```

### 3.2 单机环境详解

#### QuadcopterVelEnv（速度控制环境）

**控制目标**：控制无人机以指定速度飞行到目标位置

```python
# 观测空间 (observation_space = 6)
obs = [body2goal_w(3), root_lin_vel_w(3)]
#       到目标的相对位置    机体线速度

# 动作空间 (action_space = 2)
action = [v_x_desired, v_y_desired]  # 归一化的期望速度XY
```

**核心配置参数**：

```python
@configclass
class QuadcopterVelEnvCfg(DirectRLEnvCfg):
    # 奖励权重
    approaching_goal_reward_weight = 1.0    # 接近目标奖励
    success_reward_weight = 100.0           # 到达目标奖励
    dist_to_goal_reward_weight = 0.0        # 距离目标奖励
    
    # 频率设置
    physics_freq = 200.0    # 物理仿真频率 (Hz)
    control_freq = 100.0    # 控制器频率 (Hz)
    action_freq = 20.0      # 策略动作频率 (Hz)
    
    # 任务参数
    flight_altitude = 1.0           # 飞行高度 (m)
    success_distance_threshold = 0.5 # 成功距离阈值 (m)
    v_max = 2.0                      # 最大速度 (m/s)
```

**奖励函数设计**：

```python
reward = (
    approaching_goal_reward * approaching_goal_reward_weight +
    dist_to_goal_reward * dist_to_goal_reward_weight +
    success_reward * success_reward_weight +
    ang_vel_reward * ang_vel_penalty_weight +
    action_temporal_smoothness_reward * action_temporal_smoothness_reward_weight
)
```

### 3.3 集群环境详解

#### SwarmVelEnv（集群速度控制环境）

**核心特点**：
- 多智能体协同（默认13架无人机）
- 分布式控制（每架无人机独立决策）
- 碰撞避免（机间距离约束）
- 域随机化支持

**观测空间设计**（分布式部分可观测）：

```python
# 瞬态观测 (transient_observasion_dim = 6 + 4 * (num_drones - 1))
obs_transient = [
    # 自身观测 (6维)
    action(2),              # 上一时刻动作
    body2goal_w(2),         # 到目标的相对位置XY
    delayed_lin_vel_w(2),   # 带时延的线速度XY
    
    # 邻居观测 (4 * 12 = 48维) - 按距离排序
    rel_pos_b_noisy(3),     # 邻居在机体坐标系下的位置
    observability_mask(1),  # 可观测性掩码
]

# 历史观测 (history_length = 3)
obs_stacked = concat([obs_t-2, obs_t-1, obs_t])  # 共 54 * 3 = 162 维
```

**集群任务类型**：

```python
mission_names = ["migration", "crossover", "chaotic"]
mission_prob = [0.0, 1.0, 0.0]  # 当前只使用crossover任务

# 1. Migration（迁徙任务）
#    - 初始状态：聚集分布
#    - 目标：统一目标位置
#    - 特点：需要保持编队飞行

# 2. Crossover（交叉穿越任务）
#    - 初始状态：圆环上均匀/随机分布
#    - 目标：圆环对面位置
#    - 特点：大规模交叉避障场景

# 3. Chaotic（混沌任务）
#    - 初始状态：随机分布
#    - 目标：各自独立随机目标
#    - 特点：完全分布式任务
```

**域随机化配置**：

```python
# 观测时延
lin_vel_obs_delay_ms = 40.0    # 线速度观测时延
rel_pos_obs_delay_ms = 40.0    # 相对位置观测时延

# 观测噪声
lin_vel_noise_std = 0.1        # 线速度噪声标准差
min_dist_noise_std = 0.05      # 近距离距离噪声
max_dist_noise_std = 1.0       # 远距离距离噪声
min_bearing_noise_std = 0.1    # 近距离方位角噪声
max_bearing_noise_std = 0.25   # 远距离方位角噪声

# 观测丢包
drop_prob = 0.1                # 观测丢包概率
```

---

## 4. 控制与规划算法

### 4.1 控制器架构（controller.py）

```
高层指令 (来自RL策略)
    │
    ▼
┌─────────────────┐
│ Position Control │  PID位置控制: kPp, kPv
│  (p_desired)    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Flatness Trans. │  微分平坦性转换
│  (a_desired)    │  计算期望推力、姿态、角速度
└────────┬────────┘
         ▼
┌─────────────────┐
│ Attitude Control│  姿态反馈控制: kPR
│  (q_desired)    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Bodyrate Control│  机体角速度控制: kPw, kIw, kDw
│  (w_desired)    │  计算最终力矩
└────────┬────────┘
         ▼
    推力 & 力矩 → 物理仿真器
```

### 4.2 微分平坦性转换

```python
def flatness_with_drag(v_desired, a_desired, j_desired, yaw_desired, 
                       yaw_dot_desired, mass, gravity, cp, dv, dh, veps):
    """
    将高阶运动学量转换为低维控制输入
    
    输入：期望速度、加速度、加加速度、期望偏航角及角速度
    输出：期望推力(thrust)、期望姿态(quaternion)、期望角速度(omega)
    """
    # 核心思想：四旋翼的微分平坦特性
    # 允许通过规划位置的高阶导数来直接计算控制输入
```

### 4.3 轨迹生成（MINCO）

**MINCO**（Minimum Control Effort Optimization）：

```python
class MinJerkOpt:
    """
    最小加加速度轨迹优化器
    
    生成平滑的6阶多项式轨迹，满足：
    - 起点/终点的位置、速度、加速度约束
    - 中间路径点约束
    - 时间分配优化
    """
    
    def generate(self, inner_pts, durations):
        # 构建带状矩阵系统 BandedSystem
        # 求解线性方程组获得多项式系数
        pass
    
    def get_traj(self):
        # 返回 Trajectory 对象
        # 支持 get_pos(t), get_vel(t), get_acc(t), get_jer(t)
        pass
```

**Lissajous轨迹**：

```python
@configclass
class LissajousConfig:
    A_range = [2.5, 5.0]      # X轴振幅范围
    B_range = [2.5, 5.0]      # Y轴振幅范围
    ratio = [(1,1), (1,2), (1,3), ...]  # 频率比
    num_pieces = 128          # 轨迹段数
    max_exec_speed = 6.0      # 最大执行速度
```

---

## 5. 强化学习训练框架

### 5.1 SKRL 训练流程

```bash
# 训练命令示例
python reinforcement_learning/skrl/train.py \
    --task FAST-Swarm-Vel \
    --num_envs 10000 \
    --seed 42 \
    --max_iterations 10000 \
    --algorithm PPO
```

**训练参数**：

```python
# 来自 swarm_skrl_ppo_cfg.yaml
agent:
  rollouts: 24              # 每次更新的环境步数
  learning_epochs: 5        # 每次采样的学习轮数
  mini_batches: 4           # 小批量数量
  learning_rate: 5.0e-4     # 学习率
  
  # 探索
  experiment:
    checkpoint_interval: 1000    # 检查点保存间隔
    write_interval: 100          # 日志写入间隔

models:
  separate: True            # 是否分离Actor和Critic
  policy:
    class: GaussianMixin
    clip_actions: True
    clip_log_std: True
    min_log_std: -20.0
    max_log_std: 2.0
    input_shape: "Shape.ONE"  # 自动推断
    hiddens: [512, 512, 256, 128]
    hidden_activation: ["elu", "elu", "elu", "elu"]
    output_shape: "Shape.ACTIONS"
    output_activation: null
    output_scale: 1.0
```

### 5.2 多智能体算法支持

| 算法 | 说明 | 适用场景 |
|------|------|---------|
| PPO | 单智能体PPO | 单机控制、集群集中式训练 |
| IPPO | 独立PPO | 集群分布式训练，无通信 |
| MAPPO | 多智能体PPO | 集群训练，支持状态共享 |

> 📖 **详细网络架构**: 查看 [NETWORK_ARCHITECTURE.md](./NETWORK_ARCHITECTURE.md) 了解各算法的网络结构、层配置和参数量。

**算法选择**：

```python
# IPPO配置
swarm_skrl_ippo_cfg.yaml:
  - 每个智能体独立优化
  - 观测不共享（仅使用自身和邻居观测）
  - 适合大规模集群

# MAPPO配置  
swarm_skrl_mappo_cfg.yaml:
  - 中央Critic访问全局状态
  - Actor仅使用局部观测
  - 适合需要协作的任务
```

### 5.3 训练监控

```python
# 自动记录的指标
episode_reward/              # 回合奖励
├── meaning_to_live          # 存活奖励
├── approaching_goal         # 接近目标奖励
├── success                  # 成功到达奖励
├── mutual_collision_penalty # 碰撞惩罚
└── ...

Episode_Termination/         # 回合终止原因
├── died                     # 坠毁次数
└── time_out                 # 超时次数

# 使用 PlotJuggler 可视化
cd config/
plotjuggler --layout swarm_vel_env_plotjuggler_cfg.xml
```

---

## 6. 快速开始指南

### 6.1 环境安装

```bash
# 1. 安装 Isaac Lab (参考官方文档)
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

# 2. 安装本项目依赖
pip install -e .

# 3. 安装 ROS2 (用于调试发布)
# 参考 https://docs.ros.org/en/humble/Installation.html
```

### 6.2 单机训练示例

```bash
# 速度控制环境训练
python reinforcement_learning/skrl/train.py \
    --task FAST-Quadcopter-Vel \
    --num_envs 4096 \
    --seed 42

# 恢复训练
python reinforcement_learning/skrl/train.py \
    --task FAST-Quadcopter-Vel \
    --checkpoint outputs/skrl/FAST-Quadcopter-Vel/xxx/checkpoints/best_agent.pt
```

### 6.3 集群训练示例

```bash
# 集群速度控制（IPPO算法）
python reinforcement_learning/skrl/train.py \
    --task FAST-Swarm-Vel \
    --num_envs 10000 \
    --algorithm IPPO \
    --seed 42 \
    --run_id my_swarm_experiment

# 集群加速度控制（MAPPO算法）
python reinforcement_learning/skrl/train.py \
    --task FAST-Swarm-Acc \
    --num_envs 8192 \
    --algorithm MAPPO
```

### 6.4 模型推理

```bash
# 使用训练好的模型
python reinforcement_learning/skrl/play.py \
    --task FAST-Swarm-Vel \
    --num_envs 100 \
    --checkpoint outputs/skrl/FAST-Swarm-Vel/xxx/checkpoints/best_agent.pt
```

### 6.5 键盘遥控

```bash
# 启动键盘遥控
python scripts/teleop.py --task FAST-Quadcopter-Vel --num_envs 1

# 控制按键说明
# W/S: 前进/后退
# A/D: 左移/右移
# Q/E: 上升/下降 (仅航点模式)
```

---

## 7. 配置文件说明

### 7.1 环境配置参数对照表

#### 通用参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `episode_length_s` | float | 30.0 | 回合最大时长(秒) |
| `physics_freq` | float | 200.0 | 物理仿真频率(Hz) |
| `action_freq` | float | 20.0 | 策略控制频率(Hz) |
| `num_envs` | int | 10000 | 并行环境数量 |
| `debug_vis` | bool | True | 是否开启调试可视化 |

#### 奖励参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `approaching_goal_reward_weight` | float | 10.0 | 接近目标奖励权重 |
| `success_reward_weight` | float | 10.0 | 成功到达奖励权重 |
| `mutual_collision_penalty_weight` | float | 100.0 | 机间碰撞惩罚权重 |
| `ang_vel_penalty_weight` | float | 0.0 | 角速度惩罚权重 |

#### 域随机化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable_domain_randomization` | bool | True | 是否启用域随机化 |
| `lin_vel_obs_delay_ms` | float | 40.0 | 线速度观测时延(ms) |
| `lin_vel_noise_std` | float | 0.1 | 线速度噪声标准差 |
| `drop_prob` | float | 0.1 | 观测丢包概率 |

### 7.2 集群特定参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `num_drones` | int | 13 | 每环境无人机数量 |
| `flight_range` | float | 5.0 | 飞行范围(m) |
| `safe_dist` | float | 1.0 | 安全距离阈值(m) |
| `collide_dist` | float | 0.5 | 碰撞距离阈值(m) |
| `mission_prob` | list | [0,1,0] | 任务类型概率分布 |
| `use_custom_traj` | bool | True | 是否使用自定义轨迹 |

---

## 8. API 参考

### 8.1 环境注册

```python
# 所有环境自动注册到gymnasium
import gymnasium as gym

env = gym.make(
    "FAST-Swarm-Vel",
    cfg=env_cfg,
    render_mode="rgb_array"
)
```

### 8.2 Controller 类

```python
from utils.controller import Controller

controller = Controller(
    step_dt=0.01,           # 控制步长
    gravity=torch.tensor([0, 0, -9.81]),
    mass=torch.tensor(1.0),
    inertia=torch.eye(3),
    num_envs=1000
)

# 计算控制指令
total_des_acc, thrust, quat, omega, torque = controller.get_control(
    state=robot_state,      # [num_envs, 13] - pos(3), quat(4), vel(3), ang_vel(3)
    action=desired_state    # [num_envs, 14] - p(3), v(3), a(3), j(3), yaw(1), yaw_dot(1)
)
```

### 8.3 轨迹生成

```python
from utils.custom_trajs import generate_custom_trajs, LissajousConfig
from utils.minco import MinJerkOpt

# 生成Lissajous轨迹
cfg = LissajousConfig()
cfg.A_range = [3.0, 6.0]
cfg.num_pieces = 128

trajectories = generate_custom_trajs(
    type_id="lissajous",
    p_odom=current_pos,      # [num_trajs, 3]
    v_odom=current_vel,
    a_odom=current_acc,
    p_init=init_pos,
    custom_cfg=cfg
)

# 查询轨迹状态
pos = trajectories.get_pos(t)    # [num_trajs, 3]
vel = trajectories.get_vel(t)    # [num_trajs, 3]
```

### 8.4 观测处理工具

```python
from utils.utils import quat_to_ang_between_z_body_and_z_world

# 计算机体Z轴与世界Z轴夹角（用于判断坠毁）
angle = quat_to_ang_between_z_body_and_z_world(quat)
# angle > 80度认为是坠毁
```

---

## 9. 常见问题

### Q1: 训练时出现显存不足

```python
# 解决方案：减少并行环境数量
python reinforcement_learning/skrl/train.py \
    --task FAST-Swarm-Vel \
    --num_envs 4096  # 从10000减少
```

### Q2: 如何调整观测时延

```python
# 修改环境配置
@configclass
class SwarmVelEnvCfg(DirectMARLEnvCfg):
    lin_vel_obs_delay_ms = 20.0   # 减小到20ms
    rel_pos_obs_delay_ms = 20.0
```

### Q3: 如何添加新的无人机模型

```python
# 在 envs/quadcopter.py 中添加
NEW_DRONE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/your_drone/model.usd",
        ...
    ),
    ...
)

# 在环境配置中引用
@configclass
class YourEnvCfg(DirectRLEnvCfg):
    robot: ArticulationCfg = NEW_DRONE_CFG.replace(...)
```

### Q4: 如何可视化训练过程

```bash
# 1. 使用 TensorBoard
tensorboard --logdir outputs/skrl/

# 2. 使用 PlotJuggler
plotjuggler --layout config/swarm_vel_env_plotjuggler_cfg.xml
```

### Q5: 集群环境中如何选择任务

```python
# 在环境配置中设置任务概率
@configclass
class SwarmVelEnvCfg(DirectMARLEnvCfg):
    mission_names = ["migration", "crossover", "chaotic"]
    mission_prob = [0.0, 1.0, 0.0]  # 只使用crossover
    
    # 或者混合使用
    mission_prob = [0.3, 0.5, 0.2]  # 30% migration, 50% crossover, 20% chaotic
```

---

## 附录：术语表

| 术语 | 说明 |
|------|------|
| Isaac Lab | NVIDIA开发的机器人学习仿真框架 |
| Isaac Sim | NVIDIA的物理仿真平台 |
| PPO | Proximal Policy Optimization，近端策略优化算法 |
| IPPO | Independent PPO，独立多智能体PPO |
| MAPPO | Multi-Agent PPO，多智能体PPO |
| MINCO | Minimum Control Effort，最小控制 effort 轨迹优化 |
| Domain Randomization | 域随机化，提高sim2real迁移能力的技术 |
| Decimation | 控制频率降采样，将高频物理仿真与低频策略控制解耦 |
| Flatness | 微分平坦性，四旋翼的一种特殊动态特性 |

---

**文档版本**: v1.0  
**最后更新**: 2025-02-04  
**项目主页**: [swarm-rl](https://github.com/your-org/swarm-rl)
