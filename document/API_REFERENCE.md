# API 参考

本文档提供 FAST-Swarm-RL 核心 API 的详细参考。

---

## 环境模块 (envs)

### quadcopter.py

#### `CRAZYFLIE_CFG`

Crazyflie 四旋翼无人机配置。

```python
CRAZYFLIE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/crazyfile/cf2x.usd",
        # 质量: ~0.03 kg
        # 尺寸: 0.1m x 0.1m
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
    ),
)
```

#### `DJI_FPV_CFG`

DJI FPV 无人机配置（默认使用）。

```python
DJI_FPV_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/dji_fpv/v1/dji_fpv.usd",
        # 质量: ~0.8 kg
        # 尺寸: ~0.25m x 0.25m
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
    ),
)
```

---

### 单机环境

#### `QuadcopterVelEnv`

速度控制环境。

```python
class QuadcopterVelEnv(DirectRLEnv):
    """
    参数:
        cfg: QuadcopterVelEnvCfg 配置对象
        render_mode: 渲染模式，可选 "rgb_array" 或 None
    
    观测空间: Box(-inf, inf, (6,), float32)
        - [0:3]: body2goal_w, 到目标的相对位置
        - [3:6]: root_lin_vel_w, 机体线速度
    
    动作空间: Box(-1, 1, (2,), float32)
        - [0]: v_x_desired / v_max
        - [1]: v_y_desired / v_max
    """
    
    def __init__(self, cfg: QuadcopterVelEnvCfg, render_mode: str | None = None, **kwargs)
```

#### `QuadcopterAccEnv`

加速度控制环境。

```python
class QuadcopterAccEnv(DirectRLEnv):
    """
    观测空间: Box(-inf, inf, (9,), float32)
        - [0:3]: body2goal_w
        - [3:6]: root_lin_vel_w
        - [6:9]: root_acc_w
    
    动作空间: Box(-1, 1, (2,), float32)
        - [0]: a_x_desired / a_max
        - [1]: a_y_desired / a_max
    """
```

#### `QuadcopterWaypointEnv`

航点跟踪环境。

```python
class QuadcopterWaypointEnv(DirectRLEnv):
    """
    参数:
        num_pieces: 航点数量，默认 4
        p_max: 最大航点距离，默认 1.0m
    
    观测空间: Box(-inf, inf, (3 * num_pieces,), float32)
        - 每个航点的相对位置（机体坐标系）
    
    动作空间: Box(-1, 1, (3 * num_pieces,), float32)
        - 每个航点的相对位置 / p_max
    """
```

---

### 集群环境

#### `SwarmVelEnv`

集群速度控制环境。

```python
class SwarmVelEnv(DirectMARLEnv):
    """
    参数:
        num_drones: 无人机数量，默认 13
        history_length: 观测历史长度，默认 3
        mission_names: 任务类型列表
        mission_prob: 任务类型概率
    
    观测空间 (每架无人机): Box(-inf, inf, (history_length * transient_dim,), float32)
        transient_dim = 6 + 4 * (num_drones - 1)
        - [0:2]: 上一时刻动作
        - [2:4]: 到目标XY相对位置
        - [4:6]: 带时延的线速度XY
        - [6:]: 邻居相对位置（机体坐标系）+ 可观测性掩码
    
    动作空间 (每架无人机): Box(-1, 1, (2,), float32)
        - [0]: v_x_desired / v_max
        - [1]: v_y_desired / v_max
    
    方法:
        _get_observations() -> dict[str, torch.Tensor]
            返回每个无人机的观测字典
            
        _get_rewards() -> dict[str, torch.Tensor]
            返回每个无人机的奖励字典
            
        _get_states() -> torch.Tensor
            返回全局状态（用于中央Critic）
    """
```

---

## 控制器模块 (utils/controller.py)

### `Controller`

四旋翼无人机控制器，实现位置-姿态-角速度级联控制。

```python
class Controller:
    """
    级联控制器：位置 -> 加速度 -> 推力/姿态 -> 角速度 -> 力矩
    
    参数:
        step_dt: 控制步长（秒）
        gravity: 重力向量，shape (3,)
        mass: 无人机质量（kg）
        inertia: 转动惯量矩阵，shape (3, 3)
        num_envs: 并行环境数量
    
    属性:
        kPp: 位置控制P增益，默认 [0, 0, 10]
        kPv: 速度控制P增益，默认 [0, 0, 10]
        kPR: 姿态控制P增益，默认 [13, 13, 13]
        kPw: 角速度控制P增益，默认 [0.017, 0.01, 0.02]
        kIw: 角速度控制I增益，默认 [0, 0, 0]
        kDw: 角速度控制D增益，默认 [0, 0, 0]
        K_max_ang: 最大倾斜角（度），默认 23
        K_max_angular_acc: 最大角加速度，默认 200
    """
    
    def __init__(
        self,
        step_dt: float,
        gravity: torch.Tensor,
        mass: torch.Tensor,
        inertia: torch.Tensor,
        num_envs: int
    )
    
    def reset(self, env_ids: torch.Tensor | None = None)
        """重置控制器状态"""
    
    def get_control(
        self,
        state_: torch.Tensor,
        action_: torch.Tensor,
        env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算控制指令
        
        参数:
            state_: 当前状态 [num_envs, 13]
                - [0:3]: 位置 (p_odom)
                - [3:7]: 姿态四元数 (q_odom)
                - [7:10]: 线速度 (v_odom)
                - [10:13]: 角速度 (w_odom)
            
            action_: 期望状态 [num_envs, 14]
                - [0:3]: 期望位置 (p_desired)
                - [3:6]: 期望速度 (v_desired)
                - [6:9]: 期望加速度 (a_desired)
                - [9:12]: 期望加加速度 (j_desired)
                - [12]: 期望偏航角 (yaw_desired)
                - [13]: 期望偏航角速度 (yaw_dot_desired)
            
            env_ids: 需要计算的环境ID，None表示所有环境
        
        返回:
            total_des_acc: 期望总加速度 [num_envs, 3]
            thrust_desired: 期望推力 [num_envs]
            q_desired: 期望姿态四元数 [num_envs, 4]
            w_desired: 期望角速度 [num_envs, 3]
            torque_desired: 期望力矩 [num_envs, 3]
        """
```

### 核心函数

#### `flatness_with_drag`

微分平坦性转换（带阻力模型）。

```python
@torch.jit.script
def flatness_with_drag(
    v_desired: torch.Tensor,       # [N, 3] 期望速度
    a_desired: torch.Tensor,       # [N, 3] 期望加速度
    j_desired: torch.Tensor,       # [N, 3] 期望加加速度
    yaw_desired: torch.Tensor,     # [N] 期望偏航角
    yaw_dot_desired: torch.Tensor, # [N] 期望偏航角速度
    mass: torch.Tensor,            # 质量
    gravity: torch.Tensor,         # [3] 重力向量
    cp: float,                     # 二阶阻力系数
    dv: float,                     # 垂直方向阻力系数
    dh: float,                     # 水平方向阻力系数
    veps: float,                   # 平滑常数
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
        success: [N] 是否成功计算
        thrust: [N] 期望推力
        quat: [N, 4] 期望姿态四元数
        omega: [N, 3] 期望角速度
    """
```

#### `compute_pid_error_acc`

计算位置跟踪的PID误差加速度。

```python
@torch.jit.script
def compute_pid_error_acc(
    p_odom: torch.Tensor,      # [N, 3] 当前位置
    v_odom: torch.Tensor,      # [N, 3] 当前速度
    p_desired: torch.Tensor,   # [N, 3] 期望位置
    v_desired: torch.Tensor,   # [N, 3] 期望速度
    kPp: torch.Tensor,         # [3] 位置P增益
    kPv: torch.Tensor,         # [3] 速度P增益
) -> torch.Tensor:            # [N, 3] 误差加速度
```

---

## 轨迹生成模块 (utils/minco.py)

### `Trajectory`

多项式轨迹对象。

```python
class Trajectory:
    """
    分段多项式轨迹
    
    属性:
        num_env: 轨迹数量
        N: 段数
        durations: [num_env, N] 每段持续时间
        coeff_mats: [num_env, N, 3, 6] 多项式系数矩阵
    """
    
    def __init__(self, durations: torch.Tensor, coeff_mats: torch.Tensor)
    
    def get_total_duration(self) -> torch.Tensor
        """获取总持续时间 [num_env]"""
    
    def get_pos(self, t: torch.Tensor) -> torch.Tensor
        """
        查询位置
        
        参数:
            t: [num_env] 查询时间
        
        返回:
            pos: [num_env, 3] 位置
        """
    
    def get_vel(self, t: torch.Tensor) -> torch.Tensor
        """查询速度 [num_env, 3]"""
    
    def get_acc(self, t: torch.Tensor) -> torch.Tensor
        """查询加速度 [num_env, 3]"""
    
    def get_jer(self, t: torch.Tensor) -> torch.Tensor
        """查询加加速度 [num_env, 3]"""
```

### `MinJerkOpt`

最小加加速度轨迹优化器。

```python
class MinJerkOpt:
    """
    MINCO轨迹生成器
    
    生成满足起点/终点PVA约束的分段多项式轨迹
    """
    
    def __init__(
        self,
        head_pva: torch.Tensor,    # [num_env, 3, 3] 起点PVA
        tail_pva: torch.Tensor,    # [num_env, 3, 3] 终点PVA
        num_pieces: int            # 段数
    )
    
    def generate(
        self,
        inner_pts_: torch.Tensor,  # [num_env, 3, num_pieces-1] 中间点
        durations: torch.Tensor    # [num_env, num_pieces] 各段持续时间
    )
    
    def get_traj(self) -> Trajectory
        """获取生成的轨迹"""
```

---

## 自定义轨迹模块 (utils/custom_trajs.py)

### `LissajousConfig`

Lissajous 曲线配置。

```python
@configclass
class LissajousConfig:
    A_range: list = [2.5, 5.0]     # X轴振幅范围 [m]
    B_range: list = [2.5, 5.0]     # Y轴振幅范围 [m]
    ratio: list = [                # 频率比列表 (a, b)
        (1, 1), (1, 2), (1, 3),
        (2, 3), (3, 4), (2, 1),
        (3, 1), (3, 2), (4, 3),
    ]
    delta_range: list = [0, 2π]    # 相位差范围
    num_pieces: int = 128          # 轨迹段数
    max_exec_speed: float = 6.0    # 最大执行速度 [m/s]
```

### `generate_custom_trajs`

生成自定义轨迹。

```python
def generate_custom_trajs(
    type_id: str,                    # 轨迹类型: "lissajous" | "eight"
    p_odom: torch.Tensor | None,     # [num_trajs, 3] 起点位置
    v_odom: torch.Tensor | None,     # [num_trajs, 3] 起点速度
    a_odom: torch.Tensor | None,     # [num_trajs, 3] 起点加速度
    p_init: torch.Tensor | None,     # [num_trajs, 3] 终点位置
    custom_cfg: LissajousConfig | None = None,
    is_plotting: bool = False,
) -> Trajectory:
    """
    生成自定义轨迹
    
    参数:
        type_id: 轨迹类型标识
        p_odom: 起点位置
        v_odom: 起点速度（通常为零）
        a_odom: 起点加速度（通常为零）
        p_init: 终点位置（实际也是起点，形成闭合轨迹）
        custom_cfg: Lissajous配置
        is_plotting: 是否绘制轨迹图
    
    返回:
        Trajectory: 生成的轨迹对象
    """
```

---

## 工具函数模块 (utils/utils.py)

### `quat_to_ang_between_z_body_and_z_world`

计算机体Z轴与世界Z轴夹角。

```python
@torch.jit.script
def quat_to_ang_between_z_body_and_z_world(quat: torch.Tensor) -> torch.Tensor:
    """
    计算机体Z轴与世界坐标系Z轴的夹角
    
    参数:
        quat: [N, 4] 四元数 (w, x, y, z)
    
    返回:
        angle: [N] 夹角（弧度）
        
    用途:
        - 判断无人机是否倾覆（通常>80度认为坠毁）
    """
```

### `quat_to_yaw`

从四元数提取偏航角。

```python
@torch.jit.script
def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """
    从姿态四元数提取偏航角（Z轴旋转）
    
    参数:
        quat: [N, 4] 四元数 (w, x, y, z)
    
    返回:
        yaw: [N] 偏航角（弧度），范围 [-π, π]
    """
```

---

## 训练脚本 API

### SKRL 训练

```python
# reinforcement_learning/skrl/train.py

def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """
    SKRL训练主函数
    
    参数:
        env_cfg: 环境配置（由Hydra自动注入）
        agent_cfg: 智能体配置（由Hydra自动注入）
    """

# 命令行参数
parser.add_argument("--task", type=str, required=True, 
                   help="任务名称，如 FAST-Swarm-Vel")
parser.add_argument("--num_envs", type=int, default=1000,
                   help="并行环境数量")
parser.add_argument("--seed", type=int, default=None,
                   help="随机种子")
parser.add_argument("--max_iterations", type=int, default=None,
                   help="最大训练迭代次数")
parser.add_argument("--checkpoint", type=str, default=None,
                   help="恢复训练的检查点路径")
parser.add_argument("--algorithm", type=str, default="PPO",
                   choices=["AMP", "PPO", "IPPO", "MAPPO"],
                   help="RL算法")
parser.add_argument("--distributed", action="store_true",
                   help="启用分布式训练")
```

### SKRL 推理

```python
# reinforcement_learning/skrl/play.py

def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """
    SKRL推理主函数
    """

# 命令行参数
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True,
                   help="模型检查点路径")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video", action="store_true",
                   help="录制视频")
parser.add_argument("--video_length", type=int, default=500,
                   help="视频长度（步数）")
```

---

## 类型别名

```python
from typing import Dict, Tuple
import torch

# 观测类型
Observation = Dict[str, torch.Tensor]  # {"policy": tensor, "odom": tensor}
MultiAgentObservation = Dict[str, torch.Tensor]  # {"drone_0": tensor, ...}

# 动作类型
Action = torch.Tensor  # [num_envs, action_dim]
MultiAgentAction = Dict[str, torch.Tensor]  # {"drone_0": tensor, ...}

# 奖励类型
Reward = torch.Tensor  # [num_envs]
MultiAgentReward = Dict[str, torch.Tensor]  # {"drone_0": tensor, ...}

# 终止标志类型
Done = torch.Tensor  # [num_envs], bool
MultiAgentDone = Dict[str, torch.Tensor]
```

---

## 异常与错误码

### 环境错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Action and control decimation must be >= 1` | decimation配置错误 | 检查 physics_freq 和 action_freq |
| `Invalid task name` | 任务名称不正确 | 检查可用的任务列表 |
| `CUDA out of memory` | 显存不足 | 减少 num_envs 或减小批次大小 |

### 控制器错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Corner case: inverted flight` | 无人机倒置飞行 | 检查初始状态和重置逻辑 |
| `Retrieved timestamp out of trajectory duration` | 轨迹时间超出范围 | 检查轨迹总时长和查询时间 |

### 训练错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Unsupported skrl version` | SKRL版本过低 | 升级 SKRL: `pip install -U skrl` |
| `ModuleNotFoundError: rclpy` | ROS2未安装 | 安装 ROS2 Humble 或移除ROS依赖 |
