# 环境配置详解

本文档详细介绍所有可用的环境及其配置参数。

---

## 单机环境 (Quadcopter)

### FAST-Quadcopter-Vel

**控制目标**: 控制无人机以指定速度飞向随机生成的目标点

```python
# 观测空间: 6维
obs = [
    goal_x - robot_x,    # 目标相对X位置
    goal_y - robot_y,    # 目标相对Y位置
    goal_z - robot_z,    # 目标相对Z位置
    vel_x,               # 机体X速度
    vel_y,               # 机体Y速度
    vel_z,               # 机体Z速度
]

# 动作空间: 2维 (归一化到[-1, 1])
action = [
    v_x_desired / v_max,  # 期望X速度 / 最大速度
    v_y_desired / v_max,  # 期望Y速度 / 最大速度
]
```

**终止条件**:
- 高度超出 [0.5, 1.5] 米
- 姿态倾斜角超过 80 度
- 超时（默认 30 秒）

---

### FAST-Quadcopter-Acc

**控制目标**: 控制无人机的加速度

```python
# 观测空间: 9维
obs = [body2goal(3), root_lin_vel(3), root_acc(3)]

# 动作空间: 2维
action = [a_x_desired / a_max, a_y_desired / a_max]
```

---

### FAST-Quadcopter-Waypoint

**控制目标**: 跟踪多个航点

```python
# 观测空间: 3 * num_pieces 维
obs = [
    waypoint1_rel_body(3),
    waypoint2_rel_body(3),
    ...
]

# 动作空间: 3 * num_pieces 维
action = [
    waypoint1_x / p_max, waypoint1_y / p_max, waypoint1_z / p_max,
    waypoint2_x / p_max, waypoint2_y / p_max, waypoint2_z / p_max,
    ...
]
```

---

## 集群环境 (Swarm)

### FAST-Swarm-Vel

**核心特点**:
- 13架无人机协同
- 分布式部分可观测
- 三种任务类型

```python
# 观测空间（每架无人机）: history_length * transient_dim = 3 * 54 = 162 维
transient_obs = [
    # 自身观测 (6维)
    prev_action(2),          # 上一时刻动作
    body2goal_xy(2),         # 到目标XY相对位置
    lin_vel_xy(2),           # 线速度XY
    
    # 邻居观测 (48维 = 4 * 12)
    neighbor1_rel_pos(3), observability(1),
    neighbor2_rel_pos(3), observability(1),
    ...
]

# 动作空间: 2维
action = [v_x_desired / v_max, v_y_desired / v_max]
```

**任务类型**:

| 任务 | 描述 | 难度 |
|------|------|------|
| Migration | 编队迁徙，保持队形飞向统一目标 | ⭐⭐ |
| Crossover | 圆环上交叉穿越，大规模避障 | ⭐⭐⭐ |
| Chaotic | 随机初始状态和目标 | ⭐⭐⭐⭐ |

---

### FAST-Swarm-Acc

**核心特点**:
- 加速度级控制
- 更精细的动力学控制

```python
# 观测空间
obs = [
    prev_action(2),          # 上一时刻动作
    body2goal(3),            # 到目标相对位置
    lin_vel(3),              # 线速度
    neighbor_rel_pos(48),    # 邻居相对位置
]

# 动作空间: 2维
action = [a_x_desired / a_max, a_y_desired / a_max]
```

---

### FAST-Swarm-AJ

**核心特点**:
- 加加速度(Jerk)级控制
- 最高精度的控制

```python
# 动作空间: 2维
action = [j_x_desired / j_max, j_y_desired / j_max]
```

---

## 环境配置参数详解

### 频率相关

```python
physics_freq = 200.0    # 物理仿真频率 (Hz)
control_freq = 100.0    # 底层控制器频率 (Hz)
action_freq = 20.0      # RL策略频率 (Hz)
gui_render_freq = 50.0  # GUI渲染频率 (Hz)

decimation = physics_freq // action_freq  # 10
control_decimation = physics_freq // control_freq  # 2
```

### 奖励函数权重

```python
# 基础奖励
to_live_reward_weight = 1.0           # 存活奖励
death_penalty_weight = 0.0            # 死亡惩罚

# 任务奖励
approaching_goal_reward_weight = 10.0 # 接近目标奖励
success_reward_weight = 10.0          # 成功奖励
dist_to_goal_reward_weight = 0.0      # 距离目标奖励

# 安全奖励 (仅集群)
mutual_collision_penalty_weight = 100.0        # 碰撞惩罚
mutual_collision_avoidance_soft_penalty_weight = 0.0  # 软避障惩罚

# 平滑性奖励
ang_vel_penalty_weight = 0.0          # 角速度惩罚
action_norm_penalty_weight = 1.0      # 动作幅度惩罚
action_diff_penalty_weight = 1.0      # 动作变化惩罚
```

### 域随机化参数

```python
# 时延 (毫秒)
lin_vel_obs_delay_ms = 40.0
rel_pos_obs_delay_ms = 40.0

# 时延范围: 0.77 * max_lag ~ max_lag
# 例如 40ms / 20Hz = 2 steps, 范围 1.54 ~ 2 steps

# 噪声标准差
lin_vel_noise_std = 0.1        # 线速度噪声 (m/s)
min_dist_noise_std = 0.05      # 近距离距离噪声 (m)
max_dist_noise_std = 1.0       # 远距离距离噪声 (m)
min_bearing_noise_std = 0.1    # 近距离方位噪声 (rad)
max_bearing_noise_std = 0.25   # 远距离方位噪声 (rad)

# 丢包
drop_prob = 0.1                # 10% 概率丢失邻居观测

# 视野限制
max_visible_distance = 5.0     # 最大可见距离 (m)
max_angle_of_view = 40.0       # 最大俯仰视角 (度)
```

---

## 任务类型详解

### Migration (迁徙任务)

```
初始状态: 聚集分布（正方形或圆形）
目标: 统一随机目标位置
特点: 
- 需要保持编队
- 统一目标，协作性强
- 支持自定义轨迹（Lissajous）
```

### Crossover (交叉穿越)

```
初始状态: 圆环上分布
目标: 圆环对面位置（角度+180度）
特点:
- 大规模交叉场景
- 密集避障测试
- 两种初始分布: 均匀 / 随机
```

### Chaotic (混沌任务)

```
初始状态: 随机分布
目标: 各自独立随机目标
特点:
- 完全分布式
- 最复杂的避障场景
- 每个无人机目标不同
```

---

## 可视化配置

### 调试可视化选项

```python
debug_vis = True
debug_vis_goal = True              # 显示目标点
debug_vis_collide_dist = False     # 显示碰撞检测范围
debug_vis_rel_pos = False          # 显示相对位置观测
debug_vis_action = True            # 显示期望位置
```

### 目标可视化

- **绿色方块**: 目标位置
- **蓝色球体**: 期望位置（p_desired）
- **红色圆柱**: 碰撞检测范围

---

## 环境注册信息

所有环境通过 `gymnasium.register` 自动注册：

```python
gym.register(
    id="FAST-Swarm-Vel",
    entry_point=SwarmVelEnv,
    kwargs={
        "env_cfg_entry_point": SwarmVelEnvCfg,
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_ippo_cfg.yaml",
        ...
    },
)
```
