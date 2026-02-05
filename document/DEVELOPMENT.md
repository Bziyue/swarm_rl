# 开发指南

本文档面向希望扩展或修改 FAST-Swarm-RL 的开发者。

---

## 1. 添加新环境

### 步骤1: 创建环境文件

```python
# envs/my_custom_env.py
from __future__ import annotations
import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass

from envs.quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG


@configclass
class MyCustomEnvCfg(DirectRLEnvCfg):
    """环境配置类"""
    episode_length_s = 30.0
    physics_freq = 200.0
    action_freq = 20.0
    
    # 观测和动作空间
    observation_space = 6
    action_space = 2
    
    # 机器人配置
    robot: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="/World/envs/env_.*/Robot")


class MyCustomEnv(DirectRLEnv):
    """环境实现类"""
    cfg: MyCustomEnvCfg
    
    def __init__(self, cfg: MyCustomEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 初始化自定义变量
        
    def _setup_scene(self):
        """设置场景"""
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        # ... 其他场景设置
        
    def _pre_physics_step(self, actions: torch.Tensor):
        """物理步进前的处理"""
        self.actions = actions.clone()
        
    def _apply_action(self):
        """应用动作到机器人"""
        # 将动作转换为控制指令
        pass
        
    def _get_observations(self) -> dict:
        """获取观测"""
        obs = torch.cat([...], dim=-1)
        return {"policy": obs}
        
    def _get_rewards(self) -> torch.Tensor:
        """计算奖励"""
        reward = ...
        return reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """判断回合是否结束"""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = ...
        return died, time_out
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定环境"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # 重置逻辑
        super()._reset_idx(env_ids)


# 注册环境
gym.register(
    id="FAST-MyCustom",
    entry_point=MyCustomEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyCustomEnvCfg,
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:my_custom_skrl_ppo_cfg.yaml",
    },
)
```

### 步骤2: 创建算法配置

```yaml
# config/agents/my_custom_skrl_ppo_cfg.yaml
agent:
  rollouts: 24
  learning_epochs: 5
  mini_batches: 4
  learning_rate: 5.0e-4
  
models:
  policy:
    hiddens: [256, 256, 128]
    hidden_activation: ["relu", "relu", "relu"]
```

### 步骤3: 更新训练脚本

```python
# 在 train.py 中添加环境导入
from envs import my_custom_env
```

---

## 2. 修改奖励函数

### 示例：添加自定义奖励

```python
def _get_rewards(self) -> dict[str, torch.Tensor]:
    rewards = {}
    
    for i, agent in enumerate(self.possible_agents):
        # 原有奖励
        approaching_goal_reward = ...
        success_reward = ...
        
        # 新增：能量效率奖励
        power_consumption = torch.linalg.norm(self.actions[agent], dim=1)
        energy_efficiency_reward = -power_consumption * 0.1
        
        # 新增：编队保持奖励（集群环境）
        formation_reward = self._compute_formation_reward(agent)
        
        reward = {
            "approaching_goal": approaching_goal_reward * self.cfg.approaching_goal_reward_weight,
            "success": success_reward * self.cfg.success_reward_weight,
            "energy_efficiency": energy_efficiency_reward * self.cfg.energy_efficiency_reward_weight,
            "formation": formation_reward * self.cfg.formation_reward_weight,
        }
        
        # 记录日志
        for key, value in reward.items():
            if key in self.episode_sums:
                self.episode_sums[key] += value / self.cfg.num_drones
            else:
                self.episode_sums[key] = value / self.cfg.num_drones
        
        rewards[agent] = torch.sum(torch.stack(list(reward.values())), dim=0)
    
    return rewards

def _compute_formation_reward(self, agent: str) -> torch.Tensor:
    """计算编队保持奖励"""
    # 理想编队距离
    ideal_distance = 1.0
    
    formation_reward = torch.zeros(self.num_envs, device=self.device)
    
    # 计算与邻居的距离
    for other in self.possible_agents:
        if other == agent:
            continue
        dist = torch.linalg.norm(
            self.robots[agent].data.root_pos_w - self.robots[other].data.root_pos_w,
            dim=1
        )
        # 距离越接近理想值奖励越高
        formation_reward += torch.exp(-((dist - ideal_distance) ** 2))
    
    return formation_reward
```

---

## 3. 自定义控制器

### 修改控制器参数

```python
# utils/controller.py

class Controller:
    def __init__(self, step_dt, gravity, mass, inertia, num_envs):
        # 位置控制增益
        self.kPp = torch.tensor([2.0, 2.0, 10.0])  # 增大XY增益
        self.kPv = torch.tensor([1.0, 1.0, 10.0])
        
        # 姿态控制增益
        self.kPR = torch.tensor([15.0, 15.0, 15.0])  # 更激进的姿态响应
        
        # 角速度控制增益
        self.kPw = torch.tensor([0.02, 0.015, 0.025])
        self.kIw = torch.tensor([0.01, 0.01, 0.01])  # 添加积分项
        
        # 约束
        self.K_max_ang = 30  # 最大倾斜角（度）
        self.K_max_angular_acc = 150  # 最大角加速度
```

### 添加新的控制模式

```python
def get_velocity_control(self, state, v_desired):
    """纯速度跟踪控制器（无需完整状态）"""
    v_odom = state[:, 7:10]
    v_error = v_desired - v_odom
    
    # PD控制
    acc_cmd = self.kP_v * v_error
    
    # 转换为推力
    thrust = self.mass * (acc_cmd + self.gravity)
    
    return thrust
```

---

## 4. 添加自定义轨迹

### 步骤1: 定义轨迹生成函数

```python
# utils/custom_trajs.py

def generate_circle_trajs(p_odom, v_odom, a_odom, p_init, radius=3.0, num_points=64):
    """生成圆形轨迹"""
    device = p_odom.device
    num_envs = p_odom.shape[0]
    
    # 生成圆上的点
    angles = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
    
    # 每段轨迹的持续时间
    duration = 2 * torch.pi * radius / desired_speed
    durations = torch.full((num_envs, num_points), duration / num_points, device=device)
    
    # 计算内点
    inner_pts = torch.zeros(num_envs, 3, num_points - 1, device=device)
    for i in range(1, num_points):
        inner_pts[:, 0, i-1] = radius * torch.cos(angles[i])
        inner_pts[:, 1, i-1] = radius * torch.sin(angles[i])
        inner_pts[:, 2, i-1] = p_init[:, 2]
    
    # 构建MINCO问题
    head_pva = torch.stack([p_odom, v_odom, a_odom], dim=2)
    tail_pva = torch.stack([p_init, torch.zeros_like(p_init), torch.zeros_like(p_init)], dim=2)
    
    MJO = MinJerkOpt(head_pva, tail_pva, num_points)
    MJO.generate(inner_pts, durations)
    
    return MJO.get_traj()
```

### 步骤2: 在环境中使用

```python
# 在环境初始化中
if self.cfg.use_custom_traj:
    trajs = generate_custom_trajs(
        type_id="circle",  # 新增类型
        p_odom=sample_pos,
        v_odom=sample_vel,
        a_odom=sample_acc,
        p_init=sample_pos,
        custom_cfg={"radius": 3.0, "num_points": 64},
    )
```

---

## 5. 调试技巧

### 使用可视化工具

```python
# 在环境中添加可视化
def _set_debug_vis_impl(self, debug_vis: bool):
    if debug_vis:
        # 可视化观测范围
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/obs_range",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=self.cfg.max_visible_distance,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 0.0),  # 黄色
                        opacity=0.3,
                    ),
                )
            },
        )
        self.obs_range_visualizer = VisualizationMarkers(marker_cfg)
```

### 记录中间变量

```python
def _get_observations(self) -> dict:
    # 计算观测
    obs = ...
    
    # 记录调试信息（仅在需要时）
    if self.common_step_counter % 100 == 0:
        logger.debug(f"Observation stats: mean={obs.mean()}, std={obs.std()}")
    
    return {"policy": obs}
```

### 单元测试

```python
# tests/test_controller.py
import torch
from utils.controller import Controller

def test_flatness_transform():
    """测试微分平坦性转换"""
    gravity = torch.tensor([0.0, 0.0, -9.81])
    controller = Controller(
        step_dt=0.01,
        gravity=gravity,
        mass=torch.tensor(1.0),
        inertia=torch.eye(3),
        num_envs=10
    )
    
    # 测试输入
    v_desired = torch.zeros(10, 3)
    a_desired = torch.zeros(10, 3)
    j_desired = torch.zeros(10, 3)
    yaw_desired = torch.zeros(10)
    yaw_dot_desired = torch.zeros(10)
    
    # 调用函数
    success, thrust, quat, omega = flatness_with_drag(
        v_desired, a_desired, j_desired, 
        yaw_desired, yaw_dot_desired,
        torch.tensor(1.0), gravity,
        0.0, 0.0, 0.0, 0.02
    )
    
    # 验证输出
    assert success.all()
    assert thrust.shape == (10,)
    assert quat.shape == (10, 4)
    assert omega.shape == (10, 3)
```

---

## 6. 性能优化

### 并行环境优化

```python
# 在配置中启用物理复制
@configclass
class MyEnvCfg(DirectRLEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10000,
        env_spacing=5,
        replicate_physics=True,  # 关键：启用物理复制
    )
```

### 减少数据传输

```python
# 避免频繁的CPU-GPU传输
# 坏示例
for i in range(num_envs):
    pos = self.robot.data.root_pos_w[i].cpu().numpy()  # 慢！
    
# 好示例
all_pos = self.robot.data.root_pos_w  # 保持在GPU
```

### JIT编译

```python
# 使用torch.jit.script加速计算
@torch.jit.script
def compute_reward_fast(obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # 计算密集型操作
    return reward
```

---

## 7. 最佳实践

### 代码风格

- 使用类型注解
- 添加详细的文档字符串
- 遵循 PEP 8 规范

### 配置管理

- 所有可配置参数放在 Cfg 类中
- 使用有意义的参数名
- 添加参数注释说明

### 版本控制

```bash
# 提交前检查
./scripts/check_code.sh

# 格式化代码
black envs/ utils/ scripts/
isort envs/ utils/ scripts/
```

### 实验管理

```bash
# 命名规范
{algorithm}_{env}_{key_params}_{date}

# 示例
ppo_swarm_vel_13d_crossover_20250204
```
