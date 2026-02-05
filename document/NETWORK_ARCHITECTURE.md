# 网络架构详解

本文档详细介绍 FAST-Swarm-RL 项目中使用的各种神经网络架构。

---

## 📋 目录

1. [架构概述](#1-架构概述)
2. [SKRL 框架架构](#2-skrl-框架架构)
3. [RSL-RL 框架架构](#3-rsl-rl-框架架构)
4. [SB3 框架架构](#4-sb3-框架架构)
5. [多智能体特殊架构](#5-多智能体特殊架构)
6. [网络配置对比](#6-网络配置对比)
7. [如何修改网络](#7-如何修改网络)
8. [参数量估算](#8-参数量估算)

---

## 1. 架构概述

本项目支持三种 RL 框架，每种框架使用不同的网络架构：

| 框架 | 主要架构 | 适用场景 |
|------|---------|---------|
| **SKRL** | Gaussian Actor + Deterministic Critic | 主要推荐，支持多种算法 |
| **RSL-RL** | Actor-Critic (On-Policy) | ETH Zurich 风格训练 |
| **SB3** | DACC (Distributed Actor Centralized Critic) | Stable-Baselines3 用户 |

### 通用设计原则

```
观测输入 → [状态预处理] → 特征提取 → 策略/价值网络 → 动作/价值输出
```

**共同特点**：
- ✅ 支持连续动作空间（高斯分布）
- ✅ 运行均值/方差归一化（Running Standard Scaler）
- ✅ ELU/ReLU 激活函数
- ✅ 可选的 Actor-Critic 网络分离

---

## 2. SKRL 框架架构

### 2.1 基本架构

SKRL 使用 **高斯策略网络（Actor）** + **确定性价值网络（Critic）**：

```
                        ┌─────────────────────────────────────┐
                        │         观测输入 (obs)               │
                        │  [batch_size, observation_space]    │
                        └──────────────┬──────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │      RunningStandardScaler          │
                    │   (运行均值/方差归一化)               │
                    └──────────────────┬──────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        │                        ▼
┌───────────────────────────────┐      │      ┌───────────────────────────────┐
│      Actor (Policy)           │      │      │      Critic (Value)           │
│     GaussianMixin             │      │      │    DeterministicMixin         │
│                               │      │      │                               │
│  ┌─────────────────────────┐  │      │      │  ┌─────────────────────────┐  │
│  │  Input: normalized_obs  │  │      │      │  │  Input: normalized_obs  │  │
│  ├─────────────────────────┤  │      │      │  ├─────────────────────────┤  │
│  │  Layer 1: Linear + ELU  │  │      │      │  │  Layer 1: Linear + ELU  │  │
│  │  Layer 2: Linear + ELU  │  │      │      │  │  Layer 2: Linear + ELU  │  │
│  │  ...                    │  │      │      │  │  ...                    │  │
│  │  Layer N: Linear        │  │      │      │  │  Layer N: Linear        │  │
│  ├─────────────────────────┤  │      │      │  ├─────────────────────────┤  │
│  │  Output: mean_actions   │──┼──────┼──────┼──│  Output: state_value    │  │
│  │  Output: log_std        │  │      │      │  └─────────────────────────┘  │
│  └─────────────────────────┘  │      │      └───────────────────────────────┘
│           │                   │      │
│           ▼                   │      │
│  ┌─────────────────────────┐  │      │
│  │  Gaussian Distribution  │  │      │
│  │  (mean, exp(log_std))   │  │      │
│  ├─────────────────────────┤  │      │
│  │  Sample: action ~ N     │  │      │
│  └─────────────────────────┘  │      │
└───────────────────────────────┘      │
                                       │
                    ┌──────────────────┴──────────────────┐
                    │         输出结果                      │
                    │  action: [batch_size, action_space]  │
                    │  value:  [batch_size, 1]             │
                    └───────────────────────────────────────┘
```

### 2.2 网络层配置

#### Swarm PPO（集群集中式训练）

```yaml
# config/agents/swarm_skrl_ppo_cfg.yaml
models:
  separate: True              # Actor 和 Critic 网络分离
  
  policy:                     # Actor (策略网络)
    class: GaussianMixin      # 高斯策略
    clip_log_std: True
    min_log_std: -20.0        # 最小标准差: e^-20 ≈ 0
    max_log_std: 2.0          # 最大标准差: e^2 ≈ 7.4
    initial_log_std: 0.0      # 初始标准差: e^0 = 1.0
    
    network:
      layers: [1024, 1024, 1024, 512, 512, 256]  # 6层MLP
      activations: elu
    
  value:                      # Critic (价值网络)
    class: DeterministicMixin # 确定性输出
    network:
      layers: [1024, 1024, 1024, 512, 512, 256]  # 6层MLP
      activations: elu
```

**网络结构详情**：
- **输入维度**: 162 (Swarm-Vel环境: 54维观测 × 3帧历史)
- **Actor隐藏层**: 1024 → 1024 → 1024 → 512 → 512 → 256
- **Critic隐藏层**: 1024 → 1024 → 1024 → 512 → 512 → 256
- **输出维度**: 2 (x,y方向速度指令)

#### Swarm IPPO（集群分布式训练）

```yaml
# config/agents/swarm_skrl_ippo_cfg.yaml
models:
  separate: True
  
  policy:
    class: GaussianMixin
    min_log_std: -1.6         # 标准差 ≈ 0.2（更小的探索）
    max_log_std: 0.26         # 标准差 ≈ 1.3
    
    network:
      layers: [128, 128, 64, 32]  # 4层轻量级网络
      activations: elu
    
  value:
    class: DeterministicMixin
    network:
      layers: [128, 128, 64, 32]  # 4层轻量级网络
      activations: elu
```

**设计理由**：
- 多智能体共享同一网络参数（`param_sharing: True`）
- 减小网络规模以提高样本效率
- 更小的探索噪声（`min_log_std: -1.6`）适合协作任务

#### Swarm MAPPO（集群协作训练）

```yaml
# config/agents/swarm_skrl_mappo_cfg.yaml
models:
  separate: True
  
  policy:
    class: GaussianMixin
    network:
      layers: [256, 256, 128, 64]  # 中等规模4层网络
      activations: elu
    
  value:
    class: DeterministicMixin
    network:
      layers: [256, 256, 128, 64]  # 中等规模4层网络
      activations: elu
```

#### 单机 PPO（Quadcopter）

```yaml
# config/agents/quadcopter_skrl_ppo_cfg.yaml
models:
  separate: False             # Actor 和 Critic 共享底层网络
  
  policy:
    class: GaussianMixin
    network:
      layers: [1024, 512]     # 2层轻量网络
      activations: elu
    
  value:
    class: DeterministicMixin
    network:
      layers: [1024, 512]     # 2层轻量网络
      activations: elu
```

### 2.3 训练超参数

```yaml
agent:
  rollouts: 250               # 每次更新的步数
  learning_epochs: 3          # 每次采样的学习轮数
  mini_batches: 4             # 小批量数量
  learning_rate: 1.0e-04      # 学习率
  discount_factor: 0.99       # 折扣因子 γ
  lambda: 0.95                # GAE参数 λ
  
  state_preprocessor: RunningStandardScaler    # 观测归一化
  value_preprocessor: RunningStandardScaler    # 价值归一化
  
  grad_norm_clip: 1.0         # 梯度裁剪
  ratio_clip: 0.2             # PPO裁剪参数 ε
  entropy_loss_scale: 0.0005  # 熵奖励系数
```

---

## 3. RSL-RL 框架架构

RSL-RL 使用与 SKRL 类似的 Actor-Critic 架构，但配置方式不同：

```python
# config/agents/swarm_rsl_rl_ppo_cfg.py
@configclass
class SwarmVelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100
    max_iterations = 100000
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 1024, 1024, 512, 512, 256],  # 6层Actor
        critic_hidden_dims=[1024, 1024, 1024, 512, 512, 256], # 6层Critic
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
```

**架构对比**：
- 与 SKRL Swarm PPO 配置几乎相同
- 差异主要在训练循环实现和日志记录方式

---

## 4. SB3 框架架构

SB3 使用自定义的 **DACC（Distributed Actor Centralized Critic）** 架构，专门为多智能体设计：

```
全局观测 [batch, num_agents × obs_dim]
              │
              ▼
    ┌─────────────────────┐
    │  Features Extractor │
    └──────────┬──────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌──────────────┐ ┌──────────────┐
│  Actor Net   │ │  Critic Net  │
│ (Distributed)│ │(Centralized) │
└──────┬───────┘ └──────┬───────┘
       │                │
       ▼                ▼
动作输出            价值输出
```

### 4.1 DACC 架构详解

```python
# reinforcement_learning/sb3/policy.py

class DACCNetwork(nn.Module):
    """
    Distributed Actor Centralized Critic
    - Actor: 每个智能体独立处理，参数共享
    - Critic: 使用所有智能体的全局信息
    """
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """
        Actor 前向传播（分布式）
        
        输入: [batch_size, num_agents × obs_dim]
        输出: [batch_size, num_agents × action_dim]
        """
        batch_size = features.shape[0]
        
        # 1. 分割成每个智能体的观测
        # [batch, num_agents × obs_dim] → [batch, num_agents, obs_dim]
        split_features = features.view(
            batch_size, 
            self.num_agents, 
            self.distributed_feature_dim
        )
        
        # 2. 每个智能体独立通过同一网络（参数共享）
        latents = []
        for i in range(self.num_agents):
            agent_feature = split_features[:, i, :]      # [batch, obs_dim]
            agent_latent = self.policy_net(agent_feature) # [batch, latent_dim]
            latents.append(agent_latent)
        
        # 3. 拼接所有智能体的输出
        return th.cat(latents, dim=1)  # [batch, num_agents × latent_dim]
    
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """
        Critic 前向传播（集中式）
        
        输入: [batch_size, num_agents × obs_dim] （全局信息）
        输出: [batch_size, 1] （全局价值）
        """
        return self.value_net(features)
```

### 4.2 动作生成流程

```python
def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
    """从隐层特征生成动作分布"""
    
    if isinstance(self.action_dist, DiagGaussianDistribution):
        batch_size = latent_pi.shape[0]
        
        # 分割隐层特征
        split_latent = latent_pi.view(
            batch_size, 
            self.num_agents, 
            latent_dim_pi // self.num_agents
        )
        
        # 每个智能体独立生成动作均值
        mean_actions = []
        for i in range(self.num_agents):
            agent_latent = split_latent[:, i, :]
            agent_mean_action = self.action_net(agent_latent)
            mean_actions.append(agent_mean_action)
        
        mean_actions = th.cat(mean_actions, dim=1)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
```

### 4.3 SB3 配置

```python
# 在训练脚本中配置
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256, 128], vf=[512, 256])],  # Actor/Critic不同结构
    activation_fn=nn.ELU,
    num_agents=13,
)

model = PPO(
    DistributedActorCentralizedCriticPolicy,
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)
```

---

## 5. 多智能体特殊架构

### 5.1 IPPO (Independent PPO)

```
每个智能体独立优化，但共享网络参数

智能体1 obs ──┐
智能体2 obs ──┼──► [共享网络] ──► 动作1
智能体3 obs ──┤              ──► 动作2
...          ──┘              ──► 动作3
```

**关键配置**：
```yaml
agent:
  class: IPPO
  param_sharing: True         # 参数共享，大幅减少参数量
  rollouts: 80                # 较小的 rollout（多智能体样本多）
  mini_batches: 20
```

### 5.2 MAPPO (Multi-Agent PPO)

```
分布式Actor + 集中式Critic

智能体观测      全局状态
    │              │
    ▼              ▼
[Actor Net]   [Critic Net]
    │              │
   动作          全局价值
```

**关键配置**：
```yaml
agent:
  class: MAPPO
  param_sharing: True
  shared_state_preprocessor: RunningStandardScaler  # 全局状态预处理
```

**状态定义**（`swarm_vel_env.py`）：
```python
def _get_states(self):
    """全局状态，供中央Critic使用"""
    curr_state = []
    for agent in self.possible_agents:
        curr_state.extend([
            self.actions[agent],                    # 动作
            self.robots[agent].data.root_pos_w,     # 位置
            self.goals[agent] - self.robots[agent].data.root_pos_w,  # 相对目标
            self.robots[agent].data.root_quat_w,    # 姿态
            self.robots[agent].data.root_vel_w,     # 速度
        ])
    return torch.cat(curr_state, dim=1)  # [num_envs, 18 × num_agents]
```

---

## 6. 网络配置对比

| 配置项 | Swarm PPO | Swarm IPPO | Swarm MAPPO | 单机 PPO |
|--------|-----------|------------|-------------|----------|
| **框架** | SKRL | SKRL | SKRL | SKRL |
| **算法** | PPO | IPPO | MAPPO | PPO |
| **Actor层数** | 6 | 4 | 4 | 2 |
| **Actor维度** | [1024,1024,1024,512,512,256] | [128,128,64,32] | [256,256,128,64] | [1024,512] |
| **Critic层数** | 6 | 4 | 4 | 2 |
| **Critic维度** | [1024,1024,1024,512,512,256] | [128,128,64,32] | [256,256,128,64] | [1024,512] |
| **激活函数** | ELU | ELU | ELU | ELU |
| **网络分离** | True | True | True | False |
| **参数共享** | - | True | True | - |
| **初始log_std** | 0.0 | 0.0 | 0.0 | 0.0 |
| **min_log_std** | -20.0 | -1.6 | -1.6 | -20.0 |
| **Rollouts** | 250 | 80 | 80 | 500 |
| **Learning Epochs** | 3 | 5 | 5 | 5 |
| **Learning Rate** | 1e-4 | 1e-3 | 1e-3 | 1e-4 |

### 选择建议

| 场景 | 推荐架构 | 理由 |
|------|---------|------|
| 集群集中式训练 | SKRL PPO [1024×3,512×2,256] | 复杂任务，深层网络提取特征 |
| 集群分布式（大规模） | SKRL IPPO [128,128,64,32] | 轻量网络，高效样本利用 |
| 集群协作任务 | SKRL MAPPO [256,256,128,64] | 平衡性能和协作学习 |
| 单机简单任务 | SKRL PPO [1024,512] | 轻量快速训练 |
| SB3迁移 | DACC | 兼容Stable-Baselines3生态 |

---

## 7. 如何修改网络

### 7.1 修改隐藏层维度

编辑对应的 YAML 配置文件：

```yaml
# config/agents/swarm_skrl_ppo_cfg.yaml
models:
  policy:
    network:
      layers: [512, 512, 256]       # 改为3层
      activations: elu
    
  value:
    network:
      layers: [512, 512, 256]       # 改为3层
      activations: elu
```

### 7.2 修改激活函数

```yaml
models:
  policy:
    network:
      layers: [1024, 512, 256]
      activations: relu             # 改为 ReLU
      # 或: activations: [relu, tanh, elu]  # 每层不同
```

### 7.3 调整探索程度

```yaml
models:
  policy:
    initial_log_std: -1.0           # 初始标准差 ≈ 0.37（更保守）
    min_log_std: -2.0               # 最小标准差 ≈ 0.14
    max_log_std: 0.0                # 最大标准差 = 1.0
```

### 7.4 Actor-Critic 共享/分离

```yaml
models:
  separate: False                   # 共享底层特征提取层
```

**共享网络结构**：
```
观测输入
    │
    ├─► [共享层: 1024, 512]
    │         │
    │    ┌────┴────┐
    │    │         │
    │    ▼         ▼
Actor头部    Critic头部
    │         │
   动作      价值
```

### 7.5 动态修改（命令行）

```bash
# 调整初始探索程度
python reinforcement_learning/skrl/train.py \
    --task FAST-Swarm-Vel \
    --init_log_std -1.0
```

---

## 8. 参数量估算

### 8.1 计算公式

对于全连接层：`params = (input_dim × output_dim) + bias`

### 8.2 Swarm PPO 参数量

**Actor网络**：
```
Layer 1: 162 × 1024 + 1024 = 166,912
Layer 2: 1024 × 1024 + 1024 = 1,049,600
Layer 3: 1024 × 1024 + 1024 = 1,049,600
Layer 4: 1024 × 512 + 512 = 525,312
Layer 5: 512 × 512 + 512 = 262,656
Layer 6: 512 × 256 + 256 = 131,328
Output:  256 × 2 + 2 = 514
─────────────────────────────
Total:   ~3.18M parameters
```

**Critic网络**：
```
输入维度相同，结构相同
Total:   ~3.18M parameters
```

**总计**（分离网络）：~6.36M parameters

### 8.3 Swarm IPPO 参数量

**每个网络**：
```
Layer 1: 162 × 128 + 128 = 20,864
Layer 2: 128 × 128 + 128 = 16,512
Layer 3: 128 × 64 + 64 = 8,256
Layer 4: 64 × 32 + 32 = 2,080
Output:  32 × 2 + 2 = 66
─────────────────────────────
Total:   ~48K parameters per network
```

**总计**（13个智能体共享参数）：~96K parameters

### 8.4 参数量对比表

| 架构 | Actor参数量 | Critic参数量 | 总计 | 相对大小 |
|------|------------|-------------|------|---------|
| Swarm PPO | 3.18M | 3.18M | 6.36M | 100% |
| Swarm MAPPO | 213K | 213K | 426K | 6.7% |
| Swarm IPPO | 48K | 48K | 96K | 1.5% |
| 单机 PPO | 1.18M | 1.18M | 2.36M | 37% |

### 8.5 训练内存估算

假设 batch_size = 4096，rollouts = 250：

```
前向/反向传播内存 ≈ 参数量 × 4 bytes × 3 ( Adam + 梯度 + 动量)
                  ≈ 6.36M × 4 × 3 ≈ 76 MB

经验回放缓存 ≈ batch_size × rollouts × obs_dim × 4 bytes
            ≈ 4096 × 250 × 162 × 4 ≈ 663 MB

总计 ≈ 740 MB (仅网络)
```

**显存优化建议**：
- 减小 `rollouts` 数量
- 使用更小的网络（如 IPPO 配置）
- 启用 `separate: False` 共享部分网络

---

## 附录：激活函数对比

| 激活函数 | 公式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|------|---------|
| **ELU** | x if x>0 else α(e^x-1) | 平滑负值，收敛快 | 计算稍慢 | 默认推荐 |
| **ReLU** | max(0,x) | 计算快，稀疏激活 | 神经元死亡 | 简单任务 |
| **Tanh** | (e^x-e^-x)/(e^x+e^-x) | 输出有界 | 梯度消失 | 输出层 |
| **LeakyReLU** | max(αx,x) | 解决死亡ReLU | 负值线性 | 深层网络 |

**本项目默认使用 ELU**，因为：
1. 平滑的负值有助于稳定训练
2. 对 PPO 的 on-policy 训练更友好
3. 在机器人控制任务中表现更稳定
