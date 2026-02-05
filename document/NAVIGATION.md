# 文档导航

本文档帮助你快速找到所需信息。

---

## 🚀 我是新手，想快速上手

→ 推荐阅读顺序：

1. **[README.md](./README.md)** - 了解项目整体
   - 阅读：第1章 项目概述
   - 阅读：第2章 项目架构（看目录结构图）

2. **[QUICK_START.md](./QUICK_START.md)** - 10分钟上手
   - 按照步骤运行第一个训练任务
   - 学会基本命令和调试方法

3. **运行示例代码**
   ```bash
   python reinforcement_learning/skrl/train.py \
       --task FAST-Quadcopter-Vel \
       --num_envs 1000 \
       --seed 42
   ```

---

## 🤖 我想了解/使用某个环境

### 单机环境

| 需求 | 参考文档 |
|------|---------|
| 速度控制环境 | [ENVIRONMENTS.md](./ENVIRONMENTS.md) - FAST-Quadcopter-Vel |
| 加速度控制环境 | [ENVIRONMENTS.md](./ENVIRONMENTS.md) - FAST-Quadcopter-Acc |
| 航点跟踪环境 | [ENVIRONMENTS.md](./ENVIRONMENTS.md) - FAST-Quadcopter-Waypoint |
| 相机感知环境 | [API_REFERENCE.md](./API_REFERENCE.md) - CameraWaypointEnv |

### 集群环境

| 需求 | 参考文档 |
|------|---------|
| 了解三种任务类型 | [README.md](./README.md) - 3.3 集群环境详解 |
| 了解观测空间设计 | [ENVIRONMENTS.md](./ENVIRONMENTS.md) - FAST-Swarm-Vel |
| 域随机化配置 | [ENVIRONMENTS.md](./ENVIRONMENTS.md) - 域随机化参数 |
| 奖励函数设计 | [README.md](./README.md) - 奖励函数设计 |

---

## 🔧 我想修改/扩展项目

### 场景1: 修改奖励函数

1. [DEVELOPMENT.md](./DEVELOPMENT.md) - 第2节 修改奖励函数
2. 查看对应环境的 `_get_rewards()` 方法

### 场景2: 添加新环境

1. [DEVELOPMENT.md](./DEVELOPMENT.md) - 第1节 添加新环境
2. 参考现有环境代码（如 `envs/quadcopter_vel_env.py`）
3. [API_REFERENCE.md](./API_REFERENCE.md) - 环境API规范

### 场景3: 修改控制器参数

1. [DEVELOPMENT.md](./DEVELOPMENT.md) - 第3节 自定义控制器
2. [API_REFERENCE.md](./API_REFERENCE.md) - Controller类
3. 修改 `utils/controller.py` 中的增益参数

### 场景4: 添加自定义轨迹

1. [DEVELOPMENT.md](./DEVELOPMENT.md) - 第4节 添加自定义轨迹
2. 参考 `utils/custom_trajs.py` 中的 Lissajous 实现
3. [API_REFERENCE.md](./API_REFERENCE.md) - MINCO优化器

---

## 📊 我想调试/优化训练

### 训练监控

| 需求 | 参考 |
|------|------|
| TensorBoard使用 | [QUICK_START.md](./QUICK_START.md) - 步骤3 |
| PlotJuggler配置 | [README.md](./README.md) - 5.3 训练监控 |
| 日志级别设置 | 所有脚本的 `--verbosity` 参数 |

### 性能优化

1. [DEVELOPMENT.md](./DEVELOPMENT.md) - 第6节 性能优化
2. 减少 `num_envs` 以避免OOM
3. 启用 `replicate_physics=True`

### 常见问题

→ 查看 [README.md](./README.md) - 第9章 常见问题

---

## 📚 我需要API参考

### 核心类

| 类/函数 | 文档位置 |
|---------|---------|
| Controller | [API_REFERENCE.md](./API_REFERENCE.md) - 控制器模块 |
| MinJerkOpt | [API_REFERENCE.md](./API_REFERENCE.md) - 轨迹生成模块 |
| Trajectory | [API_REFERENCE.md](./API_REFERENCE.md) - Trajectory类 |
| generate_custom_trajs | [API_REFERENCE.md](./API_REFERENCE.md) - 自定义轨迹模块 |

### 环境类

| 环境类 | 文档位置 |
|--------|---------|
| QuadcopterVelEnv | [API_REFERENCE.md](./API_REFERENCE.md) - 单机环境 |
| SwarmVelEnv | [API_REFERENCE.md](./API_REFERENCE.md) - 集群环境 |
| All Envs | [ENVIRONMENTS.md](./ENVIRONMENTS.md) |

### 工具函数

| 函数 | 文档位置 |
|------|---------|
| quat_to_ang_between_z_body_and_z_world | [API_REFERENCE.md](./API_REFERENCE.md) - 工具函数模块 |
| quat_to_yaw | [API_REFERENCE.md](./API_REFERENCE.md) - 工具函数模块 |
| flatness_with_drag | [API_REFERENCE.md](./API_REFERENCE.md) - 核心函数 |
| compute_pid_error_acc | [API_REFERENCE.md](./API_REFERENCE.md) - 核心函数 |

---

## 🗂️ 完整文档索引

### 主文档

| 文档 | 内容 | 字数 |
|------|------|------|
| [README.md](./README.md) | 全面项目文档 | ~21000 |
| [QUICK_START.md](./QUICK_START.md) | 快速上手指南 | ~3200 |
| [ENVIRONMENTS.md](./ENVIRONMENTS.md) | 环境配置详解 | ~6200 |
| [NETWORK_ARCHITECTURE.md](./NETWORK_ARCHITECTURE.md) | 网络架构详解 | ~21000 |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | 开发扩展指南 | ~10800 |
| [API_REFERENCE.md](./API_REFERENCE.md) | API参考手册 | ~15000 |

### 代码中的文档

- 所有环境配置类都有详细注释
- 控制器实现了完整的 docstring
- 轨迹生成模块包含数学公式说明

---

## 💡 按主题查找

### 网络架构

| 需求 | 参考文档 |
|------|---------|
| 了解Actor-Critic结构 | [NETWORK_ARCHITECTURE.md](./NETWORK_ARCHITECTURE.md) - 架构概述 |
| SKRL框架配置 | [NETWORK_ARCHITECTURE.md](./NETWORK_ARCHITECTURE.md) - SKRL框架架构 |
| 多智能体DACC架构 | [NETWORK_ARCHITECTURE.md](./NETWORK_ARCHITECTURE.md) - SB3框架架构 |
| IPPO/MAPPO网络区别 | [NETWORK_ARCHITECTURE.md](./NETWORK_ARCHITECTURE.md) - 多智能体特殊架构 |
| 修改网络层数/维度 | [NETWORK_ARCHITECTURE.md](./NETWORK_ARCHITECTURE.md) - 如何修改网络 |
| 估算模型参数量 | [NETWORK_ARCHITECTURE.md](./NETWORK_ARCHITECTURE.md) - 参数量估算 |

### 强化学习算法

- [README.md](./README.md) - 5.2 多智能体算法支持
- [NETWORK_ARCHITECTURE.md](./NETWORK_ARCHITECTURE.md) - 完整网络架构详解
- [DEVELOPMENT.md](./DEVELOPMENT.md) - 实验管理

### 控制器设计

- [README.md](./README.md) - 4.1 控制器架构
- [API_REFERENCE.md](./API_REFERENCE.md) - 控制器模块

### 轨迹规划

- [README.md](./README.md) - 4.3 轨迹生成
- [API_REFERENCE.md](./API_REFERENCE.md) - 轨迹生成模块

### 域随机化

- [README.md](./README.md) - 域随机化配置
- [ENVIRONMENTS.md](./ENVIRONMENTS.md) - 域随机化参数

### Sim2Real

- [ENVIRONMENTS.md](./ENVIRONMENTS.md) - 观测时延、噪声、丢包
- [DEVELOPMENT.md](./DEVELOPMENT.md) - 调试技巧

---

## 🔗 外部资源

- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/main/)
- [SKRL 文档](https://skrl.readthedocs.io/)
- [ROS2 文档](https://docs.ros.org/en/humble/)

---

> 💡 **提示**: 使用浏览器的搜索功能（Ctrl+F）在当前文档中快速查找关键词。
