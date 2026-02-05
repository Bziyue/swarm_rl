# 快速开始指南

本文档帮助你在 10 分钟内运行第一个 FAST-Swarm-RL 训练任务。

---

## 1. 环境检查

```bash
# 检查 Isaac Lab 是否安装
python -c "import isaaclab; print('Isaac Lab version:', isaaclab.__version__)"

# 检查 PyTorch GPU 是否可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 检查 SKRL 是否安装
python -c "import skrl; print('SKRL version:', skrl.__version__)"
```

---

## 2. 单机训练（5分钟上手）

### 步骤1：启动训练

```bash
cd /home/zdp/CodeField/swarm_rl

python reinforcement_learning/skrl/train.py \
    --task FAST-Quadcopter-Vel \
    --num_envs 1000 \
    --seed 42 \
    --run_id first_try
```

### 步骤2：观察输出

训练开始后会显示类似如下输出：

```
2025-02-04 15:00:00 | INFO | Environment initialized
2025-02-04 15:00:01 | INFO | Starting training...
2025-02-04 15:00:01 | INFO | Timestep: 24000  Episode: 24  Mean reward: -12.34
...
```

### 步骤3：可视化结果

```bash
# 新开一个终端
tensorboard --logdir outputs/skrl/FAST-Quadcopter-Vel/
```

打开浏览器访问 `http://localhost:6006`

---

## 3. 集群训练（进阶）

### 基础集群训练

```bash
python reinforcement_learning/skrl/train.py \
    --task FAST-Swarm-Vel \
    --num_envs 4096 \
    --algorithm IPPO \
    --seed 42
```

### 使用多GPU训练

```bash
# 4 GPU 分布式训练
python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=4 \
    reinforcement_learning/skrl/train.py \
    --task FAST-Swarm-Vel \
    --num_envs 2500 \
    --distributed
```

---

## 4. 模型推理

### 推理单个环境

```bash
python reinforcement_learning/skrl/play.py \
    --task FAST-Quadcopter-Vel \
    --num_envs 1 \
    --checkpoint outputs/skrl/FAST-Quadcopter-Vel/first_try/checkpoints/best_agent.pt
```

### 推理并录制视频

```bash
python reinforcement_learning/skrl/play.py \
    --task FAST-Swarm-Vel \
    --num_envs 10 \
    --checkpoint outputs/skrl/FAST-Swarm-Vel/xxx/checkpoints/best_agent.pt \
    --video \
    --video_length 500
```

---

## 5. 键盘遥控测试

```bash
python scripts/teleop.py \
    --task FAST-Quadcopter-Vel \
    --num_envs 1
```

**控制说明**：
- `W/S`: 前进/后退
- `A/D`: 左移/右移
- `Q/E`: 上升/下降
- `R`: 重置环境

---

## 6. 常见问题排查

### 问题1: `ModuleNotFoundError: No module named 'isaaclab'`

**解决**: 确保已安装 Isaac Lab 并激活环境

```bash
cd /path/to/isaaclab
source setup_conda_env.sh  # 或 setup_python_env.sh
```

### 问题2: `CUDA out of memory`

**解决**: 减少并行环境数量

```bash
# 将 --num_envs 从 10000 减少到 2048
python reinforcement_learning/skrl/train.py \
    --task FAST-Swarm-Vel \
    --num_envs 2048
```

### 问题3: 训练不稳定，奖励震荡

**解决**: 调整学习率和奖励权重

```yaml
# 修改 config/agents/swarm_skrl_ppo_cfg.yaml
agent:
  learning_rate: 1.0e-4  # 降低学习率
  learning_epochs: 8     # 增加学习轮数
```

---

## 7. 下一步

- 📖 详细文档: [README.md](./README.md)
- 🔧 算法配置: [config/agents/](../config/agents/)
- 🎓 环境详解: [envs/](../envs/)
