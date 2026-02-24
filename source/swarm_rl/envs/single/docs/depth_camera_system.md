# 单机体环境深度图系统技术文档

## 1. 概述

本文档详细说明 `single_bodyrate_env.py` 中深度图（Depth Image）的获取原理、处理流程及关键实现细节。

**核心特性：**
- 使用 **Ray Casting（光线投射）** 技术而非传统光栅化渲染
- 支持 **360° 环视感知**（前后左右4个相机）
- 输出分辨率：`4 × 64 × 32`（4个相机，高64，宽32）
- 数值范围：`[0, 4.0]` 米（真实物理尺度）
- 模拟 **15% 传感器噪声** 以增强仿真真实性
- 支持 **动态地图重载**（运行时更新障碍物）

---

## 2. 深度图获取原理

### 2.1 技术选型：Ray Casting vs 光栅化

| 特性 | Ray Casting | 光栅化渲染 |
|------|-------------|-----------|
| **原理** | 从相机光心发射射线，计算与Mesh的交点 | 投影三角形到屏幕，计算像素覆盖 |
| **精度** | 精确的几何相交计算 | 离散像素近似 |
| **性能** | GPU并行加速（Warp框架） | GPU硬件光栅化 |
| **适用场景** | 需要精确距离测量的机器人感知 | 视觉真实感渲染 |

### 2.2 核心原理流程

```
┌─────────────────────────────────────────────────────────────────┐
│  相机光心 (Camera Center)                                       │
│       │                                                         │
│       ▼                                                         │
│  针孔相机模型 (Pinhole Model)                                   │
│       │  根据内参矩阵K生成射线方向                               │
│       ▼                                                         │
│  发射射线 (Ray) ───────────────────────┐                        │
│       │                                │                        │
│       ▼                                ▼                        │
│  场景Mesh (障碍物/地形) ◄────────── 射线相交检测 (Warp GPU)      │
│       │                                │                        │
│       ▼                                │                        │
│  命中点距离 ───────────────────────────┘                        │
│       │                                                         │
│       ▼                                                         │
│  距离到图像平面 (distance_to_image_plane)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 针孔相机模型参数

```python
# 相机内参矩阵 K (根据分辨率缩放)
K = [fx,   0,   cx,
     0,   fy,   cy,
     0,    0,    1]

# 原始参数 (640x480 分辨率下)
fx = 388.963, fy = 388.963  # 焦距 (像素)
cx = 317.04,  cy = 241.99   # 主点 (像素)

# 实际使用分辨率 (32x64) 缩放后
fx = 388.963 / (640/32) = 19.448
fy = 388.963 / (480/64) = 51.862
cx = 317.04 / (640/32) = 15.852
cy = 241.99 / (480/64) = 32.265
```

---

## 3. 系统架构

### 3.1 类层次结构

```
DepthCameraArray
├── DepthCameraItemCfg × 4  # 4个相机配置
├── ReloadableRayCasterCamera × 4  # 实际相机实例
│   └── RayCasterCamera (IsaacLab基类)
│       └── Warp射线相交计算
└── 后处理管道 (_read_one)
```

### 3.2 相关文件索引

| 文件路径 | 职责 |
|---------|------|
| `single_bodyrate_env.py` | 环境主类，调用深度图读取 |
| `single_bodyrate_env_cfg.py` | 深度相机配置参数 |
| `utils/depth_camera_array.py` | 多相机管理、后处理逻辑 |
| `utils/e2e_drone/reloadable_raycaster_camera.py` | 可重载的RayCaster实现 |

---

## 4. 多相机阵列配置

### 4.1 相机布局（机体坐标系）

```
            前 (Front) ↑
                    │
        左(Left) ───┼───► 右 (Right)
           ▲        │
           └────────┘
            后 (Back)

机体坐标系 (BC): X-前, Y-右, Z-下
```

### 4.2 配置详情

```python
depth_cameras: DepthCameraArrayCfg = DepthCameraArrayCfg(
    cameras = [
        # 前视相机: 朝向 +X
        DepthCameraItemCfg(
            name="front",
            pos_BC=(0.02, 0.0, 0.0),           # 机体前方 2cm
            quat_BC=(1.0, 0.0, 0.0, 0.0)       # 无旋转 (W, X, Y, Z)
        ),
        # 右视相机: 朝向 +Y (右转90°)
        DepthCameraItemCfg(
            name="right",
            pos_BC=(0.0, -0.02, 0.0),          # 机体右方 2cm
            quat_BC=(0.707, 0.0, 0.0, -0.707)  # -90°绕Z轴
        ),
        # 后视相机: 朝向 -X (转180°)
        DepthCameraItemCfg(
            name="back",
            pos_BC=(-0.02, 0.0, 0.0),          # 机体后方 2cm
            quat_BC=(0.0, 0.0, 0.0, 1.0)       # 180°绕Z轴
        ),
        # 左视相机: 朝向 -Y (左转90°)
        DepthCameraItemCfg(
            name="left",
            pos_BC=(0.0, 0.02, 0.0),           # 机体左方 2cm
            quat_BC=(0.707, 0.0, 0.0, 0.707)   # +90°绕Z轴
        ),
    ],
    resolution=(32, 64),           # (宽, 高)
    max_distance=4.0,              # 最大探测距离 4米
    normalize="none",              # 不进行归一化
    invalid_rate_max=0.15,         # 15% 最大无效像素率
)
```

---

## 5. 深度图处理流程

### 5.1 数据流总览

```
IsaacLab RayCaster (原始输出)
    │  Shape: (num_envs, H, W, 1)
    │  数值: 到图像平面的距离 (米)
    ▼
DepthCameraArray._read_one() 后处理
    ├── Step 1: NaN/Inf 处理
    ├── Step 2: 无效像素噪声注入 (模拟传感器噪声)
    ├── Step 3: 数值裁剪
    └── Step 4: 归一化 (此处为 "none")
    ▼
Stack为 (N, 4, H, W)
    ▼
输入 DeFM 网络
```

### 5.2 详细处理步骤

#### Step 1: NaN/Inf 处理
**代码位置**: `depth_camera_array.py:443`

```python
x = torch.nan_to_num(x, nan=max_d, posinf=max_d, neginf=0.0)
```

| 输入值 | 输出值 | 说明 |
|-------|-------|------|
| NaN | 4.0 (max_distance) | 射线未命中任何物体 |
| +Inf | 4.0 | 命中距离超出最大范围 |
| -Inf | 0.0 | 无效负值（不应出现） |

#### Step 2: 无效像素噪声注入
**代码位置**: `depth_camera_array.py:445-464`

```python
inv_max = 0.15  # 配置参数

# 每帧随机采样一个 0~15% 的比率
invalid_rate = torch.rand((), device=x.device).item() * inv_max

# 生成随机掩码
mask = torch.rand_like(x) < invalid_rate

# 用 max_distance (4.0米) 填充被标记的像素
fill = torch.tensor(4.0, device=x.device, dtype=x.dtype)
x = torch.where(mask, fill, x)
```

**三种采样模式：**

| 模式 | 采样粒度 | 说明 |
|------|---------|------|
| `per_frame` | 整帧一个比率 | 整帧图像使用相同的无效比例（当前配置） |
| `per_env` | 每个环境单独采样 | 不同环境有不同比例 |
| `per_pixel` | 每个像素单独采样 | 最大化随机性 |

#### Step 3: 数值裁剪
**代码位置**: `depth_camera_array.py:466-467`

```python
x = torch.clamp(x, 0.0, max_d)  # 限制在 [0, 4.0] 米
```

#### Step 4: 归一化
**代码位置**: `depth_camera_array.py:469-476`

```python
# 当前配置为 "none"，保持真实尺度
if self.cfg.normalize == "0_1":
    x = (x / max_d).clamp(0.0, 1.0)      # [0, 1]
elif self.cfg.normalize == "-1_1":
    x = (x / max_d).clamp(0.0, 1.0) * 2.0 - 1.0  # [-1, 1]
elif self.cfg.normalize == "none":
    pass  # 保持米为单位
```

### 5.3 环境观测中的调用

**代码位置**: `single_bodyrate_env.py:858-866`

```python
def _get_observations(self) -> dict:
    # 使用 stage="processed" 获取后处理后的数据
    depth_image_list = self._depth_cameras.read_batch(stage="processed")
    
    # 堆叠为 (N, 4, H, W) 格式
    image_raw = torch.stack(depth_image_list, dim=1)  # (num_envs, 4, 64, 32)
    image_noised = image_raw.clone()  # 可选：添加额外噪声
    
    return {
        "policy": {"image": image_noised, "state": policy_obs},
        "critic": {"image": image_raw,    "state": critic_obs},
    }
```

---

## 6. 特殊设计：传感器噪声模拟

### 6.1 为什么需要噪声？

真实深度传感器（如ToF相机、结构光相机）存在以下特性：
- **无效像素**：在边缘、强光、透明/反光表面无法测量
- **噪声分布**：通常随距离增加而增大
- **时间相关性**：连续帧间有相关性

仿真中加入15%噪声使策略对传感器缺陷具有鲁棒性。

### 6.2 噪声统计特性

```
噪声比例分布: Uniform(0%, 15%) 每帧独立采样
空间分布:     随机像素位置 (i.i.d.)
填充值:       4.0米 (表示"无有效测量")
```

### 6.3 为何保持真实尺度？

```python
normalize = "none"  # DeFM需要真实尺度的深度图
```

DeFM（Depth-based Feature Matching）网络设计假设：
- 输入为实际物理距离（米）
- 网络内部自行学习距离相关的特征
- 归一化会丢失绝对距离信息

---

## 7. 动态地图支持

### 7.1 问题背景

环境支持动态生成新地图（障碍物布局变化），但：
- RayCaster在初始化时加载Mesh
- 地图变化后，旧的Mesh数据不再有效

### 7.2 解决方案：动态重载

```python
# reloadable_raycaster_camera.py
class ReloadableRayCasterCamera(RayCasterCamera):
    
    def request_mesh_reload(self):
        """标记需要在下一帧重载Mesh"""
        self._mesh_reload_requested = True
    
    def reload_meshes(self) -> bool:
        """从当前USD Stage重新加载网格"""
        # 1. 清除旧Mesh
        self.meshes.clear()
        # 2. 从USD读取新几何数据
        points = np.asarray(mesh_geom.GetPointsAttr().Get())
        indices = np.asarray(mesh_geom.GetFaceVertexIndicesAttr().Get())
        # 3. 转换为Warp Mesh
        wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
        self.meshes[mesh_prim_path] = wp_mesh
```

### 7.3 调用流程

```
地图重新生成 (_regenerate_terrain)
    │
    ▼
DepthCameraArray.reload_cameras()
    │
    ├── ReloadableRayCasterCamera.request_mesh_reload()
    │
    ▼
下一帧更新时
    │
    ▼
ReloadableRayCasterCamera._update_buffers_impl()
    │
    ├── 检查 _mesh_reload_requested
    ├── 调用 reload_meshes()
    └── 继续正常更新
```

**代码位置**: `single_bodyrate_env.py:326`

```python
def _regenerate_terrain(self):
    # ... 生成新地图 ...
    self._depth_cameras.reload_cameras()  # 触发重载
```

---

## 8. 关键配置参数速查

### 8.1 分辨率与尺寸

| 参数 | 值 | 说明 |
|------|---|------|
| `image_width` | 32 | 图像宽度（像素） |
| `image_height` | 64 | 图像高度（像素） |
| `camera_num` | 4 | 相机数量 |
| 最终输入形状 | `(N, 4, 64, 32)` | batch × 相机 × 高 × 宽 |

### 8.2 深度范围

| 参数 | 值 | 说明 |
|------|---|------|
| `max_distance` | 4.0 | 最大探测距离（米） |
| `depth_clipping_behavior` | "max" | 超范围时填充max_distance |

### 8.3 噪声参数

| 参数 | 值 | 说明 |
|------|---|------|
| `invalid_rate_max` | 0.15 | 最大无效像素比例（15%） |
| `invalid_sampling` | "per_frame" | 每帧采样一次 |
| `invalid_fill_value` | "max_distance" | 无效像素填充4.0米 |

### 8.4 后处理开关

| 参数 | 值 | 说明 |
|------|---|------|
| `normalize` | "none" | 不归一化，保持米为单位 |
| `flatten` | False | 不展平，保持2D图像 |
| `clamp_to_max_distance` | True | 裁剪到[0, 4.0] |

---

## 9. 调试与可视化

### 9.1 启用可视化调试

```python
depth_cameras: DepthCameraArrayCfg = DepthCameraArrayCfg(
    debug_vis = True,  # 在视口中显示深度图
)
```

### 9.2 手动检查深度图

```python
# 在 _get_observations 中添加
import cv2

depth_image_cat = torch.cat(depth_image_list, dim=2)  # 拼接4个相机
depth_image_cat = depth_image_cat[0].squeeze(-1)      # 取第一个环境
H, W = depth_image_cat.shape

# 归一化到0-255用于显示
depth_image_cat_u8 = torch.clamp((depth_image_cat / 4.0 * 255.0), 0, 255).to(torch.uint8)

# 放大显示
scale = 6
img_big = cv2.resize(depth_image_cat_u8.cpu().numpy(), 
                     (int(W * scale), int(H * scale)), 
                     interpolation=cv2.INTER_NEAREST)
cv2.imshow("Depth", img_big)
cv2.waitKey(1)
```

---

## 10. 常见问题

### Q1: 为什么使用Ray Casting而不是渲染相机？
**A**: 
1. Ray Casting直接返回精确距离，无需从RGB-D转换
2. 更容易与物理引擎集成
3. 支持动态Mesh重载
4. 更适合机器人导航任务

### Q2: 15%噪声比例是固定的吗？
**A**: 不是。`invalid_rate_max=0.15` 是**上限**，实际每帧从 `Uniform(0, 0.15)` 采样，因此可能某帧8%，下一帧12%。

### Q3: 为什么无效像素填充4.0米而不是0？
**A**: 4.0米是max_distance，表示"探测范围内无物体"。填充0会被误解为"物体在相机表面"。

### Q4: 如何调整深度图分辨率？
**A**: 修改配置中的 `image_width` 和 `image_height`，并相应调整内参矩阵K的缩放比例。

### Q5: 可以修改噪声比例吗？
**A**: 修改 `invalid_rate_max` 参数：
- 0.0 = 无噪声
- 0.15 = 最大15%噪声（当前配置）
- 1.0 = 全图噪声（不建议）

---

**文档版本**: v1.0  
**最后更新**: 2026-02-24
