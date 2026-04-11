# 3D 顶点变换 (MVP) 实现

这个项目基于 Taichi 编程语言实现了 3D 图形中的 MVP（Model-View-Projection）变换矩阵，包含三角形和正方体两种 3D 几何体的线框渲染示例，展示了从 3D 顶点坐标到 2D 屏幕坐标的完整变换流程。

## 功能说明
- **模型变换（Model）**：实现几何体绕轴旋转（三角形绕 Z 轴、正方体绕 Y 轴）
- **视图变换（View）**：将相机（视点）移动到坐标原点
- **投影变换（Projection）**：透视投影（Perspective Projection），将 3D 透视平截头体转换为标准化设备坐标（NDC）
- **视口变换**：将 NDC 坐标映射到 2D 屏幕空间并渲染线框

## 环境依赖
- Python 3.7+
- Taichi 1.0+（推荐最新稳定版）

安装依赖：
```bash
pip install taichi
```

## 文件说明
| 文件名称 | 功能描述 |
|----------|----------|
| `mvp_transform.py` | 三角形 3D 变换示例：实现绕 Z 轴旋转的三角形 MVP 变换与渲染 |
| `mvp_transform2.py` | 正方体 3D 变换示例：实现绕 Y 轴旋转的正方体线框 MVP 变换与渲染 |

## 核心原理
MVP 变换是 3D 图形渲染的核心流程，整体公式为：

```
MVP = Projection × View × Model
```
### 1. 模型矩阵（Model Matrix）
将几何体从局部空间转换到世界空间，本项目中实现绕指定轴的旋转变换：
- 三角形：绕 Z 轴旋转
- 正方体：绕 Y 轴旋转

### 2. 视图矩阵（View Matrix）
将相机（视点）移动到坐标原点，本项目中相机固定在 `(0.0, 0.0, 5.0)`，看向 -Z 轴方向。

### 3. 投影矩阵（Projection Matrix）
实现透视投影：
1. **透视转正交（M_p2o）**：将透视平截头体挤压为长方体
2. **正交投影（M_ortho）**：将长方体缩放平移至 NDC 空间（[-1, 1]^3）

### 4. 视口变换
将 NDC 坐标映射到屏幕空间 [0, 1] × [0, 1]，最终通过 Taichi GUI 渲染。

## 运行方法
### 运行三角形示例
```bash
python mvp_transform.py
```

### 运行正方体示例
```bash
python mvp_transform2.py
```

## 交互说明
- **A 键**：几何体顺时针旋转（角度 +10°）
- **D 键**：几何体逆时针旋转（角度 -10°）
- **ESC 键**：退出程序

## 效果展示
- 三角形示例：红、绿、蓝三色线条组成的三角形绕 Z 轴旋转，呈现 3D 透视效果
- 正方体示例：白色线框正方体绕 Y 轴旋转，展示完整的 3D 正方体透视效果

## 核心代码结构
1. **矩阵定义**：`get_model_matrix` / `get_view_matrix` / `get_projection_matrix` 实现各变换矩阵
2. **并行计算**：`compute_transform` 内核函数在 Taichi 并行架构上计算顶点变换
3. **几何体初始化**：定义三角形/正方体的顶点坐标和边索引
4. **GUI 渲染**：通过 Taichi GUI 绘制变换后的 2D 线框
## 效果展示
![QOvJtboF_converted](https://github.com/user-attachments/assets/b4293000-1ed3-4658-8d43-9ce38eed03b1)
![dlUfWRki_converted](https://github.com/user-attachments/assets/37d66492-4c33-4c2f-8458-99bfdd0e868b)


## 扩展方向
- 增加平移、缩放变换
- 支持相机视角/位置调整
- 实现光照效果
- 增加纹理映射
- 扩展为面渲染（而非线框）
