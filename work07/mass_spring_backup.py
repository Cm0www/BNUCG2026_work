"""
计算机图形学实验七：质点-弹簧模型（Taichi）

功能：
1. 20x20 结构弹簧布料；
2. 显式欧拉、半隐式欧拉、固定点迭代近似的隐式欧拉；
3. 重力、阻尼、速度钳制；
4. ti.ui.Window + GGUI 控制面板，支持方法切换、暂停、重置和阻尼调整。

运行：
    pip install taichi
    python mass_spring.py

无窗口自测：
    python mass_spring.py --headless --steps 300
"""

import argparse
import math
import sys

import numpy as np
import taichi as ti


def _parse_early_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--headless", action="store_true")
    known, _ = parser.parse_known_args()
    return known


# Taichi 必须在创建 field 前初始化；这里提前读取与后端相关的两个参数。
_EARLY_ARGS = _parse_early_args()
ti.init(
    arch=ti.cpu if _EARLY_ARGS.headless else ti.gpu,
    default_fp=ti.f32,
    debug=False,
)

# ----------------------------- 1. 参数与数据场 -----------------------------
N = 20                         # 布料网格为 N x N
NV = N * N                     # 质点数
NE = 2 * N * (N - 1)           # 仅结构弹簧：横向 + 纵向
N_TRIANGLES = 2 * (N - 1) * (N - 1)

MASS = 1.0
DT = 5e-4
STIFFNESS = 1.0e4
REST_LENGTH = 0.075
MAX_VELOCITY = 50.0
EPS = 1e-6
IMPLICIT_ITERS = 12
GRAVITY = ti.Vector([0.0, -9.8, 0.0])

EXPLICIT = 0
SEMI_IMPLICIT = 1
IMPLICIT = 2

# 当前状态
x = ti.Vector.field(3, dtype=ti.f32, shape=NV)      # 位置
v = ti.Vector.field(3, dtype=ti.f32, shape=NV)      # 速度
f = ti.Vector.field(3, dtype=ti.f32, shape=NV)      # 当前受力
is_fixed = ti.field(dtype=ti.i32, shape=NV)          # 是否固定

# 隐式欧拉固定点迭代的预测状态与受力
x_guess = ti.Vector.field(3, dtype=ti.f32, shape=NV)
v_guess = ti.Vector.field(3, dtype=ti.f32, shape=NV)
f_guess = ti.Vector.field(3, dtype=ti.f32, shape=NV)

# 弹簧拓扑
spring_a = ti.field(dtype=ti.i32, shape=NE)
spring_b = ti.field(dtype=ti.i32, shape=NE)
spring_rest = ti.field(dtype=ti.f32, shape=NE)

# 渲染数据
tri_indices = ti.field(dtype=ti.i32, shape=N_TRIANGLES * 3)
line_indices = ti.field(dtype=ti.i32, shape=NE * 2)
particle_color = ti.Vector.field(3, dtype=ti.f32, shape=NV)

# 阻尼放在 Taichi 标量场中，方便 GGUI 实时修改后直接被 kernel 读取
damping = ti.field(dtype=ti.f32, shape=())


# ------------------------------- 2. 初始化 --------------------------------
@ti.kernel
def init_positions():
    """只初始化质点状态与固定点标记。"""
    for p in range(NV):
        i = p // N
        j = p % N
        # 垂直悬挂的初始布料；加入轻微 z 扰动，便于观察三维效果。
        px = (j - 0.5 * (N - 1)) * REST_LENGTH
        py = 1.15 - i * REST_LENGTH
        pz = 0.035 * ti.sin(0.55 * j)
        x[p] = ti.Vector([px, py, pz])
        v[p] = ti.Vector([0.0, 0.0, 0.0])
        f[p] = ti.Vector([0.0, 0.0, 0.0])
        x_guess[p] = x[p]
        v_guess[p] = v[p]
        f_guess[p] = ti.Vector([0.0, 0.0, 0.0])

        # 固定顶端左右两个角点
        if i == 0 and (j == 0 or j == N - 1):
            is_fixed[p] = 1
        else:
            is_fixed[p] = 0


@ti.kernel
def init_springs():
    """独立初始化横向、纵向结构弹簧。"""
    horizontal_count = N * (N - 1)
    for e in range(NE):
        if e < horizontal_count:
            i = e // (N - 1)
            j = e % (N - 1)
            spring_a[e] = i * N + j
            spring_b[e] = i * N + j + 1
        else:
            q = e - horizontal_count
            i = q // N
            j = q % N
            spring_a[e] = i * N + j
            spring_b[e] = (i + 1) * N + j
        spring_rest[e] = REST_LENGTH


@ti.kernel
def init_render_data():
    """独立初始化三角面、弹簧线段和质点颜色。"""
    for p in range(NV):
        if is_fixed[p] == 1:
            particle_color[p] = ti.Vector([1.0, 0.28, 0.18])
        else:
            particle_color[p] = ti.Vector([0.15, 0.70, 1.0])

    for e in range(NE):
        line_indices[2 * e] = spring_a[e]
        line_indices[2 * e + 1] = spring_b[e]

    for cell in range((N - 1) * (N - 1)):
        i = cell // (N - 1)
        j = cell % (N - 1)
        a = i * N + j
        b = a + 1
        c = a + N
        d = c + 1
        base = 6 * cell
        tri_indices[base] = a
        tri_indices[base + 1] = c
        tri_indices[base + 2] = b
        tri_indices[base + 3] = b
        tri_indices[base + 4] = c
        tri_indices[base + 5] = d


def reset_cloth():
    """Python 侧按顺序调用多个初始化 kernel，满足 GPU 状态同步要求。"""
    init_positions()
    init_springs()
    init_render_data()


# --------------------------- 3. 力学计算与防爆 ----------------------------
@ti.func
def clamp_velocity(vel):
    """速度钳制，防止显式欧拉等不稳定情况下数值无限增大。"""
    speed = vel.norm()
    if speed > MAX_VELOCITY:
        vel = vel * (MAX_VELOCITY / (speed + EPS))
    return vel


@ti.func
def compute_forces_on(edge):
    """为当前状态累加一根弹簧的胡克力。

    外力（重力和阻尼）在调用本函数前已经写入 f；此处以 ti.atomic_add
    同时向弹簧两端质点累加内力，避免多个弹簧并发写同一质点时发生冲突。
    """
    a = spring_a[edge]
    b = spring_b[edge]
    delta = x[a] - x[b]
    length = delta.norm()
    if length > EPS:
        spring_force = -STIFFNESS * (length - spring_rest[edge]) * delta / length
        ti.atomic_add(f[a][0], spring_force[0])
        ti.atomic_add(f[a][1], spring_force[1])
        ti.atomic_add(f[a][2], spring_force[2])
        ti.atomic_add(f[b][0], -spring_force[0])
        ti.atomic_add(f[b][1], -spring_force[1])
        ti.atomic_add(f[b][2], -spring_force[2])


@ti.func
def compute_guess_forces_on(edge):
    """隐式欧拉迭代时，使用预测状态 x_guess、v_guess 计算未来受力。"""
    a = spring_a[edge]
    b = spring_b[edge]
    delta = x_guess[a] - x_guess[b]
    length = delta.norm()
    if length > EPS:
        spring_force = -STIFFNESS * (length - spring_rest[edge]) * delta / length
        ti.atomic_add(f_guess[a][0], spring_force[0])
        ti.atomic_add(f_guess[a][1], spring_force[1])
        ti.atomic_add(f_guess[a][2], spring_force[2])
        ti.atomic_add(f_guess[b][0], -spring_force[0])
        ti.atomic_add(f_guess[b][1], -spring_force[1])
        ti.atomic_add(f_guess[b][2], -spring_force[2])


# --------------------------- 4. 三种积分求解器 ----------------------------
@ti.kernel
def step_explicit():
    """显式欧拉：x_{t+1}=x_t+v_t*dt，v_{t+1}=v_t+a_t*dt。

    一个 @ti.kernel 内按“初始化外力 -> 原子累加弹簧力 -> 更新状态”完成，
    Python 侧每个子步只有一次 kernel 调用。
    """
    for p in range(NV):
        f[p] = MASS * GRAVITY - damping[None] * v[p]

    for e in range(NE):
        compute_forces_on(e)

    for p in range(NV):
        if is_fixed[p] == 0:
            old_x = x[p]
            old_v = v[p]
            new_v = clamp_velocity(old_v + DT * f[p] / MASS)
            x[p] = old_x + DT * old_v
            v[p] = new_v
        else:
            v[p] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def step_semi_implicit():
    """半隐式欧拉：先更新 v_{t+1}，再以 v_{t+1} 更新 x_{t+1}。"""
    for p in range(NV):
        f[p] = MASS * GRAVITY - damping[None] * v[p]

    for e in range(NE):
        compute_forces_on(e)

    for p in range(NV):
        if is_fixed[p] == 0:
            new_v = clamp_velocity(v[p] + DT * f[p] / MASS)
            v[p] = new_v
            x[p] = x[p] + DT * new_v
        else:
            v[p] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def begin_implicit():
    """固定点迭代的初始猜测：使用当前时刻状态。"""
    for p in range(NV):
        x_guess[p] = x[p]
        v_guess[p] = v[p]


@ti.kernel
def step_implicit_iter():
    """隐式欧拉的一次固定点迭代。

    v^(k+1) = v_t + dt/m * F(x^(k), v^(k))
    x^(k+1) = x_t + dt * v^(k+1)
    """
    for p in range(NV):
        f_guess[p] = MASS * GRAVITY - damping[None] * v_guess[p]

    for e in range(NE):
        compute_guess_forces_on(e)

    for p in range(NV):
        if is_fixed[p] == 0:
            new_v = clamp_velocity(v[p] + DT * f_guess[p] / MASS)
            v_guess[p] = new_v
            x_guess[p] = x[p] + DT * new_v
        else:
            x_guess[p] = x[p]
            v_guess[p] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def commit_implicit():
    """将固定点迭代收敛后的预测状态提交为下一时刻状态。"""
    for p in range(NV):
        x[p] = x_guess[p]
        v[p] = v_guess[p]


def advance(method):
    """推进一个时间步。"""
    if method == EXPLICIT:
        step_explicit()
    elif method == SEMI_IMPLICIT:
        step_semi_implicit()
    else:
        begin_implicit()
        for _ in range(IMPLICIT_ITERS):
            step_implicit_iter()
        commit_implicit()


# ------------------------------ 5. 渲染与交互 ------------------------------
def run_gui():
    reset_cloth()
    damping[None] = 1.0

    window = ti.ui.Window("Mass-Spring Cloth | Taichi", res=(1120, 780), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.035, 0.045, 0.070))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    # 只设置一次初始相机；之后由 track_user_inputs 保持用户拖拽后的视角。
    camera.position(1.55, 0.18, 2.45)
    camera.lookat(0.0, 0.05, 0.0)
    camera.up(0.0, 1.0, 0.0)

    method = SEMI_IMPLICIT
    paused = False
    keep_running = True
    substeps = 4

    while window.running and keep_running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                keep_running = False

        if not paused:
            for _ in range(substeps):
                advance(method)

        # 鼠标右键拖动可旋转视角，滚轮可缩放。
        camera.track_user_inputs(window, movement_speed=0.025, hold_key=ti.ui.RMB)

        scene.set_camera(camera)
        scene.ambient_light((0.45, 0.45, 0.45))
        scene.point_light(pos=(1.5, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(-1.5, 0.5, 1.0), color=(0.45, 0.55, 1.0))
        scene.mesh(
            x,
            indices=tri_indices,
            color=(0.10, 0.30, 0.62),
            two_sided=True,
            show_wireframe=False,
        )
        scene.lines(x, width=0.0016, indices=line_indices, color=(0.92, 0.94, 1.0))
        scene.particles(x, radius=0.010, per_vertex_color=particle_color)
        canvas.scene(scene)

        gui = window.get_gui()
        with gui.sub_window("Control Panel", 0.02, 0.02, 0.30, 0.34) as g:
            method_names = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler"]
            g.text("Integration Method")
            g.text("Current: " + method_names[method])
            if g.button("Use Explicit Euler (explosive)"):
                method = EXPLICIT
            if g.button("Use Semi-Implicit Euler (stable)"):
                method = SEMI_IMPLICIT
            if g.button("Use Implicit Euler (damped)"):
                method = IMPLICIT

            new_damping = g.slider_float("Damping", float(damping[None]), 0.0, 10.0)
            damping[None] = new_damping
            g.text(f"dt={DT:.4g}, stiffness={STIFFNESS:.0f}")
            g.text(f"implicit iterations={IMPLICIT_ITERS}")

            if g.button("Resume Simulation" if paused else "Pause Simulation"):
                paused = not paused
            if g.button("Reset Cloth"):
                reset_cloth()
                paused = False

        window.show()


def run_headless(steps, method):
    """无窗口环境下的快速自检，确保三类积分核可编译并且位置不出现 NaN。"""
    reset_cloth()
    damping[None] = 1.0
    for _ in range(steps):
        advance(method)
    positions = x.to_numpy()
    velocities = v.to_numpy()
    if not (np.isfinite(positions).all() and np.isfinite(velocities).all()):
        raise RuntimeError("仿真出现 NaN / Inf，请检查参数或实现。")
    print(f"headless check passed: steps={steps}, method={method}, "
          f"y-range=[{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="不创建窗口，仅执行物理核自检")
    parser.add_argument("--steps", type=int, default=240, help="headless 模式下的时间步数")
    parser.add_argument("--method", choices=["explicit", "semi", "implicit"], default="semi")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.headless:
        method_map = {"explicit": EXPLICIT, "semi": SEMI_IMPLICIT, "implicit": IMPLICIT}
        run_headless(args.steps, method_map[args.method])
    else:
        run_gui()
