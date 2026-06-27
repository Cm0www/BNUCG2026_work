"""
计算机图形学实验七：质点-弹簧模型（Taichi）
扩展版：
1. 结构弹簧（Structural）+ 剪切弹簧（Shear）+ 弯曲弹簧（Bending）；
2. 球体空间碰撞：位置投影 + 去除向内法向速度；
3. 显式欧拉、半隐式欧拉、固定点迭代近似的隐式欧拉；
4. Taichi GGUI：积分器、阻尼、剪切/弯曲弹簧和球体碰撞均可交互切换。

运行：
    python3 -m pip install -r requirements.txt
    python3 mass_spring.py

无窗口自检：
    python3 mass_spring.py --headless --steps 300 --method semi
"""

import argparse
import math

import numpy as np
import taichi as ti


def _parse_early_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--headless", action="store_true")
    known, _ = parser.parse_known_args()
    return known


# Taichi 必须在创建 field 前初始化。
_EARLY_ARGS = _parse_early_args()
ti.init(
    arch=ti.cpu if _EARLY_ARGS.headless else ti.gpu,
    default_fp=ti.f32,
    debug=False,
)

# ----------------------------- 1. 参数与数据场 -----------------------------
N = 20
NV = N * N

# 三类弹簧数量
STRUCTURAL_H = N * (N - 1)                # 横向相邻
STRUCTURAL_V = N * (N - 1)                # 纵向相邻
SHEAR_NE = 2 * (N - 1) * (N - 1)          # 每个网格单元两条对角线
BENDING_H = N * (N - 2)                   # 横向间隔两个质点
BENDING_V = N * (N - 2)                   # 纵向间隔两个质点
STRUCTURAL_NE = STRUCTURAL_H + STRUCTURAL_V
BENDING_NE = BENDING_H + BENDING_V
NE = STRUCTURAL_NE + SHEAR_NE + BENDING_NE
N_TRIANGLES = 2 * (N - 1) * (N - 1)

# 物理参数
MASS = 1.0
DT = 5e-4
REST_LENGTH = 0.075
K_STRUCTURAL = 1.0e4
K_SHEAR = 7.0e3
K_BENDING = 2.0e3
MAX_VELOCITY = 50.0
EPS = 1e-6
IMPLICIT_ITERS = 12
GRAVITY = ti.Vector([0.0, -9.8, 0.0])

# 球体碰撞参数
SPHERE_CENTER_VALUE = (0.0, -0.28, 0.0)
SPHERE_RADIUS_VALUE = 0.34
COLLISION_DAMPING = 0.985

# 积分器编号
EXPLICIT = 0
SEMI_IMPLICIT = 1
IMPLICIT = 2

# 弹簧类型编号
STRUCTURAL = 0
SHEAR = 1
BENDING = 2

# 当前状态
x = ti.Vector.field(3, dtype=ti.f32, shape=NV)
v = ti.Vector.field(3, dtype=ti.f32, shape=NV)
f = ti.Vector.field(3, dtype=ti.f32, shape=NV)
is_fixed = ti.field(dtype=ti.i32, shape=NV)

# 隐式欧拉固定点迭代的预测状态与受力
x_guess = ti.Vector.field(3, dtype=ti.f32, shape=NV)
v_guess = ti.Vector.field(3, dtype=ti.f32, shape=NV)
f_guess = ti.Vector.field(3, dtype=ti.f32, shape=NV)

# 弹簧拓扑与属性
spring_a = ti.field(dtype=ti.i32, shape=NE)
spring_b = ti.field(dtype=ti.i32, shape=NE)
spring_rest = ti.field(dtype=ti.f32, shape=NE)
spring_k = ti.field(dtype=ti.f32, shape=NE)
spring_kind = ti.field(dtype=ti.i32, shape=NE)

# 渲染数据：为避免线框过密，只显示结构弹簧。
tri_indices = ti.field(dtype=ti.i32, shape=N_TRIANGLES * 3)
line_indices = ti.field(dtype=ti.i32, shape=STRUCTURAL_NE * 2)
particle_color = ti.Vector.field(3, dtype=ti.f32, shape=NV)

# 控制参数：使用标量 field，GUI 修改后可被 kernel 直接读取。
damping = ti.field(dtype=ti.f32, shape=())
enable_shear = ti.field(dtype=ti.i32, shape=())
enable_bending = ti.field(dtype=ti.i32, shape=())
enable_collision = ti.field(dtype=ti.i32, shape=())

# 球心和半径放进 field，方便 kernel 碰撞和 GGUI 渲染共用。
sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=1)
sphere_radius = ti.field(dtype=ti.f32, shape=())


# ------------------------------- 2. 初始化 --------------------------------
@ti.kernel
def init_positions():
    """初始化质点位置、速度、受力和固定点标记。"""
    for p in range(NV):
        i = p // N
        j = p % N
        # 垂直悬挂的初始布料，轻微 z 扰动让球体碰撞时向前/后分离。
        px = (j - 0.5 * (N - 1)) * REST_LENGTH
        py = 1.15 - i * REST_LENGTH
        pz = 0.035 * ti.sin(0.55 * j)
        x[p] = ti.Vector([px, py, pz])
        v[p] = ti.Vector([0.0, 0.0, 0.0])
        f[p] = ti.Vector([0.0, 0.0, 0.0])
        x_guess[p] = x[p]
        v_guess[p] = v[p]
        f_guess[p] = ti.Vector([0.0, 0.0, 0.0])

        # 固定顶部左右两个角点。
        if i == 0 and (j == 0 or j == N - 1):
            is_fixed[p] = 1
        else:
            is_fixed[p] = 0


@ti.kernel
def init_springs():
    """独立初始化结构、剪切和弯曲弹簧。"""
    for e in range(NE):
        if e < STRUCTURAL_H:
            # 横向结构弹簧：(i, j) <-> (i, j+1)
            i = e // (N - 1)
            j = e % (N - 1)
            spring_a[e] = i * N + j
            spring_b[e] = i * N + j + 1
            spring_rest[e] = REST_LENGTH
            spring_k[e] = K_STRUCTURAL
            spring_kind[e] = STRUCTURAL

        elif e < STRUCTURAL_NE:
            # 纵向结构弹簧：(i, j) <-> (i+1, j)
            q = e - STRUCTURAL_H
            i = q // N
            j = q % N
            spring_a[e] = i * N + j
            spring_b[e] = (i + 1) * N + j
            spring_rest[e] = REST_LENGTH
            spring_k[e] = K_STRUCTURAL
            spring_kind[e] = STRUCTURAL

        elif e < STRUCTURAL_NE + SHEAR_NE:
            # 剪切弹簧：每个单元两条对角线。
            q = e - STRUCTURAL_NE
            cell = q // 2
            diagonal = q % 2
            i = cell // (N - 1)
            j = cell % (N - 1)

            if diagonal == 0:
                spring_a[e] = i * N + j
                spring_b[e] = (i + 1) * N + (j + 1)
            else:
                spring_a[e] = i * N + (j + 1)
                spring_b[e] = (i + 1) * N + j

            spring_rest[e] = ti.sqrt(2.0) * REST_LENGTH
            spring_k[e] = K_SHEAR
            spring_kind[e] = SHEAR

        elif e < STRUCTURAL_NE + SHEAR_NE + BENDING_H:
            # 横向弯曲弹簧：(i, j) <-> (i, j+2)
            q = e - STRUCTURAL_NE - SHEAR_NE
            i = q // (N - 2)
            j = q % (N - 2)
            spring_a[e] = i * N + j
            spring_b[e] = i * N + j + 2
            spring_rest[e] = 2.0 * REST_LENGTH
            spring_k[e] = K_BENDING
            spring_kind[e] = BENDING

        else:
            # 纵向弯曲弹簧：(i, j) <-> (i+2, j)
            q = e - STRUCTURAL_NE - SHEAR_NE - BENDING_H
            i = q // N
            j = q % N
            spring_a[e] = i * N + j
            spring_b[e] = (i + 2) * N + j
            spring_rest[e] = 2.0 * REST_LENGTH
            spring_k[e] = K_BENDING
            spring_kind[e] = BENDING


@ti.kernel
def init_scene_objects():
    """初始化球体数据和用于渲染的静态索引。"""
    sphere_center[0] = ti.Vector(
        [SPHERE_CENTER_VALUE[0], SPHERE_CENTER_VALUE[1], SPHERE_CENTER_VALUE[2]]
    )
    sphere_radius[None] = SPHERE_RADIUS_VALUE

    for p in range(NV):
        if is_fixed[p] == 1:
            particle_color[p] = ti.Vector([1.0, 0.28, 0.18])
        else:
            particle_color[p] = ti.Vector([0.15, 0.70, 1.0])

    # 只显示结构弹簧，不影响剪切和弯曲弹簧的实际计算。
    for e in range(STRUCTURAL_NE):
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
    """Python 侧顺序调用多个 kernel，保证初始化状态同步。"""
    init_positions()
    init_springs()
    init_scene_objects()


# --------------------------- 3. 力学计算与碰撞 ----------------------------
@ti.func
def clamp_velocity(vel):
    """速度钳制，防止不稳定状态下速度无限增大。"""
    speed = vel.norm()
    if speed > MAX_VELOCITY:
        vel = vel * (MAX_VELOCITY / (speed + EPS))
    return vel


@ti.func
def spring_is_enabled(edge):
    """结构弹簧始终启用；剪切/弯曲弹簧可在 GUI 中切换。"""
    kind = spring_kind[edge]
    active = 1
    if kind == SHEAR and enable_shear[None] == 0:
        active = 0
    if kind == BENDING and enable_bending[None] == 0:
        active = 0
    return active


@ti.func
def resolve_sphere_collision(pos, vel):
    """球体无穿透碰撞处理。

    1. 若质点落入球内，将它投影到球面；
    2. 去除速度沿法向的向内分量，避免继续穿入；
    3. 轻微衰减速度，减小接触点抖动。
    """
    if enable_collision[None] != 0:
        offset = pos - sphere_center[0]
        distance = offset.norm()
        radius = sphere_radius[None]

        if distance < radius:
            normal = ti.Vector([0.0, 1.0, 0.0])
            if distance > EPS:
                normal = offset / distance

            # 位置投影：直接消除球体内部的穿透。
            pos = sphere_center[0] + normal * radius

            # 仅去除指向球内的法向速度分量，保留切向滑动。
            normal_speed = vel.dot(normal)
            if normal_speed < 0.0:
                vel = vel - normal_speed * normal

            vel = vel * COLLISION_DAMPING

    return pos, vel


@ti.func
def compute_forces_on(edge):
    """使用当前状态累加一根弹簧的胡克力。"""
    if spring_is_enabled(edge) == 1:
        a = spring_a[edge]
        b = spring_b[edge]
        delta = x[a] - x[b]
        length = delta.norm()

        if length > EPS:
            spring_force = -spring_k[edge] * (length - spring_rest[edge]) * delta / length
            ti.atomic_add(f[a][0], spring_force[0])
            ti.atomic_add(f[a][1], spring_force[1])
            ti.atomic_add(f[a][2], spring_force[2])
            ti.atomic_add(f[b][0], -spring_force[0])
            ti.atomic_add(f[b][1], -spring_force[1])
            ti.atomic_add(f[b][2], -spring_force[2])


@ti.func
def compute_guess_forces_on(edge):
    """隐式欧拉迭代时，基于预测状态计算未来受力。"""
    if spring_is_enabled(edge) == 1:
        a = spring_a[edge]
        b = spring_b[edge]
        delta = x_guess[a] - x_guess[b]
        length = delta.norm()

        if length > EPS:
            spring_force = -spring_k[edge] * (length - spring_rest[edge]) * delta / length
            ti.atomic_add(f_guess[a][0], spring_force[0])
            ti.atomic_add(f_guess[a][1], spring_force[1])
            ti.atomic_add(f_guess[a][2], spring_force[2])
            ti.atomic_add(f_guess[b][0], -spring_force[0])
            ti.atomic_add(f_guess[b][1], -spring_force[1])
            ti.atomic_add(f_guess[b][2], -spring_force[2])


# --------------------------- 4. 三种积分求解器 ----------------------------
@ti.kernel
def step_explicit():
    """显式欧拉：先用旧速度更新位置，再用当前力更新速度。"""
    for p in range(NV):
        f[p] = MASS * GRAVITY - damping[None] * v[p]

    for e in range(NE):
        compute_forces_on(e)

    for p in range(NV):
        if is_fixed[p] == 0:
            old_x = x[p]
            old_v = v[p]
            new_v = clamp_velocity(old_v + DT * f[p] / MASS)
            new_x = old_x + DT * old_v
            x[p], v[p] = resolve_sphere_collision(new_x, new_v)
        else:
            v[p] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def step_semi_implicit():
    """半隐式欧拉：先更新速度，再用新速度更新位置。"""
    for p in range(NV):
        f[p] = MASS * GRAVITY - damping[None] * v[p]

    for e in range(NE):
        compute_forces_on(e)

    for p in range(NV):
        if is_fixed[p] == 0:
            new_v = clamp_velocity(v[p] + DT * f[p] / MASS)
            new_x = x[p] + DT * new_v
            x[p], v[p] = resolve_sphere_collision(new_x, new_v)
        else:
            v[p] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def begin_implicit():
    """固定点迭代的初始猜测。"""
    for p in range(NV):
        x_guess[p] = x[p]
        v_guess[p] = v[p]


@ti.kernel
def step_implicit_iter():
    """隐式欧拉的一次固定点迭代。"""
    for p in range(NV):
        f_guess[p] = MASS * GRAVITY - damping[None] * v_guess[p]

    for e in range(NE):
        compute_guess_forces_on(e)

    for p in range(NV):
        if is_fixed[p] == 0:
            new_v = clamp_velocity(v[p] + DT * f_guess[p] / MASS)
            new_x = x[p] + DT * new_v
            x_guess[p], v_guess[p] = resolve_sphere_collision(new_x, new_v)
        else:
            x_guess[p] = x[p]
            v_guess[p] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def commit_implicit():
    """提交固定点迭代结果。"""
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
    damping[None] = 1.0
    enable_shear[None] = 1
    enable_bending[None] = 1
    enable_collision[None] = 1
    reset_cloth()

    window = ti.ui.Window(
        "Mass-Spring Cloth | Structural + Shear + Bending + Sphere Collision",
        res=(1120, 780),
        vsync=True,
    )
    canvas = window.get_canvas()
    canvas.set_background_color((0.035, 0.045, 0.070))

    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(1.75, 0.18, 2.85)
    camera.lookat(0.0, 0.14, 0.0)
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
        scene.lines(
            x,
            width=0.0016,
            indices=line_indices,
            color=(0.92, 0.94, 1.0),
        )
        scene.particles(x, radius=0.010, per_vertex_color=particle_color)
        # 视觉半径略小，避免渲染层面显得布料穿进球内。
        scene.particles(
            sphere_center,
            radius=float(sphere_radius[None]) * 0.97,
            color=(0.92, 0.45, 0.16),
        )
        canvas.scene(scene)

        gui = window.get_gui()
        with gui.sub_window("Control Panel", 0.02, 0.02, 0.34, 0.47) as g:
            method_names = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler"]
            g.text("Integration Method")
            g.text("Current: " + method_names[method])
            if g.button("Use Explicit Euler (explosive)"):
                method = EXPLICIT
            if g.button("Use Semi-Implicit Euler (stable)"):
                method = SEMI_IMPLICIT
            if g.button("Use Implicit Euler (damped)"):
                method = IMPLICIT

            g.text("Spring Model")
            shear_checked = g.checkbox(
                "Enable shear springs", bool(enable_shear[None])
            )
            bending_checked = g.checkbox(
                "Enable bending springs", bool(enable_bending[None])
            )
            enable_shear[None] = 1 if shear_checked else 0
            enable_bending[None] = 1 if bending_checked else 0

            collision_checked = g.checkbox(
                "Enable sphere collision", bool(enable_collision[None])
            )
            enable_collision[None] = 1 if collision_checked else 0

            new_damping = g.slider_float(
                "Damping", float(damping[None]), 0.0, 10.0
            )
            damping[None] = new_damping

            g.text(f"springs: structural={STRUCTURAL_NE}, "
                   f"shear={SHEAR_NE}, bending={BENDING_NE}")
            g.text(f"dt={DT:.4g}, k=({K_STRUCTURAL:.0f}, "
                   f"{K_SHEAR:.0f}, {K_BENDING:.0f})")
            g.text(f"implicit iterations={IMPLICIT_ITERS}")

            if g.button("Resume Simulation" if paused else "Pause Simulation"):
                paused = not paused
            if g.button("Reset Cloth"):
                reset_cloth()
                paused = False

        window.show()


def run_headless(steps, method):
    """无窗口自检：验证三类积分核、三类弹簧和碰撞不会产生 NaN/Inf。"""
    damping[None] = 1.0
    enable_shear[None] = 1
    enable_bending[None] = 1
    enable_collision[None] = 1
    reset_cloth()

    for _ in range(steps):
        advance(method)

    positions = x.to_numpy()
    velocities = v.to_numpy()
    if not (np.isfinite(positions).all() and np.isfinite(velocities).all()):
        raise RuntimeError("仿真出现 NaN / Inf，请检查参数或实现。")

    center = sphere_center.to_numpy()[0]
    radius = float(sphere_radius[None])
    min_dist = np.linalg.norm(positions - center, axis=1).min()
    print(
        f"headless check passed: steps={steps}, method={method}, "
        f"y-range=[{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}], "
        f"min-distance-to-sphere={min_dist:.4f}, radius={radius:.4f}"
    )


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
