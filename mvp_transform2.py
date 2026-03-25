import taichi as ti
import math

# 初始化 Taichi
ti.init(arch=ti.metal)

# ===================== 正方体配置 =====================
# 正方体 8 个顶点 (中心在原点，边长 2，范围 [-1, 1])
CUBE_VERTICES = 8
# 正方体 12 条边 (每两个点组成一条线)
CUBE_INDICES = ti.field(ti.i32, shape=12 * 2)

# 声明 Taichi Field
vertices = ti.Vector.field(3, dtype=ti.f32, shape=CUBE_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=CUBE_VERTICES)

# ===================== 变换矩阵 =====================
@ti.func
def get_model_matrix(angle: ti.f32):
    """绕 Y 轴旋转（3D 效果更好）"""
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    """视图变换矩阵"""
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """透视投影矩阵"""
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o

# ===================== 顶点变换 =====================
@ti.kernel
def compute_transform(angle: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    mvp = proj @ view @ model

    for i in range(CUBE_VERTICES):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        # 映射到屏幕 [0,1] 空间
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

# ===================== 初始化正方体 =====================
def init_cube():
    # 8 个顶点
    cube_points = [
        (-1, -1, -1), (1, -1, -1),
        (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1),
        (1, 1, 1), (-1, 1, 1),
    ]
    for i in range(8):
        vertices[i] = cube_points[i]

    # 12 条边索引
    edges = [
        0,1, 1,2, 2,3, 3,0,  # 前面
        4,5, 5,6, 6,7, 7,4,  # 后面
        0,4, 1,5, 2,6, 3,7   # 连接前后
    ]
    for i in range(24):
        CUBE_INDICES[i] = edges[i]

# ===================== 主程序 =====================
def main():
    init_cube()
    gui = ti.GUI("3D Cube Wireframe (Taichi)", res=(700, 700))
    angle = 0.0

    while gui.running:
        # 按键控制
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        # 计算 MVP 变换
        compute_transform(angle)

        # 绘制 12 条正方体边
        for i in range(12):
            a = screen_coords[CUBE_INDICES[i*2]]
            b = screen_coords[CUBE_INDICES[i*2+1]]
            gui.line(a, b, radius=2, color=0xffffff)

        gui.show()

if __name__ == '__main__':
    main()