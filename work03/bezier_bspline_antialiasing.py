import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000  


pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)


curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)


def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

# 2. 均匀三次B样条
def b_spline(points, samples=NUM_SEGMENTS):
    pts = np.array(points, dtype=np.float32)
    n = len(pts)
    result = []

    if n < 4:
        return []

    # 均匀三次B样条基矩阵
    M = np.array([
        [-1,  3, -3, 1],
        [ 3, -6,  3, 0],
        [-3,  0,  3, 0],
        [ 1,  4,  1, 0]
    ]) / 6.0

    seg_count = n - 3
    for i in range(seg_count):
        P = pts[i:i+4]
        for j in range(samples // seg_count):
            t = j / (samples // seg_count)
            T = np.array([t**3, t**2, t, 1], dtype=np.float32)
            point = T @ M @ P
            result.append(point)

    return result


@ti.kernel
def clear_pixels():
    """清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_antialiasing_kernel(n: ti.i32):
    """
    实验1要求：反走样（抗锯齿）绘制
    基于3×3邻域+高斯距离衰减，实现平滑边缘
    """
    for i in range(n):
        # 1. 获取精确浮点坐标（亚像素级精度）
        fx = curve_points_field[i][0] * WIDTH
        fy = curve_points_field[i][1] * HEIGHT
        
        # 2. 取3×3邻域的整数像素坐标
        x0 = ti.cast(fx - 0.5, ti.i32)
        y0 = ti.cast(fy - 0.5, ti.i32)

        # 3. 遍历3×3邻域，计算距离衰减权重
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x = x0 + dx
                y = y0 + dy
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    # 计算像素中心点与精确几何点的距离
                    dist_x = (x + 0.5) - fx
                    dist_y = (y + 0.5) - fy
                    dist_sq = dist_x**2 + dist_y**2
                    
                    # 高斯距离衰减模型（实验要求：越近权重越大）
                    weight = ti.exp(-dist_sq * 2.0)
                    
                    # 累加颜色权重，实现平滑混合
                    pixels[x, y] += ti.Vector([0.0, weight, 0.0])

def main():
    window = ti.ui.Window("Bezier + B-Spline + 反走样", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    use_bspline = False  # 按b键切换模式

    while window.running:
        # 事件处理
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
            elif e.key == 'c':
                control_points = []
                print("画布已清空")
            elif e.key == 'b':
                use_bspline = not use_bspline
                print(f"切换模式：{'B样条曲线' if use_bspline else '贝塞尔曲线'}")

        clear_pixels()
        current_count = len(control_points)

        # 计算并绘制曲线
        if current_count >= 2:
            curve_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)

            # B样条模式（需要≥4个控制点）
            if use_bspline and current_count >= 4:
                spline_points = b_spline(control_points)
                if len(spline_points) > 0:
                    curve_np = np.array(spline_points, dtype=np.float32)
            # 贝塞尔模式
            else:
                for t_int in range(NUM_SEGMENTS + 1):
                    t = t_int / NUM_SEGMENTS
                    curve_np[t_int] = de_casteljau(control_points, t)

            # 一次性上传GPU，调用反走样绘制内核
            curve_points_field.from_numpy(curve_np)
            draw_curve_antialiasing_kernel(len(curve_np))

        # 绘制控制点与控制多边形
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))

            if current_count >= 2:
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i+1])
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))

        canvas.set_image(pixels)
        window.show()

if __name__ == '__main__':
    main()