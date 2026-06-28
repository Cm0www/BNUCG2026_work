"""
计算机图形学实验五：Whitted-Style 光线追踪

必做：
1. 棋盘格无限平面、红色漫反射球、银色镜面球；
2. 迭代式多次光线弹射（throughput + final_color）；
3. 硬阴影与 Shadow Acne 偏移；
4. Light X/Y/Z、Max Bounces UI 滑条。

选做：
1. 玻璃折射（斯涅尔定律、全反射）；
2. MSAA（每像素多条随机主光线采样并平均）。

运行示例：
    python main.py
    python main.py --glass --spp 4 --max-bounces 5

程序仅提供实时交互渲染窗口，不会自动生成 PNG。
建议使用系统录屏录制演示，再自行转换为 GIF 提交。
"""

from __future__ import annotations

import argparse

import numpy as np
import taichi as ti

# 材质 ID
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

EPSILON = 1e-4
INF = 1e8
MAX_BOUNCES = 5
MAX_SPP = 8
GLASS_IOR = 1.50


@ti.data_oriented
class RayTracer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.aspect = float(width) / float(height)

        # 两个球：左球默认漫反射，可在 UI 中切换为玻璃；右球为镜面。
        self.sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.sphere_radius = ti.field(dtype=ti.f32, shape=2)
        self.sphere_material = ti.field(dtype=ti.i32, shape=2)
        self.sphere_color = ti.Vector.field(3, dtype=ti.f32, shape=2)

        self.light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.max_bounces = ti.field(dtype=ti.i32, shape=())
        self.samples_per_pixel = ti.field(dtype=ti.i32, shape=())
        self.glass_enabled = ti.field(dtype=ti.i32, shape=())
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

        self.reset_scene()

    def reset_scene(self) -> None:
        self.sphere_center[0] = (-1.5, 0.0, 0.0)
        self.sphere_radius[0] = 1.0
        self.sphere_material[0] = MAT_DIFFUSE
        self.sphere_color[0] = (0.92, 0.06, 0.08)  # 红色漫反射球

        self.sphere_center[1] = (1.5, 0.0, 0.0)
        self.sphere_radius[1] = 1.0
        self.sphere_material[1] = MAT_MIRROR
        self.sphere_color[1] = (0.82, 0.87, 0.92)  # 银色镜面球

        self.light_pos[None] = (2.0, 4.0, 3.0)
        self.max_bounces[None] = 3
        self.samples_per_pixel[None] = 1
        self.glass_enabled[None] = 0

    @ti.func
    def normalize_safe(self, v):
        return v / ti.sqrt(ti.max(v.dot(v), 1e-12))

    @ti.func
    def reflect(self, incident, normal):
        return incident - 2.0 * incident.dot(normal) * normal

    @ti.func
    def background(self, direction):
        # 简洁的天空色，使镜面球中的反射更容易观察。
        t = 0.5 * (direction[1] + 1.0)
        return (1.0 - t) * ti.Vector([0.025, 0.045, 0.070]) + t * ti.Vector([0.16, 0.34, 0.50])

    @ti.func
    def checker_color(self, point):
        # 按交点 x、z 坐标的整数部分奇偶性生成黑白棋盘格。
        ix = ti.cast(ti.floor(point[0]), ti.i32)
        iz = ti.cast(ti.floor(point[2]), ti.i32)
        parity = (ix + iz) % 2
        color = ti.Vector([0.12, 0.12, 0.12])
        if parity == 0:
            color = ti.Vector([0.88, 0.88, 0.88])
        return color

    @ti.func
    def hit_scene(self, ray_o, ray_d):
        """返回：是否相交、最近 t、交点、外法线、材质 ID、基础颜色。"""
        hit = 0
        closest_t = INF
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        hit_material = -1
        hit_color = ti.Vector([0.0, 0.0, 0.0])

        # 无限地面：y = -1.0，法线朝上 (0, 1, 0)。
        if ti.abs(ray_d[1]) > EPSILON:
            t_plane = (-1.0 - ray_o[1]) / ray_d[1]
            if t_plane > EPSILON and t_plane < closest_t:
                closest_t = t_plane
                hit = 1
                hit_point = ray_o + t_plane * ray_d
                hit_normal = ti.Vector([0.0, 1.0, 0.0])
                hit_material = MAT_DIFFUSE
                hit_color = self.checker_color(hit_point)

        # 两个球的解析相交。
        for i in ti.static(range(2)):
            center = self.sphere_center[i]
            radius = self.sphere_radius[i]
            oc = ray_o - center
            half_b = oc.dot(ray_d)
            c = oc.dot(oc) - radius * radius
            discriminant = half_b * half_b - c
            if discriminant > 0.0:
                root = ti.sqrt(discriminant)
                t_sphere = -half_b - root
                if t_sphere <= EPSILON:
                    t_sphere = -half_b + root
                if t_sphere > EPSILON and t_sphere < closest_t:
                    closest_t = t_sphere
                    hit = 1
                    hit_point = ray_o + t_sphere * ray_d
                    hit_normal = self.normalize_safe(hit_point - center)  # 球面外法线
                    hit_material = self.sphere_material[i]
                    hit_color = self.sphere_color[i]
                    if ti.static(i == 0):
                        # 选做：将左侧红球切换为玻璃球。
                        if self.glass_enabled[None] == 1:
                            hit_material = MAT_GLASS
                            hit_color = ti.Vector([0.92, 0.97, 1.00])

        return hit, closest_t, hit_point, hit_normal, hit_material, hit_color

    @ti.func
    def phong_with_hard_shadow(self, point, normal, view_dir, base_color):
        """Phong 漫反射 + 镜面高光，包含向光源发射暗影射线的硬阴影判断。"""
        ambient = 0.08 * base_color
        to_light = self.light_pos[None] - point
        light_distance = ti.sqrt(to_light.dot(to_light))
        light_dir = to_light / light_distance

        # Shadow Acne 避坑：阴影射线从 P + N * epsilon 出发。
        shadow_o = point + normal * EPSILON
        shadow_hit, shadow_t, _, _, _, _ = self.hit_scene(shadow_o, light_dir)
        in_shadow = (shadow_hit == 1) and (shadow_t < light_distance - EPSILON)

        direct = ti.Vector([0.0, 0.0, 0.0])
        if not in_shadow:
            ndotl = ti.max(normal.dot(light_dir), 0.0)
            diffuse = base_color * ndotl
            half_vec = self.normalize_safe(light_dir + view_dir)
            specular = ti.pow(ti.max(normal.dot(half_vec), 0.0), 48.0)
            direct = diffuse + 0.22 * specular * ti.Vector([1.0, 1.0, 1.0])

        return ambient + direct

    @ti.func
    def trace_ray(self, ray_o, ray_d):
        """GPU 友好的迭代式 Whitted 路径追踪：不使用递归。"""
        throughput = ti.Vector([1.0, 1.0, 1.0])
        final_color = ti.Vector([0.0, 0.0, 0.0])
        active = 1

        # 固定上限为 5；由 UI 的 max_bounces 控制实际弹射次数。
        for bounce in range(MAX_BOUNCES):
            if active == 1 and bounce < self.max_bounces[None]:
                hit, _, point, outward_normal, material, base_color = self.hit_scene(ray_o, ray_d)

                if hit == 0:
                    final_color += throughput * self.background(ray_d)
                    active = 0
                else:
                    # face_normal 始终与入射光线方向相反，用于反射、光照和偏移。
                    front_face = ray_d.dot(outward_normal) < 0.0
                    face_normal = outward_normal
                    if not front_face:
                        face_normal = -outward_normal

                    if material == MAT_DIFFUSE:
                        local_color = self.phong_with_hard_shadow(
                            point, face_normal, -ray_d, base_color
                        )
                        final_color += throughput * local_color
                        active = 0

                    elif material == MAT_MIRROR:
                        # 理想镜面反射：R = L_in - 2(L_in·N)N。
                        ray_d = self.normalize_safe(self.reflect(ray_d, face_normal))
                        # Shadow Acne 避坑：反射光线从 P + N * epsilon 出发。
                        ray_o = point + face_normal * EPSILON
                        throughput *= 0.80

                    else:  # MAT_GLASS，选做：斯涅尔定律 + 全反射。
                        eta_i = 1.0
                        eta_t = GLASS_IOR
                        if not front_face:
                            eta_i = GLASS_IOR
                            eta_t = 1.0

                        eta = eta_i / eta_t
                        cos_theta = ti.min((-ray_d).dot(face_normal), 1.0)
                        sin2_theta_t = eta * eta * (1.0 - cos_theta * cos_theta)

                        if sin2_theta_t > 1.0:
                            # 全反射：仍在当前介质中，因此沿 face_normal 偏移。
                            ray_d = self.normalize_safe(self.reflect(ray_d, face_normal))
                            ray_o = point + face_normal * EPSILON
                        else:
                            # 折射方向：eta * I + (eta*cos(theta_i)-cos(theta_t))*N。
                            cos_theta_t = ti.sqrt(1.0 - sin2_theta_t)
                            ray_d = self.normalize_safe(
                                eta * ray_d + (eta * cos_theta - cos_theta_t) * face_normal
                            )
                            # 折射后射线跨过界面，偏向另一侧以避免再次击中同一表面。
                            ray_o = point - face_normal * EPSILON

                        # 玻璃透射率略小于 1，避免无限弹射亮度失控，并带轻微蓝色调。
                        throughput *= ti.Vector([0.90, 0.95, 0.98])

        # 达到最大弹射次数仍未落在漫反射面时，补入背景色。
        if active == 1:
            final_color += throughput * self.background(ray_d)

        return final_color

    @ti.kernel
    def render(self):
        camera = ti.Vector([0.0, 1.15, 6.0])
        target = ti.Vector([0.0, -0.05, 0.0])
        forward = self.normalize_safe(target - camera)
        right = self.normalize_safe(forward.cross(ti.Vector([0.0, 1.0, 0.0])))
        up = self.normalize_safe(right.cross(forward))
        fov_scale = ti.tan(0.5 * 52.0 / 180.0 * np.pi)
        spp = self.samples_per_pixel[None]

        for i, j in self.pixels:
            color = ti.Vector([0.0, 0.0, 0.0])
            # 使用最多 8 次随机主光线采样；spp=1 时就是基础版本。
            for sample_id in range(MAX_SPP):
                if sample_id < spp:
                    jitter_x = ti.random(ti.f32)
                    jitter_y = ti.random(ti.f32)
                    u = (ti.cast(i, ti.f32) + jitter_x) / ti.cast(self.width, ti.f32)
                    v = (ti.cast(j, ti.f32) + jitter_y) / ti.cast(self.height, ti.f32)
                    screen_x = (2.0 * u - 1.0) * self.aspect * fov_scale
                    # Taichi Canvas 将字段 y=0 显示在窗口底部，因此 j=0 对应相机下方。
                    screen_y = (2.0 * v - 1.0) * fov_scale
                    direction = self.normalize_safe(forward + screen_x * right + screen_y * up)
                    color += self.trace_ray(camera, direction)

            color /= ti.cast(spp, ti.f32)
            # Gamma correction（gamma=2.0）后显示。
            self.pixels[i, j] = ti.Vector([
                ti.sqrt(ti.max(color[0], 0.0)),
                ti.sqrt(ti.max(color[1], 0.0)),
                ti.sqrt(ti.max(color[2], 0.0)),
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taichi iterative Whitted-style ray tracing demo")
    parser.add_argument("--width", type=int, default=800, help="渲染窗口宽度，默认 800")
    parser.add_argument("--height", type=int, default=600, help="渲染窗口高度，默认 600")
    parser.add_argument("--glass", action="store_true", help="启动时将左侧红色漫反射球切换为玻璃球")
    parser.add_argument("--spp", type=int, default=1, help="启动时每像素 MSAA 采样数，范围 1~8")
    parser.add_argument("--max-bounces", type=int, default=3, help="启动时最大弹射次数，范围 1~5")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 后端，适用于无 GPU 环境")
    return parser.parse_args()


def init_taichi(force_cpu: bool) -> None:
    if force_cpu:
        ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=42)
        return

    # macOS 会自动优先尝试 Metal；不支持 GPU 时可用 --cpu 运行。
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=42)
    except Exception:
        ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=42)


def run_interactive(tracer: RayTracer, args: argparse.Namespace) -> None:
    window = ti.ui.Window("Ray Tracing Demo - Work05", (tracer.width, tracer.height), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_x, light_y, light_z = 2.0, 4.0, 3.0
    bounces = max(1, min(MAX_BOUNCES, args.max_bounces))
    glass_enabled = bool(args.glass)
    spp = max(1, min(MAX_SPP, args.spp))

    while window.running:
        gui.begin("Controls", 0.71, 0.03, 0.27, 0.35)
        light_x = gui.slider_float("Light X", light_x, -6.0, 6.0)
        light_y = gui.slider_float("Light Y", light_y, 0.2, 8.0)
        light_z = gui.slider_float("Light Z", light_z, -2.0, 8.0)
        bounces = gui.slider_int("Max Bounces", bounces, 1, 5)
        gui.text("Optional")
        glass_enabled = gui.checkbox("Glass sphere", glass_enabled)
        spp = gui.slider_int("MSAA samples", spp, 1, MAX_SPP)
        gui.text("1 sample = required version")
        gui.text("Glass + 4/8 samples = optional")
        gui.end()

        tracer.light_pos[None] = (light_x, light_y, light_z)
        tracer.max_bounces[None] = bounces
        tracer.glass_enabled[None] = int(glass_enabled)
        tracer.samples_per_pixel[None] = spp

        tracer.render()
        canvas.set_image(tracer.pixels)
        window.show()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width 和 height 必须为正整数")

    init_taichi(args.cpu)
    tracer = RayTracer(args.width, args.height)
    run_interactive(tracer, args)


if __name__ == "__main__":
    main()
