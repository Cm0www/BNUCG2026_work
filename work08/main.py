#!/usr/bin/env python3
"""SMPL Linear Blend Skinning 可视化实验（必做 + 选做）。

运行：
    python main.py

模型文件默认位置：assets/models/SMPL_NEUTRAL.pkl
也可指定：
    python main.py --model-path /绝对路径/SMPL_NEUTRAL.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import imageio.v2 as imageio
import matplotlib

# 服务器或无显示器环境也可保存图片。
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import smplx
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少 smplx。请先执行：python -m pip install -r requirements.txt"
    ) from exc

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_MODEL_PATH = ROOT / "assets" / "models" / "SMPL_NEUTRAL.pkl"

# SMPL 的 24 个运动学关节名称；第 18 号为左肘，适合显示单关节权重热力图。
JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]


def to_numpy(value: torch.Tensor) -> np.ndarray:
    """将 Tensor 安全地转换到 CPU NumPy。"""
    return value.detach().cpu().numpy()


def batch_rodrigues(rot_vecs: torch.Tensor) -> torch.Tensor:
    """轴角向量批量转旋转矩阵。

    与 smplx.lbs.batch_rodrigues 的计算顺序保持一致，便于后续验证。
    """
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    skew = torch.cat(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1
    ).view(batch_size, 3, 3)
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    return ident + sin * skew + (1.0 - cos) * torch.bmm(skew, skew)


def transform_mat(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """由 R 和 t 拼接齐次 4×4 变换矩阵。"""
    top = torch.cat(
        [rotation, torch.zeros((rotation.shape[0], 1, 3), dtype=rotation.dtype, device=rotation.device)],
        dim=1,
    )
    right = torch.cat(
        [translation, torch.ones((translation.shape[0], 1, 1), dtype=translation.dtype, device=translation.device)],
        dim=1,
    )
    return torch.cat([top, right], dim=2)


def batch_rigid_transform(
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """沿 SMPL 运动学树计算关节全局位置和去除静止位姿偏移后的 A 矩阵。"""
    joints_col = joints.unsqueeze(-1)                         # B × K × 3 × 1
    rel_joints = joints_col.clone()
    rel_joints[:, 1:] -= joints_col[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1),
    ).reshape(-1, joints_col.shape[1], 4, 4)

    chain = [transforms_mat[:, 0]]
    for joint_idx in range(1, parents.shape[0]):
        parent_idx = int(parents[joint_idx].item())
        chain.append(torch.matmul(chain[parent_idx], transforms_mat[:, joint_idx]))
    transforms = torch.stack(chain, dim=1)

    joints_transformed = transforms[:, :, :3, 3]
    # 与官方 smplx 实现一致：此处使用 [J, 0]^T，仅抵消静止位姿的旋转部分。
    joints_homogeneous = torch.cat(
        [joints_col, torch.zeros_like(joints_col[:, :, :1])], dim=2
    )
    joint_offset = torch.matmul(transforms, joints_homogeneous)
    rel_transforms = transforms.clone()
    rel_transforms[:, :, :, 3:4] -= joint_offset
    return joints_transformed, rel_transforms


def hand_written_lbs(
    betas: torch.Tensor,
    full_pose: torch.Tensor,
    v_template: torch.Tensor,
    shapedirs: torch.Tensor,
    posedirs: torch.Tensor,
    j_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """明确拆出 SMPL LBS 的五个核心中间量。

    返回：v_template、v_shaped、J、pose_offsets、v_posed、verts、J_transformed 等。
    """
    batch_size = max(betas.shape[0], full_pose.shape[0])
    dtype, device = betas.dtype, betas.device

    # (b) Shape blend shapes: v_shaped = v_template + B_S(beta)
    shape_offsets = torch.einsum("bl,mkl->bmk", betas, shapedirs)
    v_shaped = v_template.unsqueeze(0) + shape_offsets

    # 关节位置由形状校正后的网格回归获得。
    joints = torch.einsum("bik,ji->bjk", v_shaped, j_regressor)

    # (c) Pose blend shapes: v_posed = v_shaped + B_P(theta)
    rot_mats = batch_rodrigues(full_pose.reshape(-1, 3)).view(batch_size, -1, 3, 3)
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).reshape(batch_size, -1)
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    v_posed = v_shaped + pose_offsets

    # (d) Forward kinematics + weighted rigid transforms + LBS。
    joints_transformed, a_mats = batch_rigid_transform(rot_mats, joints, parents)
    weights = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
    transforms = torch.matmul(weights, a_mats.view(batch_size, a_mats.shape[1], 16))
    transforms = transforms.view(batch_size, -1, 4, 4)

    homogeneous = torch.ones((batch_size, v_posed.shape[1], 1), dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogeneous], dim=2)
    verts_homo = torch.matmul(transforms, v_posed_homo.unsqueeze(-1))
    verts = verts_homo[:, :, :3, 0]

    return {
        "v_template": v_template.unsqueeze(0),
        "shape_offsets": shape_offsets,
        "v_shaped": v_shaped,
        "J": joints,
        "rot_mats": rot_mats,
        "pose_feature": pose_feature,
        "pose_offsets": pose_offsets,
        "v_posed": v_posed,
        "J_transformed": joints_transformed,
        "A": a_mats,
        "verts": verts,
    }


def display_coordinates(points: np.ndarray) -> np.ndarray:
    """将 SMPL 的 y 轴（人体高度）映射为绘图 z 轴，得到正立人体。"""
    return points[:, [0, 2, 1]]


def set_equal_3d_axes(ax: plt.Axes, vertices: np.ndarray) -> None:
    """设置等比例三维坐标，避免人体被压扁。"""
    plotted = display_coordinates(vertices)
    minimum = plotted.min(axis=0)
    maximum = plotted.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float((maximum - minimum).max()) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=8, azim=-90)
    ax.set_axis_off()
    ax.set_proj_type("ortho")


def add_mesh(
    ax: plt.Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    scalar: np.ndarray | None = None,
    vertex_rgb: np.ndarray | None = None,
    base_color: str = "#c7a177",
    alpha: float = 1.0,
) -> None:
    """向 3D 坐标轴添加三角网格，并支持按顶点标量或颜色着色。"""
    plotted = display_coordinates(vertices)
    triangles = plotted[faces]

    if vertex_rgb is not None:
        face_colors = np.clip(vertex_rgb[faces].mean(axis=1), 0.0, 1.0)
    elif scalar is not None:
        scalar = np.asarray(scalar)
        vmax = max(float(np.quantile(scalar, 0.995)), 1e-8)
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        face_values = scalar[faces].mean(axis=1)
        face_colors = plt.get_cmap("viridis")(norm(face_values))
    else:
        face_colors = base_color

    collection = Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors="none",
        linewidths=0.0,
        alpha=alpha,
    )
    ax.add_collection3d(collection)
    set_equal_3d_axes(ax, vertices)


def add_skeleton(
    ax: plt.Axes,
    joints: np.ndarray,
    parents: np.ndarray,
) -> None:
    """绘制 24 关节骨架，便于观察关节位置和姿态。"""
    points = display_coordinates(joints)
    for idx, parent in enumerate(parents):
        if parent < 0:
            continue
        ax.plot(
            [points[idx, 0], points[parent, 0]],
            [points[idx, 1], points[parent, 1]],
            [points[idx, 2], points[parent, 2]],
            color="#b3202e",
            linewidth=1.1,
            alpha=0.9,
        )
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8, c="#b3202e", depthshade=False)


def save_single_view(
    output_path: Path,
    title: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    scalar: np.ndarray | None = None,
    vertex_rgb: np.ndarray | None = None,
    joints: np.ndarray | None = None,
    parents: np.ndarray | None = None,
    base_color: str = "#c7a177",
) -> None:
    fig = plt.figure(figsize=(7, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    add_mesh(
        ax,
        vertices,
        faces,
        scalar=scalar,
        vertex_rgb=vertex_rgb,
        base_color=base_color,
    )
    if joints is not None and parents is not None:
        add_skeleton(ax, joints, parents)
    ax.set_title(title, fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_all_joint_colors(weights: np.ndarray) -> np.ndarray:
    """按主导关节给颜色、按最大权重给亮度，生成全关节辅助图。"""
    dominant = np.argmax(weights, axis=1)
    strength = np.max(weights, axis=1)
    hsv = np.zeros((weights.shape[0], 3), dtype=np.float32)
    hsv[:, 0] = dominant / float(weights.shape[1])
    hsv[:, 1] = 0.85
    hsv[:, 2] = 0.30 + 0.70 * np.clip(strength, 0.0, 1.0)
    return mcolors.hsv_to_rgb(hsv)


def make_demo_parameters(device: torch.device, dtype: torch.dtype, num_betas: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建非零形状和非零姿态，用于四阶段对比。"""
    betas = torch.zeros((1, num_betas), dtype=dtype, device=device)
    demo_betas = [1.20, -0.75, 0.45, 0.25, -0.20]
    betas[0, : min(num_betas, len(demo_betas))] = torch.tensor(
        demo_betas[:num_betas], dtype=dtype, device=device
    )

    full_pose = torch.zeros((1, 24, 3), dtype=dtype, device=device)
    # 左肩抬起、左肘弯曲，并使另一侧手臂轻微变化，便于观察 LBS 与 pose blend shapes。
    full_pose[0, 16, 2] = -1.05
    full_pose[0, 18, 2] = -0.85
    full_pose[0, 17, 2] = 0.35
    full_pose[0, 19, 2] = 0.20
    full_pose[0, 9, 1] = 0.12
    return betas, full_pose


def write_summary(
    output_path: Path,
    body_model,
    weights: torch.Tensor,
    manual: Dict[str, torch.Tensor],
    official_vertices: torch.Tensor,
    official_joints: torch.Tensor,
) -> Tuple[float, float]:
    """写入模型信息以及手写 LBS 与官方前向的一致性验证结果。"""
    vertex_error = torch.abs(manual["verts"] - official_vertices)
    joint_error = torch.abs(manual["J_transformed"] - official_joints)
    vertex_mae = float(vertex_error.mean().item())
    vertex_max = float(vertex_error.max().item())
    joint_mae = float(joint_error.mean().item())
    joint_max = float(joint_error.max().item())

    lines = [
        "SMPL Linear Blend Skinning Experiment Summary",
        "=" * 58,
        f"Vertex count: {body_model.v_template.shape[0]}",
        f"Face count: {body_model.faces.shape[0]}",
        f"Kinematic joint count: {body_model.J_regressor.shape[0]}",
        f"Betas dimension: {body_model.num_betas}",
        f"LBS weights shape: {tuple(weights.shape)}",
        f"Shape dirs shape: {tuple(body_model.shapedirs.shape)}",
        f"Pose dirs shape: {tuple(body_model.posedirs.shape)}",
        "",
        "Manual LBS vs official SMPL forward (same betas/global_orient/body_pose):",
        f"Vertex mean absolute error: {vertex_mae:.10e}",
        f"Vertex max absolute error : {vertex_max:.10e}",
        f"Joint mean absolute error : {joint_mae:.10e}",
        f"Joint max absolute error  : {joint_max:.10e}",
        "",
        "Conclusion:",
        "The hand-written pipeline follows shape blend shapes, joint regression,",
        "pose blend shapes, forward kinematics, and weighted LBS in sequence.",
        "Small floating-point error is expected; a max error below 1e-5 indicates consistency.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return vertex_mae, vertex_max


def save_comparison_grid(
    output_path: Path,
    faces: np.ndarray,
    parents: np.ndarray,
    template: np.ndarray,
    weights: np.ndarray,
    v_shaped: np.ndarray,
    joints: np.ndarray,
    v_posed: np.ndarray,
    pose_offsets: np.ndarray,
    verts: np.ndarray,
    joints_transformed: np.ndarray,
) -> None:
    """输出四个 LBS 阶段的 2×2 对比图。"""
    fig = plt.figure(figsize=(13, 12), dpi=150)

    ax = fig.add_subplot(221, projection="3d")
    add_mesh(ax, template, faces, scalar=weights[:, 18])
    ax.set_title("(a) Template + LBS Weights (left elbow)", fontsize=11, pad=8)

    ax = fig.add_subplot(222, projection="3d")
    add_mesh(ax, v_shaped, faces)
    add_skeleton(ax, joints, parents)
    ax.set_title("(b) Shape Blend + Joint Regression", fontsize=11, pad=8)

    ax = fig.add_subplot(223, projection="3d")
    add_mesh(ax, v_posed, faces, scalar=np.linalg.norm(pose_offsets, axis=1))
    ax.set_title("(c) Pose Blend Shapes", fontsize=11, pad=8)

    ax = fig.add_subplot(224, projection="3d")
    add_mesh(ax, verts, faces)
    add_skeleton(ax, joints_transformed, parents)
    ax.set_title("(d) Final Linear Blend Skinning Result", fontsize=11, pad=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_optional_animation(
    output_path: Path,
    betas: torch.Tensor,
    full_pose: torch.Tensor,
    model,
    faces: np.ndarray,
    parents: np.ndarray,
    frames: int,
) -> None:
    """选做：固定形状，让左肩从 0 平滑转到目标角度，并导出 GIF。"""
    print(f"[optional] 正在生成 GIF 动画，共 {frames} 帧……")
    images = []
    target_angle = float(full_pose[0, 16, 2].item())

    for frame_idx, angle in enumerate(np.linspace(0.0, target_angle, frames), start=1):
        animated_pose = full_pose.clone()
        animated_pose[0, 16, 2] = float(angle)
        result = hand_written_lbs(
            betas,
            animated_pose,
            model.v_template,
            model.shapedirs,
            model.posedirs,
            model.J_regressor,
            model.parents,
            model.lbs_weights,
        )

        verts = to_numpy(result["verts"][0])
        joints = to_numpy(result["J_transformed"][0])
        fig = plt.figure(figsize=(5, 6), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        add_mesh(ax, verts, faces)
        add_skeleton(ax, joints, parents)
        ax.set_title(f"Optional animation: left shoulder = {np.degrees(angle):.1f}°", fontsize=10, pad=8)
        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        images.append(image)
        plt.close(fig)
        print(f"[optional] 已完成 {frame_idx}/{frames} 帧")

    imageio.mimsave(output_path, images, duration=0.10, loop=0)


def load_smpl_model(model_path: Path, device: torch.device):
    """使用 smplx.create 加载官方 SMPL 中性模型。"""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"未找到模型文件：{model_path}\n"
            "请将 SMPL_NEUTRAL.pkl 放入 work08/assets/models/，"
            "或通过 --model-path 指定其本地路径。"
        )

    try:
        model = smplx.create(
            str(model_path),
            model_type="smpl",
            gender="neutral",
            num_betas=10,
            batch_size=1,
            create_transl=False,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "chumpy":
            raise RuntimeError(
                "该 pkl 含有旧版 Chumpy 对象，当前 smplx 无法直接读取。"
                "请使用教师提供的已转换 SMPL_NEUTRAL.pkl，或先按 smplx 官方 tools/README 转换。"
            ) from exc
        raise
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="SMPL LBS 全流程可视化实验")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="SMPL_NEUTRAL.pkl 路径")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"], help="计算设备，Mac 默认使用 cpu")
    parser.add_argument("--frames", type=int, default=20, help="选做 GIF 的帧数")
    parser.add_argument("--skip-animation", action="store_true", help="只运行必做内容，不生成选做 GIF")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA。请改用 --device cpu。")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前环境没有可用 MPS。请改用 --device cpu。")
    if args.frames < 2:
        raise ValueError("--frames 至少应为 2。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"使用设备：{device}")
    print(f"加载模型：{args.model_path}")
    model = load_smpl_model(args.model_path.expanduser().resolve(), device)
    dtype = model.v_template.dtype

    faces = np.asarray(model.faces, dtype=np.int64)
    parents = to_numpy(model.parents).astype(np.int64)
    template = to_numpy(model.v_template)
    weights = to_numpy(model.lbs_weights)

    print("模型基础信息：")
    print(f"  顶点数：{template.shape[0]}")
    print(f"  面片数：{faces.shape[0]}")
    print(f"  关节数：{model.J_regressor.shape[0]}")
    print(f"  betas 维度：{model.num_betas}")
    print(f"  lbs_weights 形状：{weights.shape}")

    betas, full_pose = make_demo_parameters(device, dtype, model.num_betas)
    manual = hand_written_lbs(
        betas,
        full_pose,
        model.v_template,
        model.shapedirs,
        model.posedirs,
        model.J_regressor,
        model.parents,
        model.lbs_weights,
    )

    v_shaped = to_numpy(manual["v_shaped"][0])
    joints = to_numpy(manual["J"][0])
    pose_offsets = to_numpy(manual["pose_offsets"][0])
    v_posed = to_numpy(manual["v_posed"][0])
    verts = to_numpy(manual["verts"][0])
    joints_transformed = to_numpy(manual["J_transformed"][0])

    print("生成必做图片……")
    selected_joint = 18
    save_single_view(
        OUTPUT_DIR / "stage_a_template_weights.png",
        f"Stage (a): Template + LBS weights ({JOINT_NAMES[selected_joint]})",
        template,
        faces,
        scalar=weights[:, selected_joint],
    )
    save_single_view(
        OUTPUT_DIR / "all_joint_weights.png",
        "All Joint LBS Weights (hue=dominant joint, brightness=max weight)",
        template,
        faces,
        vertex_rgb=make_all_joint_colors(weights),
    )
    save_single_view(
        OUTPUT_DIR / "stage_b_shaped_joints.png",
        "Stage (b): Shape Blend Shapes + Joint Regression",
        v_shaped,
        faces,
        joints=joints,
        parents=parents,
    )
    save_single_view(
        OUTPUT_DIR / "stage_c_pose_offsets.png",
        "Stage (c): Pose Blend Shapes (color = ||pose offsets||)",
        v_posed,
        faces,
        scalar=np.linalg.norm(pose_offsets, axis=1),
    )
    save_single_view(
        OUTPUT_DIR / "stage_d_lbs_result.png",
        "Stage (d): Final Linear Blend Skinning Result",
        verts,
        faces,
        joints=joints_transformed,
        parents=parents,
    )
    save_comparison_grid(
        OUTPUT_DIR / "comparison_grid.png",
        faces,
        parents,
        template,
        weights,
        v_shaped,
        joints,
        v_posed,
        pose_offsets,
        verts,
        joints_transformed,
    )

    # 与官方 smplx 前向过程对比：全局朝向为第 0 关节，其余为 body_pose。
    official_output = model(
        betas=betas,
        global_orient=full_pose[:, 0],
        body_pose=full_pose[:, 1:].reshape(1, -1),
        return_verts=True,
    )
    official_vertices = official_output.vertices
    official_joints = official_output.joints[:, : model.J_regressor.shape[0]]
    vertex_mae, vertex_max = write_summary(
        OUTPUT_DIR / "summary.txt",
        model,
        model.lbs_weights,
        manual,
        official_vertices,
        official_joints,
    )
    print(f"手写 LBS 与官方前向：MAE={vertex_mae:.3e}, MaxAE={vertex_max:.3e}")

    if not args.skip_animation:
        create_optional_animation(
            OUTPUT_DIR / "optional_pose_animation.gif",
            betas,
            full_pose,
            model,
            faces,
            parents,
            args.frames,
        )

    print("\n实验完成。输出目录：")
    for item in sorted(OUTPUT_DIR.iterdir()):
        if item.is_file():
            print(f"  outputs/{item.name}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
