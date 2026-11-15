# -*- coding: utf-8 -*-
"""
多视角 PNG 预览（不依赖 pyvista/vtk）。

做法
----
- 正交投影 + 画家算法（按三角形质心深度排序），用 2D PolyCollection 绘制；
- 每个三角形按 Lambert 光照着色，避免 mplot3d 的排序伪影；
- 每个图 2 行 × 4 列：上排 1×1×1，下排 2×2×2；四列分别是等轴测 / +X / +Y / +Z；
- 切面封口三角形用浅灰色并降低不透明度，便于看到内部通道。

这不是照片级渲染，只用于快速抽查几何形态与连通性。
"""

from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


VIEWS = [
    ("等轴测", np.array([1.0, 1.0, 1.0])),
    ("+X 视", np.array([1.0, 0.0, 0.0])),
    ("+Y 视", np.array([0.0, 1.0, 0.0])),
    ("+Z 视", np.array([0.0, 0.0, 1.0])),
]


def _camera_basis(eye: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (right, up, forward)。forward 指向屏幕内（远离相机）。"""
    fwd = center - eye
    fwd /= max(np.linalg.norm(fwd), 1e-12)
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(fwd, up_hint))) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up_hint)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, fwd)
    up /= max(np.linalg.norm(up), 1e-12)
    return right, up, fwd


def _project(verts: np.ndarray, eye: np.ndarray, center: np.ndarray):
    right, up, fwd = _camera_basis(eye, center)
    rel = verts - center
    x2 = rel @ right
    y2 = rel @ up
    depth = rel @ fwd
    return x2, y2, depth


def _draw_panel(ax, mesh, view_name: str, eye: np.ndarray, cfg, phase: str, title: str):
    verts = mesh.vertices_mm
    faces = mesh.faces
    center = (mesh.bbox_min_mm + mesh.bbox_max_mm) / 2.0

    # 相机距离 = 包围盒对角线，保证完整入镜
    diag = float(np.linalg.norm(mesh.bbox_size_mm))
    eye = center + eye / max(np.linalg.norm(eye), 1e-12) * diag * 1.9

    x2, y2, depth = _project(verts, eye, center)
    tri2 = np.stack([x2[faces], y2[faces]], axis=2)  # (m,3,2)
    tri_depth = depth[faces].mean(axis=1)

    # 法向与光照
    tri3 = verts[faces]
    n = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
    nl = np.linalg.norm(n, axis=1)
    nl[nl < 1e-15] = 1.0
    n = n / nl[:, None]
    light = np.array([-0.35, -0.45, 0.82])
    light = light / np.linalg.norm(light)
    diff = np.clip(n @ light, 0.0, 1.0)
    shade = 0.38 + 0.62 * diff

    if phase == "bulk":
        base = np.array([0.34, 0.55, 0.72])
    else:
        base = np.array([0.90, 0.53, 0.23])
    rgb = np.clip(shade[:, None] * base[None, :], 0.0, 1.0)

    cap = mesh.cap_face_mask
    colors = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)
    if cap.any():
        cap_rgb = np.array([0.86, 0.87, 0.88])
        colors[cap, :3] = 0.55 * cap_rgb[None, :] + 0.45 * rgb[cap]
        colors[cap, 3] = 0.42  # 半透明切面

    order = np.argsort(-tri_depth)  # 远（depth 大）先画
    # 用与面片相同的颜色描边，避免相邻三角形之间出现 1 像素的白色缝隙
    coll = PolyCollection(
        tri2[order],
        facecolors=colors[order],
        edgecolors=colors[order],
        linewidths=0.25,
        antialiaseds=True,
    )
    ax.add_collection(coll)

    pad = 0.12 * diag
    ax.set_xlim(float(np.min(x2)) - pad, float(np.max(x2)) + pad)
    ax.set_ylim(float(np.min(y2)) - pad, float(np.max(y2)) + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    # 坐标轴三色箭头（世界原点出发）
    axis_len = 0.28 * float(mesh.bbox_size_mm.max())
    origin = np.array([[0.0, 0.0, 0.0]])
    ax_o_x, ax_o_y, _ = _project(origin, eye, center)
    for direction, label, color in (
        (np.array([axis_len, 0.0, 0.0]), "X", "#d62728"),
        (np.array([0.0, axis_len, 0.0]), "Y", "#2ca02c"),
        (np.array([0.0, 0.0, axis_len]), "Z", "#1f77b4"),
    ):
        tip = np.array([direction])
        tx, ty, _ = _project(tip, eye, center)
        ax.annotate(
            "", xy=(tx[0], ty[0]), xytext=(ax_o_x[0], ax_o_y[0]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, shrinkA=0, shrinkB=0),
        )
        ax.text(tx[0], ty[0], label, color=color, fontsize=7, fontweight="bold",
                ha="center", va="center")

    ax.set_title(
        f"{view_name} · {mesh.tile}×{mesh.tile}×{mesh.tile} · "
        f"三角面 {mesh.n_faces:,}",
        fontsize=7.5, pad=3,
    )


def render_solution_png(
    spec,
    meshes: Dict[Tuple[str, int], "object"],
    out_path: str,
    cfg,
) -> str:
    """为一个方案输出 bulk 与 sheet 两张 8 面板 PNG。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for phase in ("bulk", "sheet"):
        keys = [(phase, 1), (phase, 2)]
        if not all(k in meshes for k in keys):
            continue

        fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.0), dpi=130)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.03, wspace=0.02, hspace=0.08)

        phase_cn = "bulk 块体相 F≤0" if phase == "bulk" else f"sheet 薄壁相 |F|≤{cfg.sheet_t:g}"
        for row, tile in enumerate((1, 2)):
            mesh = meshes[(phase, tile)]
            for col, (view_name, eye_dir) in enumerate(VIEWS):
                ax = axes[row, col]
                _draw_panel(ax, mesh, view_name, eye_dir, cfg, phase, "")

            # 行首加结构信息
            axes[row, 0].text(
                -0.02, 0.50,
                f"{spec.sid} {spec.cn_name}\n{phase_cn}\n{tile}×{tile}×{tile} 平铺",
                transform=axes[row, 0].transAxes,
                fontsize=9, va="center", ha="right", rotation=90,
                color="#222222",
            )

        fig.suptitle(
            f"{spec.sid} {spec.cn_name} — {phase_cn}\n"
            f"{spec.equation}    |    score {spec.score:.4f}  "
            f"(support {spec.support_proxy:.4f} / elastic {spec.elastic_proxy:.4f})\n"
            f"单胞 {cfg.cell_size_mm:g} mm；几何代理分，非 FEA；切面封口为半透明浅灰",
            fontsize=10.5, y=0.985,
        )
        phase_path = out_path.replace(".png", f"_{phase}.png")
        fig.savefig(phase_path, dpi=130, facecolor="white")
        plt.close(fig)
        print(f"[preview] {phase_path}")

    return out_path
