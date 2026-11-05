# -*- coding: utf-8 -*-
"""
从隐函数 F(x,y,z) 生成**实体**三角网格，并做拓扑/体积体检。

关键点
------
1. F=0 是零厚度曲面；这里用 F<=0（bulk）或 |F|<=t（sheet）作为实体相；
2. 在样品边界加“切面封口”场，使网格在盒壁处闭合（watertight）；
3. marching cubes 的顶点/面共享索引，可直接做流形检查；
4. 用有符号体积判断整体朝向，必要时翻转全部面，保证三角形朝外。

坐标单位：内部用弧度域 [0, 2π*tile]；导出时线性缩放到毫米。
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np

# 允许直接 python optimizer/geometry/xxx.py 运行：把 optimizer/ 加入 sys.path
_OPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage import measure

from geometry_config import GeometryConfig


TWO_PI = 2.0 * math.pi


@dataclass
class MeshResult:
    """一个实体网格 + 体检指标。"""

    phase: str
    tile: int
    grid_per_cell: int
    grid_nodes: int
    h_mm: float
    vertices_mm: np.ndarray
    faces: np.ndarray
    volume_mm3: float
    area_mm2: float
    bbox_min_mm: np.ndarray
    bbox_max_mm: np.ndarray
    bbox_size_mm: np.ndarray
    n_vertices: int
    n_faces: int
    n_edges: int
    boundary_edges: int
    nonmanifold_edges: int
    duplicate_directed_edges: int
    watertight: bool
    edge_manifold: bool
    orientation_consistent: bool
    surface_components: int
    euler_characteristic: int
    cap_face_mask: np.ndarray
    cap_face_count: int
    cap_area_fraction: float
    rho_voxel: float
    solid_components_periodic: int
    largest_periodic_fraction: float
    sheet_t: float
    wall_thickness_mm_est: float
    extra: Dict[str, float] = field(default_factory=dict)

    def summary_row(self, name: str) -> dict:
        """转成 manifest 表格的一行。"""
        return {
            "name": name,
            "phase": self.phase,
            "tile": f"{self.tile}x{self.tile}x{self.tile}",
            "grid_per_cell": self.grid_per_cell,
            "grid_nodes": self.grid_nodes,
            "h_mm": round(self.h_mm, 4),
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "n_edges": self.n_edges,
            "boundary_edges": self.boundary_edges,
            "nonmanifold_edges": self.nonmanifold_edges,
            "duplicate_directed_edges": self.duplicate_directed_edges,
            "watertight": self.watertight,
            "edge_manifold": self.edge_manifold,
            "orientation_consistent": self.orientation_consistent,
            "surface_components": self.surface_components,
            "euler_characteristic": self.euler_characteristic,
            "cap_face_count": self.cap_face_count,
            "cap_area_fraction": round(float(self.cap_area_fraction), 4),
            "volume_mm3": round(float(self.volume_mm3), 3),
            "area_mm2": round(float(self.area_mm2), 2),
            "bbox_size_mm": " x ".join(f"{x:.3f}" for x in self.bbox_size_mm),
            "rho_voxel": round(float(self.rho_voxel), 4),
            "solid_components_periodic": self.solid_components_periodic,
            "largest_periodic_fraction": round(float(self.largest_periodic_fraction), 4),
            "wall_thickness_mm_est": round(float(self.wall_thickness_mm_est), 3),
        }


def _sample_field(func: Callable, cells_per_cell: int, tile: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    在 [0, 2π*tile]^3 的节点网格上采样 F。

    返回 (F 值, 到盒壁最近距离 B_in, 总长度 L, 节点间距 h)。
    节点包含两个端点（0 和 L），因此数组维度是 (N+1)^3。
    """
    n_cells = int(cells_per_cell) * int(tile)
    L = TWO_PI * float(tile)
    h = L / float(n_cells)
    xs = np.linspace(0.0, L, n_cells + 1)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")

    F = func(X, Y, Z)
    # 到六个盒壁的最近距离（内部为正）
    B_in = np.minimum.reduce([X, L - X, Y, L - Y, Z, L - Z])
    return F, B_in, L, h


def build_entity_mesh(
    func: Callable,
    phase: str,
    tile: int,
    grid_per_cell: int,
    cfg: GeometryConfig,
) -> MeshResult:
    """
    生成一个实体网格。

    phase = "bulk"  -> 实体相 F <= 0
    phase = "sheet" -> 实体相 |F| <= t
    """
    phase = phase.lower()
    if phase not in ("bulk", "sheet"):
        raise ValueError(f"未知 phase: {phase}")

    F, B_in, L, h = _sample_field(func, grid_per_cell, tile)

    # 毫米/弧度比例；h_mm 是导出后的体素边长
    mm_per_rad = cfg.cell_size_mm / TWO_PI
    h_mm = h * mm_per_rad
    eps = cfg.cap_eps_voxels * h

    if phase == "bulk":
        solid_scalar = F
        mask = F <= cfg.iso_level
    else:
        solid_scalar = np.abs(F) - float(cfg.sheet_t)
        mask = np.abs(F) <= float(cfg.sheet_t)

    # 实体 = (solid_scalar <= 0) AND (在盒内)。用 eps - B_in 在盒壁处封口：
    # H = max(solid_scalar, eps - B_in)
    H = np.maximum(solid_scalar, eps - B_in)

    # marching cubes：节点坐标 0..L，spacing=h，因此顶点坐标就是物理弧度坐标
    verts, faces, _normals, _values = measure.marching_cubes(
        H, level=float(cfg.iso_level), spacing=(h, h, h),
        allow_degenerate=False, method="lewiner",
    )
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # 缩放到毫米
    verts_mm = verts * mm_per_rad

    # 整体朝向：有符号体积为负则翻转
    signed_vol = _signed_volume(verts_mm, faces)
    if signed_vol < 0:
        faces = faces[:, ::-1].copy()
        signed_vol = -signed_vol
    volume_mm3 = float(abs(signed_vol))

    # 三角面面积 / 法向
    tri = verts_mm[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    area_mm2 = float(0.5 * double_area.sum())

    # 体检
    edges, edge_counts, directed_dup, boundary_edges, nonmanifold_edges = _edge_report(faces, len(verts_mm))
    comps, labels = _mesh_components(faces, len(verts_mm))
    used = np.unique(faces)
    surface_components = int(len(np.unique(labels[used])))

    # 封口面识别：靠近盒壁 + 法向近似坐标轴
    L_mm = L * mm_per_rad
    centroids = tri.mean(axis=1)
    d_wall = np.minimum.reduce([
        centroids[:, 0], L_mm - centroids[:, 0],
        centroids[:, 1], L_mm - centroids[:, 1],
        centroids[:, 2], L_mm - centroids[:, 2],
    ])
    normals = cross / np.maximum(double_area[:, None], 1e-15)
    axis_align = np.max(np.abs(normals), axis=1)
    cos_thr = math.cos(math.radians(cfg.cap_detect_angle_deg))
    cap_mask = (d_wall < cfg.cap_detect_voxels * h_mm) & (axis_align > cos_thr)
    cap_face_count = int(cap_mask.sum())
    cap_area_fraction = float(0.5 * double_area[cap_mask].sum() / area_mm2) if area_mm2 > 0 else 0.0

    # 相对密度：用节点网格的前 N^3 个单元（避免周期端点的重复层）
    rho_voxel = float(mask[:-1, :-1, :-1].mean())
    # 实体相连通分量：用 26 邻域 + 周期边界合并。
    # 薄壁 sheet 在粗网格上厚度只有 1~2 个体素，6 邻域会把同一条斜壁判成多个碎片，
    # 26 邻域更接近“实体是否连通”的物理含义。
    solid_comp, largest = _periodic_component_stats_26(mask)

    bbox_min = verts_mm.min(axis=0)
    bbox_max = verts_mm.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # sheet 壁厚估算：2t / |∇F| * mm_per_rad；用有限差分平均梯度
    wall_est = 0.0
    if phase == "sheet":
        gx, gy, gz = np.gradient(F, h, h, h)
        gmag = np.sqrt(gx * gx + gy * gy + gz * gz)
        sel = np.abs(F) <= cfg.sheet_t
        mean_g = float(gmag[sel].mean()) if np.any(sel) else 1.0
        wall_est = (2.0 * cfg.sheet_t / max(mean_g, 1e-6)) * mm_per_rad

    return MeshResult(
        phase=phase,
        tile=int(tile),
        grid_per_cell=int(grid_per_cell),
        grid_nodes=int(F.shape[0]),
        h_mm=float(h_mm),
        vertices_mm=verts_mm,
        faces=faces,
        volume_mm3=volume_mm3,
        area_mm2=area_mm2,
        bbox_min_mm=bbox_min,
        bbox_max_mm=bbox_max,
        bbox_size_mm=bbox_size,
        n_vertices=int(len(verts_mm)),
        n_faces=int(len(faces)),
        n_edges=int(len(edges)),
        boundary_edges=int(boundary_edges),
        nonmanifold_edges=int(nonmanifold_edges),
        duplicate_directed_edges=int(directed_dup),
        watertight=bool(boundary_edges == 0 and nonmanifold_edges == 0),
        edge_manifold=bool(nonmanifold_edges == 0),
        orientation_consistent=bool(directed_dup == 0),
        surface_components=surface_components,
        euler_characteristic=int(len(used) - len(edges) + len(faces)),
        cap_face_mask=cap_mask,
        cap_face_count=cap_face_count,
        cap_area_fraction=cap_area_fraction,
        rho_voxel=rho_voxel,
        solid_components_periodic=int(solid_comp),
        largest_periodic_fraction=float(largest),
        sheet_t=float(cfg.sheet_t) if phase == "sheet" else 0.0,
        wall_thickness_mm_est=float(wall_est),
    )


def _signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    tri = verts[faces]
    vol = np.einsum("ij,ij->i", np.cross(tri[:, 0], tri[:, 1]), tri[:, 2]).sum() / 6.0
    return float(vol)


def _edge_report(faces: np.ndarray, n_vertices: int):
    """返回 (唯一边, 边计数, 重复有向边数, 边界边数, 非流形边数)。"""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    key = key[:, 0].astype(np.int64) * int(n_vertices) + key[:, 1]
    edges, counts = np.unique(key, return_counts=True)

    de = e[:, 0].astype(np.int64) * int(n_vertices) + e[:, 1]
    _ud, dcounts = np.unique(de, return_counts=True)
    duplicate_directed = int((dcounts != 1).sum())

    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    return edges, counts, duplicate_directed, boundary, nonmanifold


def _mesh_components(faces: np.ndarray, n_vertices: int):
    """按共享顶点/边做并查集，统计表面 shell 数（不等于实体连通分量）。"""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    g = coo_matrix(
        (np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1]),
        ),
        shape=(int(n_vertices), int(n_vertices)),
    )
    ncomp, labels = connected_components(g, directed=False)
    return int(ncomp), labels


def _periodic_component_stats_26(mask: np.ndarray):
    """
    26 邻域连通分量 + 周期边界合并。

    为什么不用 evaluator 里的 6 邻域版本：
    sheet 薄壁在粗网格上只有 1~2 个体素厚，6 邻域会把一条连续的斜壁切成很多碎片；
    26 邻域更接近实体连通性。周期合并处理 x/y/z 首尾面。
    """
    lab, n = measure.label(mask, connectivity=3, return_num=True)
    if n == 0:
        return 0, 0.0

    parent = list(range(int(n) + 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for axis in range(3):
        for side_a, side_b in ((0, -1),):
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[axis] = side_a
            sl1[axis] = side_b
            a = lab[tuple(sl0)].ravel()
            b = lab[tuple(sl1)].ravel()
            for x, y in zip(a, b):
                if x and y:
                    union(int(x), int(y))

    roots = np.array([find(i) for i in range(1, int(n) + 1)], dtype=np.int64)
    # 每个 label 的体素数；再按 union-find 的根累加，得到每个连通分量的体素数
    sizes = np.bincount(np.asarray(lab).ravel(), minlength=int(n) + 1).astype(np.float64)
    merged = np.bincount(roots, weights=sizes[1:], minlength=int(n) + 1)
    # 注意：根编号可能有空洞，必须统计非空 bin 的数量，不能直接用 len(merged)。
    n_components = int((merged > 0).sum())
    largest = float(merged.max()) / max(float(mask.sum()), 1.0)
    return n_components, largest
