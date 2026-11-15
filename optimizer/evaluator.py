# -*- coding: utf-8 -*-
"""
几何代理评价器（Proxy Evaluator）。

重要声明
--------
本环境没有 FEA / 材料本构 / 边界条件求解器，因此这里计算的是**几何代理指标**，
不是真实力学性能。分数只用于排序和搜索，不能当作刚度、强度或回弹模量的预测值。

计算内容
--------
1. 在单胞 [0, 2π)^3 上采样隐函数 F(x,y,z)；
2. 用 marching cubes 提取 F=0 等值面，得到面积/网格；
3. 实体相定义为 F <= 0（skeletal 约定；sheet 实体化留作后续扩展）；
4. 相对密度 rho* = 实体体素 / 总体素；
5. 周期性连通性：实体相、空相各自的最大连通分量占比；
6. 曲率：由 n = ∇F/|∇F| 计算平均曲率 H = 0.5 * div(n)，在等值面窄带内统计；
7. 把上述几何量组合成 support / elastic 两个代理子分数，再按目标权重合成 score。

这些指标与真实力学量的关系（后续 FEA 需要验证）
------------------------------------------------
- 相对密度与 Gibson–Ashby 标度律有关：拉伸主导 E*/Es ~ ρ*，弯曲主导 ~ ρ*^2；
- 低曲率/平滑曲面近似拉伸主导 → 支撑代理较高；
- 双连续、曲率均匀、无细薄韧带 → 可恢复变形能力/回弹代理较高；
- 上述都是定性代理，未经压缩试验或 FEA 标定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
from scipy import ndimage
from skimage import measure

try:  # 包导入（python -m optimizer.main）
    from .token_library import build_function
except ImportError:  # 脚本导入（python optimizer/main.py、geometry 子脚本）
    from token_library import build_function


@dataclass
class EvalMetrics:
    valid: bool
    reason: str
    rho_solid: float
    rho_void: float
    solid_components: int
    void_components: int
    solid_largest_fraction: float
    void_largest_fraction: float
    connectivity: float
    isotropy: float
    curvature_mean: float
    curvature_p95: float
    curvature_std: float
    smoothness: float
    uniformity: float
    surface_area: float
    area_per_volume: float
    n_surface_voxels: int
    f_min: float
    f_max: float
    f_std: float
    support_proxy: float = 0.0
    elastic_proxy: float = 0.0
    score: float = 0.0
    score_detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _union_find_roots(labels: np.ndarray) -> np.ndarray:
    """对 3D 标签做并查集，把跨周期边界接触的分量合并。"""
    n = int(labels.max())
    if n == 0:
        return labels

    parent = list(range(n + 1))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 三个方向的面：index 0 与 index -1 视为同一周期面
    for axis in range(3):
        sl_first = [slice(None)] * 3
        sl_last = [slice(None)] * 3
        sl_first[axis] = 0
        sl_last[axis] = -1
        first = labels[tuple(sl_first)].ravel()
        last = labels[tuple(sl_last)].ravel()
        for a, b in zip(first, last):
            if a and b:
                union(int(a), int(b))

    roots = np.array([find(i) for i in range(n + 1)], dtype=np.int32)
    # 把标签映射到根，但保留 0 为背景
    mapped = roots[labels]
    return mapped


def _component_stats(mask: np.ndarray) -> tuple:
    """返回 (组分数, 最大分量占该相的比例)；考虑周期边界。"""
    if mask.sum() == 0:
        return 0, 0.0
    labels, _ = measure.label(mask, connectivity=1, return_num=True)
    mapped = _union_find_roots(labels)
    sizes = np.bincount(mapped.ravel())
    sizes[0] = 0
    n_components = int(np.count_nonzero(sizes))
    largest = float(sizes.max() / max(1, mask.sum()))
    return n_components, largest


def _periodic_gradient(field: np.ndarray, h: float):
    """周期边界中心差分梯度。"""
    fp = np.pad(field, 1, mode="wrap")
    gx = (fp[2:, 1:-1, 1:-1] - fp[:-2, 1:-1, 1:-1]) / (2.0 * h)
    gy = (fp[1:-1, 2:, 1:-1] - fp[1:-1, :-2, 1:-1]) / (2.0 * h)
    gz = (fp[1:-1, 1:-1, 2:] - fp[1:-1, 1:-1, :-2]) / (2.0 * h)
    return gx, gy, gz


def _periodic_divergence(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray, h: float):
    """周期边界散度。"""
    nxp = np.pad(nx, 1, mode="wrap")
    nyp = np.pad(ny, 1, mode="wrap")
    nzp = np.pad(nz, 1, mode="wrap")
    dx = (nxp[2:, 1:-1, 1:-1] - nxp[:-2, 1:-1, 1:-1]) / (2.0 * h)
    dy = (nyp[1:-1, 2:, 1:-1] - nyp[1:-1, :-2, 1:-1]) / (2.0 * h)
    dz = (nzp[1:-1, 1:-1, 2:] - nzp[1:-1, 1:-1, :-2]) / (2.0 * h)
    return dx + dy + dz


class Evaluator:
    def __init__(self, config):
        self.cfg = config
        n = int(config.grid_n)
        self.n = n
        self.h = (config.domain_max - config.domain_min) / n
        # endpoint=False：避免 0 与 2π 的重复采样，保证周期性
        self.axis = np.linspace(config.domain_min, config.domain_max, n, endpoint=False)
        self._cache: dict = {}

    def _grid(self):
        x, y, z = self.axis, self.axis, self.axis
        return np.meshgrid(x, y, z, indexing="ij")

    def evaluate(self, terms: List[str], coeffs: List[float], signature: str = None,
                 profile: str = "high_support_high_elasticity") -> EvalMetrics:
        """
        计算一个候选的几何指标与代理分数。
        metrics 与 profile 无关的部分会按 signature 缓存；score 每次按目标权重重算。
        """
        key = signature or ""
        if key and key in self._cache:
            base = self._cache[key]
        else:
            base = self._compute_metrics(terms, coeffs)
            if key:
                self._cache[key] = base
        return self._apply_score(base, profile)

    # ------------------------------------------------------------------ 几何指标
    def _invalid(self, reason: str, **extra) -> dict:
        base = dict(
            valid=False, reason=reason,
            rho_solid=0.0, rho_void=0.0,
            solid_components=0, void_components=0,
            solid_largest_fraction=0.0, void_largest_fraction=0.0, connectivity=0.0,
            isotropy=0.0,
            curvature_mean=0.0, curvature_p95=0.0, curvature_std=0.0,
            smoothness=0.0, uniformity=0.0,
            surface_area=0.0, area_per_volume=0.0, n_surface_voxels=0,
            f_min=0.0, f_max=0.0, f_std=0.0,
        )
        base.update(extra)
        return base

    def _compute_metrics(self, terms: List[str], coeffs: List[float]) -> dict:
        if not terms or len(terms) != len(coeffs):
            return self._invalid("terms 为空或与 coeffs 长度不一致")
        try:
            X, Y, Z = self._grid()
            F = build_function(terms, coeffs)(X, Y, Z)
        except Exception as exc:
            return self._invalid(f"函数构造失败：{exc}")
        if not np.all(np.isfinite(F)):
            return self._invalid("函数值包含 NaN/Inf")

        f_min, f_max = float(F.min()), float(F.max())
        f_std = float(F.std())
        if f_std < 1e-9:
            return self._invalid("函数近似常数，无法定义曲面", f_min=f_min, f_max=f_max, f_std=f_std)

        solid = F <= float(self.cfg.iso_level)
        rho_solid = float(solid.mean())
        rho_void = 1.0 - rho_solid
        if rho_solid < self.cfg.min_phase_fraction or rho_void < self.cfg.min_phase_fraction:
            return self._invalid(
                f"实体相或空相占比过小（rho_solid={rho_solid:.4f}）",
                rho_solid=rho_solid, rho_void=rho_void,
                f_min=f_min, f_max=f_max, f_std=f_std,
            )

        # ---- marching cubes：F=0 等值面 ----
        try:
            verts, faces, _, _ = measure.marching_cubes(
                F, level=float(self.cfg.iso_level),
                spacing=(self.h, self.h, self.h),
                allow_degenerate=False,
            )
        except Exception as exc:
            return self._invalid(f"marching cubes 失败：{exc}",
                                 rho_solid=rho_solid, rho_void=rho_void,
                                 f_min=f_min, f_max=f_max, f_std=f_std)
        if len(faces) == 0:
            return self._invalid("等值面没有三角面片",
                                 rho_solid=rho_solid, rho_void=rho_void,
                                 f_min=f_min, f_max=f_max, f_std=f_std)

        tri = verts[faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        surface_area = float(0.5 * np.linalg.norm(cross, axis=1).sum())
        volume = (self.cfg.domain_max - self.cfg.domain_min) ** 3
        area_per_volume = surface_area / volume

        # ---- 连通性（周期边界） ----
        solid_components, solid_largest = _component_stats(solid)
        void_components, void_largest = _component_stats(~solid)
        connectivity = float(min(solid_largest, void_largest))

        # ---- 曲率：H = 0.5 * div(n)，n = ∇F/|∇F| ----
        gx, gy, gz = _periodic_gradient(F, self.h)
        gmag = np.sqrt(gx * gx + gy * gy + gz * gz)
        eps = max(1e-12, float(gmag.max()) * 1e-9)
        safe = gmag > eps
        nx = np.zeros_like(F)
        ny = np.zeros_like(F)
        nz = np.zeros_like(F)
        nx[safe] = gx[safe] / gmag[safe]
        ny[safe] = gy[safe] / gmag[safe]
        nz[safe] = gz[safe] / gmag[safe]
        div_n = _periodic_divergence(nx, ny, nz, self.h)
        H = 0.5 * div_n

        # 只在与等值面距离约 1.5 个体素的窄带内统计，避免远处噪声
        band = np.abs(F) <= self.cfg.band_voxels * gmag * self.h
        band &= safe
        if band.sum() < 200:
            # 兜底：用 |F| 最小的 5% 体素
            thresh = np.percentile(np.abs(F), 5)
            band = np.abs(F) <= thresh
        H_band = np.abs(H[band])
        # 砍掉极端数值尖峰（离散化伪影），但保留 p99 作为尾部指标
        h_cap = float(np.percentile(H_band, 99))
        H_clip = np.clip(H_band, 0.0, max(h_cap, 1e-6))
        curvature_mean = float(H_clip.mean())
        curvature_p95 = float(np.percentile(H_clip, 95))
        curvature_std = float(H_clip.std())

        k = float(self.cfg.curvature_k)
        smooth_mean = math.exp(-k * curvature_mean)
        smooth_tail = math.exp(-k * curvature_p95)
        smoothness = float(0.6 * smooth_mean + 0.4 * smooth_tail)
        uniformity = float(math.exp(-k * curvature_std))

        # ---- 法向各向异性/三维度：防止 1D 层状或 2D 柱状退化结构刷分 ----
        # 用等值面窄带内单位法向 n 的协方差特征值衡量方向分散程度；
        # 各向同性曲面三个特征值接近，1D 层状结构只有一个方向 → isotropy≈0。
        w = band.astype(np.float64)
        wsum = float(w.sum())
        if wsum > 0:
            cov = np.array([
                [float((w * nx * nx).sum()), float((w * nx * ny).sum()), float((w * nx * nz).sum())],
                [float((w * ny * nx).sum()), float((w * ny * ny).sum()), float((w * ny * nz).sum())],
                [float((w * nz * nx).sum()), float((w * nz * ny).sum()), float((w * nz * nz).sum())],
            ], dtype=np.float64) / wsum
            eig = np.linalg.eigvalsh(cov)
            eig = np.clip(eig, 0.0, None)
            isotropy = float(eig[0] / eig[-1]) if eig[-1] > 1e-12 else 0.0
        else:
            isotropy = 0.0

        return dict(
            valid=True, reason="ok",
            rho_solid=rho_solid, rho_void=rho_void,
            solid_components=int(solid_components), void_components=int(void_components),
            solid_largest_fraction=float(solid_largest),
            void_largest_fraction=float(void_largest),
            connectivity=connectivity, isotropy=isotropy,
            curvature_mean=curvature_mean, curvature_p95=curvature_p95,
            curvature_std=curvature_std,
            smoothness=smoothness, uniformity=uniformity,
            surface_area=surface_area, area_per_volume=area_per_volume,
            n_surface_voxels=int(band.sum()),
            f_min=f_min, f_max=f_max, f_std=f_std,
        )

    # ------------------------------------------------------------------ 打分
    def _apply_score(self, base: dict, profile: str) -> EvalMetrics:
        cfg = self.cfg
        if not base.get("valid", False):
            return EvalMetrics(**{**base, "support_proxy": 0.0, "elastic_proxy": 0.0,
                                  "score": 0.0, "score_detail": "invalid"})

        rho = base["rho_solid"]
        # 支撑：密度越高越能承载，但用 density_ref 归一化，避免无上限地奖励“越密越好”
        density_norm = float(np.clip((rho - 0.10) / max(1e-6, cfg.density_ref - 0.10), 0.0, 1.0))
        # 弹性：偏好适中的相对密度窗口（过密无孔隙、过疏薄壁屈曲）
        density_window = float(math.exp(-0.5 * ((rho - cfg.density_target) / 0.15) ** 2))
        # 比表面积窗口：过低像退化层状/平板，过高像细碎高频网络，TPMS 通常在 0.4~0.7
        area_window = float(math.exp(-0.5 * ((base["area_per_volume"] - cfg.area_target) / cfg.area_sigma) ** 2))

        sw = cfg.support_subweights
        support = (
            sw["density"] * density_norm
            + sw["connectivity"] * base["connectivity"]
            + sw["smoothness"] * base["smoothness"]
            + sw["isotropy"] * base["isotropy"]
            + sw["area_window"] * area_window
        )
        ew = cfg.elastic_subweights
        elastic = (
            ew["connectivity"] * base["connectivity"]
            + ew["smoothness"] * base["smoothness"]
            + ew["density_window"] * density_window
            + ew["uniformity"] * base["uniformity"]
            + ew["isotropy"] * base["isotropy"]
            + ew["area_window"] * area_window
        )
        weights = cfg.profile_weights[profile]
        score = weights["support"] * support + weights["elastic"] * elastic
        score = float(np.clip(score, 0.0, 1.0))

        detail = (
            f"rho={rho:.3f}; conn={base['connectivity']:.3f}; iso={base['isotropy']:.3f}; "
            f"H_mean={base['curvature_mean']:.3f}; H_p95={base['curvature_p95']:.3f}; "
            f"area/V={base['area_per_volume']:.3f}; "
            f"support={support:.3f}; elastic={elastic:.3f}"
        )
        return EvalMetrics(
            **{k: v for k, v in base.items() if k not in ("support_proxy", "elastic_proxy", "score", "score_detail")},
            support_proxy=float(support), elastic_proxy=float(elastic),
            score=score, score_detail=detail,
        )
