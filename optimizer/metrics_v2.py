# -*- coding: utf-8 -*-
"""
独立的 v2 几何代理规则（与 v1 完全隔离，默认不启用）。

启用方式：`--metric-rule v2`。v1 的公式、权重与历史分数保持原样，不做任何修改。

本模块只解决审查报告 R-12 / R-13 / R-14 指出的三个问题
------------------------------------------------------------
R-12  退化 1D/2D 结构靠其他项补分：
      各向同性从“加权项”改为**乘法门** gate ∈ [0,1]，直接乘在 support / elastic 上。
      层状/柱状结构即使密度、连通性、平滑度满分，也会被 gate 压到接近 0。

R-13  isotropy 实际只有 1.000 / 0.000 两个值：
      改用归一化特征值熵
          p_i = λ_i / Σλ,  entropy = -Σ p_i ln p_i / ln 3
      各向同性 → 1.0；平面型 → ln2/ln3 ≈ 0.631；线型 → 0.0，可连续区分程度。
      gate = clip((entropy - floor) / (1 - floor), 0, 1) ** power。

R-14  skeletal 约定下相对密度恒为 ≈0.5，密度项对所有候选贡献常数：
      改用 sheet 型实体化 |F| <= t·|∇F|，并按 cfg.rho_targets 扫描多个目标密度。
      密度不再作为打分维度（由构造控制），分数反映候选在多个密度下是否仍然连通、平滑。

注意：这仍然是**几何代理**，不是 FEA / 压缩试验结果；只能在同一条规则内排序，
v1 与 v2 的绝对分数不可直接比较。
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def isotropy_entropy(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray,
                     band: np.ndarray, floor: float, power: float) -> Tuple[float, float]:
    """返回 (归一化方向熵, 乘法门 gate)。"""
    w = band.astype(np.float64)
    wsum = float(w.sum())
    if wsum <= 0:
        return 0.0, 0.0
    cov = np.array([
        [float((w * nx * nx).sum()), float((w * nx * ny).sum()), float((w * nx * nz).sum())],
        [float((w * ny * nx).sum()), float((w * ny * ny).sum()), float((w * ny * nz).sum())],
        [float((w * nz * nx).sum()), float((w * nz * ny).sum()), float((w * nz * nz).sum())],
    ], dtype=np.float64) / wsum
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = float(eig.sum())
    if total <= 1e-12:
        return 0.0, 0.0
    p = eig / total
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum() / math.log(3.0))
    entropy = float(np.clip(entropy, 0.0, 1.0))
    gate = float(np.clip((entropy - floor) / max(1e-9, 1.0 - floor), 0.0, 1.0)) ** float(power)
    return entropy, gate


def sheet_phase_stats(F: np.ndarray, gmag: np.ndarray, rho_targets: Sequence[float],
                      component_stats: Callable[[np.ndarray], Tuple[int, float]]) -> Dict:
    """
    sheet 型实体化 |F| <= t·|∇F|，按目标密度反解厚度 t。

    对每个目标 ρ*，取 ratio = |F|/|∇F| 的 ρ* 分位数作为 t，使该密度被精确命中；
    再统计实体相/空相的周期连通性。返回各目标的均值与逐项明细。
    """
    ratio = np.abs(F) / np.maximum(gmag, 1e-12)
    rows: List[Dict] = []
    for rho_t in rho_targets:
        t = float(np.quantile(ratio, float(rho_t)))
        mask = ratio <= t
        s_comp, s_largest = component_stats(mask)
        v_comp, v_largest = component_stats(~mask)
        rows.append({
            "rho_target": float(rho_t),
            "thickness": t,
            "rho": float(mask.mean()),
            "solid_components": int(s_comp),
            "void_components": int(v_comp),
            "solid_largest": float(s_largest),
            "void_largest": float(v_largest),
            # sheet 型实体化会把空相天然分成片层两侧，两侧各自连通但互不连通，
            # 因此这里只看实体相连通性，不用 min(两相) 把结构统一压到 0.5。
            "connectivity": float(s_largest),
        })

    n = max(1, len(rows))
    return {
        "rho_solid": float(np.mean([r["rho"] for r in rows])),
        "rho_void": 1.0 - float(np.mean([r["rho"] for r in rows])),
        "solid_components": int(round(np.mean([r["solid_components"] for r in rows]))),
        "void_components": int(round(np.mean([r["void_components"] for r in rows]))),
        "solid_largest_fraction": float(np.mean([r["solid_largest"] for r in rows])),
        "void_largest_fraction": float(np.mean([r["void_largest"] for r in rows])),
        # 取均值：低密度或高密度任一端断开都会拉低分数
        "connectivity": float(np.mean([r["connectivity"] for r in rows])),
        "rho_scan": "; ".join(
            f"rho*={r['rho_target']:.2f}->{r['rho']:.3f},conn={r['connectivity']:.3f}"
            for r in rows),
    }


def score(base: Dict, profile: str, cfg) -> Tuple[float, float, float, str]:
    """v2 打分：各项先归一化，再统一乘各向同性 gate。"""
    smooth = float(base["smoothness"])
    conn = float(base["connectivity"])
    uniform = float(base["uniformity"])
    area = float(math.exp(
        -0.5 * ((base["area_per_volume"] - cfg.area_target) / cfg.area_sigma) ** 2))
    gate = float(base.get("isotropy_gate", 0.0))

    sw = cfg.support_v2_subweights
    support = (sw["connectivity"] * conn
               + sw["smoothness"] * smooth
               + sw["area_window"] * area) * gate

    ew = cfg.elastic_v2_subweights
    elastic = (ew["connectivity"] * conn
               + ew["smoothness"] * smooth
               + ew["uniformity"] * uniform
               + ew["area_window"] * area) * gate

    weights = cfg.profile_weights[profile]
    score_value = float(np.clip(weights["support"] * support + weights["elastic"] * elastic,
                                0.0, 1.0))
    detail = (
        f"rule=v2; iso_entropy={base.get('isotropy_entropy', 0.0):.3f}; "
        f"gate={gate:.3f}; conn={conn:.3f}; H_mean={base['curvature_mean']:.3f}; "
        f"H_p95={base['curvature_p95']:.3f}; area/V={base['area_per_volume']:.3f}; "
        f"support={support:.3f}; elastic={elastic:.3f}"
    )
    return float(support), float(elastic), score_value, detail
