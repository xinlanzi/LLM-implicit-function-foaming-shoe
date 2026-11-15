# -*- coding: utf-8 -*-
"""
五个“高支撑 + 高弹性”组合结构的定义。

数据来源
--------
- terms / coeffs / 轮次 / 来源：`outputs/candidates.csv` 中实际搜索产生的候选；
- score / support / elastic：由 `optimizer/evaluator.py` 实际计算的几何代理分；
- 这里保留 4 位小数的系数，构建网格时可用 `--verify-candidates` 与 CSV 交叉核对。

注意：score 是几何代理分，不是 FEA 结果。详见
`outputs/high_support_high_elasticity_solutions.md`。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolutionSpec:
    sid: str                 # 文件编号
    name: str                # 英文短名（用于文件名）
    cn_name: str             # 中文名
    terms: tuple
    coeffs: tuple
    score: float             # 参考分（构建时可被 CSV 实际值覆盖）
    support_proxy: float
    elastic_proxy: float
    round: int
    provenance: str
    equation: str            # 便于报告展示的方程字符串
    notes: str               # 组合理由 / 适用场景（用于 viewer 与 README）


SOLUTIONS = [
    SolutionSpec(
        sid="01",
        name="GyroidFlip",
        cn_name="Gyroid 反号变体",
        terms=("sin_x_cos_y", "sin_y_cos_z", "sin_z_cos_x"),
        coeffs=(-0.9446, 0.9723, 0.9816),
        score=0.9449,
        support_proxy=0.9747,
        elastic_proxy=0.9085,
        round=24,
        provenance="LLM(模拟) r24 · 模板 Gyroid + flip[0]",
        equation="F = -0.9446·sin(x)cos(y) + 0.9723·sin(y)cos(z) + 0.9816·sin(z)cos(x) = 0",
        notes=(
            "Gyroid 三条螺旋分量近等幅，三向互锁；第一项反号是本方案的关键变异，"
            "在保持三维连通与各向同性的同时改变两侧实体相分布。"
            "支撑来自三轴连续承力路径，弹性来自螺旋曲面的负高斯曲率可侧向让位。"
            "适用：轻量化承力芯、换热/混合骨架、需要各向同性的场合。"
        ),
    ),
    SolutionSpec(
        sid="02",
        name="DiamondBalanced",
        cn_name="Diamond 等幅变体",
        terms=("cos_x_cos_y_cos_z", "sin_x_sin_y_sin_z"),
        coeffs=(0.9235, -0.9301),
        score=0.9338,
        support_proxy=0.9664,
        elastic_proxy=0.8940,
        round=18,
        provenance="LLM(模拟) r18 · 模板 Diamond",
        equation="F = 0.9235·cos(x)cos(y)cos(z) - 0.9301·sin(x)sin(y)sin(z) = 0",
        notes=(
            "三轴余弦节点网络（承载集中）+ 三轴正弦鞍面（连接节点）；"
            "等幅反对称即 Schwarz Diamond 的标准形式。"
            "节点提供高支撑，鞍面双曲曲率提供可恢复变形。"
            "适用：弯曲主导、需要高比强度的夹芯/支撑结构。"
        ),
    ),
    SolutionSpec(
        sid="03",
        name="DiamondPlusX",
        cn_name="Diamond + x 向余弦",
        terms=("cos_x_cos_y_cos_z", "sin_x_sin_y_sin_z", "cos_x"),
        coeffs=(1.0472, -1.0811, 0.1942),
        score=0.9194,
        support_proxy=0.9533,
        elastic_proxy=0.8779,
        round=1,
        provenance="LLM(模拟) r1 · 模板 Diamond_plus_x",
        equation=(
            "F = 1.0472·cos(x)cos(y)cos(z) - 1.0811·sin(x)sin(y)sin(z) "
            "+ 0.1942·cos(x) = 0"
        ),
        notes=(
            "在 D 面上叠加小幅 cos(x) 相位偏置，沿 x 向增加定向承载能力；"
            "isotropy 由 1.000 降到约 0.975，是典型的“支撑换弹性”。"
            "适用：主载荷方向已知的单向压缩支柱、定向导热/流动结构。"
        ),
    ),
    SolutionSpec(
        sid="04",
        name="DiamondPlusY",
        cn_name="Diamond + y 向余弦",
        terms=("cos_x_cos_y_cos_z", "sin_x_sin_y_sin_z", "cos_y"),
        coeffs=(1.0953, -0.9327, 0.3112),
        score=0.8986,
        support_proxy=0.9346,
        elastic_proxy=0.8545,
        round=46,
        provenance="LLM(模拟) r46 · 模板 Diamond_plus_y",
        equation=(
            "F = 1.0953·cos(x)cos(y)cos(z) - 0.9327·sin(x)sin(y)sin(z) "
            "+ 0.3112·cos(y) = 0"
        ),
        notes=(
            "与 D+x 同族，但 y 向偏置更强（0.31），各向异性更明显、弹性代理更低；"
            "适合单向剪切/弯曲梁，也可作为“偏置过强”的对照。"
            "适用：需要沿 y 向增强刚度的梁/板。"
        ),
    ),
    SolutionSpec(
        sid="05",
        name="GyroidDiamondMix",
        cn_name="Gyroid × Diamond 交叉",
        terms=("cos_x_cos_y_cos_z", "sin_x_sin_y_sin_z", "sin_x_cos_y"),
        coeffs=(1.0295, -0.8589, 0.8835),
        score=0.7983,
        support_proxy=0.8350,
        elastic_proxy=0.7533,
        round=17,
        provenance="LLM(模拟) r17 · crossover S003 × S001",
        equation=(
            "F = 1.0295·cos(x)cos(y)cos(z) - 0.8589·sin(x)sin(y)sin(z) "
            "+ 0.8835·sin(x)cos(y) = 0"
        ),
        notes=(
            "跨族交叉：D 的节点/鞍面 + Gyroid 的螺旋项。"
            "结果 isotropy 降到约 0.60、曲率升高，提示结构开始柱状化。"
            "保留为反例：跨族混合不一定优于任一母体。"
            "适用：需要混合拓扑/梯度过渡的研究样件，而非性能最优件。"
        ),
    ),
]


def find_solution(sid_or_name: str) -> SolutionSpec:
    """按编号或英文名查找方案。"""
    key = str(sid_or_name).lower()
    for s in SOLUTIONS:
        if s.sid == key or s.name.lower() == key:
            return s
    raise KeyError(f"未知方案: {sid_or_name}（可选: {[s.sid for s in SOLUTIONS]}）")
