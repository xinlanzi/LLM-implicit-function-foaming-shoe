# -*- coding: utf-8 -*-
"""
隐函数基函数词表（token library）。

技术文档给出的 12 个核心基函数在 `CORE_TOKENS` 中严格保留；
Excel《Structure-Token Mapping》里还真实出现了 sin*sin、三乘积、倍频 cos(2X)/sin(2X)
等三角闭式表达式，这些表达式被解析为 `LIBRARY_TOKENS` 的扩展 token。
这样既能严格复现文档的 core 搜索空间，也能忠实读取库内的 11 个三角闭式种子结构。

每个 token 都对应一个可执行的 f(x, y, z) 基函数；隐式曲面由
    F(x, y, z) = Σ coeff_i * token_i(x, y, z) = 0
给出。
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# token 规格表
# --------------------------------------------------------------------------- #
# name       : 优化器内部使用的规范 token 名
# expr       : 与 Excel Token_Expression 完全一致的表达式（用于种子解析）
# family     : 便于解释的族名
# role       : 几何/力学作用简述（写入 prompt 与报告）
TOKEN_SPECS: List[dict] = [
    # ---- 单轴基函数：P 面 / 直通道 ----
    dict(name="sin_x", expr="sin(X)", family="single", role="x 向正弦波，形成沿 x 的周期性起伏"),
    dict(name="cos_x", expr="cos(X)", family="single", role="x 向余弦波，形成沿 x 的直通道/承载骨架"),
    dict(name="sin_y", expr="sin(Y)", family="single", role="y 向正弦波，形成沿 y 的周期性起伏"),
    dict(name="cos_y", expr="cos(Y)", family="single", role="y 向余弦波，形成沿 y 的直通道/承载骨架"),
    dict(name="sin_z", expr="sin(Z)", family="single", role="z 向正弦波，形成沿 z 的周期性起伏"),
    dict(name="cos_z", expr="cos(Z)", family="single", role="z 向余弦波，形成沿 z 的直通道/承载骨架"),

    # ---- 双轴 sin*cos：G 面型螺旋/手性 ----
    dict(name="sin_x_cos_y", expr="sin(X)cos(Y)", family="chiral", role="x-y 螺旋耦合，形成 G 面型手性通道"),
    dict(name="sin_x_cos_z", expr="sin(X)cos(Z)", family="chiral", role="x-z 手性耦合（核心词表补充项）"),
    dict(name="sin_y_cos_z", expr="sin(Y)cos(Z)", family="chiral", role="y-z 螺旋耦合，形成 G 面型手性通道"),
    dict(name="cos_x_sin_y", expr="cos(X)sin(Y)", family="chiral", role="x-y 反向手性耦合"),
    dict(name="cos_x_sin_z", expr="cos(X)sin(Z)", family="chiral", role="x-z 反向手性耦合"),
    dict(name="cos_y_sin_z", expr="cos(Y)sin(Z)", family="chiral", role="y-z 反向手性耦合"),
    dict(name="sin_z_cos_x", expr="sin(Z)cos(X)", family="chiral", role="z-x 螺旋耦合，G 面的第三项"),

    # ---- 双轴 cos*cos：D/FCC 型对角余弦 ----
    dict(name="cos_x_cos_y", expr="cos(X)cos(Y)", family="diagonal", role="x-y 对角余弦，形成 D/FCC 型对角韧带"),
    dict(name="cos_x_cos_z", expr="cos(Z)cos(X)", family="diagonal", role="x-z 对角余弦，形成对角韧带"),
    dict(name="cos_y_cos_z", expr="cos(Y)cos(Z)", family="diagonal", role="y-z 对角余弦，形成对角韧带"),

    # ---- 双轴 sin*sin：剪切型交叉项 ----
    dict(name="sin_x_sin_y", expr="sin(X)sin(Y)", family="shear", role="x-y 剪切型交叉项，制造斜向节点"),
    dict(name="sin_x_sin_z", expr="sin(X)sin(Z)", family="shear", role="x-z 剪切型交叉项"),
    dict(name="sin_y_sin_z", expr="sin(Y)sin(Z)", family="shear", role="y-z 剪切型交叉项"),

    # ---- 三乘积：D 面 / 4 重网络 ----
    dict(name="cos_x_cos_y_cos_z", expr="cos(X)cos(Y)cos(Z)", family="triple", role="三轴余弦（D 面主项），形成三维棋盘式弯曲主导网络"),
    dict(name="sin_x_sin_y_sin_z", expr="sin(X)sin(Y)sin(Z)", family="triple", role="三轴正弦（D 面第二项），与三轴余弦构成 D 面"),
    dict(name="cos_x_cos_y_sin_z", expr="cos(X)cos(Y)sin(Z)", family="triple", role="三轴混合项，打破 z 向对称性"),
    dict(name="cos_x_sin_y_sin_z", expr="cos(X)sin(Y)sin(Z)", family="triple", role="三轴混合项，打破 y/z 对称性"),

    # ---- 倍频项：细化单胞、提高比表面积 ----
    dict(name="cos_2x", expr="cos(2X)", family="harmonic", role="x 向二倍频，细化周期性、提高比表面积，但增加曲率"),
    dict(name="cos_2y", expr="cos(2Y)", family="harmonic", role="y 向二倍频，细化周期性、提高比表面积，但增加曲率"),
    dict(name="cos_2z", expr="cos(2Z)", family="harmonic", role="z 向二倍频，细化周期性、提高比表面积，但增加曲率"),
    dict(name="sin_2x", expr="sin(2X)", family="harmonic", role="x 向二倍频正弦（库内未出现，作为对称扩展）"),
    dict(name="sin_2y", expr="sin(2Y)", family="harmonic", role="y 向二倍频正弦（对称扩展）"),
    dict(name="sin_2z", expr="sin(2Z)", family="harmonic", role="z 向二倍频正弦（对称扩展）"),

    # ---- 倍频双轴/三轴：复杂 4 重网络 ----
    dict(name="cos_2x_cos_2y", expr="cos(2X)cos(2Y)", family="harmonic_diag", role="2x-2y 对角余弦，细化对角韧带"),
    dict(name="cos_2y_cos_2z", expr="cos(2Y)cos(2Z)", family="harmonic_diag", role="2y-2z 对角余弦"),
    dict(name="cos_2z_cos_2x", expr="cos(2Z)cos(2X)", family="harmonic_diag", role="2z-2x 对角余弦"),
    dict(name="sin_2x_sin_2y", expr="sin(2X)sin(2Y)", family="harmonic_shear", role="2x-2y 剪切交叉项，形成细密斜向节点"),
    dict(name="sin_2y_sin_2z", expr="sin(2Y)sin(2Z)", family="harmonic_shear", role="2y-2z 剪切交叉项"),
    dict(name="sin_2x_sin_2z", expr="sin(2X)sin(2Z)", family="harmonic_shear", role="2x-2z 剪切交叉项"),
    dict(name="cos_2x_cos_2y_cos_2z", expr="cos(2X)cos(2Y)cos(2Z)", family="harmonic_triple", role="2 倍频三轴余弦，高比表面积 4 重网络"),
    dict(name="sin_2x_sin_2y_sin_2z", expr="sin(2X)sin(2Y)sin(2Z)", family="harmonic_triple", role="2 倍频三轴正弦，与倍频余弦组合成 4 重网络"),
    dict(name="sin_2x_cos_y_sin_z", expr="sin(2X)cos(Y)sin(Z)", family="hybrid", role="x 倍频 + y/z 基频混合，形成强各向异性复杂网络"),
    dict(name="sin_x_sin_2y_cos_z", expr="sin(X)sin(2Y)cos(Z)", family="hybrid", role="y 倍频 + x/z 基频混合"),
    dict(name="cos_x_cos_y_sin_2z", expr="cos(X)cos(Y)sin(2Z)", family="hybrid", role="z 倍频 + x/y 基频混合"),
    dict(name="cos_x_sin_y_sin_2z", expr="cos(X)sin(Y)sin(2Z)", family="hybrid", role="z 倍频 + 手性混合"),
]

# 技术文档明确给出的 12 个核心基函数（顺序与文档一致）
CORE_TOKEN_NAMES: Tuple[str, ...] = (
    "sin_x", "cos_x", "sin_y", "cos_y", "sin_z", "cos_z",
    "sin_x_cos_y", "sin_x_cos_z", "sin_y_cos_z",
    "cos_x_cos_y", "cos_x_cos_z", "cos_y_cos_z",
)

# name -> 规格
TOKEN_BY_NAME: Dict[str, dict] = {spec["name"]: spec for spec in TOKEN_SPECS}

# Excel 表达式 -> 规范 token 名（允许缺少空格/大小写不完全一致）
_EXPR_ALIASES: Dict[str, str] = {}
for _spec in TOKEN_SPECS:
    _expr = _spec["expr"]
    _EXPR_ALIASES[_expr] = _spec["name"]
    _EXPR_ALIASES[_expr.replace(" ", "")] = _spec["name"]
    _EXPR_ALIASES[_expr.lower()] = _spec["name"]

# 几组等价的写法，Excel 里可能出现顺序不同但数学上相同的乘积
_EXTRA_ALIASES = {
    "cos(Y)cos(X)": "cos_x_cos_y",
    "cos(Z)cos(Y)": "cos_y_cos_z",
    "cos(X)cos(Z)": "cos_x_cos_z",
    "sin(Y)cos(X)": "sin_x_cos_y",
    "sin(Z)cos(Y)": "sin_y_cos_z",
    "sin(X)cos(Z)": "sin_x_cos_z",
    "cos(Y)sin(X)": "cos_x_sin_y",
    "cos(Z)sin(X)": "cos_x_sin_z",
    "cos(Z)sin(Y)": "cos_y_sin_z",
    "cos(Y)cos(X)cos(Z)": "cos_x_cos_y_cos_z",
}
_EXPR_ALIASES.update(_EXTRA_ALIASES)


# --------------------------------------------------------------------------- #
# 基函数实现
# --------------------------------------------------------------------------- #
def _f(name: str) -> Callable:
    """按规范名返回 numpy 向量化函数 f(x, y, z)。"""
    sin, cos = np.sin, np.cos
    table = {
        # 单轴
        "sin_x": lambda x, y, z: sin(x),
        "cos_x": lambda x, y, z: cos(x),
        "sin_y": lambda x, y, z: sin(y),
        "cos_y": lambda x, y, z: cos(y),
        "sin_z": lambda x, y, z: sin(z),
        "cos_z": lambda x, y, z: cos(z),
        # 双轴 sin*cos
        "sin_x_cos_y": lambda x, y, z: sin(x) * cos(y),
        "sin_x_cos_z": lambda x, y, z: sin(x) * cos(z),
        "sin_y_cos_z": lambda x, y, z: sin(y) * cos(z),
        "cos_x_sin_y": lambda x, y, z: cos(x) * sin(y),
        "cos_x_sin_z": lambda x, y, z: cos(x) * sin(z),
        "cos_y_sin_z": lambda x, y, z: cos(y) * sin(z),
        "sin_z_cos_x": lambda x, y, z: sin(z) * cos(x),
        # 双轴 cos*cos
        "cos_x_cos_y": lambda x, y, z: cos(x) * cos(y),
        "cos_x_cos_z": lambda x, y, z: cos(x) * cos(z),
        "cos_y_cos_z": lambda x, y, z: cos(y) * cos(z),
        # 双轴 sin*sin
        "sin_x_sin_y": lambda x, y, z: sin(x) * sin(y),
        "sin_x_sin_z": lambda x, y, z: sin(x) * sin(z),
        "sin_y_sin_z": lambda x, y, z: sin(y) * sin(z),
        # 三乘积
        "cos_x_cos_y_cos_z": lambda x, y, z: cos(x) * cos(y) * cos(z),
        "sin_x_sin_y_sin_z": lambda x, y, z: sin(x) * sin(y) * sin(z),
        "cos_x_cos_y_sin_z": lambda x, y, z: cos(x) * cos(y) * sin(z),
        "cos_x_sin_y_sin_z": lambda x, y, z: cos(x) * sin(y) * sin(z),
        # 倍频
        "cos_2x": lambda x, y, z: cos(2 * x),
        "cos_2y": lambda x, y, z: cos(2 * y),
        "cos_2z": lambda x, y, z: cos(2 * z),
        "sin_2x": lambda x, y, z: sin(2 * x),
        "sin_2y": lambda x, y, z: sin(2 * y),
        "sin_2z": lambda x, y, z: sin(2 * z),
        # 倍频双轴/三轴
        "cos_2x_cos_2y": lambda x, y, z: cos(2 * x) * cos(2 * y),
        "cos_2y_cos_2z": lambda x, y, z: cos(2 * y) * cos(2 * z),
        "cos_2z_cos_2x": lambda x, y, z: cos(2 * z) * cos(2 * x),
        "sin_2x_sin_2y": lambda x, y, z: sin(2 * x) * sin(2 * y),
        "sin_2y_sin_2z": lambda x, y, z: sin(2 * y) * sin(2 * z),
        "sin_2x_sin_2z": lambda x, y, z: sin(2 * x) * sin(2 * z),
        "cos_2x_cos_2y_cos_2z": lambda x, y, z: cos(2 * x) * cos(2 * y) * cos(2 * z),
        "sin_2x_sin_2y_sin_2z": lambda x, y, z: sin(2 * x) * sin(2 * y) * sin(2 * z),
        # 混合倍频
        "sin_2x_cos_y_sin_z": lambda x, y, z: sin(2 * x) * cos(y) * sin(z),
        "sin_x_sin_2y_cos_z": lambda x, y, z: sin(x) * sin(2 * y) * cos(z),
        "cos_x_cos_y_sin_2z": lambda x, y, z: cos(x) * cos(y) * sin(2 * z),
        "cos_x_sin_y_sin_2z": lambda x, y, z: cos(x) * sin(y) * sin(2 * z),
    }
    if name not in table:
        raise KeyError(f"未实现的 token: {name}")
    return table[name]


# 预绑定函数，减少重复查表
TOKEN_FUNCS: Dict[str, Callable] = {spec["name"]: _f(spec["name"]) for spec in TOKEN_SPECS}


def expr_to_name(expr: str) -> str:
    """把 Excel 中的表达式字符串转成规范 token 名；找不到则抛 KeyError。"""
    if expr is None:
        raise KeyError("空表达式")
    raw = str(expr).strip()
    candidates = [raw, raw.replace(" ", ""), raw.lower()]
    for cand in candidates:
        if cand in _EXPR_ALIASES:
            return _EXPR_ALIASES[cand]
    raise KeyError(f"无法解析的基函数表达式: {expr!r}")


def get_allowed_tokens(token_set: str = "library") -> List[str]:
    """返回当前搜索空间允许的 token 名列表（升序，便于复现）。"""
    token_set = token_set.lower()
    if token_set == "core":
        return list(CORE_TOKEN_NAMES)
    if token_set == "library":
        return sorted(TOKEN_BY_NAME.keys())
    raise ValueError(f"未知 token_set: {token_set}")


def token_roles(tokens: Iterable[str]) -> str:
    """把 token 列表转成一句可读的作用说明，用于 prompt/报告。"""
    parts = []
    for t in tokens:
        role = TOKEN_BY_NAME.get(t, {}).get("role", "未知基函数")
        parts.append(f"{t}（{role}）")
    return "；".join(parts)


def build_function(terms: List[str], coeffs: List[float]) -> Callable:
    """
    技术文档模块 5 的 build_function：
    把 token 序列还原成可执行的隐式曲面函数 F(x, y, z)。
    """
    if len(terms) != len(coeffs):
        raise ValueError("terms 与 coeffs 长度不一致")
    funcs = []
    for term, coeff in zip(terms, coeffs):
        if term not in TOKEN_FUNCS:
            raise KeyError(f"非法 token: {term}")
        funcs.append((float(coeff), TOKEN_FUNCS[term]))

    def F(x, y, z):
        result = np.zeros_like(np.asarray(x, dtype=float))
        for coeff, fn in funcs:
            result = result + coeff * fn(x, y, z)
        return result

    return F


def canonical_signature(terms: List[str], coeffs: List[float], ndigits: int = 6) -> str:
    """
    生成用于去重的规范签名。

    隐式曲面 F=0 对整体正比例缩放不变（F 与 2F 是同一张曲面），
    因此先把系数按最大绝对值归一化；负号不忽略，因为 F<=0 的实体相会翻转，
    几何/密度指标会改变。
    """
    if not terms or len(terms) != len(coeffs):
        return ""
    merged: Dict[str, float] = {}
    for t, c in zip(terms, coeffs):
        merged[t] = merged.get(t, 0.0) + float(c)
    merged = {t: c for t, c in merged.items() if abs(c) > 1e-12}
    if not merged:
        return ""
    scale = max(abs(c) for c in merged.values())
    # 固定项顺序，避免 ["a","b"] 与 ["b","a"] 被视为不同
    items = sorted(merged.items(), key=lambda kv: kv[0])
    norm = [(t, round(c / scale, ndigits)) for t, c in items]
    return ";".join(f"{t}:{c}" for t, c in norm)


def terms_coeffs_to_json(terms: List[str], coeffs: List[float]) -> Tuple[str, str]:
    """输出给 CSV 用的紧凑 JSON 字符串。"""
    import json
    return json.dumps(list(terms), ensure_ascii=False), json.dumps([round(float(c), 6) for c in coeffs])
