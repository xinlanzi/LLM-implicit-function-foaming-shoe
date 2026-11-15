# -*- coding: utf-8 -*-
"""
从 Excel 知识库读取种子结构、标签与目标性能定义。

真实表结构与技术文档的“理想接口”不同，因此这里按实际列名适配：
- Structure Master           : 结构元数据、原始方程、分类
- Structure-Tag Mapping      : 每个结构的多标签（中文标签）
- Structure-Token Mapping    : Token_Sequence（JSON 字符串，+/-、系数、基函数表达式交替）
- Structure-Performance Mapping : C1~C7 文献评分（仅作背景，不替代评价器）

只把“可由三角闭式基函数表达”的结构放进可优化种子池；
FRep / 图论 / 坐标参数化等无三角闭式的结构保留在元数据中，但不参与本次隐函数搜索。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import openpyxl

try:  # 兼容包导入与脚本导入
    from .token_library import expr_to_name, canonical_signature
except ImportError:
    from token_library import expr_to_name, canonical_signature


# --------------------------------------------------------------------------- #
# 目标性能定义
# --------------------------------------------------------------------------- #
# 说明：
# “高支撑” → 高刚度/高承载/拉伸主导（stretching-dominated）；
# “高弹性” → 高可恢复变形/高回弹/变形稳定（resilience），注意它并不等于“低刚度柔顺”。
# 两者在力学上存在刚度—柔度权衡，组合目标必须有折中。
PROFILE_TAGS: Dict[str, List[str]] = {
    "high_support": [
        "高刚度", "高比刚度", "高比强度", "承载能力", "拉伸主导",
        "高屈服强度", "刚度增强", "可定制刚度",
    ],
    "high_elasticity": [
        "高柔顺性", "变形稳定", "高延伸率", "可重复吸能", "滞回",
        "压缩诱导拉伸", "压缩诱导拉伸(CIS)",
    ],
    "high_support_high_elasticity": [],  # 运行时取并集
}

PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "high_support": (
        "高支撑（高刚度/高承载）：希望载荷沿连续曲面骨架传递，尽量以拉伸/压缩主导，"
        "减少弯曲与局部屈曲；曲面曲率平缓、实体相连续，并保持足够的相对密度。"
    ),
    "high_elasticity": (
        "高弹性（高可恢复变形/高回弹）：希望双连续通道、曲率平滑且分布均匀，"
        "没有细薄韧带与尖角，变形以可恢复的弯曲/剪切为主而不发生局部屈服或屈曲；"
        "相对密度适中，避免过密导致脆性压溃、过疏导致薄壁失稳。"
    ),
    "high_support_high_elasticity": (
        "同时兼顾高支撑与高弹性：用连续、低曲率的承载骨架提供刚度（支撑），"
        "同时保持平滑、均匀、双连续的曲面以降低弯曲应力集中、保证变形可恢复（弹性）。"
        "两者在相对密度与曲率上存在权衡：密度越高支撑越强但孔隙与可恢复变形空间越小，"
        "倍频/高曲率项能提高比表面积却会损害弹性，因此需要折中而非单指标最大化。"
    ),
}


def _profile_tag_union(name: str) -> List[str]:
    if name == "high_support_high_elasticity":
        return sorted(set(PROFILE_TAGS["high_support"]) | set(PROFILE_TAGS["high_elasticity"]))
    return PROFILE_TAGS[name]


@dataclass
class SeedStructure:
    """一个可被评价器执行的种子结构。"""
    sid: str
    name: str
    chinese_name: str
    category: str
    original_equation: str
    terms: List[str]
    coeffs: List[float]
    tags: List[str]
    source_sequence: str
    notes: str = ""
    metrics: Optional[dict] = None
    profile_match: Dict[str, bool] = field(default_factory=dict)


def _parse_token_sequence(seq_json: str) -> Tuple[List[str], List[float], str]:
    """
    解析交替序列，例如：
        ["+", "1.0", "sin(X)cos(Y)", "+", "1.0", "sin(Y)cos(Z)", ...]
    返回 (terms, coeffs, 说明)。无法解析的标记（如 "|DD2|"）会被跳过并记录。
    """
    try:
        seq = json.loads(seq_json)
    except Exception:
        return [], [], "Token_Sequence 不是合法 JSON"
    if not isinstance(seq, list):
        return [], [], "Token_Sequence 不是列表"

    terms: List[str] = []
    coeffs: List[float] = []
    notes: List[str] = []
    i = 0
    sign = 1.0
    while i < len(seq):
        token = seq[i]
        if isinstance(token, str) and token in ("+", "-"):
            sign = 1.0 if token == "+" else -1.0
            i += 1
            # 期望：系数、表达式
            if i + 1 >= len(seq):
                notes.append("序列在 +/- 后缺少系数或表达式")
                break
            raw_coeff, raw_expr = seq[i], seq[i + 1]
            try:
                coeff = float(raw_coeff)
            except (TypeError, ValueError):
                notes.append(f"系数不是数字：{raw_coeff!r}")
                i += 2
                continue
            try:
                name = expr_to_name(raw_expr)
            except KeyError:
                notes.append(f"基函数不可用：{raw_expr!r}")
                i += 2
                continue
            terms.append(name)
            coeffs.append(sign * coeff)
            i += 2
        else:
            # 例如 S009 的 "|DD2|" 分隔标记
            if isinstance(token, str) and token.strip():
                notes.append(f"跳过非运算标记：{token!r}")
            i += 1
    return terms, coeffs, "；".join(notes)


def load_excel(excel_path: str):
    """读取四张表，返回元数据、标签、token 序列、性能矩阵。"""
    wb = openpyxl.load_workbook(excel_path, data_only=True)

    # 1) Structure Master
    ws = wb["Structure Master"]
    header = [c for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    meta: Dict[str, dict] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        rec = dict(zip(header, row))
        sid = str(rec.get("Structure_ID", "")).strip()
        if sid:
            meta[sid] = {
                "name": rec.get("Structure_Name") or "",
                "chinese_name": rec.get("Chinese_Name") or "",
                "category": rec.get("Structure_Category") or "",
                "original_equation": rec.get("Original_Equation") or "",
                "notes": rec.get("Notes") or "",
            }

    # 2) Structure-Tag Mapping
    ws = wb["Structure-Tag Mapping"]
    tag_header = [c for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    tags: Dict[str, List[str]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        rec = dict(zip(tag_header, row))
        sid = str(rec.get("Structure_ID", "")).strip()
        tag = str(rec.get("Tag_Name") or "").strip()
        if sid and tag:
            tags.setdefault(sid, [])
            if tag not in tags[sid]:
                tags[sid].append(tag)

    # 3) Structure-Token Mapping
    ws = wb["Structure-Token Mapping"]
    tok_header = [c for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    sequences: Dict[str, List[str]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        rec = dict(zip(tok_header, row))
        sid = str(rec.get("Structure_ID", "")).strip()
        seq = rec.get("Token_Sequence")
        if sid and seq:
            sequences.setdefault(sid, [])
            if str(seq) not in sequences[sid]:
                sequences[sid].append(str(seq))

    # 4) Structure-Performance Mapping（仅统计背景，不伪造 FEA）
    ws = wb["Structure-Performance Mapping"]
    perf_header = [c for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    performance: Dict[str, dict] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        rec = dict(zip(perf_header, row))
        sid = str(rec.get("Structure_ID", "")).strip()
        if sid:
            performance[sid] = {
                "C1_SEA": rec.get("C1 SEA上限"),
                "C2_energy_window": rec.get("C2 稳定吸能密度窗口"),
                "C3_stiffness": rec.get("C3 比刚度上限"),
                "C4_load_tunability": rec.get("C4 承载可调区间"),
                "C5_monotonicity": rec.get("C5 参数单调性"),
                "C6_feasible_space": rec.get("C6 几何可行空间"),
                "C7_failure_risk": rec.get("C7 失效模式突变风险"),
            }
    wb.close()
    return meta, tags, sequences, performance


def load_seeds(excel_path: str) -> Tuple[List[SeedStructure], List[dict], Dict[str, dict]]:
    """
    返回：
      seeds        —— 可被三角闭式隐函数评价的种子（已去重）
      excluded     —— 无法用隐函数 token 表达的结构（保留元数据，便于报告说明）
      performance  —— 原始 C1~C7 文献评分（背景信息）
    """
    meta, tags, sequences, performance = load_excel(excel_path)
    seeds: List[SeedStructure] = []
    excluded: List[dict] = []
    seen_signatures: Dict[str, str] = {}

    for sid in sorted(meta.keys()):
        if sid not in sequences:
            excluded.append({
                "sid": sid, "name": meta[sid]["name"],
                "reason": "Token 库中没有 Token_Sequence（FRep/几何构造/无闭式）",
            })
            continue

        chosen = None
        chosen_terms = chosen_coeffs = None
        notes = []
        for seq in sequences[sid]:
            terms, coeffs, note = _parse_token_sequence(seq)
            if terms:
                chosen, chosen_terms, chosen_coeffs = seq, terms, coeffs
                notes.append(note)
                break
            notes.append(note)
        if not chosen or not chosen_terms:
            excluded.append({
                "sid": sid, "name": meta[sid]["name"],
                "reason": "Token 序列无法解析为三角闭式基函数：" + "；".join(n for n in notes if n),
            })
            continue

        # 去重：F 与正比例缩放 F 视为同一张隐式曲面
        sig = canonical_signature(chosen_terms, chosen_coeffs)
        if sig and sig in seen_signatures:
            excluded.append({
                "sid": sid, "name": meta[sid]["name"],
                "reason": f"与 {seen_signatures[sig]} 几何等价（F=0 对正比例缩放不变）",
            })
            continue
        if sig:
            seen_signatures[sig] = sid

        seed = SeedStructure(
            sid=sid,
            name=meta[sid]["name"],
            chinese_name=meta[sid]["chinese_name"],
            category=meta[sid]["category"],
            original_equation=meta[sid]["original_equation"],
            terms=chosen_terms,
            coeffs=chosen_coeffs,
            tags=tags.get(sid, []),
            source_sequence=chosen,
            notes=meta[sid]["notes"],
        )
        # 预计算每个目标标签的匹配情况（生成候选默认视为匹配当前目标）
        for profile in PROFILE_TAGS:
            tag_union = _profile_tag_union(profile)
            seed.profile_match[profile] = any(t in tag_union for t in seed.tags)
        seeds.append(seed)

    return seeds, excluded, performance


def describe_seed_table(seeds: List[SeedStructure]) -> List[dict]:
    """把种子转成便于写 CSV / 报告的行。"""
    rows = []
    for s in seeds:
        rows.append({
            "sid": s.sid,
            "name": s.name,
            "chinese_name": s.chinese_name,
            "category": s.category,
            "terms": json.dumps(s.terms, ensure_ascii=False),
            "coeffs": json.dumps([round(c, 6) for c in s.coeffs], ensure_ascii=False),
            "n_terms": len(s.terms),
            "tags": " | ".join(s.tags),
            "original_equation": s.original_equation,
            "notes": s.notes,
        })
    return rows
