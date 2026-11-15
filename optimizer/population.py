# -*- coding: utf-8 -*-
"""
候选池（Population）——技术文档模块 1 与模块 6。

规则
----
- 按目标 profile 分池（本次主实验是 high_support_high_elasticity 单池）；
- 每池最多 pool_size 个候选；
- 新候选分数 > 池内最差分数才替换；
- 去重使用“几何等价”规范签名（F=0 对整体正比例缩放不变）。

注意：种子结构的 score 使用评价器的真实几何代理分，而不是文档示例里的 0.5 常数。
这样池内比较才是真实可追溯的；报告中会单独列出种子分数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

try:  # 兼容包导入与脚本导入
    from .token_library import canonical_signature
except ImportError:
    from token_library import canonical_signature


@dataclass
class Candidate:
    label: str
    terms: List[str]
    coeffs: List[float]
    score: float
    source: str                     # seed / llm / random
    round: int
    accepted: bool
    reason: str
    signature: str
    metrics: Optional[dict] = None
    provenance: dict = field(default_factory=dict)
    sid: str = ""
    tags: str = ""
    status: str = "evaluated"       # evaluated / invalid / parse_failed

    @property
    def support_proxy(self) -> float:
        return float((self.metrics or {}).get("support_proxy", 0.0))

    @property
    def elastic_proxy(self) -> float:
        return float((self.metrics or {}).get("elastic_proxy", 0.0))

    def to_row(self) -> dict:
        m = self.metrics or {}
        return {
            "label": self.label,
            "sid": self.sid,
            "terms": json.dumps(list(self.terms), ensure_ascii=False),
            "coeffs": json.dumps([round(float(c), 6) for c in self.coeffs], ensure_ascii=False),
            "n_terms": len(self.terms),
            "score": round(float(self.score), 6) if self.score is not None else None,
            "source": self.source,
            "round": int(self.round),
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "status": self.status,
            "support_proxy": round(float(m.get("support_proxy", 0.0)), 6) if m else None,
            "elastic_proxy": round(float(m.get("elastic_proxy", 0.0)), 6) if m else None,
            "rho_solid": round(float(m.get("rho_solid", 0.0)), 6) if m else None,
            "connectivity": round(float(m.get("connectivity", 0.0)), 6) if m else None,
            "isotropy": round(float(m.get("isotropy", 0.0)), 6) if m else None,
            "solid_components": m.get("solid_components"),
            "void_components": m.get("void_components"),
            "curvature_mean": round(float(m.get("curvature_mean", 0.0)), 6) if m else None,
            "curvature_p95": round(float(m.get("curvature_p95", 0.0)), 6) if m else None,
            "area_per_volume": round(float(m.get("area_per_volume", 0.0)), 6) if m else None,
            "smoothness": round(float(m.get("smoothness", 0.0)), 6) if m else None,
            "uniformity": round(float(m.get("uniformity", 0.0)), 6) if m else None,
            "provenance": json.dumps(self.provenance, ensure_ascii=False),
            "tags": self.tags,
            "signature": self.signature,
        }


class Population:
    def __init__(self, label: str, pool_size: int):
        self.label = label
        self.pool_size = int(pool_size)
        self.items: List[Candidate] = []
        self.signatures: Set[str] = set()

    def contains_signature(self, sig: str) -> bool:
        return bool(sig) and sig in self.signatures

    def add(self, cand: Candidate) -> tuple:
        """
        返回 (accepted, reason)。候选必须已经算好 score / metrics。
        """
        if not cand.signature:
            return False, "空签名（terms/coeffs 无效）"
        if cand.signature in self.signatures:
            return False, "与池内候选几何等价（去重）"

        if len(self.items) < self.pool_size:
            self.items.append(cand)
            self.signatures.add(cand.signature)
            return True, "池未满，直接入池"

        worst = min(self.items, key=lambda c: c.score)
        if cand.score > worst.score:
            self.items.remove(worst)
            self.signatures.discard(worst.signature)
            self.items.append(cand)
            self.signatures.add(cand.signature)
            return True, f"优于池内最差（{cand.score:.4f} > {worst.score:.4f}）"
        return False, f"不优于池内最差（{cand.score:.4f} <= {worst.score:.4f}）"

    def best(self) -> Optional[Candidate]:
        return max(self.items, key=lambda c: c.score) if self.items else None

    def worst(self) -> Optional[Candidate]:
        return min(self.items, key=lambda c: c.score) if self.items else None

    def sorted_items(self) -> List[Candidate]:
        return sorted(self.items, key=lambda c: -c.score)

    def profile_matched_items(self) -> List[Candidate]:
        """用于 prompt 高分组：优先取标签匹配当前目标的结构。"""
        matched = [c for c in self.items if "seed" in c.source and c.tags and _tags_match(c.tags, self.label)]
        return sorted(matched, key=lambda c: -c.score)

    def summary(self) -> dict:
        scores = [c.score for c in self.items]
        return {
            "label": self.label,
            "size": len(self.items),
            "best": round(max(scores), 6) if scores else None,
            "mean": round(float(sum(scores) / len(scores)), 6) if scores else None,
            "worst": round(min(scores), 6) if scores else None,
        }


def _tags_match(tags: str, profile: str) -> bool:
    from seed_loader import PROFILE_TAGS, _profile_tag_union
    tag_set = {t.strip() for t in tags.split("|") if t.strip()}
    union = set(_profile_tag_union(profile))
    return bool(tag_set & union)
