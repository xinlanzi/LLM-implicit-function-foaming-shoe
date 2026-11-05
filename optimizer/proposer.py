# -*- coding: utf-8 -*-
"""
提议器（Proposer）：负责“根据 prompt / 历史样本生成新 token 组合”。

三种实现
--------
1. MockLLMProposer
   本地“上下文感知”模拟器：读取 prompt 里的高分样本与随机样本，做系数扰动、
   基函数替换、交叉组合；**不调用评价器**。它的作用是在没有 LLM API 凭证/网络时
   让整条优化流水线真实跑通，并作为“会利用历史信息”的提议器与随机基线对照。
   它不是真实大模型，报告中必须明确标注为模拟器。

2. OpenAICompatibleProposer
   任意 OpenAI-compatible /chat/completions 服务（豆包、GPT、Claude 代理、本地 ollama 等）。
   通过环境变量 LLM_API_KEY + Config.llm_base_url/llm_model 接入；temperature=0.7。

3. RandomProposer
   技术文档要求的随机基线：从允许 token 中随机抽 1~3 个，系数 U[-2,2]。

所有提议器只返回文本 JSON，解析与合法性检查在 optimizer 中统一完成。
"""

from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from token_library import (
    TOKEN_BY_NAME,
    canonical_signature,
    get_allowed_tokens,
)


# --------------------------------------------------------------------------- #
# 小工具：token 轴、族、邻居
# --------------------------------------------------------------------------- #
_AXES = ("x", "y", "z")

_FAMILY_NEIGHBORS = {
    "single": {"single", "harmonic"},
    "chiral": {"chiral", "diagonal", "triple", "shear"},
    "diagonal": {"diagonal", "chiral", "triple", "shear"},
    "shear": {"shear", "diagonal", "chiral"},
    "triple": {"triple", "diagonal", "chiral"},
    "harmonic": {"harmonic", "single", "hybrid"},
    "harmonic_diag": {"harmonic_diag", "harmonic_triple", "diagonal"},
    "harmonic_shear": {"harmonic_shear", "shear", "harmonic_diag"},
    "harmonic_triple": {"harmonic_triple", "harmonic_diag", "triple"},
    "hybrid": {"hybrid", "harmonic", "triple"},
}

# 领域模板：真实 LLM 从种子库/示例中会联想到这些经典 TPMS 组合。
# 模拟器只做系数扰动/少量项替换，不调用评价器；模板本身来自公开 TPMS 文献与 Excel 种子。
TEMPLATE_SPECS = [
    ("P_surface", ["cos_x", "cos_y", "cos_z"], [1.0, 1.0, 1.0]),
    ("Gyroid", ["sin_x_cos_y", "sin_y_cos_z", "sin_z_cos_x"], [1.0, 1.0, 1.0]),
    ("Diamond", ["cos_x_cos_y_cos_z", "sin_x_sin_y_sin_z"], [1.0, -1.0]),
    ("Diamond_plus_x", ["cos_x_cos_y_cos_z", "sin_x_sin_y_sin_z", "cos_x"], [1.0, -1.0, 0.20]),
    ("Diamond_plus_y", ["cos_x_cos_y_cos_z", "sin_x_sin_y_sin_z", "cos_y"], [1.0, -1.0, 0.30]),
    ("Gyroid_Diamond_mix", ["sin_x_cos_y", "sin_y_cos_z", "cos_x_cos_y_cos_z"], [1.0, 1.0, 0.20]),
    ("Harmonic_D", ["cos_2x_cos_2y_cos_2z", "sin_2x_sin_2y_sin_2z"], [1.0, -1.0]),
    ("Neovius_like", ["cos_x", "cos_y", "cos_x_cos_y_cos_z"], [1.0, 1.0, 1.0]),
]


def term_axes(term: str) -> Set[str]:
    axes = set()
    for a in _AXES:
        if f"_{a}" in term:
            axes.add(a)
    return axes


def token_family(term: str) -> str:
    return TOKEN_BY_NAME.get(term, {}).get("family", "unknown")


def token_neighbors(term: str, allowed: Sequence[str]) -> List[str]:
    fam = token_family(term)
    fams = _FAMILY_NEIGHBORS.get(fam, {fam})
    return [t for t in allowed if t != term and token_family(t) in fams]


def axis_diversity(terms: Sequence[str]) -> int:
    axes = set()
    for t in terms:
        axes |= term_axes(t)
    return len(axes)


def _clamp_coeff(c: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, c)))


def _non_tiny(c: float, rng: np.random.Generator) -> float:
    """避免系数接近 0（近乎删项，容易产生退化几何）。"""
    if abs(c) < 0.12:
        return float(rng.choice([-1, 1]) * rng.uniform(0.25, 1.0))
    return float(c)


def _trim_terms(terms: List[str], coeffs: List[float], max_terms: int) -> Tuple[List[str], List[float]]:
    """把超过 max_terms 的组合裁剪到上限：优先保留 |系数| 大的项，并补充轴多样性。"""
    if len(terms) <= max_terms:
        return terms, coeffs
    order = sorted(range(len(terms)), key=lambda i: -abs(coeffs[i]))
    chosen: List[int] = []
    axes = set()
    # 先保留能扩大轴覆盖的项，再按 |coeff| 填充
    for i in order:
        if len(chosen) >= max_terms:
            break
        if term_axes(terms[i]) - axes or not chosen:
            chosen.append(i)
            axes |= term_axes(terms[i])
    for i in order:
        if len(chosen) >= max_terms:
            break
        if i not in chosen:
            chosen.append(i)
    chosen = sorted(chosen)
    return [terms[i] for i in chosen], [coeffs[i] for i in chosen]


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def parse_llm_output(text: str) -> dict:
    """
    技术文档模块 3 的解析：
    - 支持纯 JSON、```json ... ``` 代码块、以及 JSON 前后夹带解释文字；
    - 只取第一个完整的 JSON 对象。
    """
    if text is None:
        raise ValueError("空输出")
    s = str(text).strip()
    if not s:
        raise ValueError("空输出")
    # 去掉 markdown 代码围栏
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    start = s.find("{")
    if start < 0:
        raise ValueError("输出中没有 JSON 对象")
    depth = 0
    end = -1
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("JSON 对象不完整")
    obj = json.loads(s[start:end])
    if not isinstance(obj, dict):
        raise ValueError("JSON 顶层不是对象")
    if "terms" not in obj or "coeffs" not in obj:
        raise ValueError("JSON 缺少 terms/coeffs")
    terms = obj["terms"]
    coeffs = obj["coeffs"]
    if not isinstance(terms, list) or not isinstance(coeffs, list):
        raise ValueError("terms/coeffs 不是列表")
    return {"terms": terms, "coeffs": coeffs}


# --------------------------------------------------------------------------- #
# Mock LLM 提议器
# --------------------------------------------------------------------------- #
@dataclass
class Proposal:
    text: str
    provenance: dict


class BaseProposer:
    def __init__(self, config, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng
        self.allowed = get_allowed_tokens(config.token_set)

    def propose(self, prompt: str, examples: List[dict], profile: str,
                round_num: int, seen_signatures: Set[str]) -> Proposal:
        raise NotImplementedError


class MockLLMProposer(BaseProposer):
    """
    本地模拟“上下文感知提议器”。

    行为逻辑（仅使用 prompt 中可见的样本与分数，不查询评价器）：
      - 70% exploit：基于最高分样本，做系数扰动 / 基函数替换 / 增加一项；
      - 20% crossover：把两个高分样本的项交叉组合，裁剪到 ≤3 项；
      - 10% explore：随机探索，但优先选择能补足 x/y/z 轴多样性的模板。

    真实 LLM 会看到同样的 prompt；这里用显式启发式模拟“利用上下文”的能力。
    """

    def propose(self, prompt: str, examples: List[dict], profile: str,
                round_num: int, seen_signatures: Set[str]) -> Proposal:
        for attempt in range(self.cfg.max_attempts):
            op = self._choose_op()
            if op == "exploit":
                terms, coeffs, prov = self._exploit(examples)
            elif op == "crossover":
                terms, coeffs, prov = self._crossover(examples)
            elif op == "template":
                terms, coeffs, prov = self._template()
            else:
                terms, coeffs, prov = self._explore()
            terms, coeffs = _trim_terms(terms, coeffs, self.cfg.max_terms)
            coeffs = [_non_tiny(_clamp_coeff(c, self.cfg.coeff_min, self.cfg.coeff_max), self.rng) for c in coeffs]
            if not terms or len(terms) > self.cfg.max_terms:
                continue
            if len(set(terms)) != len(terms):
                # 去重同一 token（合并系数）
                merged: Dict[str, float] = {}
                for t, c in zip(terms, coeffs):
                    merged[t] = merged.get(t, 0.0) + c
                terms, coeffs = list(merged.keys()), list(merged.values())
                if len(terms) > self.cfg.max_terms:
                    continue
            sig = canonical_signature(terms, coeffs)
            if not sig or sig in seen_signatures:
                continue
            prov.update({"round": round_num, "profile": profile, "attempt": attempt + 1})
            return Proposal(text=json.dumps({"terms": terms, "coeffs": [round(c, 4) for c in coeffs]},
                                            ensure_ascii=False),
                            provenance=prov)
        # 极端情况下回退到随机，保证循环不中断
        terms, coeffs = self._random_candidate()
        return Proposal(text=json.dumps({"terms": terms, "coeffs": [round(c, 4) for c in coeffs]},
                                        ensure_ascii=False),
                        provenance={"operation": "fallback_random", "round": round_num})

    # ------------------------------------------------------------------ 策略
    def _choose_op(self) -> str:
        r = float(self.rng.random())
        if r < 0.60:
            return "exploit"
        if r < 0.78:
            return "crossover"
        if r < 0.95:
            return "template"
        return "explore"

    def _template(self) -> Tuple[List[str], List[float], dict]:
        """从经典 TPMS 模板出发做小幅系数扰动（模拟 LLM 的结构先验）。"""
        available = [
            (name, terms, coeffs) for (name, terms, coeffs) in TEMPLATE_SPECS
            if all(t in self.allowed for t in terms)
        ]
        if not available:
            return self._explore()
        name, base_terms, base_coeffs = available[int(self.rng.integers(0, len(available)))]
        terms = list(base_terms)
        # 每个系数独立扰动，避免整体缩放被去重规范化吃掉
        coeffs = [float(c) * float(self.rng.uniform(0.90, 1.10)) for c in base_coeffs]
        detail = name
        if self.rng.random() < 0.25:
            target = int(self.rng.integers(0, len(coeffs)))
            coeffs[target] = -coeffs[target]
            detail += f"+flip[{target}]"
        return terms, coeffs, {"operation": "template", "detail": detail}

    def _exploit(self, examples: List[dict]) -> Tuple[List[str], List[float], dict]:
        if not examples:
            return self._explore()
        ex = max(examples, key=lambda e: e.get("score", 0.0) if e.get("score") is not None else -1)
        terms = list(ex.get("terms", []))
        coeffs = [float(c) for c in ex.get("coeffs", [])]
        if not terms:
            return self._explore()
        parent = ex.get("sid") or ex.get("source", "?")
        op_detail = "coefficient_jitter"

        action = self.rng.choice(["jitter", "replace", "add", "flip"], p=[0.45, 0.25, 0.20, 0.10])
        if action == "jitter":
            target = int(self.rng.integers(0, len(coeffs)))
            factor = float(self.rng.uniform(0.65, 1.35))
            coeffs[target] = coeffs[target] * factor
            op_detail = f"coeff_jitter[{target}]"
        elif action == "replace":
            target = int(self.rng.integers(0, len(terms)))
            neigh = token_neighbors(terms[target], self.allowed)
            if neigh:
                new = str(self.rng.choice(neigh))
                # 保持系数大小，随机换符号的概率较低
                terms[target] = new
                op_detail = f"replace[{target}]:{new}"
        elif action == "add" and len(terms) < self.cfg.max_terms:
            # 优先补足缺失轴，避免 1D/2D 退化
            missing_axes = set(_AXES) - set().union(*[term_axes(t) for t in terms]) if terms else set(_AXES)
            candidates = [t for t in self.allowed if term_axes(t) & missing_axes]
            if not candidates:
                candidates = token_neighbors(terms[0], self.allowed) or self.allowed
            new = str(self.rng.choice(candidates))
            new_c = float(self.rng.choice([-1, 1]) * self.rng.uniform(0.2, 0.9))
            terms.append(new)
            coeffs.append(new_c)
            op_detail = f"add:{new}"
        elif action == "flip":
            target = int(self.rng.integers(0, len(coeffs)))
            coeffs[target] = -coeffs[target]
            op_detail = f"sign_flip[{target}]"

        # 如果三项来自同一族的 x/y/z 对称模板，偶尔把系数幅值对齐以提高各向同性
        if self.rng.random() < 0.35:
            fams = {token_family(t) for t in terms}
            axes = set().union(*[term_axes(t) for t in terms])
            if len(fams) == 1 and len(axes) == 3:
                mag = float(np.mean([abs(c) for c in coeffs]))
                signs = [1 if c >= 0 else -1 for c in coeffs]
                coeffs = [s * mag for s in signs]
                op_detail += "+symmetrize"

        return terms, coeffs, {"operation": "exploit", "detail": op_detail, "parents": [str(parent)]}

    def _crossover(self, examples: List[dict]) -> Tuple[List[str], List[float], dict]:
        if len(examples) < 2:
            return self._exploit(examples)
        # 按分数加权抽两个不同样本
        scores = np.array([max(0.01, float(e.get("score", 0.01)) ) for e in examples], dtype=float)
        p = scores / scores.sum()
        idx = self.rng.choice(len(examples), size=min(2, len(examples)), replace=False, p=p)
        a, b = examples[int(idx[0])], examples[int(idx[1])]
        terms: List[str] = []
        coeffs: List[float] = []
        for ex in (a, b):
            for t, c in zip(ex.get("terms", []), ex.get("coeffs", [])):
                if t not in terms:
                    terms.append(t)
                    coeffs.append(float(c) * float(self.rng.uniform(0.8, 1.25)))
                elif self.rng.random() < 0.5:
                    coeffs[terms.index(t)] += float(c) * 0.5
        # 裁剪并补足轴多样性
        terms, coeffs = _trim_terms(terms, coeffs, self.cfg.max_terms)
        if len(terms) < self.cfg.max_terms and self.rng.random() < 0.5:
            neigh = token_neighbors(terms[0], self.allowed)
            if neigh:
                new = str(self.rng.choice(neigh))
                if new not in terms:
                    terms.append(new)
                    coeffs.append(float(self.rng.choice([-1, 1]) * self.rng.uniform(0.2, 0.8)))
        parents = [str(a.get("sid") or a.get("source", "?")), str(b.get("sid") or b.get("source", "?"))]
        return terms, coeffs, {"operation": "crossover", "parents": parents}

    def _explore(self) -> Tuple[List[str], List[float], dict]:
        """随机探索，但优先构造 3D 模板而不是 1D 退化结构。"""
        n = int(self.rng.integers(2, self.cfg.max_terms + 1))
        # 选一个族模板，保证三项覆盖 x/y/z 或属于三轴族
        fam_pref = str(self.rng.choice(["single", "chiral", "diagonal", "triple", "harmonic_diag", "harmonic_triple"]))
        pool = [t for t in self.allowed if token_family(t) == fam_pref]
        if not pool:
            pool = list(self.allowed)
        terms: List[str] = []
        axes = set()
        for _ in range(n * 3):
            cand = str(self.rng.choice(pool))
            if cand in terms:
                continue
            if term_axes(cand) - axes or len(terms) < 2:
                terms.append(cand)
                axes |= term_axes(cand)
            if len(terms) >= n:
                break
        if not terms:
            terms = [str(self.rng.choice(self.allowed))]
        # 系数等幅值/随机符号，保持较好对称性
        base = float(self.rng.uniform(0.4, 1.8))
        coeffs = [float(self.rng.choice([-1, 1]) * base * self.rng.uniform(0.8, 1.2)) for _ in terms]
        return terms, coeffs, {"operation": "explore", "detail": f"family={fam_pref}"}

    def _random_candidate(self) -> Tuple[List[str], List[float]]:
        n = int(self.rng.integers(1, self.cfg.max_terms + 1))
        terms = list(self.rng.choice(self.allowed, size=n, replace=False))
        coeffs = [float(self.rng.uniform(self.cfg.coeff_min, self.cfg.coeff_max)) for _ in terms]
        return terms, coeffs


# --------------------------------------------------------------------------- #
# 随机基线
# --------------------------------------------------------------------------- #
class RandomProposer(BaseProposer):
    """技术文档模块 7 的随机基线：不看任何历史，直接随机组合。"""

    def propose(self, prompt: str, examples: List[dict], profile: str,
                round_num: int, seen_signatures: Set[str]) -> Proposal:
        for attempt in range(self.cfg.max_attempts):
            n = int(self.rng.integers(1, self.cfg.max_terms + 1))
            terms = list(self.rng.choice(self.allowed, size=n, replace=False))
            coeffs = [float(self.rng.uniform(self.cfg.coeff_min, self.cfg.coeff_max)) for _ in terms]
            sig = canonical_signature(terms, coeffs)
            if not sig or sig in seen_signatures:
                continue
            return Proposal(text=json.dumps({"terms": terms, "coeffs": [round(c, 4) for c in coeffs]},
                                            ensure_ascii=False),
                            provenance={"operation": "random", "round": round_num, "attempt": attempt + 1})
        terms = [str(self.rng.choice(self.allowed))]
        return Proposal(text=json.dumps({"terms": terms, "coeffs": [1.0]}, ensure_ascii=False),
                        provenance={"operation": "random_fallback", "round": round_num})


# --------------------------------------------------------------------------- #
# 真实 LLM（OpenAI-compatible）
# --------------------------------------------------------------------------- #
class OpenAICompatibleProposer(BaseProposer):
    """
    通过标准 /chat/completions 接口接入真实 LLM。

    运行前设置：
        set LLM_API_KEY=...
        set LLM_BASE_URL=https://api.openai.com/v1   （或豆包/ollama 的兼容地址）
        set LLM_MODEL=gpt-4o-mini
    并在 Config 中把 proposer_backend 设为 "openai"。
    """

    def _call(self, prompt: str) -> str:
        key = os.environ.get(self.cfg.llm_api_key_env, "")
        if not key:
            raise RuntimeError(
                f"未设置环境变量 {self.cfg.llm_api_key_env}，无法调用真实 LLM；"
                f"请改用 proposer_backend='mock' 或配置 API 凭证。"
            )
        url = self.cfg.llm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.llm_model,
            "temperature": self.cfg.temperature,
            "messages": [
                {"role": "system", "content": "你是严格的 JSON 输出器，只输出一个 JSON 对象。"},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def propose(self, prompt: str, examples: List[dict], profile: str,
                round_num: int, seen_signatures: Set[str]) -> Proposal:
        last_err = None
        for attempt in range(2):
            try:
                text = self._call(prompt)
                obj = parse_llm_output(text)
                terms = [str(t) for t in obj["terms"]]
                coeffs = [float(c) for c in obj["coeffs"]]
                sig = canonical_signature(terms, coeffs)
                if sig and sig not in seen_signatures:
                    return Proposal(text=text, provenance={
                        "operation": "llm_api", "model": self.cfg.llm_model,
                        "round": round_num, "attempt": attempt + 1,
                    })
            except Exception as exc:  # 网络/解析异常都记录，继续重试
                last_err = exc
        raise RuntimeError(f"LLM 调用或解析失败：{last_err}")


def make_proposer(config, arm: str, seed_offset: int = 0):
    """
    arm = "llm" -> Mock / 真实 LLM；arm = "random" -> RandomProposer。
    随机流独立：LLM 与随机基线不会因为共享 rng 而产生相关性。
    """
    rng = np.random.default_rng(config.seed + seed_offset)
    if arm == "random":
        return RandomProposer(config, rng)
    if config.proposer_backend == "openai":
        return OpenAICompatibleProposer(config, rng)
    return MockLLMProposer(config, rng)
