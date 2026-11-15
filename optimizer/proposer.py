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
   固定调用 SiliconFlow 的 OpenAI-compatible /chat/completions 接口。
   密钥通过 tkinter 安全弹窗输入，只问一次并缓存在进程内存；**不写盘、不读环境变量**。

3. RandomProposer
   技术文档要求的随机基线：从允许 token 中随机抽 1~3 个，系数 U[-2,2]。

所有提议器只返回文本 JSON，解析与合法性检查在 optimizer 中统一完成。
"""

from __future__ import annotations

import http.client
import json
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit

import numpy as np

try:  # 兼容包导入与脚本导入
    from .token_library import (
        TOKEN_BY_NAME,
        canonical_signature,
        get_allowed_tokens,
    )
    from .config import LLM_PROVIDERS
except ImportError:
    from token_library import (
        TOKEN_BY_NAME,
        canonical_signature,
        get_allowed_tokens,
    )
    from config import LLM_PROVIDERS

LLM_PROVIDER_NAMES = frozenset(k for k in LLM_PROVIDERS if k != "mock")


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
    # 去掉 markdown 代码围栏（```json ... ``` 或 ``` ... ```）
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    # 少数服务会把整个 JSON 再包一层字符串："{\"terms\": ...}"，先解一层
    if s.startswith('"') and s.endswith('"'):
        try:
            inner = json.loads(s)
            if isinstance(inner, str):
                s = inner.strip()
        except Exception:
            pass
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
    segment = s[start:end]
    try:
        obj = json.loads(segment)
    except json.JSONDecodeError:
        # 容错：模型偶尔把引号/换行二次转义成 \" 或 \\n 字面量
        obj = json.loads(segment.replace('\\"', '"').replace("\\n", " "))
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
            if not terms or len(terms) > self.cfg.max_terms:
                continue
            if len(set(terms)) != len(terms):
                # 先合并同名项，再统一 clamp；否则合并后系数可能超出上限被整轮丢弃
                merged: Dict[str, float] = {}
                for t, c in zip(terms, coeffs):
                    merged[t] = merged.get(t, 0.0) + c
                terms, coeffs = list(merged.keys()), list(merged.values())
                if len(terms) > self.cfg.max_terms:
                    continue
            coeffs = [_non_tiny(_clamp_coeff(c, self.cfg.coeff_min, self.cfg.coeff_max), self.rng)
                      for c in coeffs]
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
# 凭证：安全弹窗 / 进程内缓存 / 打码
# --------------------------------------------------------------------------- #
class CredentialError(RuntimeError):
    """无法创建弹窗，或用户取消、未输入密钥。"""

    def __init__(self, msg: str):
        super().__init__(msg)
        self.status = "credential_error"


# 弹窗输入的 key 只问一次，只存在内存，不落盘。
_KEY_CACHE: Dict[str, str] = {"value": ""}
# --api-key-stdin：无桌面环境（SSH/CI）时的显式备选；不读环境变量、不进 shell 历史
_STDIN_KEY_ENABLED: Dict[str, bool] = {"value": False}


def enable_stdin_key() -> None:
    """显式开启“从标准输入读取一次密钥”（仅由 --api-key-stdin 调用）。"""
    _STDIN_KEY_ENABLED["value"] = True


def _read_key_from_stdin() -> str:
    import sys
    key = (sys.stdin.readline() or "").strip()
    if not key:
        raise CredentialError("--api-key-stdin 未从标准输入读到密钥，已中止。")
    return key


def mask_secret(value: str) -> str:
    """凭证打码：只保留前 4 位与后 4 位，避免日志/终端泄露完整 key。"""
    if not value:
        return "(未设置)"
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}****{value[-4:]}"


def scrub_secrets(text: str) -> str:
    """把文本中可能出现的完整 key 替换为打码形式（用于异常信息与原始响应落盘）。"""
    if not text:
        return text
    cand = _KEY_CACHE.get("value", "")
    if cand and len(cand) >= 8 and cand in text:
        text = text.replace(cand, mask_secret(cand))
    return text


def estimate_tokens(text: str) -> int:
    """
    粗略估计 token 数（用于成本预估与上游不返回 usage 时的退化统计）。

    口径：ASCII 按 4 字符/token，非 ASCII（中文等）按 1.6 字符/token。
    这是经验值，英文/代码偏准、中文略保守；精确值以服务端 usage 为准。
    """
    if not text:
        return 0
    ascii_n = sum(1 for ch in text if ord(ch) < 128)
    other_n = len(text) - ascii_n
    return max(1, int(ascii_n / 4 + other_n / 1.6 + 0.5))


def extract_usage(data: dict, prompt: str, content: str) -> dict:
    """
    从响应 JSON 读 usage；缺失时退化为字符数估算。

    返回 {"prompt_tokens","completion_tokens","total_tokens","usage_source"}，
    usage_source 为 "reported"（服务端返回）或 "estimated"（本地估算）。
    """
    u = data.get("usage") if isinstance(data, dict) else None
    if isinstance(u, dict) and isinstance(u.get("total_tokens"), (int, float)):
        p = int(u.get("prompt_tokens") or 0)
        c = int(u.get("completion_tokens") or 0)
        return {"prompt_tokens": p, "completion_tokens": c,
                "total_tokens": int(u.get("total_tokens") or (p + c)),
                "usage_source": "reported"}
    p = estimate_tokens(prompt)
    c = estimate_tokens(content)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
            "usage_source": "estimated"}


def _ask_key(label: str = "SiliconFlow API 密钥") -> str:
    """用密码遮蔽弹窗读取密钥；取消、空输入或无法创建弹窗时安全中止。"""
    try:
        import tkinter as tk
        from tkinter import simpledialog
    except ImportError as exc:
        raise CredentialError("当前 Python 缺少 tkinter，无法创建密钥输入弹窗。") from exc

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        key = simpledialog.askstring(
            "LLM 密钥输入",
            f"请输入{label}：\n\n密钥仅以掩码显示，只保存在本进程内存，\n"
            "不会写入文件、环境变量或日志。",
            show="*",
            parent=root,
        )
    except tk.TclError as exc:
        raise CredentialError("无法创建密钥输入弹窗，请确认当前桌面环境可用。") from exc
    finally:
        if root is not None:
            root.destroy()

    key = (key or "").strip()
    if not key:
        raise CredentialError("未输入密钥或已取消弹窗，已中止。")
    return key


def resolve_api_key(cfg) -> Tuple[str, str]:
    """从安全弹窗（或显式 stdin）获取密钥并缓存于进程内存，返回 (key, 来源说明)。"""
    cached = _KEY_CACHE.get("value", "")
    if cached:
        return cached, "内存缓存（本进程此前已输入）"

    if _STDIN_KEY_ENABLED["value"]:
        key = _read_key_from_stdin()
        _KEY_CACHE["value"] = key
        return key, "--api-key-stdin 标准输入（仅进程内缓存）"

    key = _ask_key()
    _KEY_CACHE["value"] = key
    return key, "安全弹窗输入（仅进程内缓存，后续轮次不再询问）"


# --------------------------------------------------------------------------- #
# 真实 LLM（固定 SiliconFlow，OpenAI-compatible）
# --------------------------------------------------------------------------- #
class LLMAuthError(RuntimeError):
    """401/403：密钥错误或无权限，重试没有意义。"""


class LLMFatalError(RuntimeError):
    """400/404：请求体或端点/模型名错误，重试没有意义。"""


class LLMRetryableError(RuntimeError):
    """429 / 5xx / 超时 / 网络 / 解析失败：可指数退避重试。"""


class LLMEmptyContentError(LLMRetryableError):
    """
    HTTP 200 但 choices[0].message.content 为空。

    典型原因：
      - 推理模型把输出额度全用在思维链上（finish_reason="length"），content 尚未开始；
      - 服务端把答案放在 reasoning_content 而不是 content；
      - 内容被安全策略清空。
    带上 finish_reason / reasoning_content / 原始响应，便于上层给出可定位的 status。
    """

    def __init__(self, msg: str, finish_reason: Optional[str] = None,
                 reasoning_content: str = "", raw: str = ""):
        super().__init__(msg)
        self.finish_reason = finish_reason
        self.reasoning_content = reasoning_content
        self.raw = raw
        # 被 max_tokens 截断与"纯粹空响应"分开标注
        self.status = "truncated" if finish_reason == "length" else "empty_content"


def build_chat_payload(cfg, prompt: str, stream: Optional[bool] = None,
                       max_tokens: Optional[int] = None,
                       include_usage: bool = True) -> dict:
    """
    构造 /chat/completions 请求体（正式调用与 --debug-one-call 共用，保证一致）。

    - stream：默认取 cfg.llm_stream（默认 True）。流式便于观测首字节/首 token/块间间隔；
    - include_usage：流式时加 `stream_options={"include_usage": true}`，让最后一个
      chunk 带上 usage；若上游不认该字段而返回 400，chat_once 会自动去掉它重发一次；
    - enable_thinking：只有显式开启时才发送，避免部分模型/服务直接 400。
    """
    use_stream = bool(getattr(cfg, "llm_stream", True)) if stream is None else bool(stream)
    mt = int(cfg.llm_max_tokens) if max_tokens is None else int(max_tokens)
    payload = {
        "model": cfg.llm_model,
        "temperature": cfg.temperature,
        "max_tokens": mt,
        "stream": use_stream,
        "messages": [
            {"role": "system", "content": "你是严格的 JSON 输出器，只输出一个 JSON 对象。"},
            {"role": "user", "content": prompt},
        ],
    }
    if use_stream and include_usage:
        payload["stream_options"] = {"include_usage": True}
    if cfg.proposer_backend == "siliconflow" and bool(cfg.llm_enable_thinking):
        payload["enable_thinking"] = True
    return payload


def _http_target(base_url: str) -> Tuple[str, str, int, str]:
    """把 base_url 拆成 (scheme, host, port, path)，支持 http:// 的本地假服务器。"""
    u = urlsplit(base_url.rstrip("/") + "/chat/completions")
    if not u.hostname:
        raise ValueError(f"base_url 不合法：{base_url!r}")
    scheme = (u.scheme or "https").lower()
    port = u.port or (443 if scheme == "https" else 80)
    path = u.path or "/chat/completions"
    if u.query:
        path += "?" + u.query
    return scheme, u.hostname, port, path


def _new_conn(scheme: str, host: str, port: int, timeout: float):
    if scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=timeout)
    return http.client.HTTPConnection(host, port, timeout=timeout)


def _result_skeleton(stream: bool, max_tokens: int) -> dict:
    """chat_once 的返回结构：所有字段固定存在，便于调试输出与上层映射。"""
    return {
        "ok": False, "stream": bool(stream), "max_tokens": int(max_tokens),
        "http_status": None, "elapsed": 0.0,
        "t_connect": None, "t_first_byte": None, "t_first_token": None,
        "chunks": 0, "content": "", "reasoning": "",
        "finish_reason": None, "usage": None, "usage_raw": None,
        "usage_reported": False, "usage_unavailable": False,
        "usage_unavailable_reason": None, "downgraded": False,
        "error_kind": None, "timeout_phase": None, "error": "",
        "raw": "", "body": "", "last_event": "", "message_keys": [],
        "source": "", "key_mask": "", "payload": {},
    }


def _note_timeout(out: dict, phase: str, t0: float) -> None:
    """把超时阶段、已等时长、已收 chunk 数与最后一段内容写进错误信息。"""
    out["error_kind"] = "timeout"
    out["timeout_phase"] = phase
    out["elapsed"] = time.perf_counter() - t0
    label = {"connect": "连接超时", "first_byte": "首字节超时",
             "idle": "块间空闲超时", "total": "总时长超时"}.get(phase, "超时")
    detail = [f"{label}：已等 {out['elapsed']:.2f}s", f"收到 {out['chunks']} 个 chunk"]
    if out.get("t_connect") is not None:
        detail.append(f"建连 {out['t_connect']:.2f}s")
    if out.get("t_first_byte") is not None:
        detail.append(f"首字节 {out['t_first_byte']:.2f}s")
    if out.get("content"):
        detail.append(f"最后 content 片段={out['content'][-60:]!r}")
    if out.get("reasoning"):
        detail.append(f"最后 reasoning 片段={out['reasoning'][-60:]!r}")
    out["error"] = "；".join(detail)


def _apply_usage(out: dict, prompt: str) -> None:
    """reported（服务端 usage）优先；缺失时按字符数估算并标记 usage_unavailable。"""
    u = out.get("usage_raw")
    if isinstance(u, dict) and isinstance(u.get("total_tokens"), (int, float)):
        p = int(u.get("prompt_tokens") or 0)
        c = int(u.get("completion_tokens") or 0)
        out["usage"] = {"prompt_tokens": p, "completion_tokens": c,
                        "total_tokens": int(u.get("total_tokens") or (p + c)),
                        "usage_source": "reported"}
        out["usage_reported"] = True
        return
    # 流式没拿到 usage（或上游不支持 include_usage）：用 content+reasoning 估算
    p = estimate_tokens(prompt)
    c = estimate_tokens(out.get("content", "") + out.get("reasoning", ""))
    out["usage"] = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                    "usage_source": "estimated"}
    if out.get("stream"):
        out["usage_unavailable"] = True
        out["usage_unavailable_reason"] = out.get("usage_unavailable_reason") or "no_usage_field"


def _read_sse(out: dict, resp, conn, t0: float, deadline: float,
              first_to: float, idle_to: float) -> None:
    """逐行解析 SSE：`data: {...}` 累加 delta，跳过 `data: [DONE]`。"""
    finish_reason = None
    saw_done = False
    raw_head: List[str] = []
    while True:
        remain = deadline - time.monotonic()
        if remain <= 0:
            out["finish_reason"] = finish_reason
            _note_timeout(out, "total", t0)
            return
        # 首个 chunk 之前用 first_token 超时，之后用块间空闲超时
        wait = first_to if out["chunks"] == 0 else idle_to
        try:
            if conn.sock is not None:
                conn.sock.settimeout(max(0.5, min(wait, remain)))
        except OSError:
            pass
        try:
            line = resp.readline()
        except socket.timeout:
            out["finish_reason"] = finish_reason
            _note_timeout(out, "first_byte" if out["chunks"] == 0 else "idle", t0)
            return
        except (http.client.HTTPException, OSError) as exc:
            out["finish_reason"] = finish_reason
            out["error_kind"] = "network_error"
            out["elapsed"] = time.perf_counter() - t0
            out["error"] = (f"流中途断连（{type(exc).__name__}: {exc}）："
                            f"已收 {out['chunks']} 个 chunk，content {len(out['content'])} 字符")
            return
        if not line:
            break       # 服务端关闭流
        if out["t_first_byte"] is None:
            out["t_first_byte"] = time.perf_counter() - t0
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not text or text.startswith(":") or not text.startswith("data:"):
            continue    # 空行 / SSE 注释 / event: 等非 data 行
        if len(raw_head) < 200:
            raw_head.append(text)
        out["last_event"] = text[:400]
        payload_s = text[5:].strip()
        if payload_s == "[DONE]":
            saw_done = True
            break
        try:
            obj = json.loads(payload_s)
        except Exception:
            continue    # 单行 JSON 不完整（极少见）：跳过，不中断整条流
        if not isinstance(obj, dict):
            continue
        u = obj.get("usage")
        if isinstance(u, dict) and u:
            out["usage_raw"] = u
        choices = obj.get("choices") or []
        if not choices:
            continue    # include_usage 的收尾 chunk：只有 usage，没有 choices
        choice = choices[0] or {}
        # 兼容 delta（标准流式）与 message（少数服务直接给 message）
        delta = choice.get("delta") or choice.get("message") or {}
        piece = delta.get("content")
        if isinstance(piece, list):
            piece = "".join(p.get("text", "") for p in piece if isinstance(p, dict))
        rpiece = delta.get("reasoning_content")
        if rpiece is None:
            rpiece = delta.get("reasoning")
        if isinstance(rpiece, list):
            rpiece = "".join(str(p) for p in rpiece)
        out["chunks"] += 1
        if piece:
            out["content"] += str(piece)
        if rpiece:
            out["reasoning"] += str(rpiece)
        if (piece or rpiece) and out["t_first_token"] is None:
            out["t_first_token"] = time.perf_counter() - t0
        fr = choice.get("finish_reason")
        if fr:
            finish_reason = fr
    out["finish_reason"] = finish_reason
    out["raw"] = scrub_secrets("\n".join(raw_head))[:4000]
    if not saw_done and finish_reason is None:
        # 服务端在既没有 [DONE] 也没有 finish_reason 的情况下断开 = 流被截断
        out["error_kind"] = "network_error"
        out["elapsed"] = time.perf_counter() - t0
        out["error"] = (f"流意外结束（未收到 [DONE]/finish_reason）："
                        f"已收 {out['chunks']} 个 chunk，content {len(out['content'])} 字符")


def _chat_http(cfg, prompt: str, key: str, stream: bool, max_tokens: int,
               include_usage: bool = True) -> dict:
    """
    发一次 HTTP 请求（流式或非流式），返回结构化 dict；**从不抛网络异常**，
    所有失败都写进 error_kind/error，供 chat_once 决定是否降级、_call 决定映射成哪种异常。

    分阶段超时（流式）：
        connect  —— 建连（cfg.llm_connect_timeout，默认 10s）
        first_byte —— 请求发出到第一个数据块（cfg.llm_first_token_timeout，默认 60s）
        idle     —— 数据块之间的最大空闲（cfg.llm_idle_timeout，默认 30s）
        total    —— 兜底总时长（cfg.llm_timeout，默认 180s）
    非流式只有 connect + total 两段。
    """
    out = _result_skeleton(stream, max_tokens)
    out["payload"] = build_chat_payload(cfg, prompt, stream=stream,
                                        max_tokens=max_tokens,
                                        include_usage=include_usage)
    try:
        scheme, host, port, path = _http_target(cfg.llm_base_url)
    except ValueError as exc:
        out["error_kind"] = "bad_response"
        out["error"] = str(exc)
        return out

    body = json.dumps(out["payload"], ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "Authorization": f"Bearer {key}",
    }
    connect_to = float(getattr(cfg, "llm_connect_timeout", 10.0))
    first_to = float(getattr(cfg, "llm_first_token_timeout", 60.0))
    idle_to = float(getattr(cfg, "llm_idle_timeout", 30.0))
    total_to = float(cfg.llm_timeout)
    deadline = time.monotonic() + total_to
    t0 = time.perf_counter()

    conn = _new_conn(scheme, host, port, connect_to)
    try:
        conn.connect()
    except socket.timeout:
        _note_timeout(out, "connect", t0)
        return out
    except (http.client.HTTPException, OSError) as exc:
        out["error_kind"] = "network_error"
        out["elapsed"] = time.perf_counter() - t0
        out["error"] = f"连接失败（{type(exc).__name__}: {exc}）"
        return out
    out["t_connect"] = time.perf_counter() - t0

    try:
        if conn.sock is not None:
            # 等响应头/首字节用首字节超时，而不是 connect+first 拼起来
            conn.sock.settimeout(max(1.0, min(first_to, total_to)))
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
    except socket.timeout:
        _note_timeout(out, "first_byte", t0)
        return out
    except (http.client.HTTPException, OSError) as exc:
        out["error_kind"] = "network_error"
        out["elapsed"] = time.perf_counter() - t0
        out["error"] = f"发送/等待响应失败（{type(exc).__name__}: {exc}）"
        return out

    status = int(getattr(resp, "status", 0) or 0)
    out["http_status"] = status
    ctype = (resp.getheader("Content-Type") or "").lower()

    if status != 200:
        raw_err = ""
        try:
            if conn.sock is not None:
                conn.sock.settimeout(max(1.0, min(idle_to, deadline - time.monotonic())))
            raw_err = resp.read(8000).decode("utf-8", errors="replace")
        except Exception:
            pass
        out["raw"] = scrub_secrets(raw_err)[:4000]
        out["body"] = scrub_secrets(raw_err)[:1200]
        out["error_kind"] = "http_error"
        out["elapsed"] = time.perf_counter() - t0
        out["error"] = f"HTTP {status}"
        return out

    is_sse = bool(stream) and ("text/event-stream" in ctype or not ctype)

    if not is_sse:
        # --no-stream，或服务端无视 stream 直接返回整包 JSON
        try:
            if conn.sock is not None:
                remain = max(1.0, deadline - time.monotonic())
                conn.sock.settimeout(remain)
            raw_bytes = resp.read()
        except socket.timeout:
            _note_timeout(out, "total", t0)
            return out
        except (http.client.HTTPException, OSError) as exc:
            out["error_kind"] = "network_error"
            out["elapsed"] = time.perf_counter() - t0
            out["error"] = f"读响应失败（{type(exc).__name__}: {exc}）：已收 {len(out['content'])} 字符"
            return out
        out["t_first_byte"] = time.perf_counter() - t0
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        out["raw"] = scrub_secrets(raw_text)[:4000]
        try:
            data = json.loads(raw_text)
        except Exception:
            out["error_kind"] = "bad_response"
            out["elapsed"] = time.perf_counter() - t0
            out["error"] = f"HTTP {status} 响应不是 JSON：{scrub_secrets(raw_text[:300])}"
            return out
        try:
            choice = (data.get("choices") or [])[0]
            msg = choice.get("message") or {}
        except Exception:
            out["error_kind"] = "bad_response"
            out["elapsed"] = time.perf_counter() - t0
            out["error"] = (f"HTTP {status} 响应不是标准 chat/completions 结构："
                            f"{scrub_secrets(raw_text[:300])}")
            return out
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        out["content"] = "" if content is None else str(content)
        out["reasoning"] = str(msg.get("reasoning_content") or "")
        out["finish_reason"] = choice.get("finish_reason")
        out["message_keys"] = list(msg.keys())
        out["chunks"] = 1 if (out["content"] or out["reasoning"]) else 0
        out["t_first_token"] = out["t_first_byte"] if out["chunks"] else None
        u = data.get("usage")
        if isinstance(u, dict) and u:
            out["usage_raw"] = u
        out["elapsed"] = time.perf_counter() - t0
        _apply_usage(out, prompt)
        out["ok"] = True
        return out

    _read_sse(out, resp, conn, t0, deadline, first_to, idle_to)
    out["elapsed"] = time.perf_counter() - t0
    _apply_usage(out, prompt)
    if out["error_kind"]:
        return out
    if out["chunks"] == 0:
        out["error_kind"] = "bad_response"
        out["error"] = ("流式响应里没有任何 data 事件（不是标准 SSE；"
                        f"Content-Type={ctype or '未给出'}）")
        return out
    out["ok"] = True
    return out


def chat_once(cfg, prompt: str, stream: Optional[bool] = None,
              max_tokens: Optional[int] = None,
              include_usage: bool = True) -> dict:
    """
    发一次真实请求并返回完整诊断结构（不做重试、不解析 JSON、不写盘）。

    流式 + include_usage 时，若上游因不认识 `stream_options` 返回 400，
    自动去掉该字段重发一次；成功后标记 usage_unavailable（用字符估算兜底）。
    """
    key, source = resolve_api_key(cfg)
    if stream is None:
        stream = bool(getattr(cfg, "llm_stream", True))
    if max_tokens is None:
        max_tokens = int(cfg.llm_max_tokens)
    stream = bool(stream)
    max_tokens = int(max_tokens)

    out = _chat_http(cfg, prompt, key, stream, max_tokens, include_usage)
    out["source"] = source
    out["key_mask"] = mask_secret(key)

    need_downgrade = (stream and include_usage and out.get("http_status") == 400
                      and "stream_options" in (out.get("payload") or {}))
    if not need_downgrade:
        return out
    first_err = out.get("error") or ""
    out2 = _chat_http(cfg, prompt, key, stream, max_tokens, include_usage=False)
    out2["downgraded"] = True
    out2["first_error"] = first_err
    out2["source"] = source
    out2["key_mask"] = mask_secret(key)
    if out2.get("http_status") == 200 and not out2.get("usage_reported"):
        out2["usage_unavailable"] = True
        out2["usage_unavailable_reason"] = "stream_options_removed"
    return out2


class OpenAICompatibleProposer(BaseProposer):
    """
    通过标准 /chat/completions 接入真实 LLM。

    真实接口固定为 https://api.siliconflow.cn/v1/chat/completions。
    SiliconFlow 特有的 enable_thinking 默认不发送，避免部分模型直接返回 400；
    推理模型的思维链可能占满 max_tokens 并导致 content 为空。
    """

    def _payload(self, prompt: str) -> dict:
        return build_chat_payload(self.cfg, prompt)

    def _call(self, prompt: str) -> Tuple[str, dict, dict]:
        """
        发一次请求（默认流式），返回 (content, usage, meta)。

        HTTP/网络/超时/空内容都映射成带 status 属性的异常，`status` 取值：
          http_error / network_error / timeout / bad_response / empty_content
        """
        res = chat_once(self.cfg, prompt)
        usage = res.get("usage") or {}
        meta = {
            "http_status": res.get("http_status"),
            "finish_reason": res.get("finish_reason"),
            "reasoning_content": res.get("reasoning", ""),
            "message_keys": res.get("message_keys") or [],
            "raw": res.get("raw", ""),
            "chunks": res.get("chunks", 0),
            "elapsed": res.get("elapsed", 0.0),
            "t_first_byte": res.get("t_first_byte"),
            "t_first_token": res.get("t_first_token"),
            "max_tokens": res.get("max_tokens"),
            "usage_unavailable": res.get("usage_unavailable", False),
            "downgraded": res.get("downgraded", False),
        }

        kind = res.get("error_kind")
        if kind:
            msg = res.get("error") or kind
            body = res.get("body") or ""
            status = res.get("http_status")
            if kind == "http_error":
                if status in (401, 403):
                    err = LLMAuthError(
                        f"HTTP {status} 认证失败：密钥无效/过期/无权限。原始响应：{body}")
                elif status in (400, 404):
                    err = LLMFatalError(
                        f"HTTP {status}：请求体参数/端点路径/模型名可能不对。原始响应：{body}")
                elif status == 429:
                    err = LLMRetryableError(f"HTTP 429 限流或余额不足：{body}")
                else:
                    err = LLMRetryableError(f"HTTP {status}：{body}")
                err.status = "http_error"
                err.http_status = status
                err.raw = res.get("raw", "")
            elif kind == "timeout":
                err = LLMRetryableError(msg)
                err.status = "timeout"
                err.timeout_phase = res.get("timeout_phase")
                err.http_status = status
                err.raw = res.get("raw", "")
            elif kind == "network_error":
                err = LLMRetryableError(msg)
                err.status = "network_error"
                err.http_status = status
                err.raw = res.get("raw", "")
            else:       # bad_response
                err = LLMRetryableError(msg)
                err.status = "bad_response"
                err.http_status = status
                err.raw = res.get("raw", "")
            err.chunks = res.get("chunks", 0)
            err.partial = {
                "chunks": res.get("chunks", 0),
                "content": res.get("content", "")[:300],
                "reasoning": res.get("reasoning", "")[:300],
                "elapsed": round(float(res.get("elapsed") or 0.0), 2),
                "timeout_phase": res.get("timeout_phase"),
            }
            err.usage = usage
            raise err from None

        text = res.get("content", "")
        reasoning = res.get("reasoning", "")
        finish_reason = res.get("finish_reason")
        if not text.strip():
            hint = (f"finish_reason={finish_reason!r}，max_tokens={res.get('max_tokens')}，"
                    f"chunks={res.get('chunks')}，reasoning_content 有 {len(reasoning)} 字符")
            raw_head = scrub_secrets(res.get("raw", ""))[:300]
            err = LLMEmptyContentError(
                f"HTTP {res.get('http_status')} 但 content 为空（{hint}）。原始响应前 300 字符："
                f"{raw_head}",
                finish_reason=finish_reason, reasoning_content=reasoning,
                raw=scrub_secrets(res.get("raw", "")))
            err.usage = usage
            raise err
        return text, usage, meta

    def propose(self, prompt: str, examples: List[dict], profile: str,
                round_num: int, seen_signatures: Set[str]) -> Proposal:
        retries = max(1, int(self.cfg.llm_max_retries))
        last_err: Optional[Exception] = None
        # 重试也会消耗 token：把每次尝试的 usage 累加，避免少记
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                       "usage_source": None, "requests": 0}
        sources = set()

        def _accumulate(usage: Optional[dict]) -> None:
            if not usage:
                return
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_total[k] += int(usage.get(k, 0))
            usage_total["requests"] += max(1, int(usage.get("requests") or 1))
            sources.add(usage.get("usage_source", "estimated"))

        attempts_made = 0
        for attempt in range(retries):
            attempts_made = attempt + 1
            meta: dict = {}
            try:
                text, usage, meta = self._call(prompt)
            except (LLMAuthError, LLMFatalError, LLMEmptyContentError, CredentialError) as exc:
                # 认证/端点错误、以及"content 为空/被截断"都不重试：
                # 同样的 prompt + max_tokens 再来一次只会再烧一遍 token。
                _accumulate(getattr(exc, "usage", None))
                raise
            except Exception as exc:        # 429/5xx/超时/解析失败：指数退避重试
                _accumulate(getattr(exc, "usage", None))
                last_err = exc
            else:
                _accumulate(usage)
                try:
                    obj = parse_llm_output(text)
                    terms = [str(t) for t in obj["terms"]]
                    coeffs = [float(c) for c in obj["coeffs"]]
                    sig = canonical_signature(terms, coeffs)
                    if sig and sig not in seen_signatures:
                        usage_total["usage_source"] = ("mixed" if len(sources) > 1
                                                       else next(iter(sources)))
                        return Proposal(text=text, provenance={
                            "operation": "llm_api", "provider": self.cfg.proposer_backend,
                            "model": self.cfg.llm_model, "round": round_num,
                            "attempt": attempt + 1, "usage": usage_total,
                        })
                    dup = ValueError("候选与已见签名重复")
                    dup.status = "duplicate"
                    last_err = dup
                    # 同一 prompt 重发很可能再次给出重复候选，直接结束重试以省 token
                    break
                except ValueError as exc:
                    # finish_reason=length 时解析失败，直接标为"被 max_tokens 截断"
                    exc.status = ("truncated" if meta.get("finish_reason") == "length"
                                  else "parse_failed")
                    exc.raw = text
                    exc.finish_reason = meta.get("finish_reason")
                    last_err = exc
                    if exc.status == "truncated":
                        break       # 同一 prompt 重试只会再被截断，白花钱
            if attempt < retries - 1:
                time.sleep(min(2.0 ** attempt, 8.0))

        status = getattr(last_err, "status", None) or (
            "http_error" if isinstance(last_err, LLMRetryableError) else "proposer_failed")
        err = RuntimeError(
            f"LLM 调用或解析失败（status={status}，已尝试 {attempts_made} 次）："
            f"{scrub_secrets(str(last_err))}")
        err.status = status
        err.raw = getattr(last_err, "raw", "") or ""
        err.http_status = getattr(last_err, "http_status", None)
        err.finish_reason = getattr(last_err, "finish_reason", None)
        err.reasoning_head = (getattr(last_err, "reasoning_content", "") or "")[:300]
        # 失败的调用也消耗了 token：把 usage 挂在异常上，由调用方计入运行级汇总
        usage_total["usage_source"] = ("mixed" if len(sources) > 1
                                       else (next(iter(sources)) if sources else "none"))
        err.usage = usage_total
        raise err


def check_llm_endpoint(cfg) -> Tuple[str, str]:
    """
    自检：复用正式请求路径（chat_once）验证凭证 / 端点 / 模型 / content 可用性。

    与正式运行使用同一个 max_tokens，避免推理模型思维链占满额度时出现
    “自检通过、正式运行 content 全空”的假阳性。

    返回 (状态码, 人类可读说明)，状态码用于进程退出码：
        ok / no_key / empty_content / auth / not_found / quota /
        timeout / network / server / bad_response
    说明文本中只出现打码后的 key，绝不打印完整凭证。
    """
    base = (f"provider: {cfg.proposer_backend}\n"
            f"端点   : {cfg.llm_base_url.rstrip('/')}/chat/completions\n"
            f"模型   : {cfg.llm_model}\n")

    try:
        key, source = resolve_api_key(cfg)
    except CredentialError as exc:
        return "no_key", base + f"凭证   : (未获取)\n✗ {exc}"

    head = base + (f"凭证   : {mask_secret(key)}（来源：{source}）\n"
                   f"max_tokens: {cfg.llm_max_tokens} | 超时: {cfg.llm_timeout}s\n")

    out = chat_once(cfg, "只回复 OK 两个字符。", stream=bool(cfg.llm_stream),
                    max_tokens=int(cfg.llm_max_tokens))
    usage = out.get("usage") or {}
    usage_line = (f"usage  : {int(usage.get('total_tokens') or 0):,} token"
                  f"（{usage.get('usage_source', 'none')}）\n")
    content = (out.get("content") or "").strip()
    reasoning = (out.get("reasoning") or "").strip()
    kind = out.get("error_kind")

    if not kind and content:
        return "ok", head + usage_line + "✓ 连通成功：凭证、端点、模型均可用，content 非空。"

    if not kind and not content:
        extra = "；已有思维链输出，说明额度被推理过程占满" if reasoning else ""
        return "empty_content", head + usage_line + (
            f"✗ HTTP 200 但 content 为空{extra}（finish_reason="
            f"{out.get('finish_reason')}）。\n"
            f"  处理：提高 --max-tokens（当前 {cfg.llm_max_tokens}），或改用非推理模型。")

    http_status = out.get("http_status")
    body = scrub_secrets(str(out.get("body") or out.get("error") or ""))[:400]

    if kind == "http_error":
        if http_status in (401, 403):
            return "auth", head + (f"✗ HTTP {http_status} 认证失败：密钥无效、已过期或无权限。\n"
                                   f"  处理：到平台重新生成密钥，确认没有多余空格/引号。\n  {body}")
        if http_status == 404:
            return "not_found", head + (f"✗ HTTP 404：端点路径或模型名不存在。\n"
                                        f"  处理：确认模型名拼写"
                                        f"（SiliconFlow 形如 deepseek-ai/DeepSeek-V4-Flash）。\n  {body}")
        if http_status == 429:
            return "quota", head + f"✗ HTTP 429 配额/频率限制：请充值或降低调用频率。\n  {body}"
        if isinstance(http_status, int) and 500 <= http_status < 600:
            return "server", head + f"✗ HTTP {http_status} 服务端错误：稍后重试或联系服务商。\n  {body}"
        return "server", head + f"✗ HTTP {http_status}：{body}"

    if kind == "timeout":
        phase = out.get("timeout_phase") or "-"
        label = {"connect": "连接超时", "first_byte": "首字节超时",
                 "idle": "块间空闲超时", "total": "总时长超时"}.get(phase, "超时")
        return "timeout", head + f"✗ {label}（{phase}）：网络不通或服务端无响应。"

    if kind == "network_error":
        return "network", head + f"✗ 网络不可达：{out.get('error') or '连接失败'}。\n" \
                                 f"  处理：检查代理/防火墙，或确认主机名可达。"

    return "bad_response", head + f"✗ 未预期的响应：{body or kind}"


def make_proposer(config, arm: str, seed_offset: int = 0):
    """
    arm = "llm" -> Mock / 真实 LLM；arm = "random" -> RandomProposer。
    随机流独立：LLM 与随机基线不会因为共享 rng 而产生相关性。
    """
    rng = np.random.default_rng(config.seed + seed_offset)
    if arm == "random":
        return RandomProposer(config, rng)
    if config.proposer_backend in LLM_PROVIDER_NAMES:
        return OpenAICompatibleProposer(config, rng)
    return MockLLMProposer(config, rng)
