# -*- coding: utf-8 -*-
"""
组合优化器主循环 —— 技术文档模块 2/3/4/5/6/7。

流程
----
1. 从 Excel 读取可优化的三角闭式种子，按目标 profile 初始化候选池；
2. 每轮：选高分组样本 + 1 个随机样本 → 构造 prompt → 提议器输出 JSON；
3. 解析 → 合法性检查 → 评价器算几何代理分 → 与池内最差比较 → 入池/拒绝；
4. 记录每轮结果；跑满 50 轮；
5. LLM 组与随机基线使用**完全相同的评价器、种子池、入池规则和轮数**。

关于“LLM”的说明
----------------
默认 proposer_backend="mock"，即本地上下文感知模拟器；它不使用评价器，
只根据 prompt 中的高分样本做变异/交叉。若配置真实 LLM API，则同一套流程可直接替换。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import Config
from evaluator import EvalMetrics, Evaluator
from population import Candidate, Population
from proposer import Proposal, make_proposer, parse_llm_output
from seed_loader import PROFILE_DESCRIPTIONS, PROFILE_TAGS, SeedStructure, _profile_tag_union
from token_library import TOKEN_BY_NAME, canonical_signature, get_allowed_tokens, token_roles


# --------------------------------------------------------------------------- #
# Prompt 构造（技术文档模块 2）
# --------------------------------------------------------------------------- #
def _format_example(idx: int, ex: dict, tag: str = "历史样本") -> str:
    terms = json.dumps(ex.get("terms", []), ensure_ascii=False)
    coeffs = json.dumps([round(float(c), 4) for c in ex.get("coeffs", [])], ensure_ascii=False)
    score = ex.get("score")
    score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
    tags = ex.get("tags", "") or ""
    sid = ex.get("sid", "") or ""
    head = f"{tag}{idx}"
    if sid:
        head += f"（{sid}）"
    return (
        f"{head}：\n"
        f"  terms = {terms}\n"
        f"  coeffs = {coeffs}\n"
        f"  score = {score_s}\n"
        f"  tags = {tags}"
    )


def build_prompt(profile: str, examples: List[dict], random_example: Optional[dict],
                 round_num: int, cfg: Config) -> str:
    allowed = get_allowed_tokens(cfg.token_set)
    weights = cfg.profile_weights[profile]

    # 按族列出 token，便于真实 LLM 理解每个基函数的作用
    family_lines = []
    seen_fams = []
    for fam in [TOKEN_BY_NAME[t]["family"] for t in allowed]:
        if fam not in seen_fams:
            seen_fams.append(fam)
    for fam in seen_fams:
        items = [t for t in allowed if TOKEN_BY_NAME[t]["family"] == fam]
        desc = "；".join(f"{t}={TOKEN_BY_NAME[t]['expr']}" for t in items)
        family_lines.append(f"- {fam}: {desc}")

    examples_text = "\n\n".join(_format_example(i + 1, ex) for i, ex in enumerate(examples))
    random_text = _format_example(1, random_example, "随机多样性样本") if random_example else "（无）"

    prompt = f"""你是一个 TPMS 结构设计专家。我们正在设计具有特定力学性能的隐式曲面结构。

目标性能标签：{profile}
这个标签对应的力学行为：{PROFILE_DESCRIPTIONS[profile]}

当前是第 {round_num} 轮。评分是几何代理指标（0~1，越高越好）：
- support 代理：相对密度、实体/空相周期性连通性、曲率平滑度、法向各向同性、比表面积窗口；
- elastic  代理：双连续连通性、曲率平滑/均匀、适中相对密度、法向各向同性、比表面积窗口；
- 最终 score = {weights['support']:.2f} * support + {weights['elastic']:.2f} * elastic。
注意：高支撑与高弹性存在权衡，不要只追求“越密越好”；同时避免 1D 层状/2D 柱状等退化结构。

下面是一些已有的、表现良好的结构：

{examples_text}

随机多样性样本（用于增加探索方向）：

{random_text}

允许使用的基函数（token 名 = 表达式）：
{chr(10).join(family_lines)}

请你基于这些结构，设计一个新的、可能具有更好性能的隐式曲面方程。
要求：
1. 只能使用上面列出的 token 名；
2. 最多组合 {cfg.max_terms} 个基函数；
3. 每个系数必须在 [{cfg.coeff_min}, {cfg.coeff_max}] 范围内；
4. 不要与上面样本完全相同；
5. 输出严格的 JSON 格式，不要有任何其他文字、不要解释过程：
{{
  "terms": ["基函数1", "基函数2"],
  "coeffs": [系数1, 系数2]
}}
"""
    return prompt


# --------------------------------------------------------------------------- #
# 合法性检查（技术文档模块 4）
# --------------------------------------------------------------------------- #
def validate_candidate(obj: dict, cfg: Config, allowed: List[str],
                       pool: Population, seen: Set[str]) -> Tuple[Optional[List[str]], Optional[List[float]], str, str]:
    """
    返回 (terms, coeffs, signature, reason)；失败时 terms/coeffs 为 None，reason 说明原因。
    """
    terms = obj.get("terms")
    coeffs = obj.get("coeffs")
    if not isinstance(terms, list) or not isinstance(coeffs, list):
        return None, None, "", "terms/coeffs 不是列表"
    if len(terms) == 0:
        return None, None, "", "terms 为空"
    if len(terms) != len(coeffs):
        return None, None, "", "terms 与 coeffs 长度不一致"
    if len(terms) > cfg.max_terms:
        return None, None, "", f"terms 数量 {len(terms)} 超过上限 {cfg.max_terms}"

    terms_clean: List[str] = []
    for t in terms:
        if not isinstance(t, str):
            return None, None, "", f"token 不是字符串：{t!r}"
        if t not in allowed:
            return None, None, "", f"token 不在允许词表：{t!r}"
        terms_clean.append(t)
    if len(set(terms_clean)) != len(terms_clean):
        return None, None, "", "terms 中存在重复 token"

    coeffs_clean: List[float] = []
    for c in coeffs:
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            return None, None, "", f"系数不是数字：{c!r}"
        c = float(c)
        if not math.isfinite(c):
            return None, None, "", f"系数不是有限数：{c!r}"
        if c < cfg.coeff_min or c > cfg.coeff_max:
            return None, None, "", f"系数 {c} 超出 [{cfg.coeff_min}, {cfg.coeff_max}]"
        if abs(c) < 1e-9:
            return None, None, "", "系数为 0（等价删项，视为无效）"
        coeffs_clean.append(c)

    sig = canonical_signature(terms_clean, coeffs_clean)
    if not sig:
        return None, None, "", "无法生成几何签名"
    if pool.contains_signature(sig):
        return None, None, sig, "与池内候选几何等价（去重）"
    if sig in seen:
        return None, None, sig, "本实验已生成过等价候选（去重）"
    return terms_clean, coeffs_clean, sig, "ok"


# --------------------------------------------------------------------------- #
# 样本选择（模块 2）
# --------------------------------------------------------------------------- #
def select_examples(pop: Population, profile: str, round_num: int, cfg: Config,
                    rng: np.random.Generator) -> Tuple[List[dict], Optional[dict]]:
    """
    前 10 轮：4 个高分样本 + 1 个随机样本（利用已有知识）；
    之后：2 个高分样本 + 2 个随机样本（增加探索）。

    高分组优先：先取标签匹配当前目标的种子，再取所有高分候选（含生成候选），
    避免“标签组”为空时无法构造 prompt。
    """
    n_high = 4 if round_num <= 10 else 2
    n_random = 1 if round_num <= 10 else 2

    tag_union = set(_profile_tag_union(profile))
    matched_seeds = [c for c in pop.items if c.source == "seed" and
                     bool({t.strip() for t in c.tags.split("|") if t.strip()} & tag_union)]
    matched_seeds.sort(key=lambda c: -c.score)
    generated = [c for c in pop.items if c.source != "seed"]
    generated.sort(key=lambda c: -c.score)
    other = [c for c in pop.items if c.source == "seed" and c not in matched_seeds]
    other.sort(key=lambda c: -c.score)

    ordered = matched_seeds + generated + other
    high = ordered[:n_high]

    remaining = [c for c in pop.items if c not in high]
    rng.shuffle(remaining)
    randoms = remaining[:n_random]

    def to_dict(c: Candidate) -> dict:
        return {
            "terms": list(c.terms), "coeffs": list(c.coeffs),
            "score": float(c.score) if c.score is not None else None,
            "tags": c.tags, "sid": c.sid, "source": c.source,
        }

    examples = [to_dict(c) for c in high]
    random_example = to_dict(randoms[0]) if randoms else (to_dict(remaining[0]) if remaining else None)
    return examples, random_example


# --------------------------------------------------------------------------- #
# 种子池初始化
# --------------------------------------------------------------------------- #
def init_population(seeds: List[SeedStructure], evaluator: Evaluator, cfg: Config,
                    profile: str) -> Tuple[Population, List[Candidate]]:
    pop = Population(label=profile, pool_size=cfg.pool_size)
    seed_candidates: List[Candidate] = []
    for seed in seeds:
        sig = canonical_signature(seed.terms, seed.coeffs)
        metrics = evaluator.evaluate(seed.terms, seed.coeffs, sig, profile)
        if not metrics.valid:
            continue
        cand = Candidate(
            label=profile, terms=list(seed.terms), coeffs=list(seed.coeffs),
            score=float(metrics.score), source="seed", round=0, accepted=True,
            reason="初始种子（按评价器实际打分）", signature=sig,
            metrics=metrics.to_dict(), sid=seed.sid,
            tags=" | ".join(seed.tags),
            provenance={"operation": "seed", "original_equation": seed.original_equation},
        )
        accepted, reason = pop.add(cand)
        cand.accepted = accepted
        cand.reason = "初始种子：" + reason
        seed_candidates.append(cand)
    return pop, seed_candidates


# --------------------------------------------------------------------------- #
# 单臂运行（LLM 或随机基线）
# --------------------------------------------------------------------------- #
def run_arm(seeds: List[SeedStructure], evaluator: Evaluator, cfg: Config,
            profile: str, arm: str, seed_offset: int = 0) -> dict:
    pop, seed_candidates = init_population(seeds, evaluator, cfg, profile)
    allowed = get_allowed_tokens(cfg.token_set)
    proposer = make_proposer(cfg, arm, seed_offset=seed_offset)
    prompt_rng = np.random.default_rng(cfg.seed + 500 + seed_offset)

    seen: Set[str] = set(pop.signatures)
    history: List[dict] = []
    all_candidates: List[Candidate] = list(seed_candidates)
    consecutive_reject = 0

    for round_num in range(1, cfg.rounds + 1):
        examples, random_example = select_examples(pop, profile, round_num, cfg, prompt_rng)
        prompt = build_prompt(profile, examples, random_example, round_num, cfg)

        # ---- 调用提议器 ----
        try:
            proposal: Proposal = proposer.propose(prompt, examples, profile, round_num, seen)
            raw_text = proposal.text
            provenance = proposal.provenance
        except Exception as exc:
            history.append({
                "round": round_num, "label": profile, "status": "proposer_failed",
                "score": None, "accepted": False, "reason": f"提议器失败：{exc}",
                "terms": [], "coeffs": [],
            })
            all_candidates.append(Candidate(
                label=profile, terms=[], coeffs=[], score=None, source=arm,
                round=round_num, accepted=False, reason=f"提议器失败：{exc}",
                signature="", status="proposer_failed",
            ))
            consecutive_reject += 1
            continue

        # ---- 解析 ----
        try:
            obj = parse_llm_output(raw_text)
        except Exception as exc:
            history.append({
                "round": round_num, "label": profile, "status": "parse_failed",
                "score": None, "accepted": False, "reason": f"JSON 解析失败：{exc}",
                "terms": [], "coeffs": [], "provenance": provenance,
            })
            all_candidates.append(Candidate(
                label=profile, terms=[], coeffs=[], score=None, source=arm,
                round=round_num, accepted=False, reason=f"JSON 解析失败：{exc}",
                signature="", status="parse_failed", provenance=provenance,
            ))
            consecutive_reject += 1
            continue

        # ---- 合法性检查 ----
        terms, coeffs, sig, reason = validate_candidate(obj, cfg, allowed, pop, seen)
        if terms is None:
            status = "duplicate" if "去重" in reason else "invalid"
            history.append({
                "round": round_num, "label": profile, "status": status,
                "score": None, "accepted": False, "reason": reason,
                "terms": obj.get("terms", []), "coeffs": obj.get("coeffs", []),
                "provenance": provenance,
            })
            all_candidates.append(Candidate(
                label=profile, terms=obj.get("terms", []) or [],
                coeffs=obj.get("coeffs", []) or [], score=None, source=arm,
                round=round_num, accepted=False, reason=reason,
                signature=sig, status=status, provenance=provenance,
            ))
            if sig:
                seen.add(sig)
            consecutive_reject += 1
            continue

        # ---- 评价器 ----
        metrics: EvalMetrics = evaluator.evaluate(terms, coeffs, sig, profile)
        if not metrics.valid:
            hist_score = None
            status = "invalid_geometry"
        else:
            hist_score = float(metrics.score)
            status = "evaluated"

        cand = Candidate(
            label=profile, terms=terms, coeffs=coeffs,
            score=hist_score, source=arm, round=round_num,
            accepted=False, reason=metrics.reason if not metrics.valid else "",
            signature=sig, metrics=metrics.to_dict(),
            provenance=provenance, status=status,
        )

        if metrics.valid:
            accepted, add_reason = pop.add(cand)
            cand.accepted = accepted
            cand.reason = add_reason
        else:
            accepted = False
            cand.accepted = False
            cand.reason = "评价器判定退化几何：" + metrics.reason

        all_candidates.append(cand)
        seen.add(sig)
        if accepted:
            consecutive_reject = 0
        else:
            consecutive_reject += 1

        best = pop.best()
        history.append({
            "round": round_num, "label": profile, "status": status,
            "score": hist_score, "accepted": accepted, "reason": cand.reason,
            "terms": terms, "coeffs": coeffs,
            "support_proxy": float(metrics.support_proxy),
            "elastic_proxy": float(metrics.elastic_proxy),
            "rho_solid": float(metrics.rho_solid),
            "connectivity": float(metrics.connectivity),
            "isotropy": float(metrics.isotropy),
            "curvature_mean": float(metrics.curvature_mean),
            "area_per_volume": float(metrics.area_per_volume),
            "pool_best": float(best.score) if best else None,
            "pool_best_terms": list(best.terms) if best else [],
            "pool_best_coeffs": list(best.coeffs) if best else [],
            "provenance": provenance,
        })

        if cfg.convergence_patience and consecutive_reject >= cfg.convergence_patience:
            history.append({
                "round": round_num, "label": profile, "status": "early_stop",
                "score": None, "accepted": False,
                "reason": f"连续 {consecutive_reject} 轮无入池，提前停止",
            })
            break

    summary = summarize_arm(pop, history, seed_candidates, arm, profile, cfg)
    return {
        "arm": arm, "profile": profile,
        "population": pop, "seed_candidates": seed_candidates,
        "history": history, "all_candidates": all_candidates,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# 汇总与收敛
# --------------------------------------------------------------------------- #
def _valid_scores(history: List[dict]) -> List[float]:
    return [float(h["score"]) for h in history if h.get("status") == "evaluated" and h.get("score") is not None]


def _running_best(history: List[dict], seed_best: float) -> List[float]:
    best = seed_best
    out = []
    for h in history:
        if h.get("status") == "evaluated" and h.get("score") is not None:
            best = max(best, float(h["score"]))
        out.append(best)
    return out


def summarize_arm(pop: Population, history: List[dict], seed_candidates: List[Candidate],
                  arm: str, profile: str, cfg: Config) -> dict:
    seed_scores = [c.score for c in seed_candidates if c.score is not None]
    seed_best = max(seed_scores) if seed_scores else 0.0
    valid = _valid_scores(history)
    best_cand = pop.best()
    running = _running_best(history, seed_best)
    final_best = max(running) if running else seed_best

    # 收敛轮次：池内最高分首次达到最终最优的 99%（或差距 < 0.002）
    conv_round = 0
    if best_cand is not None:
        target = final_best - max(0.002, 0.01 * abs(final_best))
        for i, val in enumerate(running, start=1):
            if val >= target:
                conv_round = i
                break

    status_counts: Dict[str, int] = {}
    for h in history:
        status_counts[h.get("status", "?")] = status_counts.get(h.get("status", "?"), 0) + 1

    return {
        "arm": arm,
        "profile": profile,
        "rounds_run": len([h for h in history if h.get("status") != "early_stop"]),
        "seed_best": round(float(seed_best), 6),
        "seed_mean": round(float(np.mean(seed_scores)), 6) if seed_scores else None,
        "best_score": round(float(best_cand.score), 6) if best_cand else None,
        "best_terms": list(best_cand.terms) if best_cand else [],
        "best_coeffs": [round(float(c), 6) for c in best_cand.coeffs] if best_cand else [],
        "best_sid": best_cand.sid if best_cand else "",
        "candidate_mean": round(float(np.mean(valid)), 6) if valid else None,
        "candidate_median": round(float(np.median(valid)), 6) if valid else None,
        "candidate_max": round(float(np.max(valid)), 6) if valid else None,
        "valid_count": len(valid),
        "accepted_count": int(sum(1 for h in history if h.get("accepted"))),
        "acceptance_rate": round(float(np.mean([1.0 if h.get("accepted") else 0.0
                                                for h in history if h.get("status") in
                                                ("evaluated", "invalid_geometry")])), 4) if history else None,
        "status_counts": status_counts,
        "convergence_round": int(conv_round),
        "pool_summary": pop.summary(),
        "pool_top": [
            {
                "sid": c.sid, "terms": c.terms, "coeffs": [round(float(x), 6) for x in c.coeffs],
                "score": round(float(c.score), 6), "source": c.source, "round": c.round,
                "support": round(float(c.support_proxy), 6),
                "elastic": round(float(c.elastic_proxy), 6),
                "provenance": c.provenance,
            }
            for c in pop.sorted_items()
        ],
    }


def compare_arms(llm_summary: dict, random_summary: dict) -> dict:
    """给出 LLM(模拟) 与随机基线的对比结论数据。"""
    out = {
        "best_llm": llm_summary.get("best_score"),
        "best_random": random_summary.get("best_score"),
        "best_delta": None,
        "mean_llm": llm_summary.get("candidate_mean"),
        "mean_random": random_summary.get("candidate_mean"),
        "mean_delta": None,
        "acceptance_llm": llm_summary.get("acceptance_rate"),
        "acceptance_random": random_summary.get("acceptance_rate"),
        "convergence_llm": llm_summary.get("convergence_round"),
        "convergence_random": random_summary.get("convergence_round"),
        "improved_over_seed": False,
    }
    if out["best_llm"] is not None and out["best_random"] is not None:
        out["best_delta"] = round(out["best_llm"] - out["best_random"], 6)
    if out["mean_llm"] is not None and out["mean_random"] is not None:
        out["mean_delta"] = round(out["mean_llm"] - out["mean_random"], 6)
    out["improved_over_seed"] = bool(
        (out["best_llm"] or 0) > (llm_summary.get("seed_best") or 0)
    )
    return out
