# -*- coding: utf-8 -*-
"""
入口脚本：加载 Excel → 运行 LLM(模拟) 与随机基线 → 输出 candidates.csv / comparison.png / run_log.json。

运行示例
--------
    D:\\anaconda\\python.exe optimizer\\main.py
    D:\\anaconda\\python.exe optimizer\\main.py --rounds 50 --profile high_support_high_elasticity
    D:\\anaconda\\python.exe optimizer\\main.py --llm-backend openai   # 需先设置 LLM_API_KEY

可选：--extra-profiles high_support,high_elasticity 会额外各跑 50 轮 LLM-only，
用于展示“同一套优化器支持不同目标性能输入”。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

# 允许直接 python optimizer/main.py 运行
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import Config
from evaluator import Evaluator
from optimizer import compare_arms, run_arm
from seed_loader import PROFILE_DESCRIPTIONS, describe_seed_table, load_seeds
from token_library import get_allowed_tokens


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _rolling_mean(scores: List[float], window: int = 5) -> List[float]:
    out = []
    vals = []
    for s in scores:
        if s is not None and not (isinstance(s, float) and np.isnan(s)):
            vals.append(float(s))
        if len(vals) > window:
            vals.pop(0)
        out.append(float(np.mean(vals)) if vals else np.nan)
    return out


def plot_comparison(llm_res: dict, random_res: dict, out_path: str, cfg: Config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    profile = llm_res["profile"]
    llm_hist, rnd_hist = llm_res["history"], random_res["history"]
    llm_seed_best = llm_res["summary"]["seed_best"]
    rnd_seed_best = random_res["summary"]["seed_best"]

    def series(hist, key):
        xs, ys = [], []
        for h in hist:
            if h.get("status") == "evaluated" and h.get("score") is not None:
                xs.append(h["round"])
                ys.append(float(h["score"]))
        return xs, ys

    def pool_best_series(hist, seed_best):
        xs = [0] + [h["round"] for h in hist if h.get("status") != "early_stop"]
        ys = [seed_best]
        current = seed_best
        for h in hist:
            if h.get("status") == "early_stop":
                break
            pb = h.get("pool_best")
            if pb is not None:
                current = max(current, float(pb))
            ys.append(current)
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # ---- A. 每轮候选得分 ----
    for res, color, label in [(llm_res, "#d62728", "LLM（模拟）"), (random_res, "#1f77b4", "随机基线")]:
        xs, ys = series(res["history"], "score")
        ax1.scatter(xs, ys, s=16, alpha=0.55, color=color, label=f"{label} 每轮得分")
        ax1.plot(range(1, len(res["history"]) + 1),
                 _rolling_mean([h.get("score") if h.get("status") == "evaluated" else None
                                for h in res["history"]]),
                 color=color, linewidth=1.8, label=f"{label} 5轮滑动均值")
    ax1.set_xlabel("轮次")
    ax1.set_ylabel("候选得分（几何代理）")
    ax1.set_title("A. 每轮候选得分")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="lower right")

    # ---- B. 池内最高分（含种子基线） ----
    xl, yl = pool_best_series(llm_hist, llm_seed_best)
    xr, yr = pool_best_series(rnd_hist, rnd_seed_best)
    ax2.plot(xl, yl, color="#d62728", linewidth=2.2, label="LLM（模拟）池内最高分")
    ax2.plot(xr, yr, color="#1f77b4", linewidth=2.2, label="随机基线池内最高分")
    ax2.axhline(llm_seed_best, color="gray", linestyle=":", linewidth=1.2,
                label=f"种子最高分 {llm_seed_best:.3f}")
    ax2.set_xlabel("轮次")
    ax2.set_ylabel("池内最高得分（几何代理）")
    ax2.set_title("B. 池内最高分随轮次变化")
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="lower right")

    # ---- C. 汇总柱状图 ----
    labels = ["池内最高分", "候选均分", "入池率"]
    llm_vals = [llm_res["summary"].get("best_score") or 0,
                llm_res["summary"].get("candidate_mean") or 0,
                llm_res["summary"].get("acceptance_rate") or 0]
    rnd_vals = [random_res["summary"].get("best_score") or 0,
                random_res["summary"].get("candidate_mean") or 0,
                random_res["summary"].get("acceptance_rate") or 0]
    xpos = np.arange(len(labels))
    ax3.bar(xpos - 0.18, llm_vals, width=0.34, color="#d62728", label="LLM（模拟）")
    ax3.bar(xpos + 0.18, rnd_vals, width=0.34, color="#1f77b4", label="随机基线")
    for i, (a, b) in enumerate(zip(llm_vals, rnd_vals)):
        ax3.text(i - 0.18, a + 0.015, f"{a:.3f}", ha="center", fontsize=8)
        ax3.text(i + 0.18, b + 0.015, f"{b:.3f}", ha="center", fontsize=8)
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 1.05)
    ax3.set_title("C. 汇总指标")
    ax3.grid(alpha=0.25, axis="y")
    ax3.legend(fontsize=8, loc="lower right")

    # ---- D. 最终池：support vs elastic（看权衡） ----
    llm_top = llm_res["population"].sorted_items()
    rnd_top = random_res["population"].sorted_items()
    seed_points = [(c.support_proxy, c.elastic_proxy, c.sid) for c in llm_top if c.source == "seed"]
    for pool, color, label in [(llm_top, "#d62728", "LLM（模拟）"), (rnd_top, "#1f77b4", "随机基线")]:
        pts = [(c.support_proxy, c.elastic_proxy) for c in pool if c.source != "seed"]
        if pts:
            ax4.scatter([p[0] for p in pts], [p[1] for p in pts], s=28, alpha=0.65,
                        color=color, label=f"{label} 生成候选")
    if seed_points:
        ax4.scatter([p[0] for p in seed_points], [p[1] for p in seed_points],
                    s=55, marker="*", color="#2ca02c", label="种子结构")
        for sx, sy, sid in seed_points:
            ax4.annotate(sid, (sx, sy), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax4.set_xlabel("support 代理")
    ax4.set_ylabel("elastic 代理")
    ax4.set_title("D. 最终池的支撑—弹性权衡")
    ax4.set_xlim(0, 1.02)
    ax4.set_ylim(0, 1.02)
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"TPMS 隐函数组合优化：LLM(模拟) vs 随机基线（{profile}，{cfg.rounds} 轮）",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_candidates_dataframe(results: List[dict]) -> pd.DataFrame:
    rows = []
    seed_signatures = set()
    for res in results:
        arm = res["arm"]
        profile = res["profile"]
        for cand in res["all_candidates"]:
            if cand.source == "seed":
                if cand.signature in seed_signatures:
                    continue
                seed_signatures.add(cand.signature)
                row_run = "seed"
            else:
                row_run = f"{arm}:{profile}"
            row = cand.to_row()
            row["run"] = row_run
            rows.append(row)
    df = pd.DataFrame(rows)
    # 稳定列顺序
    front = ["run", "label", "sid", "round", "source", "status", "terms", "coeffs",
             "n_terms", "score", "support_proxy", "elastic_proxy", "accepted", "reason"]
    cols = front + [c for c in df.columns if c not in front]
    return df[cols]


def main():
    parser = argparse.ArgumentParser(description="LLM 驱动的 TPMS 隐函数组合优化器")
    parser.add_argument("--rounds", type=int, default=None, help="每臂轮数（默认 50）")
    parser.add_argument("--profile", type=str, default=None, help="目标性能 profile")
    parser.add_argument("--token-set", type=str, default=None, choices=["core", "library"])
    parser.add_argument("--llm-backend", type=str, default=None, choices=["mock", "openai"])
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--extra-profiles", type=str, default="high_support,high_elasticity",
                        help="额外跑 LLM-only 的目标，逗号分隔；设为空字符串关闭")
    args = parser.parse_args()

    cfg = Config()
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.profile is not None:
        cfg.default_profile = args.profile
    if args.token_set is not None:
        cfg.token_set = args.token_set
    if args.llm_backend is not None:
        cfg.proposer_backend = args.llm_backend
    if args.seed is not None:
        cfg.seed = args.seed
    if args.outdir is not None:
        cfg.output_dir = args.outdir

    os.makedirs(cfg.output_dir, exist_ok=True)
    profile = cfg.profile()

    print("=" * 78)
    print("TPMS 隐函数组合优化器")
    print("=" * 78)
    print(f"目标 profile : {profile}")
    print(f"目标描述     : {PROFILE_DESCRIPTIONS[profile]}")
    print(f"轮数         : {cfg.rounds}")
    print(f"token 词表   : {cfg.token_set}（{len(get_allowed_tokens(cfg.token_set))} 个 token）")
    print(f"提议器后端   : {cfg.proposer_backend}")
    print(f"随机种子     : {cfg.seed}")
    print(f"评价网格     : {cfg.grid_n}^3，等值面 F=0，实体相 F<=0")
    print(f"输出目录     : {cfg.output_dir}")

    # ---- 1. 读取种子 ----
    seeds, excluded, performance = load_seeds(cfg.excel_path)
    print(f"\n可优化三角闭式种子: {len(seeds)} 个；未纳入隐函数搜索: {len(excluded)} 个")
    for s in seeds:
        print(f"  {s.sid} {s.name:24s} terms={s.terms} coeffs={[round(c,3) for c in s.coeffs]}")

    evaluator = Evaluator(cfg)

    # ---- 2. 主实验：LLM(模拟) vs 随机基线 ----
    print("\n" + "-" * 78)
    print("主实验：LLM(模拟) 50 轮 vs 随机基线 50 轮（同一评价器/种子/入池规则）")
    print("-" * 78)
    llm_res = run_arm(seeds, evaluator, cfg, profile, arm="llm", seed_offset=0)
    random_res = run_arm(seeds, evaluator, cfg, profile, arm="random", seed_offset=100)

    comparison = compare_arms(llm_res["summary"], random_res["summary"])
    print(f"\n种子最高分: {llm_res['summary']['seed_best']:.4f}")
    print(f"LLM(模拟) : 池内最高 {llm_res['summary']['best_score']:.4f} | "
          f"候选均分 {llm_res['summary']['candidate_mean']:.4f} | "
          f"入池率 {llm_res['summary']['acceptance_rate']:.3f} | "
          f"收敛轮 {llm_res['summary']['convergence_round']}")
    print(f"随机基线  : 池内最高 {random_res['summary']['best_score']:.4f} | "
          f"候选均分 {random_res['summary']['candidate_mean']:.4f} | "
          f"入池率 {random_res['summary']['acceptance_rate']:.3f} | "
          f"收敛轮 {random_res['summary']['convergence_round']}")
    print(f"差值      : best Δ={comparison['best_delta']:+.4f} | mean Δ={comparison['mean_delta']:+.4f}")

    # ---- 3. 额外目标（LLM-only）展示目标可切换 ----
    extra_results = []
    if args.extra_profiles.strip() and args.extra_profiles.strip().lower() not in ("none", "-"):
        for i, p in enumerate([x.strip() for x in args.extra_profiles.split(",") if x.strip()]):
            if p not in cfg.profile_weights or p == profile:
                continue
            print("\n" + "-" * 78)
            print(f"额外目标：{p}（LLM-only，{cfg.rounds} 轮）")
            print("-" * 78)
            res = run_arm(seeds, evaluator, cfg, p, arm="llm", seed_offset=200 + i * 50)
            extra_results.append(res)
            print(f"  池内最高 {res['summary']['best_score']:.4f} | "
                  f"top terms={res['summary']['best_terms']} coeffs={res['summary']['best_coeffs']}")

    # ---- 4. 输出 candidates.csv ----
    all_results = [llm_res, random_res] + extra_results
    df = build_candidates_dataframe(all_results)
    csv_path = os.path.join(cfg.output_dir, "candidates.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n已写出候选表: {csv_path}（{len(df)} 行）")

    # ---- 5. 输出 comparison.png ----
    png_path = os.path.join(cfg.output_dir, "comparison.png")
    if cfg.make_plots:
        plot_comparison(llm_res, random_res, png_path, cfg)
        print(f"已写出对比图: {png_path}")

    # ---- 6. 输出 run_log.json（完整参数、种子、历史、池） ----
    log = {
        "config": asdict(cfg),
        "profile": profile,
        "profile_description": PROFILE_DESCRIPTIONS[profile],
        "seeds": describe_seed_table(seeds),
        "excluded_structures": excluded,
        "performance_background": performance,
        "summary": {
            "llm": llm_res["summary"],
            "random": random_res["summary"],
            "comparison": comparison,
        },
        "extra_profiles": [
            {"profile": res["profile"], "summary": res["summary"], "history": res["history"]}
            for res in extra_results
        ],
        "histories": {
            "llm": llm_res["history"],
            "random": random_res["history"],
        },
        "final_pools": {
            "llm": [
                {
                    "terms": c.terms, "coeffs": [round(float(x), 6) for x in c.coeffs],
                    "score": round(float(c.score), 6), "source": c.source,
                    "round": c.round, "sid": c.sid, "tags": c.tags,
                    "support": round(float(c.support_proxy), 6),
                    "elastic": round(float(c.elastic_proxy), 6),
                    "provenance": c.provenance,
                    "metrics": c.metrics,
                }
                for c in llm_res["population"].sorted_items()
            ],
            "random": [
                {
                    "terms": c.terms, "coeffs": [round(float(x), 6) for x in c.coeffs],
                    "score": round(float(c.score), 6), "source": c.source,
                    "round": c.round, "sid": c.sid, "tags": c.tags,
                    "support": round(float(c.support_proxy), 6),
                    "elastic": round(float(c.elastic_proxy), 6),
                    "provenance": c.provenance,
                    "metrics": c.metrics,
                }
                for c in random_res["population"].sorted_items()
            ],
        },
    }
    log_path = os.path.join(cfg.output_dir, "run_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"已写出运行日志: {log_path}")

    print("\n主实验最终池（LLM 模拟）前 5：")
    for i, c in enumerate(llm_res["population"].sorted_items()[:5], 1):
        print(f"  {i}. {c.score:.4f} | {c.source} r{c.round} | terms={c.terms} | "
              f"coeffs={[round(x,3) for x in c.coeffs]} | support={c.support_proxy:.3f} elastic={c.elastic_proxy:.3f}")
    print("\n完成。")


if __name__ == "__main__":
    main()
