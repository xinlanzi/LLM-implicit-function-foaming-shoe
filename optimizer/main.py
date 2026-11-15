# -*- coding: utf-8 -*-
"""
入口脚本：加载 Excel → 运行 LLM(模拟) 与随机基线 → 输出 candidates.csv / comparison.png / run_log.json。

运行示例
--------
    D:\\anaconda\\python.exe optimizer\\main.py
    D:\\anaconda\\python.exe optimizer\\main.py --rounds 50 --profile high_support_high_elasticity
    D:\\anaconda\\python.exe optimizer\\main.py --llm-backend mock   # 离线自检，不访问 LLM

可选：--extra-profiles high_support,high_elasticity 会额外各跑 50 轮 LLM-only，
用于展示“同一套优化器支持不同目标性能输入”。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

# 允许直接 python optimizer/main.py 运行
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import Config, LLM_PROVIDERS
from evaluator import Evaluator
from optimizer import UsageMeter, build_prompt, compare_arms, run_arm
from proposer import (
    CredentialError,
    build_chat_payload,
    chat_once,
    check_llm_endpoint,
    estimate_tokens,
    mask_secret,
    parse_llm_output,
    resolve_api_key,
)
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
    llm_name = "LLM（模拟）" if cfg.proposer_backend == "mock" else f"LLM（{cfg.proposer_backend}）"

    def series(hist, key):
        xs, ys = [], []
        for h in hist:
            if h.get("status") == "evaluated" and h.get("score") is not None:
                xs.append(h["round"])
                ys.append(float(h["score"]))
        return xs, ys

    def pool_best_series(hist, seed_best):
        base = 0.0 if seed_best is None else float(seed_best)
        xs = [0] + [h["round"] for h in hist if h.get("status") != "early_stop"]
        ys = [base]
        current = base
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
    for res, color, label in [(llm_res, "#d62728", llm_name), (random_res, "#1f77b4", "随机基线")]:
        xs, ys = series(res["history"], "score")
        ax1.scatter(xs, ys, s=16, alpha=0.55, color=color, label=f"{label} 每轮得分")
        ax1.plot(range(1, len(res["history"]) + 1),
                 _rolling_mean([h.get("score") if h.get("status") == "evaluated" else None
                                for h in res["history"]]),
                 color=color, linewidth=1.8, label=f"{label} 5轮滑动均值")
    if not any(h.get("status") == "evaluated" for h in list(llm_hist) + list(rnd_hist)):
        ax1.text(0.5, 0.5, "无数据：本次没有任何候选通过解析与评价",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=12, color="#888888")
    ax1.set_xlabel("轮次")
    ax1.set_ylabel("候选得分（几何代理）")
    ax1.set_title("A. 每轮候选得分")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="lower right")

    # ---- B. 池内最高分（含种子基线） ----
    xl, yl = pool_best_series(llm_hist, llm_seed_best)
    xr, yr = pool_best_series(rnd_hist, rnd_seed_best)
    ax2.plot(xl, yl, color="#d62728", linewidth=2.2, label=f"{llm_name}池内最高分")
    ax2.plot(xr, yr, color="#1f77b4", linewidth=2.2, label="随机基线池内最高分")
    ax2.axhline(0.0 if llm_seed_best is None else llm_seed_best,
                color="gray", linestyle=":", linewidth=1.2,
                label=f"种子最高分 {_fmt_num(llm_seed_best, '.3f')}")
    ax2.set_xlabel("轮次")
    ax2.set_ylabel("池内最高得分（几何代理）")
    ax2.set_title("B. 池内最高分随轮次变化")
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="lower right")

    # ---- C. 汇总柱状图 ----
    def _num(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    labels = ["池内最高分", "候选均分", "入池率"]
    keys = ("best_score", "candidate_mean", "acceptance_rate")
    llm_raw = [llm_res["summary"].get(k) for k in keys]
    rnd_raw = [random_res["summary"].get(k) for k in keys]
    llm_vals = [0.0 if _num(v) is None else _num(v) for v in llm_raw]
    rnd_vals = [0.0 if _num(v) is None else _num(v) for v in rnd_raw]
    xpos = np.arange(len(labels))
    ax3.bar(xpos - 0.18, llm_vals, width=0.34, color="#d62728", label=llm_name)
    ax3.bar(xpos + 0.18, rnd_vals, width=0.34, color="#1f77b4", label="随机基线")
    for i, (a_raw, b_raw) in enumerate(zip(llm_raw, rnd_raw)):
        ax3.text(i - 0.18, (0.0 if _num(a_raw) is None else _num(a_raw)) + 0.015,
                 _fmt_num(a_raw, ".3f", "n/a"), ha="center", fontsize=8)
        ax3.text(i + 0.18, (0.0 if _num(b_raw) is None else _num(b_raw)) + 0.015,
                 _fmt_num(b_raw, ".3f", "n/a"), ha="center", fontsize=8)
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
    for pool, color, label in [(llm_top, "#d62728", llm_name), (rnd_top, "#1f77b4", "随机基线")]:
        pts = [(c.support_proxy, c.elastic_proxy) for c in pool if c.source != "seed"]
        if pts:
            ax4.scatter([p[0] for p in pts], [p[1] for p in pts], s=28, alpha=0.65,
                        color=color, label=f"{label} 生成候选")
    if seed_points:
        ax4.scatter([p[0] for p in seed_points], [p[1] for p in seed_points],
                    s=55, marker="*", color="#2ca02c", label="种子结构")
        for sx, sy, sid in seed_points:
            ax4.annotate(sid, (sx, sy), fontsize=7, xytext=(4, 4), textcoords="offset points")
    if not seed_points and not any(c.source != "seed" for c in list(llm_top) + list(rnd_top)):
        ax4.text(0.5, 0.5, "无数据：池内没有可用候选", transform=ax4.transAxes,
                 ha="center", va="center", fontsize=12, color="#888888")
    ax4.set_xlabel("support 代理")
    ax4.set_ylabel("elastic 代理")
    ax4.set_title("D. 最终池的支撑—弹性权衡")
    ax4.set_xlim(0, 1.02)
    ax4.set_ylim(0, 1.02)
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"TPMS 隐函数组合优化：{llm_name} vs 随机基线（{profile}，{cfg.rounds} 轮）",
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
    if df.empty:
        return pd.DataFrame(columns=front)
    cols = front + [c for c in df.columns if c not in front]
    return df[cols]


def _fmt_num(value, spec: str = ".4f", dash: str = "—") -> str:
    """安全格式化：None / NaN / inf 一律显示占位符，避免格式化崩溃。"""
    if value is None:
        return dash
    try:
        v = float(value)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(v):
        return dash
    return format(v, spec)


def _debug_one_call(cfg: Config, prompt: str, prompt_note: str = "") -> int:
    """
    `--debug-one-call`：用真实 prompt 发一次请求，打印全部诊断信息后退出。
    不跑优化、不写 outputs；密钥由用户交互输入，只显示打码。
    流式下额外打印首字节/首 token/块间耗时与 chunk 数；超时也打印已收到的部分内容。
    """
    mode = "流式 SSE（stream=true）" if cfg.llm_stream else "非流式（stream=false）"
    print("LLM 单次调试（只发 1 次真实请求；不跑优化、不写 outputs）")
    print("-" * 78)
    print(f"端点     : {cfg.llm_base_url.rstrip('/')}/chat/completions")
    print(f"模型     : {cfg.llm_model}")
    print(f"模式     : {mode}")
    print(f"本次 max_tokens = {cfg.llm_max_tokens}")
    print(f"提示词   : {len(prompt)} 字符 ≈ {estimate_tokens(prompt)} token（粗估）")
    if prompt_note:
        print(f"提示词说明: {prompt_note}")
    print(f"超时     : 建连 {cfg.llm_connect_timeout:.0f}s | 首字节 "
          f"{cfg.llm_first_token_timeout:.0f}s | 块间空闲 {cfg.llm_idle_timeout:.0f}s | "
          f"总上限 {cfg.llm_timeout:.0f}s")

    preview = build_chat_payload(cfg, prompt)
    preview = json.loads(json.dumps(preview, ensure_ascii=False))
    preview["messages"] = [{"role": m.get("role"),
                            "content_len": len(str(m.get("content", "")))}
                           for m in preview.get("messages", [])]
    print("\n实际请求体（messages 内容折叠为长度；不含 Authorization 头）：")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    try:
        res = chat_once(cfg, prompt)
    except CredentialError as exc:
        print(f"\n✗ 无法获取凭证：{exc}")
        return 2

    def _sec(v):
        return "—" if v is None else f"{v:.2f}s"

    print(f"\nHTTP 状态 : {res.get('http_status')} | 凭证来源 {res.get('source')} | "
          f"{res.get('key_mask')}")
    print(f"耗时      : 建连 {_sec(res.get('t_connect'))} | 首字节 "
          f"{_sec(res.get('t_first_byte'))} | 首 token {_sec(res.get('t_first_token'))} | "
          f"总 {res.get('elapsed', 0.0):.2f}s")
    print(f"流统计    : chunk 数 {res.get('chunks', 0)} | content {len(res.get('content',''))} 字符 | "
          f"reasoning_content {len(res.get('reasoning',''))} 字符 | "
          f"finish_reason={res.get('finish_reason')!r}")
    if res.get("downgraded"):
        print(f"降级      : 带 stream_options 的请求失败（{res.get('first_error')}），"
              "已去掉该字段重发一次")
    if res.get("error"):
        print(f"错误      : [{res.get('error_kind')}"
              f"{('/' + str(res.get('timeout_phase'))) if res.get('timeout_phase') else ''}] "
              f"{res.get('error')}")

    usage_raw = res.get("usage_raw")
    if isinstance(usage_raw, dict):
        print("\nusage 原文（服务端返回）：")
        print(json.dumps(usage_raw, ensure_ascii=False, indent=2))
    else:
        reason = res.get("usage_unavailable_reason") or "no_usage_field"
        print(f"\nusage：响应里没有 usage 字段（{reason}；token 只能按字符数本地估算）")
    if res.get("usage"):
        u = res["usage"]
        print(f"usage 采用: {u.get('usage_source')} | prompt {u.get('prompt_tokens')} + "
              f"completion {u.get('completion_tokens')} = {u.get('total_tokens')}")

    if res.get("body"):
        print("\n⚠ 错误体（响应正文前 800 字符）：")
        print(str(res["body"])[:800])

    content = res.get("content") or ""
    reasoning = res.get("reasoning") or ""
    if content:
        print(f"\ncontent 前 300 字符：{content[:300]!r}（总长 {len(content)}）")
    else:
        print('\ncontent = "" （长度 0）')
    if reasoning:
        print(f"reasoning_content 前 300 字符：{reasoning[:300]!r}（总长 {len(reasoning)}）")
    if res.get("message_keys"):
        print(f"message 的键 = {res['message_keys']}")

    raw = res.get("raw") or ""
    print(f"\n原始响应/事件流前 2000 字符（已脱敏）：\n{raw[:2000] if raw else '(空)'}")

    # ---- 诊断结论 ----
    print("\n诊断结论：")
    status = res.get("http_status")
    if res.get("error_kind") == "timeout":
        print(f"  ✗ 超时（阶段={res.get('timeout_phase')}）：{res.get('error')}")
        if res.get("chunks"):
            print(f"    → 已收到 {res['chunks']} 个 chunk，说明模型在推理但没在时限内吐出完整答案；"
                  "可先 --max-tokens 512 二分，或加大 LLM_FIRST_TOKEN_TIMEOUT/LLM_IDLE_TIMEOUT。")
        else:
            print("    → 一个 chunk 都没收到：更可能是网络/网关/模型不可用；"
                  "可先用 --debug-prompt-chars 300 --max-tokens 512 排除 prompt 过长。")
    elif status is None:
        print("  ✗ 没有拿到 HTTP 响应：先检查网络/代理/端点主机是否可达。")
    elif int(status) != 200:
        print(f"  ✗ HTTP {status}：请求被拒绝。")
        if int(status) in (401, 403):
            print("    → 密钥无效/过期/无权限：重新生成密钥。")
        elif int(status) in (400, 404):
            print("    → 多为请求体参数（如 enable_thinking/stream_options）或模型名/端点路径不被接受，"
                  "对照上面的请求体与错误体核对。")
        elif int(status) == 429:
            print("    → 限流或余额不足。")
        else:
            print("    → 服务端错误，稍后重试。")
    elif not content.strip():
        if reasoning.strip():
            prefix = "流式" if res.get("stream") else "非流式"
            print(f"  ✗ {prefix}返回 chunk 数 {res.get('chunks')}，但 content 为空、"
                  f"reasoning_content 有 {len(reasoning)} 字符：思维链占满了输出额度。")
            print("    → 提高 --max-tokens（如 4096），或确认该模型能否关闭思考。")
        elif res.get("finish_reason") == "length":
            print("  ✗ content 为空且 finish_reason=length：输出被 max_tokens 截断。")
            print("    → 提高 --max-tokens 后重试。")
        else:
            print("  ✗ content 为空：模型没有返回答案（可能被安全策略清空或服务异常）。")
    else:
        try:
            obj = parse_llm_output(content)
            print(f"  ✓ content 非空且可解析为候选：terms={obj['terms']} coeffs={obj['coeffs']}")
        except Exception as exc:
            print(f"  ✗ content 非空但解析失败：{exc}")
            print("    → 看上面的 content 片段；常见是 JSON 被截断或夹带解释文字。")
    if res.get("finish_reason") == "length":
        print("  ！finish_reason=length：本次输出被 max_tokens 截断（即使 content 非空也可能不完整）。")
    if prompt_note:
        print("  ！本次是截断/短 prompt 探活，结果不能当作正式结论。")

    print("\n提示：本模式不写任何输出文件；确认上面的结论后，再去跑 3 轮冒烟。")
    ok = (status == 200 and not res.get("error_kind"))
    return 0 if ok else 3


def main():
    parser = argparse.ArgumentParser(description="LLM 驱动的 TPMS 隐函数组合优化器")
    parser.add_argument("--rounds", type=int, default=None, help="每臂轮数（默认 50）")
    parser.add_argument("--profile", type=str, default=None, help="目标性能 profile")
    parser.add_argument("--token-set", type=str, default=None, choices=["core", "library"])
    parser.add_argument("--llm-backend", "--llm-provider", dest="llm_backend", type=str,
                        default=None, choices=list(LLM_PROVIDERS),
                        help="默认 siliconflow；mock 仅用于无需密钥的离线自检")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="单次生成上限 token（默认 2048；推理模型的思维链会先占额度）")
    parser.add_argument("--stream", dest="stream", action="store_true", default=None,
                        help="流式 SSE 接收（默认开；便于观测首字节/首 token/块间空闲）")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="回退到非流式整包响应（用于对比诊断）")
    parser.add_argument("--debug-prompt-chars", type=int, default=None,
                        help="--debug-one-call 时把 prompt 截断到 N 字符（探活用，结果不能当正式结论）")
    parser.add_argument("--check-llm", action="store_true",
                        help="只做一次最小请求自检（凭证/端点/模型名），不跑优化流程")
    parser.add_argument("--debug-one-call", action="store_true",
                        help="用真实的第一条 prompt 发一次请求并打印全部诊断（usage/message/"
                             "finish_reason/原始响应），不跑优化、不写 outputs")
    parser.add_argument("--max-total-tokens", type=int, default=None,
                        help="运行级 token 预算：累计 total_tokens 达到该值后，当前轮完成后停机"
                             "（默认不限制）")
    parser.add_argument("--stop-after-first-response", action="store_true",
                        help="冒烟用：第一次成功解析出候选后停机（本轮评价/入池/记录照常完成）")
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
    if args.max_tokens is not None:
        cfg.llm_max_tokens = int(args.max_tokens)
    if args.stream is not None:
        cfg.llm_stream = bool(args.stream)
    cfg.max_total_tokens = args.max_total_tokens
    cfg.stop_after_first_response = bool(args.stop_after_first_response)
    # 真实 LLM 的端点与模型固定；密钥仅由运行时安全弹窗读取。
    if args.seed is not None:
        cfg.seed = args.seed
    if args.outdir is not None:
        cfg.output_dir = args.outdir

    # 自检模式：只发一次极短请求验证凭证/端点/模型，不进入优化流程
    if args.check_llm:
        status, detail = check_llm_endpoint(cfg)
        print("LLM 连通性自检")
        print("-" * 60)
        print("密钥输入  : 安全弹窗（仅进程内存，不落盘）")
        print(detail)
        code = {"ok": 0, "no_key": 2, "auth": 3, "not_found": 4,
                "timeout": 5, "network": 5, "quota": 6}.get(status, 7)
        sys.exit(code)

    # 单次调试模式：用真实 prompt 打一次，不跑优化、不写盘
    if args.debug_one_call:
        if cfg.proposer_backend == "mock":
            print("--debug-one-call 只对真实 SiliconFlow 接口有意义；当前是 mock，无需调试。")
            sys.exit(2)
        profile = cfg.profile()
        seeds, _excluded, _perf = load_seeds(cfg.excel_path)
        _examples = [{"sid": s.sid, "terms": s.terms, "coeffs": s.coeffs,
                      "score": None, "tags": ""} for s in seeds]
        _prompt = build_prompt(profile, _examples[:max(1, cfg.n_high_examples)],
                               _examples[-1] if _examples else None, 1, cfg)
        _note = ""
        n_chars = args.debug_prompt_chars
        if n_chars is not None and 0 < int(n_chars) < len(_prompt):
            _prompt = _prompt[:int(n_chars)]
            _note = f"prompt 已截断到 {int(n_chars)} 字符（短 prompt 探活，结果不能当正式结论）"
        sys.exit(_debug_one_call(cfg, _prompt, prompt_note=_note))

    os.makedirs(cfg.output_dir, exist_ok=True)
    profile = cfg.profile()

    print("=" * 78)
    print("TPMS 隐函数组合优化器")
    print("=" * 78)
    print(f"目标 profile : {profile}")
    print(f"目标描述     : {PROFILE_DESCRIPTIONS[profile]}")
    print(f"轮数         : {cfg.rounds}")
    print(f"token 词表   : {cfg.token_set}（{len(get_allowed_tokens(cfg.token_set))} 个 token）")
    print(f"提议器后端   : {cfg.proposer_backend}"
          f"（{LLM_PROVIDERS.get(cfg.proposer_backend, {}).get('label', '')}）")
    if cfg.proposer_backend != "mock":
        print(f"LLM 端点     : {cfg.llm_base_url.rstrip('/')}/chat/completions")
        print(f"LLM 模型     : {cfg.llm_model}（max_tokens={cfg.llm_max_tokens}，"
              f"temperature={cfg.temperature}，"
              f"{'流式' if cfg.llm_stream else '非流式'}）")
        print(f"超时         : 建连 {cfg.llm_connect_timeout:.0f}s / 首字节 "
              f"{cfg.llm_first_token_timeout:.0f}s / 块间 {cfg.llm_idle_timeout:.0f}s / "
              f"总上限 {cfg.llm_timeout:.0f}s")
        print("密钥输入     : 安全弹窗输入一次，仅进程内缓存，不落盘")
        print(f"token 预算   : "
              f"{('不限' if cfg.max_total_tokens is None else f'{cfg.max_total_tokens:,}')}"
              f" | 冒烟停止: {'开' if cfg.stop_after_first_response else '关'}")
    print(f"随机种子     : {cfg.seed}")
    print(f"评价网格     : {cfg.grid_n}^3，等值面 F=0，实体相 F<=0")
    print(f"输出目录     : {cfg.output_dir}")

    # ---- 1. 读取种子 ----
    seeds, excluded, performance = load_seeds(cfg.excel_path)
    print(f"\n可优化三角闭式种子: {len(seeds)} 个；未纳入隐函数搜索: {len(excluded)} 个")
    for s in seeds:
        print(f"  {s.sid} {s.name:24s} terms={s.terms} coeffs={[round(c,3) for c in s.coeffs]}")

    evaluator = Evaluator(cfg)

    # ---- 1b. 真实 LLM：先给成本预估，再要密钥 ----
    if cfg.proposer_backend != "mock":
        _examples = [{"sid": s.sid, "terms": s.terms, "coeffs": s.coeffs,
                      "score": None, "tags": ""} for s in seeds]
        _prompt = build_prompt(profile, _examples[:max(1, cfg.n_high_examples)],
                               _examples[-1] if _examples else None, 1, cfg)
        per_prompt = estimate_tokens(_prompt)
        per_round = per_prompt + int(cfg.llm_max_tokens)
        worst = cfg.rounds * per_round
        print("\n" + "-" * 78)
        print("token 成本预估（粗估；实际以服务端 usage 为准）")
        print("-" * 78)
        print(f"  prompt ≈ {per_prompt:,} token（按实际 build_prompt 首条统计；"
              f"ASCII 4 字符/token、中文 1.6 字符/token）")
        print(f"  每轮 1 次请求 × (prompt {per_prompt:,} + max_tokens {cfg.llm_max_tokens}) "
              f"≈ {per_round:,} token")
        if cfg.stop_after_first_response:
            print(f"  本次开启 --stop-after-first-response：预计 1 轮 ≈ {per_round:,} token")
        else:
            print(f"  最坏 {cfg.rounds} 轮 ≈ {worst:,} token（不含重试；重试会额外消耗）")
        if cfg.max_total_tokens is not None and per_round > 0:
            trigger_at = min(cfg.rounds, max(1, -(-int(cfg.max_total_tokens) // per_round)))
            print(f"  预算 --max-total-tokens={cfg.max_total_tokens:,} → "
                  f"预计第 {trigger_at} 轮后触发停机")
        try:
            _key, _src = resolve_api_key(cfg)
            print(f"  凭证来源: {_src} | {mask_secret(_key)}")
        except CredentialError as exc:
            print("\n✗ 无法获取 LLM 凭证，已中止：")
            print(str(exc))
            sys.exit(9)

    # ---- 2. 主实验：LLM(模拟) vs 随机基线 ----
    arm_label = "LLM(模拟)" if cfg.proposer_backend == "mock" else f"LLM({cfg.proposer_backend})"
    print("\n" + "-" * 78)
    print(f"主实验：{arm_label} {cfg.rounds} 轮 vs 随机基线 {cfg.rounds} 轮（同一评价器/种子/入池规则）")
    print("-" * 78)
    meter = UsageMeter()
    stop_state = {"triggered": False, "reason": None}
    llm_res = run_arm(seeds, evaluator, cfg, profile, arm="llm", seed_offset=0,
                      meter=meter, stop_state=stop_state)
    random_res = run_arm(seeds, evaluator, cfg, profile, arm="random", seed_offset=100,
                         meter=meter)

    comparison = compare_arms(llm_res["summary"], random_res["summary"])
    print(f"\n种子最高分: {_fmt_num(llm_res['summary']['seed_best'])}")
    print(f"{arm_label} : 池内最高 {_fmt_num(llm_res['summary']['best_score'])} | "
          f"候选均分 {_fmt_num(llm_res['summary']['candidate_mean'])} | "
          f"入池率 {_fmt_num(llm_res['summary']['acceptance_rate'], '.3f')} | "
          f"收敛轮 {llm_res['summary']['convergence_round']}")
    print(f"随机基线  : 池内最高 {_fmt_num(random_res['summary']['best_score'])} | "
          f"候选均分 {_fmt_num(random_res['summary']['candidate_mean'])} | "
          f"入池率 {_fmt_num(random_res['summary']['acceptance_rate'], '.3f')} | "
          f"收敛轮 {random_res['summary']['convergence_round']}")
    print(f"差值      : best Δ={_fmt_num(comparison['best_delta'], '+.4f')} | "
          f"mean Δ={_fmt_num(comparison['mean_delta'], '+.4f')}")
    counts = llm_res["summary"].get("status_counts") or {}
    if counts:
        print("LLM 每轮状态: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if llm_res["summary"].get("valid_count", 0) == 0:
        print("  注意：本次 LLM 臂没有任何候选通过解析与评价（全部失败/非法），"
              "“—”表示无数据而非 0 分；原因见 run_log.json 的 histories.llm[].status。")

    # ---- 3. 额外目标（LLM-only）展示目标可切换 ----
    extra_results = []
    extra_skipped = []
    if stop_state.get("triggered"):
        print("\n⏹ 已触发停机，跳过额外目标（" + str(args.extra_profiles) + "），"
              "以保证预算不被继续消耗。")
    elif args.extra_profiles.strip() and args.extra_profiles.strip().lower() not in ("none", "-"):
        for i, p in enumerate([x.strip() for x in args.extra_profiles.split(",") if x.strip()]):
            if p not in cfg.profile_weights or p == profile:
                continue
            if stop_state.get("triggered"):
                extra_skipped.append(p)
                continue
            print("\n" + "-" * 78)
            print(f"额外目标：{p}（LLM-only，{cfg.rounds} 轮）")
            print("-" * 78)
            res = run_arm(seeds, evaluator, cfg, p, arm="llm", seed_offset=200 + i * 50,
                          meter=meter, stop_state=stop_state)
            extra_results.append(res)
            print(f"  池内最高 {_fmt_num(res['summary']['best_score'])} | "
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
        "stop_reason": stop_state,
        "llm_usage": meter.to_dict(),
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

    # ---- 7. 运行级 token 用量汇总 ----
    usage = meter.to_dict()
    print("\nLLM token 用量：")
    print(f"  请求 {usage['requests']} 次 | prompt {usage['prompt_tokens']:,} + "
          f"completion {usage['completion_tokens']:,} = total {usage['total_tokens']:,}"
          f"（口径: {usage['usage_source']}）")
    if stop_state.get("triggered"):
        print(f"  停机: {stop_state.get('reason')} | 第 {stop_state.get('round')} 轮后 | "
              f"已完成 {stop_state.get('rounds_completed')}/{cfg.rounds} 轮")

    print(f"\n主实验最终池（{arm_label}）前 5：")
    top_pool = llm_res["population"].sorted_items()[:5]
    if not top_pool:
        print("  （空池：本次没有候选入池）")
    for i, c in enumerate(top_pool, 1):
        print(f"  {i}. {_fmt_num(c.score)} | {c.source} r{c.round} | terms={c.terms} | "
              f"coeffs={[round(x,3) for x in c.coeffs]} | "
              f"support={_fmt_num(c.support_proxy, '.3f')} "
              f"elastic={_fmt_num(c.elastic_proxy, '.3f')}")
    print("\n完成。")


if __name__ == "__main__":
    main()
