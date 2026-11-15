# -*- coding: utf-8 -*-
"""
集中配置：所有可调参数都在这里，保证结果可复现。

设计原则
--------
1. 所有随机性都由固定 seed 派生，主流程、LLM 模拟提议器、随机基线各自使用独立随机流；
2. 目标性能由 `profile` 切换，支持 "高支撑" / "高弹性" / "高支撑+高弹性" 三种需求；
3. 参数集中在 Config 中，命令行只做少量覆盖，避免散落在业务代码里。
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _env_str(name: str, default: str) -> str:
    """读取非密钥运行参数；未设置时使用代码默认值。"""
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# 固定 LLM 接口；mock 仅用于离线自检，不访问网络
# --------------------------------------------------------------------------- #
# SiliconFlow 的 endpoint 与认证方式来自官方 API 文档：
# https://api-docs.siliconflow.cn/docs/api/chat-completions-post
LLM_PROVIDERS = {
    "mock": {
        "label": "本地模拟器（仅离线自检，无需凭证）",
        "base_url": "",
        "model": "",
    },
    "siliconflow": {
        "label": "硅基流动 SiliconFlow",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
    },
}


@dataclass
class Config:
    # ------------------------------------------------------------------ 数据与路径
    # 原始 Excel 只读，不做任何修改
    excel_path: str = r"c:\Users\hp\Desktop\LLM\晶格结构统一知识库_Structure_Master_V1.0.xlsx"
    output_dir: str = r"c:\Users\hp\Desktop\LLM\outputs"

    # token 词表模式：
    #   "core"    —— 严格使用技术文档给出的 12 个基函数
    #   "library" —— 使用 Excel Token 库中真实出现的全部三角闭式基函数（默认）
    # 说明：库内结构 S003/S004/S005/S006/S008/S009/S010/S011 含有 sin*sin、
    #       三乘积、倍频 cos(2X) 等表达式，若只用 core 会丢失这些真实种子。
    token_set: str = "library"

    # ------------------------------------------------------------------ 搜索循环
    seed: int = 20260101
    rounds: int = 50                 # 技术文档要求的轮数（LLM / 随机基线各 50 轮）
    pool_size: int = 10              # 每个目标池最多保留 10 个候选
    n_high_examples: int = 3         # prompt 中放入的高分样本数（文档建议 2~4）
    n_random_examples: int = 1       # prompt 中放入的随机多样性样本数
    max_terms: int = 3               # 生成候选最多 3 个基函数
    coeff_min: float = -2.0          # 生成候选系数下界
    coeff_max: float = 2.0           # 生成候选系数上界
    max_attempts: int = 8            # 提议器单轮最多尝试次数（用于去重/合法性重试）
    convergence_patience: int = None # 连续多少轮无入池则提前停止；None 表示固定跑满 rounds（MVP 默认）

    # ------------------------------------------------------------------ 评价器
    grid_n: int = 64                 # 单胞采样网格 64^3
    iso_level: float = 0.0           # 隐式曲面 F(x,y,z)=0
    domain_min: float = 0.0          # 单胞定义域 [0, 2π]
    domain_max: float = 6.283185307179586
    band_voxels: float = 1.5         # 曲率统计使用的等值面带宽（以网格步长为单位）
    min_phase_fraction: float = 0.02 # 实体相 / 空相各至少占 2%，否则视为退化几何
    density_ref: float = 0.35        # 支撑代理的相对密度参考值（用于归一化）
    density_target: float = 0.35     # 弹性代理偏好的相对密度中心

    # ------------------------------------------------------------------ 打分权重
    # 三个目标配置；score = w_support * support + w_elastic * elastic
    profile_weights: dict = field(default_factory=lambda: {
        "high_support":               {"support": 0.80, "elastic": 0.20},
        "high_elasticity":            {"support": 0.20, "elastic": 0.80},
        "high_support_high_elasticity": {"support": 0.55, "elastic": 0.45},
    })
    default_profile: str = "high_support_high_elasticity"

    # support / elastic 子分数内部权重（与 profile 无关的结构指标）
    # isotropy：法向分布的三维各向同性程度，防止 1D 层状/2D 柱状结构刷分；
    # area_window：比表面积是否落在 TPMS 合理窗口内（过低=退化层状，过高=细碎难制造）。
    support_subweights: dict = field(default_factory=lambda: {
        "density": 0.32, "connectivity": 0.18, "smoothness": 0.20,
        "isotropy": 0.22, "area_window": 0.08,
    })
    elastic_subweights: dict = field(default_factory=lambda: {
        "connectivity": 0.25, "smoothness": 0.25, "density_window": 0.15,
        "uniformity": 0.08, "isotropy": 0.19, "area_window": 0.08,
    })
    area_target: float = 0.50         # 比表面积（面积/单胞体积）偏好中心
    area_sigma: float = 0.22          # 比表面积窗口宽度
    # 曲率平滑度 exp(-k * H)，k 越大越惩罚高曲率
    curvature_k: float = 2.0

    # ------------------------------------------------------------------ 提议器
    # 默认使用固定的 SiliconFlow chat/completions 接口；base_url 与模型名不接受
    # 命令行或环境变量覆盖。mock 只用于离线自检，不是真实大模型。
    # 密钥只由 proposer.resolve_api_key 的安全弹窗读取，进程内缓存，不写入 Config。
    proposer_backend: str = "siliconflow"
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7))
    llm_base_url: str = LLM_PROVIDERS["siliconflow"]["base_url"]
    llm_model: str = LLM_PROVIDERS["siliconflow"]["model"]
    # ---------------- 流式与超时 ----------------
    # 默认走 SSE 流式：能拿到首字节/首 token/块间间隔的实测数据，
    # 且长 prompt + 推理模型不会因为“整包响应”在 60 秒处被一刀切。
    llm_stream: bool = field(
        default_factory=lambda: _env_str("LLM_STREAM", "1") not in ("0", "", "false", "False"))
    # 分阶段超时（流式下任一阶段超时才算 timeout）：
    #   connect   —— TCP/TLS 建连
    #   first     —— 请求发出后等首字节（含等待服务端开始吐 token）
    #   idle      —— 两个数据块之间的最大空闲
    #   timeout   —— 兜底总时长（非流式的唯一超时）
    llm_connect_timeout: float = field(
        default_factory=lambda: _env_float("LLM_CONNECT_TIMEOUT", 10.0))
    llm_first_token_timeout: float = field(
        default_factory=lambda: _env_float("LLM_FIRST_TOKEN_TIMEOUT", 60.0))
    llm_idle_timeout: float = field(
        default_factory=lambda: _env_float("LLM_IDLE_TIMEOUT", 30.0))
    llm_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 180.0))
    llm_max_retries: int = 2         # 单轮生成最多尝试次数（429/5xx/超时指数退避重试）
    # 候选 JSON 本身很短（约 30~60 token），但推理模型会先花大量 token 写思维链
    # （reasoning_content），max_tokens 偏小会导致 content 为空 / finish_reason="length"。
    # 默认 2048 给思维链留出额度；可用 --max-tokens 或 LLM_MAX_TOKENS 覆盖。
    llm_max_tokens: int = field(default_factory=lambda: int(_env_float("LLM_MAX_TOKENS", 2048)))
    # SiliconFlow 的 enable_thinking：默认不发送该字段（部分模型不接受，会直接 400）。
    # 显式开启时才传 true。
    llm_enable_thinking: bool = field(
        default_factory=lambda: _env_str("LLM_ENABLE_THINKING", "0") not in ("0", "", "false", "False"))
    # ------------------------------------------------------------------ token 预算与停机
    # None = 不限制（避免意外截断长跑）
    # 达到或超过该值后，**当前轮完成后**优雅停机，不中断正在进行的请求
    max_total_tokens: Optional[int] = None
    # 冒烟用：第一次成功解析出候选后停机（本轮的评价/入池/记录照常完成）
    stop_after_first_response: bool = False

    # ------------------------------------------------------------------ 输出
    run_llm: bool = True
    run_random: bool = True
    make_plots: bool = True

    def profile(self, name: str = None) -> str:
        name = name or self.default_profile
        if name not in self.profile_weights:
            raise ValueError(f"未知目标 profile: {name}")
        return name

    def clone(self, **kwargs):
        """返回一个覆盖了部分参数的副本，避免修改全局默认值。"""
        from dataclasses import replace
        return replace(self, **kwargs)
