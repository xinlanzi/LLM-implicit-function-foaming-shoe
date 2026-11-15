# -*- coding: utf-8 -*-
"""TPMS 隐函数组合优化器包。

显式导出主流程符号，使下面两种调用方式等价：
    python optimizer/main.py
    python -m optimizer.main
"""

from .optimizer import (  # noqa: F401
    UsageMeter,
    build_prompt,
    compare_arms,
    init_population,
    run_arm,
    select_examples,
    summarize_arm,
    validate_candidate,
)
