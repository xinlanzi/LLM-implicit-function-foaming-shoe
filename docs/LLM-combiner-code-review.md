# LLM-combiner 分支代码审查报告

| 项目 | 内容 |
|---|---|
| 仓库 | `xinlanzi/LLM-implicit-function-foaming-shoe`（私有） |
| 审查分支 | `LLM-combiner`（`248b286be5f0bdb44dea66da0ac3df336b375d51`） |
| 对比基线 | `main`（`6c7df8f220276054c30f8786dfd6f69a4db3fedf`） |
| 审查时间 | 2026-11-15 |
| 审查范围 | `optimizer/` 全部 10 个模块 + `组合优化器技术文档.md` + `outputs/` 产物 + `.gitignore` / `.gitattributes` |
| 审查方式 | 全量源码静态审查 + 关键公式复刻实测 + 导入结构实测 |
| 未覆盖 | `optimizer/geometry/` 子模块（约 76KB，STL/STEP 导出与查看器，未纳入本轮） |

---

## 一、结论摘要

**一句话结论：代码架构和文档质量都不错（评价器口径诚实、模拟器设计有想法），但目前这个分支在别人的机器上跑不起来，且"真实 LLM 接入"没有任何一次成功调用的证据。**

严重度分布：

| 级别 | 数量 | 含义 |
|---|---|---|
| P0 阻断 | 4 | 直接导致无法运行，或使核心结论无效 |
| P1 高 | 7 | 逻辑/成本/工程缺陷，会产生错误结论或浪费 |
| P2 中 | 3 | 代理指标退化，影响排序可信度 |
| P3 低 | 4 | 规范与卫生问题 |
| 观察项 | 1 | 边界行为，暂不建议改动 |

三个最该先处理的点：

1. **本分支的所有实验结论都建立在 `proposer_backend=mock` 上**，而 commit 标题写的是"完善组合优化器的真实 LLM 接入" —— 名实不符，这是最容易被追问的地方。
2. **数据源（Excel）既不在版本库里、路径又是硬编码的 Windows 绝对路径** —— 换机器必挂。
3. **`outputs/` 里 87MB 的 zip 已经进了 git 历史**，且没走 LFS。

---

## 二、P0 阻断级（4 条）

### R-01 真实 LLM 路径从未被实测

**位置**：`optimizer/proposer.py` → `OpenAICompatibleProposer`、`chat_once`；`outputs/run_log.json`

**现象**：本分支的提交信息是 `feat: 完善组合优化器的真实 LLM 接入`（新增 1468 行），但仓库内的实验结果全部来自本地模拟器。

**证据**：

- `outputs/high_support_high_elasticity_solutions.md` 第 5.3 节原文：
  > 本次没有真实 LLM API 凭证/网络，`proposer_backend=mock` 是本地上下文感知模拟器……接入真实 API 的代码路径已实现但未实测。
- 同一文档第 1 节表格里"提议器"一栏 = `默认 mock`。
- 由此推论：`run_log.json` 中不存在 `usage_source: "reported"` 的记录（若有真实调用，服务端会返回 usage）。

**影响**：报告里"LLM(模拟) 候选均分 0.8456 vs 随机基线 0.5533、p≈7.3×10⁻¹⁶"这个结论**只能支撑"上下文感知模拟器优于随机"**，不能支撑"LLM 优于随机"。这两件事在答辩语境下差别很大。

**建议**：

- 先跑一次最小真实调用并留证，再决定是否保留这个 commit message：

```bash
python optimizer/main.py --rounds 3 --stop-after-first-response --max-total-tokens 20000
```

验收标准：`run_log.json` 里出现 `"usage_source": "reported"`，且 `histories.llm[].status` 出现 `evaluated`。

- 若短期拿不到凭证，建议把 commit message 与文档统一改成"新增真实 LLM 接入路径（未实测）"，避免名实不符。

---

### R-02 Excel 数据源为硬编码 Windows 绝对路径，且数据不在版本库

**位置**：`optimizer/config.py` → `Config.excel_path` / `Config.output_dir`；`.gitignore`

**现象**：

```python
excel_path: str = r"c:\Users\hp\Desktop\LLM\晶格结构统一知识库_Structure_Master_V1.0.xlsx"
output_dir: str = r"c:\Users\hp\Desktop\LLM\outputs"
```

同时 `.gitignore` 第一条：

```
*.xlsx
```

**根因**：数据文件被主动排除在版本库外，但唯一的读取路径又写死在别人的桌面。`main.py` 里**没有 `--excel` 参数**，只有 `--outdir`。

**影响**：

- 任何新克隆 → `load_seeds()` 抛 `FileNotFoundError`，整个流水线无法启动，且无法通过参数绕过。
- 在 macOS / Linux 上还有二次伤害：`main()` 的执行顺序是 `os.makedirs(cfg.output_dir)` 在前、`load_seeds()` 在后。反斜杠在 macOS 上是合法文件名字符，所以会先在当前目录**创建一个名叫 `c:\Users\hp\Desktop\LLM\outputs` 的目录**，然后才崩。

**建议**：

1. Excel 放进 `database/`（`main` 分支原本就有这个目录），或放到 `data/` 并在 `.gitignore` 里改成白名单；
2. 默认路径改相对：

```python
excel_path: str = str(Path(__file__).resolve().parent.parent / "data" / "Structure_Master_V1.0.xlsx")
```

3. `main.py` 增加 `--excel` 参数，缺省回落到上面的路径。

---

### R-03 `python -m optimizer.main` 必然 ImportError（已实测）

**位置**：`optimizer/__init__.py`（空壳）、`optimizer/main.py` 顶部导入

**现象**：

```python
# main.py
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import Config, LLM_PROVIDERS
from optimizer import UsageMeter, build_prompt, compare_arms, run_arm   # ← 这里
```

`optimizer/` 目录下同时存在 `optimizer.py`（模块）和 `__init__.py`（包）。用 `python optimizer/main.py` 时，`sys.path[0]` 是脚本所在目录，`import optimizer` 命中 `optimizer.py`，恰好能跑；用 `python -m optimizer.main` 时，`sys.path[0]` 是当前工作目录，`import optimizer` 命中的是**包**，而 `__init__.py` 只有一句 docstring。

**实测证据**（按原目录结构复刻后执行）：

```
python optimizer/main.py    -> exit=0 | IMPORT OK
python -m optimizer.main    -> exit=1 | ImportError: cannot import name 'UsageMeter' from 'optimizer'
```

**影响**：任何 IDE 运行配置、CI、`pytest`、别人按标准方式调用都会失败；而作者本人用 `python optimizer/main.py` 恰好绕过了，属于典型的"只在一个人机器上能跑"。

**建议**：`optimizer/__init__.py` 显式导出，使两种调用方式都成立：

```python
"""TPMS 隐函数组合优化器包。"""
from .optimizer import (  # noqa: F401
    UsageMeter, run_arm, build_prompt, validate_candidate,
    select_examples, init_population, summarize_arm, compare_arms,
)
```

同时把 `main.py` 里的 `from config import ...` 统一改为包内相对导入（或加 `try/except ImportError` 兼容两种入口）。

---

### R-04 87MB 的 zip 被提交进 git，且没有走 LFS

**位置**：`outputs/combination_optimizer_v1.zip`（873 855 189 bytes，随 `f653983` 引入）

**现象**：

- `outputs/` 下有两个大 zip：`combination_optimizer_v1.zip`（87MB）与 `outputs/geometry/combination_optimizer_batch1_geometry.zip`；
- `.gitattributes` 只写了 `*.zip binary`，**没有任何 `filter=lfs` 规则**；
- `.gitignore` 只挡了 STL（注释还写着"大体积 STL 不进 git 历史"），**漏掉了 zip**。

**影响**：仓库克隆体积被永久性放大；即使事后 `git rm`，历史里仍然存在，无法通过普通提交回收。

**建议**：

1. 立即把 zip 加进 `.gitignore` 并移出版本库：

```gitignore
outputs/**/*.zip
*.zip
```

2. 如果确实需要版本化大文件，改用 LFS：

```
*.zip filter=lfs diff=lfs merge=lfs -text
*.stl filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
```

3. 已进历史的 87MB 需要用 `git filter-repo`（或 BFG）重写历史后再强推 —— 这一步会改写所有提交哈希，**必须提前和协作者对齐**。

---

## 三、P1 高级（7 条）

### R-05 本分支是"孤儿历史"，与 `main` 无共同祖先

**现象**：`LLM-combiner` 的提交历史**只有 2 个提交**（`248b286`、`f653983`），`main` 有 6 个；且该分支的树里**没有 `README.md`、没有 `database/`**，而 `main` 两者都有。

**证据**：`f653983` 的提交统计显示 **全部为 `status: added`、零删除**（新增 1 053 995 行），却在结果树上不含 `README.md` / `database/` —— 说明它的父提交本来就不含这些文件；即该分支不是从 `main` HEAD 分出来的。

**影响**：

- `git merge main` 会直接报 `refusing to merge unrelated histories`；
- 答辩或交付时若拿 `LLM-combiner` 当"最新完整版本"，会丢 `database/`（知识库表结构与 scripts）和 `README.md`。

**建议**：先定主线。若以 `main` 为主线，则：

```bash
git checkout main
git checkout -b integrate-optimizer
git checkout LLM-combiner -- optimizer outputs 组合优化器技术文档.md
git commit -m "feat: 合入组合优化器（来自 LLM-combiner）"
```

不要尝试用 `--allow-unrelated-histories` 硬合，那样会得到两份无关的根提交。

---

### R-06 默认命令实际消耗是成本预估的 3 倍

**位置**：`optimizer/main.py` → `--extra-profiles` 默认值；成本预估段

**现象**：

```python
parser.add_argument("--extra-profiles", type=str,
                    default="high_support,high_elasticity", ...)
```

即默认会额外各跑 50 轮 LLM-only。合计 **3 个 profile × 50 轮 = 150 次请求**。

而成本预估只算了主 profile：

```python
worst = cfg.rounds * per_round
print(f"  最坏 {cfg.rounds} 轮 ≈ {worst:,} token（不含重试；重试会额外消耗）")
```

**影响**：真正接上 `DeepSeek-V4-Flash` 后，实际花费会是屏幕上提示数字的约 3 倍。这类"预估与真实不符"在科研经费/额度场景里很致命。

**建议**：

```python
n_profiles = 1 + len([p for p in args.extra_profiles.split(",") if p.strip()])
worst = n_profiles * cfg.rounds * per_round
print(f"  最坏 {n_profiles} 个目标 × {cfg.rounds} 轮 ≈ {worst:,} token")
```

或把 `--extra-profiles` 默认改成空字符串，需要时显式开启。

---

### R-07 `n_high_examples` / `n_random_examples` 是死配置，且与预估逻辑不一致

**位置**：`optimizer/config.py`（声明）vs `optimizer/optimizer.py` → `select_examples`

**现象**：

```python
# config.py
n_high_examples: int = 3
n_random_examples: int = 1

# optimizer.py → select_examples
n_high = 4 if round_num <= 10 else 2      # ← 硬编码，没读 cfg
n_random = 1 if round_num <= 10 else 2    # ← 硬编码，没读 cfg
```

**连带问题**：`main.py` 的成本预估与 `--debug-one-call` **却按 `cfg.n_high_examples` 构造 prompt**：

```python
_prompt = build_prompt(profile, _examples[:max(1, cfg.n_high_examples)], ...)
```

**影响**：屏幕上"prompt ≈ N token"的统计对象，和真正每轮发出去的 prompt 不是同一个东西；且代码注释声称"文档建议 2~4"，实际跑的是 4 和 2，配置项形同虚设。

**建议**：让 `select_examples` 真正读配置，并把"前 10 轮加权"改成显式参数：

```python
n_high = cfg.n_high_examples if round_num <= cfg.warmup_rounds else cfg.n_high_examples_late
n_random = cfg.n_random_examples
```

---

### R-08 重复候选被当作"提议器失败"，并用同一 prompt 重试

**位置**：`optimizer/proposer.py` → `OpenAICompatibleProposer.propose`

**现象**：

```python
sig = canonical_signature(terms, coeffs)
if sig and sig not in seen_signatures:
    return Proposal(...)
dup = ValueError("候选与已见签名重复")
dup.status = "duplicate"
last_err = dup
...
# 循环末尾
if attempt < retries - 1:
    time.sleep(...)          # ← 用完全相同的 prompt 再来一次
```

**影响链**：

1. 同一 prompt 重发，模型很可能再次给出重复候选 → 白烧一次 token；
2. 走的是 `except Exception` 分支，最终抛 `RuntimeError`，`run_arm` 记为**提议器失败**（`reason="提议器失败：..."`），而状态却是 `duplicate`；
3. `consecutive_reject += 1`，污染收敛统计与 `status_counts`。

**建议**：重复应当是一次"正常但未入池的轮次"，而不是异常：

- 在 `propose` 里把 `seen` 转成提示注入 prompt：`"以下组合已出现过，请避开：..."`；
- 或者在 `run_arm` 里捕获 `status == "duplicate"` 并记为 `status="duplicate"`、`reason="与已见组合重复"`，不计入失败。

---

### R-09 Mock 提议器"先截断系数、后合并同名项"，导致越界

**位置**：`optimizer/proposer.py` → `MockLLMProposer.propose`

**现象**（顺序即问题）：

```python
terms, coeffs = _trim_terms(terms, coeffs, self.cfg.max_terms)
coeffs = [_non_tiny(_clamp_coeff(c, self.cfg.coeff_min, self.cfg.coeff_max), self.rng) for c in coeffs]
# ↑ 此处已 clamp 到 [-2, 2]

if len(set(terms)) != len(terms):
    merged: Dict[str, float] = {}
    for t, c in zip(terms, coeffs):
        merged[t] = merged.get(t, 0.0) + c        # ← 合并，结果可能 1.5 + 2.0 = 3.5
    terms, coeffs = list(merged.keys()), list(merged.values())
```

**影响**：合并后系数可达 3.5，超出 `coeff_max=2.0`，随后 `validate_candidate` 判"系数超出范围"直接丢弃 —— 一轮白跑。

**建议**：把合并提到 clamp 之前，或在合并后再 clamp 一次：

```python
if len(set(terms)) != len(terms):
    merged = {}
    for t, c in zip(terms, coeffs):
        merged[t] = merged.get(t, 0.0) + c
    terms, coeffs = list(merged.keys()), list(merged.values())
    coeffs = [_non_tiny(_clamp_coeff(c, self.cfg.coeff_min, self.cfg.coeff_max), self.rng) for c in coeffs]
```

---

### R-10 `--check-llm` 自检"看起来过了"，但验证不到关键项

**位置**：`optimizer/proposer.py` → `check_llm_endpoint`

**现象**：

```python
payload = {"model": cfg.llm_model, "temperature": 0, "max_tokens": 16, "stream": False,
           "messages": [{"role": "user", "content": "ping"}]}
...
data = json.loads(raw)
ok = bool(data.get("choices"))        # ← 只判断 choices 字段存在
```

**影响**：

- 完全不看 `content` 是否为空、是否被 `finish_reason=length` 截断、是否能解析出 JSON；
- `max_tokens=16` 对带思维链的模型（`DeepSeek-V4-Flash` 支持 Non-Think / Think High / Think Max）严重不足；
- 结果可能是：自检打印"✓ 连通成功：凭证、端点、模型名均可用"，正式跑 50 轮却因为 content 全空而全部失败。

这与 `config.py` 里作者自己的注释（"推理模型的思维链会占满 max_tokens 并导致 content 为空"）互相矛盾。

**建议**：自检复用正式路径，并把结论分级：

```python
out = chat_once(cfg, "ping", stream=False, max_tokens=cfg.llm_max_tokens)
if out["content"].strip():
    return "ok", ...
elif out["reasoning"]:
    return "empty_content", "HTTP 200 但 content 为空（思维链占满额度），建议提高 --max-tokens"
```

---

### R-11 文档里的"接入真实 LLM"命令与当前代码不符

**位置**：`outputs/high_support_high_elasticity_solutions.md` 第 1 节

**现象**：文档写：

```powershell
$env:LLM_API_KEY="你的 key"
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_MODEL="gpt-4o-mini"
D:\anaconda\python.exe optimizer\main.py --llm-backend openai
```

但代码实际：

- `argparse` 的 `choices=list(LLM_PROVIDERS)` 只有 `mock`、`siliconflow` → 传 `openai` 直接报 `invalid choice`；
- `config.py` 中 `llm_base_url` / `llm_model` **明确拒绝环境变量覆盖**（"base_url 与模型名不接受命令行或环境变量覆盖"）；
- 密钥只由 `proposer.resolve_api_key` 的 tkinter 弹窗读取，`LLM_API_KEY` 环境变量**根本不会被读**。

**影响**：任何人照着文档复现都会失败，且会被误导成"环境变量方式可用"。

**建议**：重写该段为实际可执行的命令：

```powershell
python optimizer\main.py --llm-backend siliconflow --max-tokens 2048
# 运行后会弹出密钥输入框（仅进程内存，不落盘）
```

---

## 四、P2 中级（3 条）—— 代理指标退化

以下结论来自**复刻 `evaluator.py` 公式后的实测**（64³ 网格，与源码同参数），可与仓库内报告的数字交叉验算（gyroid 算出 0.9511、P 算出 0.9087、Diamond 算出 0.9339，与 `solutions.md` 表格完全一致，说明复刻可信）。

### R-12 "1D 层状/2D 柱状结构防刷分"未真正生效

实测结果：

| 结构 | score | 备注 |
|---|---|---|
| gyroid（种子 S001） | **0.9511** | 与仓库报告一致 |
| Diamond | 0.9339 | 一致 |
| P surface | 0.9087 | 一致 |
| **`cos_x`（两层平板）** | **0.7417** | 最退化结构，却高于下一行 |
| Neovius-like（合法三维结构） | 0.7309 | |
| `cos2x`（三层平板） | 0.6507 | |
| `cos_x_cos_y`（柱状） | 0.4786 | |

**根因**：平面上 `curvature = 0`，于是 `smoothness = 1.000`、`uniformity = 1.000`；周期性折返让 `connectivity = 1.000`、`density_window` 也满分。唯一在扣分的只有 `isotropy`（权重 0.22 / 0.19），4 项满分把它拉了回来。

**建议**：把 `isotropy` 从"加权项"改为"乘法门"，让退化结构无法靠其他项补偿：

```python
score = (w_support * support + w_elastic * elastic) * (0.35 + 0.65 * isotropy)
```

改完后需要重跑基准，因为所有结构的绝对分数都会变（相对排序基本保持）。

### R-13 `isotropy` 实际是二值开关，不是连续指标

实测：

| 结构 | isotropy |
|---|---|
| gyroid / P / Diamond / `cos2x cos2y cos2z` | **1.000** |
| `cos_x` / `cos_x_cos_y` / `cos2x`（层面/柱状） | **0.000** |

立方对称结构的法向协方差正好正比于单位阵，`eig[0]/eig[-1]` 必然等于 1。所以这个量只能回答"是不是各向同性"，**不提供程度信息**。

**影响**：报告中 `0.972 / 0.945 / 0.601` 这类三位小数的解读，实际是离散化噪声，不是"各向同性程度"的连续差异。若被问"0.972 与 1.000 的物理差别"，无法作答。

**建议**：要么承认它是布尔门（改名 `is_isotropic`，权重意义重写），要么换成真正连续的度量，例如法向分布的球面熵：

```python
# 用 n 在球面上的分布熵代替特征值比
p = histogram_on_sphere(nx, ny, nz, bins=32)
iso_entropy = entropy(p) / np.log(bins)   # 0~1，且连续
```

### R-14 `ρ* ≈ 0.5` 恒定导致密度维度提前饱和

**现象**：`skeletal` 约定（实体相 `F <= 0`）下，所有极小曲面的相对密度都接近 0.500（实测 gyroid 0.500、P 0.501、Diamond 0.500），因此：

- `density_norm = clip((0.500-0.10)/(0.35-0.10), 0, 1) = 1.0`（恒满）
- `density_window = exp(-0.5*((0.500-0.35)/0.15)^2) ≈ 0.607`（恒为同一常数）

**影响**：support 里 0.32 的权重、elastic 里 0.15 的权重**对全部候选贡献相同常数**，等于这两项在排序中不起作用。"支撑—弹性权衡"实际上只剩曲率、连通、各向同性三维在起作用。

作者在 `solutions.md` 5.3 节已承认此局限，但代码层面没有缓解。

**建议**：（与作者的后续计划一致）先做 sheet 型实体化 `|F| <= t·|∇F|`，让 ρ* 成为可扫描参数（0.15 / 0.25 / 0.35），这两项才会真正产生区分度。

---

## 五、P3 低级（4 条）+ 观察项（1 条）

### R-15 缺依赖声明，文档示例全是 Windows 路径

- 无 `requirements.txt` / `environment.yml`。实际依赖为 `numpy`、`scipy`、`scikit-image`、`pandas`、`matplotlib`、`openpyxl`、`tkinter`。
- `main.py` docstring 的示例全是 `D:\anaconda\python.exe optimizer\main.py`。

**建议**：补 `requirements.txt`，并在 `README` 里给出 macOS / Windows 两套命令。

### R-16 允许把运行产物提交进版本库

`outputs/run_log.json`（323KB）、`outputs/candidates.csv`（99KB）、`outputs/comparison.png`（357KB）均已入库。每跑一次就会再叠一份，历史持续膨胀。

**建议**：`outputs/*.{json,csv,png}` 进 `.gitignore`，只保留结论性文档（如 `solutions.md`）；或在 `outputs/` 下建 `.gitattributes` 走 LFS。

### R-17 对比图的中文字体在 macOS / Linux 上会缺字

```python
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
```

前两者只存在于 Windows；`DejaVu Sans` 不含 CJK 字形，回落结果是方块（豆腐块）。

**建议**：加跨平台候选 + 显式探测：

```python
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Microsoft YaHei",
                                   "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
```

### R-18 密钥只能靠 tkinter 弹窗，无桌面环境直接失败

`resolve_api_key()` → `_ask_key()` 在没有显示器的 SSH / CI / 后台进程里抛 `CredentialError` 并中止。

这是**安全设计上的正确取舍**（不落盘、不读环境变量），但缺少可用性备选。

**建议**：保留弹窗为默认，额外提供显式的 `--api-key-stdin`（从标准输入读一次，不进历史、不进日志），仅在用户主动指定时启用。

### R-19（观察项）`parse_llm_output` 的两个边界

实测 11 组输入，9 组正确解析（含 ```json 围栏、前后夹解释文字、字符串内含 `}`、转义引号、二次编码、数组包对象）。两个失败：

| 输入 | 结果 |
|---|---|
| JSON 前出现未配对的 `{`（如 `说明：{ 见下 {json}`） | `JSON 对象不完整` |
| 尾逗号 `{"terms": [...], "coeffs": [...],}` | `JSONDecodeError` |

两者都会进入"解析失败 → 重试/丢弃"路径，不会崩流程，**暂不建议改**。若要兜底，可在 `json.loads` 失败后追加一次尾逗号清洗（`re.sub(r",\s*[}\]]", lambda m: m.group(0)[-1], seg)`）。

---

## 六、附：实测记录（可复现）

为排除误报，以下两条**经实测确认不是 bug**，一并记录以免后续重复排查：

| 曾经的怀疑 | 实测结论 |
|---|---|
| `area_target = 0.50` 量纲错了（怀疑实际应为 0.0125） | **不成立**。gyroid 实测 `area/V = 0.473`，P 为 `0.363`，Diamond 为 `0.567` —— 与 `area_target=0.50, sigma=0.22` 完全匹配 |
| `measure.label(connectivity=1)` 少算了周期连通 | **不成立**。`cos_x_cos_y` 用 6-连通得 0.500、26-连通得 1.000，差别来自**点/棱接触**，物理上不应算连通，6-连通是正确选择 |

复现方式（Linux + `numpy` + `scikit-image`，与源码同参数 64³ / level=0 / spacing=h）：

```python
n, dom = 64, 2*np.pi ; h = dom/n
ax = np.linspace(0, dom, n, endpoint=False)
X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
F = np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)   # gyroid
v, f, _, _ = measure.marching_cubes(F, 0.0, spacing=(h, h, h), allow_degenerate=False)
area = 0.5 * np.linalg.norm(np.cross(v[f][:,1]-v[f][:,0], v[f][:,2]-v[f][:,0]), axis=1).sum()
print(area / dom**3)    # -> 0.473
```

---

## 七、建议的修复顺序与验收清单

| 顺序 | 动作 | 对应条目 | 验收标准 |
|---|---|---|---|
| 1 | 数据进库 + 相对路径 + `--excel` | R-02 | 新克隆可在 macOS / Windows 直接跑 `--llm-backend mock` |
| 2 | 补 `requirements.txt` | R-15 | 干净虚拟环境 `pip install -r` 后无 ImportError |
| 3 | `__init__.py` 导出 + 导入统一 | R-03 | `python -m optimizer.main` 与 `python optimizer/main.py` 都 exit=0 |
| 4 | 清理 zip 与产物 + LFS 规则 | R-04 / R-16 | 新克隆体积 < 10MB；`git count-objects -vH` 无异常 |
| 5 | 跑通 3 轮真实 LLM 并留证 | R-01 | `run_log.json` 出现 `usage_source: "reported"` |
| 6 | 修成本预估 + `extra-profiles` 默认 | R-06 | 预估数字 = 实际请求数 × 每轮 token 之和（误差 < 5%） |
| 7 | 修 `select_examples` 读配置 | R-07 | 改 `n_high_examples` 后 prompt 长度确实变化 |
| 8 | 修重复处理与 clamp 顺序 | R-08 / R-09 | 重复轮次状态为 `duplicate`，不再计入 `proposer_failed` |
| 9 | 改写文档接入示例 | R-11 | 按文档命令复制粘贴即可跑通 |
| 10 | `isotropy` 改乘法门 + 重跑基准 | R-12 / R-13 | `cos_x` 得分显著低于 P / D；基准数字全部重算 |
| 11 | 决定 `main` 与 `LLM-combiner` 的主线归属 | R-05 | `git log --graph --all` 可见单一主线 |

**给答辩的一句话建议**：现阶段不要把"LLM 优于随机基线"当作结论陈述；更稳的说法是"上下文感知提示策略优于随机采样（模拟器口径，p < 0.001，效应量 0.93），真实 LLM 接入路径已打通、待接入凭证后实测"。前者会被追问，后者不会。

---

## 八、本轮未覆盖

- `optimizer/geometry/` 下 7 个模块（`build_geometry` / `build_viewer` / `faceted_step` / `mesh_builder` / `mesh_io` / `render_png` / `solutions`，约 76KB）以及 vendored 的 `three.min.js` / `OrbitControls.js` —— 属于 STL/STEP 导出与 HTML 查看器子系统，与组合优化主流程无直接依赖，未做代码级审查。
- `outputs/run_log.json`（323KB）与 `candidates.csv`（99KB）仅做了摘要核对，未逐行审计。
- 未做实际 FEA / 制造可行性评估（本仓库本身也不包含这些能力）。

如需继续，建议下一轮优先审 `optimizer/geometry/mesh_io.py` 与 `faceted_step.py`（文件写出与几何重建，最易出静默错误）。
