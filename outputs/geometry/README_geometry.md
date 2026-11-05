# TPMS 组合结构几何导出说明

> 本目录由 `optimizer/geometry/build_geometry.py` 生成，原始 Excel 与技术文档未修改。  
> **所有 score 都是优化器里的几何代理指标，不是 FEA 结果。** 这里只导出几何，
> 没有计算应力、应变、屈曲或疲劳。

## 1. 实际导出了什么

| 格式 | 数量 | 状态 | 说明 |
|---|---|---|---|
| 二进制 STL（mm） | 20 个 | ✅ 已实际写出并回读校验面数 | 5 方案 × (bulk/sheet) × (1×1×1/2×2×2) |
| faceted STEP（AP203） | 5 个 | ⚠️ 降级方案，未在 CAD 中打开验证 | 三角面 B-Rep，不是 NURBS 光滑 B-Rep |
| 多视角 PNG | 10 张 | ✅ 已生成 | 每个方案 bulk/sheet 各一张，4 视角 × 2 平铺 |
| 单文件 3D 查看页 | 1 个 | ✅ 已生成 | three.js 内嵌，`file://` 可离线打开，约 38.4 MB |
| manifest | CSV + JSON | ✅ | 全部网格体检指标与配置 |

**STEP 说明（重要）**：本机没有 `OCP` / `pythonocc-core` / `cadquery`，无法生成真正的
NURBS B-Rep。这里输出的是 `FACETED_BREP`（三角面 B-Rep），用 40³/单胞
的低分辨率网格控制体积。它适合在 CAD 中查看/测量，**不适合高精度曲面加工或参数化编辑**；
并且未在 SolidWorks/CATIA 中实际打开验证。若需要真正 B-Rep，请安装 CAD 内核后自行转换：

```powershell
# 方案 A：conda 安装 pythonocc-core（推荐，Windows 需 conda-forge）
conda install -c conda-forge pythonocc-core

# 方案 B：pip 安装 cadquery（自带 OCP 内核，体积较大）
D:\anaconda\python.exe -m pip install cadquery
```

安装后可把 STL 导入 CAD 做“网格→曲面”重建，或直接用 OCC 从三角面缝合为 shell。

## 2. 实体化定义（为什么 STL 是实体而不是零厚度片）

- `F(x,y,z)=0` 本身是零厚度极小曲面，直接 marching cubes 只有开放曲面片；
- **bulk 主交付**：实体相 `F<=0`（与优化器评价器的实体相约定一致），ρ* 约 0.5；
- **sheet 附带**：`|F|<=t`，`t=0.13`（隐函数场单位，不是毫米）。
  估算壁厚 ≈ `2t/|∇F| × cell/2π`；本批实际估算值见 manifest 的
  `wall_thickness_mm_est`（单胞 20 mm 时约 0.56~0.86 mm，约为单胞的 2.8%~4.3%）；
  **t 的选择依据**：建议区间是 0.10~0.15，0.13 取其中段；在这个取值下，
  sheet 1×1 的 96 体素/单胞、2×2 的 48 体素/单胞时，壁厚大约跨 3~4 个节点，
  marching cubes 能稳定分辨而不会断壁；t 偏小会更接近真实薄壁、但粗网格容易断成碎片，
  t 偏大则逐渐退化成块体，失去“薄壁件”外观。
- 盒子边界用 `eps - B_in`（`eps=0.25` 体素）封口，得到
  watertight 实体；封口面是**样品切断处的材料截面**，不是额外结构。
- 朝向：用有符号体积判断整体朝向，必要时翻转全部面，保证法向朝外。

## 3. 五个方案（分数来自优化器实际评价）

| 编号 | 名称 | 文件短名 | score | support / elastic | 轮次 |
|---|---|---|---|---|---|
| 01 | Gyroid 反号变体 | `GyroidFlip` | 0.944925 | 0.9747 / 0.9085 | 24 |
| 02 | Diamond 等幅变体 | `DiamondBalanced` | 0.933838 | 0.9664 / 0.8940 | 18 |
| 03 | Diamond + x 向余弦 | `DiamondPlusX` | 0.919396 | 0.9533 / 0.8779 | 1 |
| 04 | Diamond + y 向余弦 | `DiamondPlusY` | 0.898590 | 0.9346 / 0.8545 | 46 |
| 05 | Gyroid × Diamond 交叉 | `GyroidDiamondMix` | 0.798256 | 0.8350 / 0.7533 | 17 |

方程与组合理由（完整版见 `../high_support_high_elasticity_solutions.md`）：

- **01 Gyroid 反号变体**：`F = -0.9446·sin(x)cos(y) + 0.9723·sin(y)cos(z) + 0.9816·sin(z)cos(x) = 0`
  Gyroid 三条螺旋分量近等幅，三向互锁；第一项反号是本方案的关键变异，在保持三维连通与各向同性的同时改变两侧实体相分布。支撑来自三轴连续承力路径，弹性来自螺旋曲面的负高斯曲率可侧向让位。适用：轻量化承力芯、换热/混合骨架、需要各向同性的场合。
- **02 Diamond 等幅变体**：`F = 0.9235·cos(x)cos(y)cos(z) - 0.9301·sin(x)sin(y)sin(z) = 0`
  三轴余弦节点网络（承载集中）+ 三轴正弦鞍面（连接节点）；等幅反对称即 Schwarz Diamond 的标准形式。节点提供高支撑，鞍面双曲曲率提供可恢复变形。适用：弯曲主导、需要高比强度的夹芯/支撑结构。
- **03 Diamond + x 向余弦**：`F = 1.0472·cos(x)cos(y)cos(z) - 1.0811·sin(x)sin(y)sin(z) + 0.1942·cos(x) = 0`
  在 D 面上叠加小幅 cos(x) 相位偏置，沿 x 向增加定向承载能力；isotropy 由 1.000 降到约 0.975，是典型的“支撑换弹性”。适用：主载荷方向已知的单向压缩支柱、定向导热/流动结构。
- **04 Diamond + y 向余弦**：`F = 1.0953·cos(x)cos(y)cos(z) - 0.9327·sin(x)sin(y)sin(z) + 0.3112·cos(y) = 0`
  与 D+x 同族，但 y 向偏置更强（0.31），各向异性更明显、弹性代理更低；适合单向剪切/弯曲梁，也可作为“偏置过强”的对照。适用：需要沿 y 向增强刚度的梁/板。
- **05 Gyroid × Diamond 交叉**：`F = 1.0295·cos(x)cos(y)cos(z) - 0.8589·sin(x)sin(y)sin(z) + 0.8835·sin(x)cos(y) = 0`
  跨族交叉：D 的节点/鞍面 + Gyroid 的螺旋项。结果 isotropy 降到约 0.60、曲率升高，提示结构开始柱状化。保留为反例：跨族混合不一定优于任一母体。适用：需要混合拓扑/梯度过渡的研究样件，而非性能最优件。

## 4. 网格体检表

`watertight=是` 表示无边界边且无非流形边；`solid_components_periodic` 是按周期边界
合并后的**实体相**连通分量（期望=1），与表面 shell 分量不是同一个概念。

| 文件 | 相 | 平铺 | 三角面 | watertight | 边界边 | 非流形边 | 表面分量 | 周期实体分量 | 体积 mm³ | ρ* | bbox mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `01_GyroidFlip_bulk_1x1.stl` | bulk | 1x1x1 | 251,468 | 是 | 0 | 0 | 1 | 1 | 3950.8 | 0.5000 | 19.922 x 19.922 x 19.922 |
| `01_GyroidFlip_bulk_2x2.stl` | bulk | 2x2x2 | 406,836 | 是 | 0 | 0 | 1 | 1 | 31592.0 | 0.5000 | 39.844 x 39.844 x 39.844 |
| `01_GyroidFlip_sheet_1x1.stl` | sheet | 1x1x1 | 180,000 | 是 | 0 | 0 | 1 | 1 | 678.3 | 0.0873 | 19.896 x 19.896 x 19.896 |
| `01_GyroidFlip_sheet_2x2.stl` | sheet | 2x2x2 | 351,408 | 是 | 0 | 0 | 1 | 1 | 5374.1 | 0.0877 | 39.792 x 39.792 x 39.792 |
| `02_DiamondBalanced_bulk_1x1.stl` | bulk | 1x1x1 | 279,128 | 是 | 0 | 0 | 1 | 1 | 3948.2 | 0.5000 | 19.922 x 19.922 x 19.922 |
| `02_DiamondBalanced_bulk_2x2.stl` | bulk | 2x2x2 | 452,656 | 是 | 0 | 0 | 1 | 1 | 31561.0 | 0.5000 | 39.844 x 39.844 x 39.844 |
| `02_DiamondBalanced_sheet_1x1.stl` | sheet | 1x1x1 | 234,736 | 是 | 0 | 0 | 1 | 1 | 1256.3 | 0.1636 | 19.896 x 19.896 x 19.896 |
| `02_DiamondBalanced_sheet_2x2.stl` | sheet | 2x2x2 | 436,832 | 是 | 0 | 0 | 1 | 1 | 9913.1 | 0.1657 | 39.792 x 39.792 x 39.792 |
| `03_DiamondPlusX_bulk_1x1.stl` | bulk | 1x1x1 | 271,496 | 是 | 0 | 0 | 1 | 1 | 3953.7 | 0.5000 | 19.922 x 19.922 x 19.922 |
| `03_DiamondPlusX_bulk_2x2.stl` | bulk | 2x2x2 | 454,704 | 是 | 0 | 0 | 1 | 1 | 31607.4 | 0.5000 | 39.844 x 39.844 x 39.844 |
| `03_DiamondPlusX_sheet_1x1.stl` | sheet | 1x1x1 | 229,136 | 是 | 0 | 0 | 1 | 1 | 1097.7 | 0.1414 | 19.896 x 19.896 x 19.896 |
| `03_DiamondPlusX_sheet_2x2.stl` | sheet | 2x2x2 | 432,864 | 是 | 0 | 0 | 1 | 1 | 8666.0 | 0.1411 | 39.792 x 39.792 x 39.792 |
| `04_DiamondPlusY_bulk_1x1.stl` | bulk | 1x1x1 | 266,664 | 是 | 0 | 0 | 1 | 1 | 3955.7 | 0.5000 | 19.922 x 19.922 x 19.922 |
| `04_DiamondPlusY_bulk_2x2.stl` | bulk | 2x2x2 | 447,216 | 是 | 0 | 0 | 1 | 1 | 31623.9 | 0.5000 | 39.844 x 39.844 x 39.844 |
| `04_DiamondPlusY_sheet_1x1.stl` | sheet | 1x1x1 | 226,800 | 是 | 0 | 0 | 1 | 1 | 1161.4 | 0.1498 | 19.896 x 19.896 x 19.896 |
| `04_DiamondPlusY_sheet_2x2.stl` | sheet | 2x2x2 | 430,688 | 是 | 0 | 0 | 1 | 1 | 9189.0 | 0.1500 | 39.792 x 39.792 x 39.792 |
| `05_GyroidDiamondMix_bulk_1x1.stl` | bulk | 1x1x1 | 271,756 | 是 | 0 | 0 | 1 | 1 | 3949.3 | 0.5000 | 19.922 x 19.922 x 19.922 |
| `05_GyroidDiamondMix_bulk_2x2.stl` | bulk | 2x2x2 | 441,942 | 是 | 0 | 0 | 1 | 1 | 31574.0 | 0.5000 | 39.844 x 39.844 x 39.844 |
| `05_GyroidDiamondMix_sheet_1x1.stl` | sheet | 1x1x1 | 213,776 | 是 | 0 | 0 | 1 | 1 | 1094.2 | 0.1404 | 19.896 x 19.896 x 19.896 |
| `05_GyroidDiamondMix_sheet_2x2.stl` | sheet | 2x2x2 | 410,496 | 是 | 0 | 0 | 1 | 1 | 8673.4 | 0.1424 | 39.792 x 39.792 x 39.792 |

> 说明：单胞 bbox 略小于标称尺寸（切面封口在盒壁内约 0.25 体素）；
> 体积是网格封闭体积，ρ* 是节点体素法向密度，两者会有小幅差异（见 `rho_mesh_bbox`）。
>
> **关于 `surface_components`（表面分量）**：它是三角网格的**表面 shell 数**，
> 可能大于 1，因为一个实体内部的封闭空腔（void）会贡献额外的内表面 shell。
> 它**不等于**实体相是否连通。判断“实体是否连通”请只看
> `solid_components_periodic`：这是对体素实体相做 26 邻域标注、并在 x/y/z
> 周期边界上合并首尾面后得到的**周期实体连通分量**，期望值为 1。

## 5. 生成参数

| 参数 | 值 |
|---|---|
| 单胞尺寸 | 20 mm（2×2×2 为 40 mm） |
| bulk 1×1×1 网格 | 128 体素/单胞（节点 129³） |
| bulk 2×2×2 网格 | 64 体素/单胞（总计 128³） |
| sheet 1×1×1 网格 | 96 体素/单胞 |
| sheet 2×2×2 网格 | 48 体素/单胞 |
| sheet t | 0.13（隐函数场单位） |
| 切面封口 eps | 0.25 体素 |
| 随机种子 | 20260101（marching cubes 无随机性，记录以便复现） |
| 等值面 | F = 0 |
| 提交通道 | binary STL，坐标单位按 mm 解释 |

> 实测耗时（本机，`D:\anaconda\python.exe`，Python 3.9）：128³ bulk 单网格约 1.0~1.4 s，
> 96³ sheet 约 0.6~0.7 s；20 个主 STL + 5 个 faceted STEP + 10 张 PNG + 离线 viewer
> 全量约 8 分钟。129³ 的 float64 标量场约 17 MB/数组，主交付分辨率可以放心用 128³。

## 6. 怎么打开

- **SolidWorks**：`File → Open → 选择 .stl`，导入选项选“实体/图形体”，单位 mm。
  faceted STEP 可能被识别为“图形体/曲面”，不是参数化实体。
- **Blender**：`File → Import → STL`（3.5+ 在 Import → Mesh → STL）。导入后
  `Shade Smooth` 可看光滑效果，`Flat` 看三角面。2×2×2 版本更适合观察连通性。
- **Meshmixer / Cura / PrusaSlicer**：直接拖入 STL；用 Meshmixer 的
  `Analysis → Inspector` 检查孔洞/非流形边（本批应为 0）。
- **Paraview**：`File → Open → STL`，Apply；可用 `Clip`/`Slice` 查看内部通道。
- **查看器**：双击 `viewer/index.html`（离线、无需服务器）。可切换结构、相、
  平铺和“隐藏切面”；隐藏切面后能直接看到盒子切开的内部通道。

## 7. 已知缺陷与抽查看点

1. **切面封口不是自由表面**：盒壁处是人为封口（材料截面），不能当成真实样品的
   外表面；做力学样件时应按实际夹具/切割方式重新处理边界。
2. **faceted STEP 未验证**：没有 CAD 内核，未在 SolidWorks/CATIA 中打开过；
   三角面 B-Rep 在大模型里可能被识别为图形体，且文件较大。
3. **分辨率损失**：marching cubes 会漏掉小于约 1 个体素的特征；sheet 薄壁在
   粗网格下可能局部变薄/变厚。完整精度看 bulk 128³ / sheet 96³ 的 STL。
4. **未做自相交检查**：STL 只保证边流形与闭合，没有检查三角形自相交；
   sheet 在曲率极高处可能出现自相交（真实薄壁件也需检查最小壁厚）。
5. **STL 无单位/材质元数据**：单位按 mm 解释；颜色、材料、公差都不在 STL 里。
6. **单胞看起来像断片是正常的**：周期结构在单胞边界被切断；要看三向连通性请用
   2×2×2 版本，或在查看器里隐藏切面。
7. **分数不是力学性能**：score/support/elastic 都是几何代理；未做 FEA，
   不能据此声称刚度、强度或回弹模量。后续验证清单见
   `../high_support_high_elasticity_solutions.md`。

## 8. 复现

```powershell
D:\anaconda\python.exe optimizer\geometry\build_geometry.py
# 只导某几个方案
D:\anaconda\python.exe optimizer\geometry\build_geometry.py --only 01 03
# 跳过 STEP / viewer（快速重跑）
D:\anaconda\python.exe optimizer\geometry\build_geometry.py --no-step --no-viewer
```
