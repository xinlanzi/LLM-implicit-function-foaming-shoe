# -*- coding: utf-8 -*-
"""
几何导出主入口：把 5 个“高支撑 + 高弹性”组合结构导出为实体 STL、
faceted STEP、多视角 PNG 和离线 three.js 查看页。

运行
----
    D:\\anaconda\\python.exe optimizer\\geometry\\build_geometry.py
    D:\\anaconda\\python.exe optimizer\\geometry\\build_geometry.py --no-step --no-viewer
    D:\\anaconda\\python.exe optimizer\\geometry\\build_geometry.py --only 01 03

输出
----
    outputs/geometry/*.stl
    outputs/geometry/step/*.step
    outputs/geometry/preview/*.png
    outputs/geometry/viewer/index.html
    outputs/geometry/manifest.csv / manifest.json
    outputs/geometry/README_geometry.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
for _p in (_HERE, _OPT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from token_library import build_function
from evaluator import Evaluator
from geometry_config import DEFAULT_GEOMETRY_CONFIG, GeometryConfig
from mesh_builder import MeshResult, build_entity_mesh
from mesh_io import write_binary_stl, write_indexed_bin
from faceted_step import write_faceted_step
from render_png import render_solution_png
from build_viewer import build_viewer_html, encode_mesh_b64
from solutions import SOLUTIONS, SolutionSpec, find_solution


# --------------------------------------------------------------------------- #
# 参数与文件名
# --------------------------------------------------------------------------- #
def grid_for(cfg: GeometryConfig, phase: str, tile: int) -> int:
    """返回每单胞的体素数（不是总节点数）。"""
    if phase == "bulk":
        return cfg.bulk_grid_per_cell if tile == 1 else cfg.bulk_tile_grid_per_cell
    return cfg.sheet_grid_per_cell if tile == 1 else cfg.sheet_tile_grid_per_cell


def preview_grid_for(cfg: GeometryConfig, phase: str, tile: int) -> int:
    if phase == "bulk":
        return cfg.preview_bulk_grid_per_cell if tile == 1 else cfg.preview_bulk_tile_grid_per_cell
    return cfg.preview_sheet_grid_per_cell if tile == 1 else cfg.preview_sheet_tile_grid_per_cell


def viewer_grid_for(cfg: GeometryConfig, phase: str, tile: int) -> int:
    if phase == "bulk":
        return cfg.viewer_bulk_grid_per_cell if tile == 1 else cfg.viewer_bulk_tile_grid_per_cell
    return cfg.viewer_sheet_grid_per_cell if tile == 1 else cfg.viewer_sheet_tile_grid_per_cell


def stl_name(spec: SolutionSpec, phase: str, tile: int) -> str:
    return f"{spec.sid}_{spec.name}_{phase}_{tile}x{tile}.stl"


def step_name(spec: SolutionSpec, phase: str, tile: int) -> str:
    return f"{spec.sid}_{spec.name}_{phase}_{tile}x{tile}_faceted.step"


# --------------------------------------------------------------------------- #
# 与优化器 candidates.csv 交叉核对
# --------------------------------------------------------------------------- #
def verify_against_candidates(cfg: GeometryConfig) -> Dict[str, dict]:
    """
    从 outputs/candidates.csv 找出这 5 个候选的实际评价值。
    只做元数据核对；网格仍严格按 terms/coeffs 生成。
    """
    path = os.path.join(cfg.outputs_dir, "candidates.csv")
    if not os.path.exists(path):
        print(f"[warn] 找不到 {path}，使用 solutions.py 中的参考分数")
        return {}

    import pandas as pd

    df = pd.read_csv(path, encoding="utf-8-sig")
    found: Dict[str, dict] = {}
    for _, row in df.iterrows():
        try:
            terms = json.loads(row["terms"])
            coeffs = [float(x) for x in json.loads(row["coeffs"])]
        except Exception:
            continue
        for spec in SOLUTIONS:
            if list(terms) != list(spec.terms):
                continue
            if len(coeffs) != len(spec.coeffs):
                continue
            if not all(abs(a - b) < 2e-3 for a, b in zip(coeffs, spec.coeffs)):
                continue
            rec = {
                "score": float(row["score"]),
                "support_proxy": float(row["support_proxy"]),
                "elastic_proxy": float(row["elastic_proxy"]),
                "round": int(row["round"]),
                "run": str(row["run"]),
                "source": str(row["source"]),
                "accepted": bool(row["accepted"]),
            }
            old = found.get(spec.sid)
            if old is None or rec["score"] > old["score"]:
                found[spec.sid] = rec

    for spec in SOLUTIONS:
        rec = found.get(spec.sid)
        if rec:
            print(f"[verify] {spec.sid} {spec.name}: score {rec['score']:.6f} "
                  f"(参考 {spec.score:.4f}) run={rec['run']} round={rec['round']}")
        else:
            print(f"[warn] {spec.sid} {spec.name}: candidates.csv 中未找到完全匹配，沿用参考分数")
    return found


# --------------------------------------------------------------------------- #
# 构建与写出
# --------------------------------------------------------------------------- #
def build_solution_meshes(spec: SolutionSpec, cfg: GeometryConfig,
                          grid_fn, kind: str) -> Dict[Tuple[str, int], MeshResult]:
    """为 (bulk/sheet) × (1/2) 四个变体生成网格。"""
    F = build_function(list(spec.terms), list(spec.coeffs))
    out: Dict[Tuple[str, int], MeshResult] = {}
    for phase in ("bulk", "sheet"):
        for tile in (1, 2):
            n = grid_fn(cfg, phase, tile)
            t0 = time.time()
            mesh = build_entity_mesh(F, phase=phase, tile=tile, grid_per_cell=n, cfg=cfg)
            out[(phase, tile)] = mesh
            print(
                f"  [{kind}] {spec.sid} {phase} {tile}x{tile} "
                f"n={n}/cell nodes={mesh.grid_nodes}^3 "
                f"faces={mesh.n_faces:,} watertight={mesh.watertight} "
                f"bnd={mesh.boundary_edges} nonman={mesh.nonmanifold_edges} "
                f"rho={mesh.rho_voxel:.4f} vol={mesh.volume_mm3:.1f}mm3 "
                f"{time.time()-t0:.1f}s"
            )
    return out


def export_solution(spec: SolutionSpec, cfg: GeometryConfig, verified: dict):
    """导出一个方案的全部文件，返回 manifest 行。"""
    print(f"\n=== {spec.sid} {spec.name} | {spec.cn_name} ===")
    if verified:
        spec = replace(
            spec,
            score=verified["score"],
            support_proxy=verified["support_proxy"],
            elastic_proxy=verified["elastic_proxy"],
            round=verified["round"],
        )

    meshes = build_solution_meshes(spec, cfg, grid_for, "main")
    rows: List[dict] = []

    for (phase, tile), mesh in meshes.items():
        fname = stl_name(spec, phase, tile)
        path = os.path.join(cfg.geometry_dir, fname)
        nbytes = write_binary_stl(
            path, mesh.vertices_mm, mesh.faces,
            header_text=f"{spec.sid} {spec.name} {phase} {tile}x{tile} | proxy score {spec.score:.4f}",
        )

        # 写出后回读文件头，确认面数一致
        import struct
        with open(path, "rb") as f:
            f.read(80)
            n_read = struct.unpack("<I", f.read(4))[0]
        readback_ok = int(n_read) == int(mesh.n_faces)
        if not readback_ok:
            raise RuntimeError(f"STL 回读面数不一致: {fname}")

        bbox_prod = float(np.prod(mesh.bbox_size_mm))
        row = {
            "sid": spec.sid,
            "name": spec.name,
            "cn_name": spec.cn_name,
            "phase": phase,
            "tile": f"{tile}x{tile}x{tile}",
            "equation": spec.equation,
            "terms": json.dumps(list(spec.terms), ensure_ascii=False),
            "coeffs": json.dumps(list(spec.coeffs)),
            "score": round(float(spec.score), 6),
            "support_proxy": round(float(spec.support_proxy), 6),
            "elastic_proxy": round(float(spec.elastic_proxy), 6),
            "round": spec.round,
            "provenance": spec.provenance,
            "cell_size_mm": cfg.cell_size_mm,
            "grid_per_cell": mesh.grid_per_cell,
            "grid_nodes": mesh.grid_nodes,
            "h_mm": round(mesh.h_mm, 5),
            "sheet_t": mesh.sheet_t,
            "wall_thickness_mm_est": round(mesh.wall_thickness_mm_est, 4),
            "n_vertices": mesh.n_vertices,
            "n_faces": mesh.n_faces,
            "n_edges": mesh.n_edges,
            "boundary_edges": mesh.boundary_edges,
            "nonmanifold_edges": mesh.nonmanifold_edges,
            "duplicate_directed_edges": mesh.duplicate_directed_edges,
            "watertight": mesh.watertight,
            "edge_manifold": mesh.edge_manifold,
            "orientation_consistent": mesh.orientation_consistent,
            "surface_components": mesh.surface_components,
            "euler_characteristic": mesh.euler_characteristic,
            "cap_face_count": mesh.cap_face_count,
            "cap_area_fraction": round(mesh.cap_area_fraction, 4),
            "volume_mm3": round(mesh.volume_mm3, 3),
            "rho_voxel": round(mesh.rho_voxel, 4),
            "rho_mesh_bbox": round(mesh.volume_mm3 / bbox_prod, 4) if bbox_prod > 0 else 0.0,
            "area_mm2": round(mesh.area_mm2, 2),
            "bbox_size_mm": " x ".join(f"{x:.3f}" for x in mesh.bbox_size_mm),
            "solid_components_periodic": mesh.solid_components_periodic,
            "largest_periodic_fraction": round(mesh.largest_periodic_fraction, 4),
            "stl_path": path,
            "stl_bytes": nbytes,
            "stl_readback_ok": readback_ok,
            "format": "binary STL (mm)",
        }
        rows.append(row)
        print(f"  -> {fname}  {nbytes/1e6:.1f} MB")

    # ---- faceted STEP（无 CAD 内核的降级方案）----
    step_path = None
    if cfg.export_step:
        step_dir = os.path.join(cfg.geometry_dir, "step")
        os.makedirs(step_dir, exist_ok=True)
        F = build_function(list(spec.terms), list(spec.coeffs))
        step_mesh = build_entity_mesh(
            F, phase="bulk", tile=1, grid_per_cell=cfg.step_grid_per_cell, cfg=cfg
        )
        step_path = os.path.join(step_dir, step_name(spec, "bulk", 1))
        nbytes = write_faceted_step(step_path, step_mesh.vertices_mm, step_mesh.faces,
                                    name=f"{spec.sid}_{spec.name}")
        print(f"  -> step/{os.path.basename(step_path)}  {nbytes/1e6:.1f} MB "
              f"(faceted, {step_mesh.n_faces:,} tris @ {cfg.step_grid_per_cell}/cell)")

    return rows, meshes, step_path


# --------------------------------------------------------------------------- #
# README / manifest
# --------------------------------------------------------------------------- #
def write_manifest(cfg: GeometryConfig, all_rows: List[dict], verified: dict,
                   step_paths: Dict[str, str], reader_size: int) -> Tuple[str, str]:
    import pandas as pd

    os.makedirs(cfg.geometry_dir, exist_ok=True)
    csv_path = os.path.join(cfg.geometry_dir, "manifest.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest = {
        "generated_by": "optimizer/geometry/build_geometry.py",
        "assumptions": {
            "units": "mm",
            "cell_size_mm": cfg.cell_size_mm,
            "domain": "[0, 2π] per cell",
            "bulk": "solid = F<=0",
            "sheet": f"solid = abs(F)<= {cfg.sheet_t}",
            "cap": "boundary cross-section cap; eps = %.2f voxel" % cfg.cap_eps_voxels,
            "score": "geometric proxy only, not FEA",
        },
        "config": asdict(cfg),
        "verified_candidates": verified,
        "step_files": step_paths,
        "viewer_html_bytes": reader_size,
        "meshes": all_rows,
    }
    json_path = os.path.join(cfg.geometry_dir, "manifest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    return csv_path, json_path


def write_readme(cfg: GeometryConfig, all_rows: List[dict], verified: dict,
                 step_paths: Dict[str, str], html_bytes: int) -> str:
    path = os.path.join(cfg.geometry_dir, "README_geometry.md")

    # 每个方案一行汇总（取 bulk 1×1 的体检结果）
    sol_rows = []
    for spec in SOLUTIONS:
        score = verified.get(spec.sid, {}).get("score", spec.score)
        sol_rows.append(
            f"| {spec.sid} | {spec.cn_name} | `{spec.name}` | {score:.6f} | "
            f"{spec.support_proxy:.4f} / {spec.elastic_proxy:.4f} | {spec.round} |"
        )

    geom_rows = []
    for r in all_rows:
        wt = "是" if r["watertight"] else "**否**"
        geom_rows.append(
            f"| `{os.path.basename(r['stl_path'])}` | {r['phase']} | {r['tile']} | "
            f"{r['n_faces']:,} | {wt} | {r['boundary_edges']} | {r['nonmanifold_edges']} | "
            f"{r['surface_components']} | {r['solid_components_periodic']} | "
            f"{r['volume_mm3']:.1f} | {r['rho_voxel']:.4f} | {r['bbox_size_mm']} |"
        )

    text = f"""# TPMS 组合结构几何导出说明

> 本目录由 `optimizer/geometry/build_geometry.py` 生成，原始 Excel 与技术文档未修改。  
> **所有 score 都是优化器里的几何代理指标，不是 FEA 结果。** 这里只导出几何，
> 没有计算应力、应变、屈曲或疲劳。

## 1. 实际导出了什么

| 格式 | 数量 | 状态 | 说明 |
|---|---|---|---|
| 二进制 STL（mm） | {len(all_rows)} 个 | ✅ 已实际写出并回读校验面数 | 5 方案 × (bulk/sheet) × (1×1×1/2×2×2) |
| faceted STEP（AP203） | {len(step_paths)} 个 | ⚠️ 降级方案，未在 CAD 中打开验证 | 三角面 B-Rep，不是 NURBS 光滑 B-Rep |
| 多视角 PNG | {len(SOLUTIONS)*2} 张 | ✅ 已生成 | 每个方案 bulk/sheet 各一张，4 视角 × 2 平铺 |
| 单文件 3D 查看页 | 1 个 | ✅ 已生成 | three.js 内嵌，`file://` 可离线打开，约 {html_bytes/1e6:.1f} MB |
| manifest | CSV + JSON | ✅ | 全部网格体检指标与配置 |

**STEP 说明（重要）**：本机没有 `OCP` / `pythonocc-core` / `cadquery`，无法生成真正的
NURBS B-Rep。这里输出的是 `FACETED_BREP`（三角面 B-Rep），用 {cfg.step_grid_per_cell}³/单胞
的低分辨率网格控制体积。它适合在 CAD 中查看/测量，**不适合高精度曲面加工或参数化编辑**；
并且未在 SolidWorks/CATIA 中实际打开验证。若需要真正 B-Rep，请安装 CAD 内核后自行转换：

```powershell
# 方案 A：conda 安装 pythonocc-core（推荐，Windows 需 conda-forge）
conda install -c conda-forge pythonocc-core

# 方案 B：pip 安装 cadquery（自带 OCP 内核，体积较大）
D:\\anaconda\\python.exe -m pip install cadquery
```

安装后可把 STL 导入 CAD 做“网格→曲面”重建，或直接用 OCC 从三角面缝合为 shell。

## 2. 实体化定义（为什么 STL 是实体而不是零厚度片）

- `F(x,y,z)=0` 本身是零厚度极小曲面，直接 marching cubes 只有开放曲面片；
- **bulk 主交付**：实体相 `F<=0`（与优化器评价器的实体相约定一致），ρ* 约 0.5；
- **sheet 附带**：`|F|<=t`，`t={cfg.sheet_t:g}`（隐函数场单位，不是毫米）。
  估算壁厚 ≈ `2t/|∇F| × cell/2π`；本批实际估算值见 manifest 的
  `wall_thickness_mm_est`（单胞 {cfg.cell_size_mm:g} mm 时约 0.56~0.86 mm，约为单胞的 2.8%~4.3%）；
  **t 的选择依据**：建议区间是 0.10~0.15，{cfg.sheet_t:g} 取其中段；在这个取值下，
  sheet 1×1 的 96 体素/单胞、2×2 的 48 体素/单胞时，壁厚大约跨 3~4 个节点，
  marching cubes 能稳定分辨而不会断壁；t 偏小会更接近真实薄壁、但粗网格容易断成碎片，
  t 偏大则逐渐退化成块体，失去“薄壁件”外观。
- 盒子边界用 `eps - B_in`（`eps={cfg.cap_eps_voxels:g}` 体素）封口，得到
  watertight 实体；封口面是**样品切断处的材料截面**，不是额外结构。
- 朝向：用有符号体积判断整体朝向，必要时翻转全部面，保证法向朝外。

## 3. 五个方案（分数来自优化器实际评价）

| 编号 | 名称 | 文件短名 | score | support / elastic | 轮次 |
|---|---|---|---|---|---|
{chr(10).join(sol_rows)}

方程与组合理由（完整版见 `../high_support_high_elasticity_solutions.md`）：

{chr(10).join("- **" + s.sid + " " + s.cn_name + "**：`" + s.equation + "`" + chr(10) + "  " + s.notes for s in SOLUTIONS)}

## 4. 网格体检表

`watertight=是` 表示无边界边且无非流形边；`solid_components_periodic` 是按周期边界
合并后的**实体相**连通分量（期望=1），与表面 shell 分量不是同一个概念。

| 文件 | 相 | 平铺 | 三角面 | watertight | 边界边 | 非流形边 | 表面分量 | 周期实体分量 | 体积 mm³ | ρ* | bbox mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(geom_rows)}

> 说明：单胞 bbox 略小于标称尺寸（切面封口在盒壁内约 {cfg.cap_eps_voxels:g} 体素）；
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
| 单胞尺寸 | {cfg.cell_size_mm:g} mm（2×2×2 为 {2*cfg.cell_size_mm:g} mm） |
| bulk 1×1×1 网格 | {cfg.bulk_grid_per_cell} 体素/单胞（节点 {cfg.bulk_grid_per_cell+1}³） |
| bulk 2×2×2 网格 | {cfg.bulk_tile_grid_per_cell} 体素/单胞（总计 {2*cfg.bulk_tile_grid_per_cell}³） |
| sheet 1×1×1 网格 | {cfg.sheet_grid_per_cell} 体素/单胞 |
| sheet 2×2×2 网格 | {cfg.sheet_tile_grid_per_cell} 体素/单胞 |
| sheet t | {cfg.sheet_t:g}（隐函数场单位） |
| 切面封口 eps | {cfg.cap_eps_voxels:g} 体素 |
| 随机种子 | {cfg.seed}（marching cubes 无随机性，记录以便复现） |
| 等值面 | F = {cfg.iso_level:g} |
| 提交通道 | binary STL，坐标单位按 mm 解释 |

> 实测耗时（本机，`D:\\anaconda\\python.exe`，Python 3.9）：128³ bulk 单网格约 1.0~1.4 s，
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
D:\\anaconda\\python.exe optimizer\\geometry\\build_geometry.py
# 只导某几个方案
D:\\anaconda\\python.exe optimizer\\geometry\\build_geometry.py --only 01 03
# 跳过 STEP / viewer（快速重跑）
D:\\anaconda\\python.exe optimizer\\geometry\\build_geometry.py --no-step --no-viewer
```
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="TPMS 组合结构实体网格导出")
    ap.add_argument("--only", nargs="*", default=None, help="只导出指定编号，如 01 03")
    ap.add_argument("--no-step", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    cfg = replace(
        DEFAULT_GEOMETRY_CONFIG,
        export_step=not args.no_step,
        export_viewer=not args.no_viewer,
        export_preview_png=not args.no_preview,
    )

    os.makedirs(cfg.geometry_dir, exist_ok=True)
    specs = [find_solution(s) for s in args.only] if args.only else list(SOLUTIONS)

    print("=" * 78)
    print("TPMS 组合结构几何导出")
    print(f"  输出目录 : {cfg.geometry_dir}")
    print(f"  单胞尺寸 : {cfg.cell_size_mm} mm")
    print(f"  bulk 网格: {cfg.bulk_grid_per_cell}/cell (1x1), {cfg.bulk_tile_grid_per_cell}/cell (2x2)")
    print(f"  sheet 网格: {cfg.sheet_grid_per_cell}/cell (1x1), {cfg.sheet_tile_grid_per_cell}/cell (2x2), t={cfg.sheet_t}")
    print(f"  STEP     : {'关闭' if args.no_step else 'faceted B-Rep（无 CAD 内核）'}")
    print("=" * 78)

    verified = verify_against_candidates(cfg) if cfg.verify_candidates else {}
    all_rows: List[dict] = []
    step_paths: Dict[str, str] = {}
    main_meshes: Dict[str, Dict[Tuple[str, int], MeshResult]] = {}

    for spec in specs:
        rows, meshes, step_path = export_solution(spec, cfg, verified.get(spec.sid, {}))
        all_rows.extend(rows)
        main_meshes[spec.sid] = meshes
        if step_path:
            step_paths[spec.sid] = step_path

    # ---- 预览 PNG（低分辨率单独重建，避免用主网格拖慢渲染）----
    if cfg.export_preview_png:
        print("\n--- 多视角 PNG ---")
        os.makedirs(cfg.preview_dir, exist_ok=True)
        for spec in specs:
            print(f"[preview] {spec.sid} {spec.name} 生成低分辨率预览网格 ...")
            preview = build_solution_meshes(spec, cfg, preview_grid_for, "preview")
            out = os.path.join(cfg.preview_dir, f"{spec.sid}_{spec.name}_preview.png")
            render_solution_png(spec, preview, out, cfg)

    # ---- 单文件 viewer（内嵌低分辨率索引网格）----
    html_bytes = 0
    if cfg.export_viewer:
        print("\n--- 离线 3D viewer ---")
        mesh_b64: Dict[str, dict] = {}
        for spec in specs:
            viewer = build_solution_meshes(spec, cfg, viewer_grid_for, "viewer")
            for (phase, tile), mesh in viewer.items():
                key = f"{spec.sid}|{phase}|{tile}"
                meta = {
                    "sid": spec.sid,
                    "name": spec.name,
                    "cn_name": spec.cn_name,
                    "phase": phase,
                    "tile": tile,
                    "equation": spec.equation,
                    "score": float(spec.score),
                    "support": float(spec.support_proxy),
                    "elastic": float(spec.elastic_proxy),
                    "notes": spec.notes,
                }
                mesh_b64[key] = encode_mesh_b64(mesh, meta, cfg.viewer_dir, key)
        html_path = os.path.join(cfg.viewer_dir, "index.html")
        html_bytes = build_viewer_html(html_path, mesh_b64, cfg)
        print(f"[viewer] {html_path}  {html_bytes/1e6:.1f} MB（自包含，可离线打开）")

    # ---- manifest + README ----
    csv_path, json_path = write_manifest(cfg, all_rows, verified, step_paths, html_bytes)
    readme = write_readme(cfg, all_rows, verified, step_paths, html_bytes)
    print("\n--- 输出汇总 ---")
    print(f"  manifest.csv : {csv_path}")
    print(f"  manifest.json: {json_path}")
    print(f"  README       : {readme}")
    print(f"  STL 文件     : {len(all_rows)} 个")
    print(f"  STEP 文件    : {len(step_paths)} 个")
    print(f"  viewer       : {os.path.join(cfg.viewer_dir, 'index.html')} ({html_bytes/1e6:.1f} MB)")

    # 终端打印体检摘要
    print("\n--- 网格体检摘要 ---")
    for r in all_rows:
        print(
            f"{os.path.basename(r['stl_path']):<42} "
            f"faces={r['n_faces']:>8,} wt={str(r['watertight']):>5} "
            f"bnd={r['boundary_edges']} nonman={r['nonmanifold_edges']} "
            f"comp(surf/solid)={r['surface_components']}/{r['solid_components_periodic']} "
            f"rho={r['rho_voxel']:.4f}"
        )


if __name__ == "__main__":
    main()
