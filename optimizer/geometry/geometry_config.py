# -*- coding: utf-8 -*-
"""
几何导出配置：所有网格参数集中在这里，保证可复现。

单位约定
--------
- 隐函数域：每个单胞为 [0, 2π]^3（与优化器评价器一致）；
- 导出网格：把 2π 映射为 `cell_size_mm` 毫米，因此 2×2×2 平铺的外形尺寸是
  `2 * cell_size_mm`；
- sheet 版本的 t 是**隐函数场单位**（不是毫米），实体壁厚约为
  `2*t/|∇F| * cell_size_mm/(2π)`，实际厚度见 manifest 中的估算。

关于 “水密”
-----------
F=0 本身是零厚度曲面，不是实体。这里用两个实体化定义：
1. bulk  ：F(x,y,z) <= 0 的块体相（与优化器评价器的实体相约定一致）；
2. sheet ：|F(x,y,z)| <= t 的薄壁相（更接近真实 TPMS 薄壁件）。
在盒子边界处再加一个“切面封口”场 `eps - B_in`（B_in 为到盒壁的距离），
使网格在样品切面处闭合，从而得到 watertight、可 3D 打印的实体。
封口面是样品被切断处的材料截面，不是凭空添加的结构。
"""

from dataclasses import dataclass, field


@dataclass
class GeometryConfig:
    # ---------------------------------------------------------------- 路径
    # 只读：优化器输出目录（用于读取 candidates.csv 交叉核对分数）
    outputs_dir: str = r"c:\Users\hp\Desktop\LLM\outputs"
    geometry_dir: str = r"c:\Users\hp\Desktop\LLM\outputs\geometry"
    viewer_dir: str = r"c:\Users\hp\Desktop\LLM\outputs\geometry\viewer"
    step_dir: str = r"c:\Users\hp\Desktop\LLM\outputs\geometry\step"
    preview_dir: str = r"c:\Users\hp\Desktop\LLM\outputs\geometry\preview"

    # ---------------------------------------------------------------- 可复现性
    # marching cubes 本身无随机性；这里仍记录固定种子，便于与优化器 run_log 对齐
    seed: int = 20260101
    cell_size_mm: float = 20.0      # 单胞边长（毫米）；2×2×2 即 40 mm
    domain_max: float = 6.283185307179586  # 2π

    # ---------------------------------------------------------------- 主交付分辨率
    bulk_grid_per_cell: int = 128   # bulk 1×1×1：每单胞 128 体素（节点 129）
    bulk_tile_grid_per_cell: int = 64  # bulk 2×2×2：每单胞 64 体素（总计 128）
    sheet_grid_per_cell: int = 96   # sheet 1×1×1
    sheet_tile_grid_per_cell: int = 48  # sheet 2×2×2（总计 96）

    # ---------------------------------------------------------------- 实体化参数
    sheet_t: float = 0.13           # |F| <= t；t 在隐函数场单位下
    iso_level: float = 0.0
    cap_eps_voxels: float = 0.25    # 切面封口平面距盒壁的距离（体素倍数）

    # ---------------------------------------------------------------- 预览 / viewer 分辨率
    # 预览网格只用于 PNG 和 HTML 查看器，不替代主交付 STL
    preview_bulk_grid_per_cell: int = 56
    preview_bulk_tile_grid_per_cell: int = 32
    preview_sheet_grid_per_cell: int = 48
    preview_sheet_tile_grid_per_cell: int = 28

    # viewer 内嵌：每个结构嵌入 4 个变体的低分辨率预览网格
    viewer_bulk_grid_per_cell: int = 52
    viewer_bulk_tile_grid_per_cell: int = 30
    viewer_sheet_grid_per_cell: int = 44
    viewer_sheet_tile_grid_per_cell: int = 26

    # ---------------------------------------------------------------- STEP
    # 无 CAD 内核，导出 faceted B-Rep；用较低分辨率控制文件体积
    step_grid_per_cell: int = 40
    step_export_sheet: bool = False  # 默认只导 bulk 1×1×1，避免 STEP 体积失控

    # ---------------------------------------------------------------- 输出开关
    export_stl: bool = True
    export_step: bool = True
    export_preview_png: bool = True
    export_viewer: bool = True
    verify_candidates: bool = True

    # ---------------------------------------------------------------- 顶点法向/封口识别
    # 封口三角形：质心到最近盒壁距离 < cap_detect_voxels * h，
    # 且法向与某个坐标轴夹角 < cap_detect_angle_deg
    cap_detect_voxels: float = 1.0
    cap_detect_angle_deg: float = 25.0


DEFAULT_GEOMETRY_CONFIG = GeometryConfig()
