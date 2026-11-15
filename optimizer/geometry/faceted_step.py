# -*- coding: utf-8 -*-
"""
FACETED STEP (AP203) 写出器 —— 没有 CAD 内核时的降级方案。

重要说明
--------
本环境没有 OCP / pythonocc-core / cadquery，无法生成真正的 NURBS B-Rep。
这里把三角网格写成 STEP 的 `FACETED_BREP`：
- 每个三角面是一个 `FACE_SURFACE` + `PLANE` + `POLY_LOOP`；
- 拓扑是 `CLOSED_SHELL`，整体是 `FACETED_BREP`；
- 它是“三角面 B-Rep”，不是光滑曲面 B-Rep，适合在 CAD 中查看/测量，
  不适合做高精度曲面加工或参数化编辑。

STEP 文本按 ISO-10303-21 组织；生成后会做一次引用完整性自检，
但**没有在任何 CAD 软件中实际打开验证**（环境缺 CAD 内核）。
"""

from __future__ import annotations

import math
import os
from typing import List

import numpy as np


def _fmt(x: float, nd: int = 6) -> str:
    """STEP 实数格式：避免科学计数法中的多余字符。"""
    v = float(x)
    if abs(v) < 10.0 ** (-nd):
        v = 0.0
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    if "." not in s and "E" not in s and "e" not in s:
        s += "."
    return s


def _vec(v: np.ndarray, nd: int = 6) -> str:
    return "(" + ",".join(_fmt(x, nd) for x in v) + ")"


def _perp(n: np.ndarray) -> np.ndarray:
    """构造与 n 垂直的单位向量。"""
    ref_axis = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    d = np.cross(n, ref_axis)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        d = np.cross(n, np.array([0.0, 0.0, 1.0]))
        norm = np.linalg.norm(d)
    return d / max(norm, 1e-12)


def write_faceted_step(
    path: str,
    verts_mm: np.ndarray,
    faces: np.ndarray,
    name: str = "TPMS",
) -> int:
    """写出 faceted STEP，返回文件字节数。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    verts = np.asarray(verts_mm, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = verts[faces]

    # 每个三角形取一个统一的外法向（已保证面片朝外）
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths < 1e-15] = 1.0
    normals = normals / lengths[:, None]

    out: List[str] = []
    out.append("ISO-10303-21;")
    out.append("HEADER;")
    out.append("FILE_DESCRIPTION(('faceted triangle B-Rep exported by LLM TPMS geometry exporter'),'2;1');")
    out.append(
        "FILE_NAME('" + name.replace("'", "") + ".step','2026-11-04T00:00:00',"
        "('LLM TPMS optimizer'),(''),'faceted STEP writer','','');"
    )
    out.append("FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));")
    out.append("ENDSEC;")
    out.append("DATA;")

    nid = 0

    def eid() -> int:
        nonlocal nid
        nid += 1
        return nid

    # ---- 产品/单位/上下文骨架 ----
    app_ctx = eid()
    out.append(f"#{app_ctx}=APPLICATION_CONTEXT('configuration controlled 3d designs');")
    app_proto = eid()
    out.append(f"#{app_proto}=APPLICATION_PROTOCOL_DEFINITION('international standard','config_control_design',1994,#{app_ctx});")
    prod_ctx = eid()
    out.append(f"#{prod_ctx}=PRODUCT_CONTEXT('',#{app_ctx},'mechanical');")
    product = eid()
    safe_name = name.replace("'", "")
    out.append(f"#{product}=PRODUCT('{safe_name}','{safe_name}','',(#{prod_ctx}));")
    pdf = eid()
    out.append(f"#{pdf}=PRODUCT_DEFINITION_FORMATION('','',#{product});")
    pd_ctx = eid()
    out.append(f"#{pd_ctx}=PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx},'design');")
    pd = eid()
    out.append(f"#{pd}=PRODUCT_DEFINITION('design','',#{pdf},#{pd_ctx});")
    pds = eid()
    out.append(f"#{pds}=PRODUCT_DEFINITION_SHAPE('','',#{pd});")

    length_unit = eid()
    out.append(f"#{length_unit}=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));")
    angle_unit = eid()
    out.append(f"#{angle_unit}=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));")
    solid_unit = eid()
    out.append(f"#{solid_unit}=(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());")
    uncertainty = eid()
    out.append(f"#{uncertainty}=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-6),#{length_unit},'distance_accuracy_value','');")
    geom_ctx = eid()
    out.append(
        f"#{geom_ctx}=(GEOMETRIC_REPRESENTATION_CONTEXT(3) "
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty})) "
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length_unit},#{angle_unit},#{solid_unit})) "
        f"REPRESENTATION_CONTEXT('',''));"
    )

    # ---- 唯一顶点 ----
    point_ids: List[int] = []
    for v in verts:
        pid = eid()
        point_ids.append(pid)
        out.append(f"#{pid}=CARTESIAN_POINT('',{_vec(v)});")

    face_ids: List[int] = []
    for fi, (f, n) in enumerate(zip(faces, normals)):
        # 三个顶点引用
        p0, p1, p2 = (point_ids[int(i)] for i in f)

        loop_id = eid()
        out.append(f"#{loop_id}=POLY_LOOP('',(#{p0},#{p1},#{p2}));")
        bound_id = eid()
        out.append(f"#{bound_id}=FACE_OUTER_BOUND('',#{loop_id},.T.);")

        # 平面坐标系：原点 = 顶点 0，axis = 外法向，ref = 任一垂直方向
        origin_id = eid()
        out.append(f"#{origin_id}=CARTESIAN_POINT('',{_vec(verts[int(f[0])])});")
        axis_id = eid()
        out.append(f"#{axis_id}=DIRECTION('',{_vec(n)});")
        ref_id = eid()
        out.append(f"#{ref_id}=DIRECTION('',{_vec(_perp(n))});")
        place_id = eid()
        out.append(f"#{place_id}=AXIS2_PLACEMENT_3D('',#{origin_id},#{axis_id},#{ref_id});")
        plane_id = eid()
        out.append(f"#{plane_id}=PLANE('',#{place_id});")

        face_id = eid()
        out.append(f"#{face_id}=FACE_SURFACE('',(#{bound_id}),#{plane_id},.T.);")
        face_ids.append(face_id)

    shell_id = eid()
    out.append(f"#{shell_id}=CLOSED_SHELL('',({','.join(f'#{i}' for i in face_ids)}));")
    brep_id = eid()
    out.append(f"#{brep_id}=FACETED_BREP('{safe_name}',#{shell_id});")

    shape_rep = eid()
    out.append(f"#{shape_rep}=SHAPE_REPRESENTATION('',(#{brep_id}),#{geom_ctx});")
    shape_def = eid()
    out.append(f"#{shape_def}=SHAPE_DEFINITION_REPRESENTATION(#{pds},#{shape_rep});")

    out.append("ENDSEC;")
    out.append("END-ISO-10303-21;")

    text = "\n".join(out) + "\n"
    with open(path, "w", encoding="ascii", errors="ignore") as f:
        f.write(text)

    _self_check(text, nid)
    return os.path.getsize(path)


def _self_check(text: str, max_id: int) -> None:
    """轻量自检：括号平衡 + 引用都在 1..max_id 范围内。"""
    if text.count("(") != text.count(")"):
        raise ValueError("STEP 自检失败：括号不平衡")
    import re
    refs = re.findall(r"#(\d+)", text)
    if not refs:
        raise ValueError("STEP 自检失败：没有实体引用")
    bad = [r for r in refs if not (1 <= int(r) <= max_id)]
    if bad:
        raise ValueError(f"STEP 自检失败：越界引用 {bad[:5]}")
