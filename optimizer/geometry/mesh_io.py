# -*- coding: utf-8 -*-
"""
网格文件读写：
- 二进制 STL（主交付，兼容 SolidWorks / Blender / Meshmixer / Paraview）；
- 自定义索引二进制（viewer 用，顶点共享、体积更小）。

STL 是三角面格式，不含拓扑；索引二进制保留顶点索引，便于 three.js 直接渲染。
"""

from __future__ import annotations

import os
import struct

import numpy as np


def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    norm[norm < 1e-15] = 1.0
    return n / norm[:, None]


def write_binary_stl(path: str, verts_mm: np.ndarray, faces: np.ndarray, header_text: str = "") -> int:
    """写出二进制 STL，返回文件字节数。三角形法向按右手定则计算。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    verts = np.asarray(verts_mm, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    normals = _face_normals(verts, faces)

    header = (header_text or "LLM TPMS geometry export").encode("ascii", "ignore")[:80]
    header = header + b" " * (80 - len(header))

    rec = np.zeros(
        len(faces),
        dtype=[("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")],
    )
    rec["normal"] = normals.astype(np.float32)
    rec["v"] = verts[faces].astype(np.float32)
    rec["attr"] = 0

    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", int(len(faces))))
        rec.tofile(f)

    return os.path.getsize(path)


def read_binary_stl(path: str):
    """读取二进制 STL，返回 (顶点数组, 面片索引, 三角形法向)。仅用于自检。"""
    with open(path, "rb") as f:
        header = f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        rec = np.fromfile(
            f,
            dtype=[("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")],
            count=n,
        )
    verts = rec["v"].reshape(-1, 3).astype(np.float64)
    faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
    return verts, faces, rec["normal"].astype(np.float64)


def write_indexed_bin(
    path: str,
    verts_mm: np.ndarray,
    faces: np.ndarray,
    cap_face_mask: np.ndarray,
) -> int:
    """
    viewer 用的紧凑格式（小端）：
        magic 'TPMS' (4s) | version uint32 | nverts uint32 | ntris uint32
        positions float32[nverts*3] | indices uint32[ntris*3] | flags uint8[ntris]
    flags=1 表示切面封口三角形。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    verts = np.asarray(verts_mm, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    flags = np.asarray(cap_face_mask, dtype=np.uint8)

    with open(path, "wb") as f:
        f.write(b"TPMS")
        f.write(struct.pack("<III", 1, len(verts), len(faces)))
        f.write(verts.reshape(-1).tobytes(order="C"))
        f.write(faces.reshape(-1).tobytes(order="C"))
        f.write(flags.tobytes(order="C"))

    return os.path.getsize(path)
