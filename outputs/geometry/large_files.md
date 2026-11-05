# 大文件说明（STL 不进入 git 历史）

第一批 5 个组合结构导出的 20 个 STL 合计约 **319 MB**，远超 GitHub 单文件 100 MB 限制，
也不适合写入 git 历史。因此：

- **STL 原文件不提交到 git**，仅保留在本机 `outputs/geometry/` 目录；
- 分支内提交的是本说明、体检表（`manifest.csv/json`）和打包后的几何压缩包
  `combination_optimizer_batch1_geometry.zip`；
- 需要原始 STL 时，可用下面的命令重新生成（marching cubes 无随机性，参数固定）。

## 生成方式（可复现）

```powershell
D:\anaconda\python.exe optimizer\geometry\build_geometry.py
```

生成参数集中在 `optimizer/geometry/geometry_config.py`：

| 项目 | 取值 |
|---|---|
| 单胞盒 | `[0, 2π)^3`，映射到 20 mm 立方体 |
| bulk 实体 | `F <= 0`，bulk 1×1 用 128 体素/单胞，2×2 用 64 体素/单胞 |
| sheet 实体 | `|F| <= t`，`t = 0.13`；sheet 1×1 用 96 体素/单胞，2×2 用 48 体素/单胞 |
| 网格算法 | `skimage.measure.marching_cubes(method="lewiner")`，无随机性 |
| 封口 | 盒壁 `eps = 0.25` 体素，保证网格 watertight |

## 文件命名

```
outputs/geometry/<编号>_<名称>_<bulk|sheet>_<1x1|2x2>.stl
```

完整清单与每个文件的三角面数、watertight、周期连通分量、体积、ρ* 等，
见 `outputs/geometry/manifest.csv` 与 `README_geometry.md`。

## 压缩包取舍

`combination_optimizer_batch1_geometry.zip` 装的是 **5 个 `bulk_2x2` + 5 个 `sheet_1x1`**，
共 10 个 STL（压缩后约 56 MB，未压缩约 164 MB）：
`bulk_2x2` 是观察三向周期性连通性最关键的版本；`sheet_1x1` 提供最接近真实薄壁件外观的版本。
1×1 bulk 会被周期边界切成断片、看起来不完整，`sheet_2x2` 体积更大，均未放入压缩包。
若需要其余 10 个 STL，按上面的命令重新生成即可（无随机性，结果可复现）。
