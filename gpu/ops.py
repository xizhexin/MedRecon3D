# -*- coding: utf-8 -*-
"""GPU 算子层：把重建管线里真正吃算力的体素运算搬到 GPU。

为什么是这些算子、而不是整条管线
--------------------------------
先说结论，免得后人以为是偷懒：**不是所有步骤 GPU 化都更快**。

| 步骤 | 实现 | 理由 |
|---|---|---|
| 各向同性重采样 | **GPU**（`resample_volume_gpu`） | 纯 element-wise 采样，450³ 级别，GPU 快一个量级 |
| HU 阈值分割 | **GPU** | 纯比较运算，天然适合并行 |
| 逐层孔洞填充 | CPU（scipy + 线程池） | `binary_fill_holes` 内部是**串行** flood fill。GPU 上要用
| | | 迭代膨胀模拟，450 层 × 数百次迭代反而更慢；scipy 的 C 实现是线性时间，
| | | 且 **ndimage 会释放 GIL**，用线程池能真并行 |
| 连通域标记 | CPU（scipy） | 同上：并查集类算法在 GPU 上要反复 atomic 传播，规模不够大时不划算 |
| Marching Cubes | CPU（skimage） | 见下方说明 |
| 多级配准 | **GPU**（`gpu/registration.py`） | 全局最热点，且梯度可用 autograd 自动求，收益最大 |
| 网格体积/面积 | CPU（trimesh） | 已经是 numpy 向量化的 O(面数) 运算，没必要搬 |

关于 Marching Cubes
-------------------
skimage 的 `marching_cubes` 是 CPU 实现。GPU 版本需要 CUDA 扩展
（如 torchmcubes，要 nvcc 现场编译），在按小时计费的租用实例上
编译成本 > 收益。而且 MC 本身在 450³ 体素上只要几秒，
GPU 化整条管线后它已经不是瓶颈。所以**保留 CPU 实现，不硬凑 GPU**。

关于数值一致性（这部分很重要）
------------------------------
GPU 版和 CPU 版算出来的体积必须能对上，否则"加速"就没意义。
两个必须对齐的点：

1. **重采样的索引映射**。SimpleITK `ResampleImageFilter` 的语义是
   「输出索引 i → 物理坐标 origin + i·new_spacing → 回查输入索引」，
   因为新旧 origin 相同，化简后就是 `src_idx = i · new_spacing / old_spacing`。
   这里用 `grid_sample(align_corners=True)` 严格复现这个线性映射，
   **不是** `F.interpolate` —— 后者的 `align_corners=False` 是
   「半像素对半像素」对齐，端点行为不同，边界会出现半像素错位。
2. **输出网格尺寸**：走 `volume_grid.isotropic_size()`。**绝对不能用
   `round(size·old_sp/new_sp)`** —— 那个公式会让最后一层落到输入范围之外，
   而两边对超界的处理不同（ITK 补 `defaultPixelValue`、grid_sample 补 0），
   会在末层造成几十到几百 HU 的差异，并且给分割塞进一块假的"零 HU 组织"。
   详见 `volume_grid.py` 头部。
3. **插值补值**。采样前把体数据整体 +1000 HU，采完减回来 —— grid_sample 的
   `padding_mode="zeros"` 于是等价于补 −1000 HU（空气）。与 CPU 侧
   `defaultPixelValue=PAD_HU` 语义一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .backend import TORCH_AVAILABLE

# 输出网格规则来自项目根目录的单一来源模块（CPU/GPU 共用，避免两处各写一遍）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from volume_grid import PAD_HU, isotropic_size                      # noqa: E402

if TORCH_AVAILABLE:  # pragma: no cover - 环境相关
    import torch
    import torch.nn.functional as F


def _as_tensor(arr: np.ndarray, device) -> "torch.Tensor":
    return torch.as_tensor(np.ascontiguousarray(arr, dtype=np.float32), device=device)


# --------------------------------------------------------------- 重采样
def resample_volume_gpu(arr: np.ndarray, src_spacing_zyx, dst_spacing_mm: float, device):
    """把 (Z, Y, X) 体数据重采样到各向同性 spacing（mm）。

    返回 (重采样后的 (Z,Y,X) float32 numpy 数组, 实际 spacing 三元组, (nz,ny,nx))。

    索引映射严格对齐 SimpleITK：
        new_size = isotropic_size(...)          ← 见 volume_grid.py，不是 round()
        out[i]   = linear_interp(in, i · new_spacing / old_spacing)

    关于边界
    --------
    `volume_grid.isotropic_size()` 保证输出网格**不超出**输入体素中心张成的
    物理范围，所以正常情况下 grid_sample 永远不会采到图外。这里仍然先把
    体数据整体抬高 1000 HU 再采样、采完再减回来 —— 这不是装饰：grid_sample 的
    `padding_mode="zeros"` 补的是 0，而在抬高后的坐标里 0 恰好等于 −1000 HU，
    也就是**空气**。万一某个轴因为浮点舍入差半个体素，补进来的也是空气，
    而不是会污染分割的"0 HU 水"。

    注意输出是**在 GPU 上算完再拷回 CPU** —— 后续的 scipy 连通域、skimage
    的 Marching Cubes 都还是 CPU 的，留在显存里没有意义。
    """
    if len(src_spacing_zyx) != 3:
        raise ValueError("spacing 必须是 3 元组")
    # 输入本应是 (Z,Y,X)：spacing 按 z,y,x 给
    old_size = np.array(arr.shape, dtype=np.int64)
    old_spacing = np.array(src_spacing_zyx, dtype=np.float64)
    new_spacing = np.full(3, float(dst_spacing_mm), dtype=np.float64)
    new_size = isotropic_size(old_size, old_spacing, new_spacing)

    nz, ny, nx = (int(v) for v in new_size)
    z, y, x = (int(v) for v in old_size)
    # 输出索引 → 源索引的线性系数（每个轴一个）
    sz = float(new_spacing[0] / old_spacing[0])
    sy = float(new_spacing[1] / old_spacing[1])
    sx = float(new_spacing[2] / old_spacing[2])

    bias = -float(PAD_HU)                                          # = +1000
    t = _as_tensor(arr, device)[None, None] + bias                 # (1,1,Z,Y,X)

    # 归一化到 grid_sample 的 [-1,1]（align_corners=True 时 -1↔0, +1↔size-1）
    iz = torch.arange(nz, device=device, dtype=torch.float32) * sz
    iy = torch.arange(ny, device=device, dtype=torch.float32) * sy
    ix = torch.arange(nx, device=device, dtype=torch.float32) * sx
    nz_ = (2.0 * iz / max(z - 1, 1)) - 1.0
    ny_ = (2.0 * iy / max(y - 1, 1)) - 1.0
    nx_ = (2.0 * ix / max(x - 1, 1)) - 1.0

    gz, gy, gx = torch.meshgrid(nz_, ny_, nx_, indexing="ij")
    # grid_sample 的最后一维是 (x, y, z)，与 (W, H, D) 对应
    grid = torch.stack([gx, gy, gz], dim=-1)[None]                  # (1,nz,ny,nx,3)

    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=True)
    torch.cuda.synchronize() if device.type == "cuda" else None
    res = (out[0, 0] - bias).detach().cpu().numpy()
    return res, tuple(float(v) for v in new_spacing), (nz, ny, nx)


# --------------------------------------------------------------- 阈值分割
def hu_thresholds_gpu(arr: np.ndarray, device,
                      hu_tissue: float = -500.0,
                      hu_lung: float = -400.0,
                      hu_bone: float = 200.0):
    """在 GPU 上算三个阈值掩膜，返回 **CPU bool 数组**。

    为什么算完就搬回 CPU：布尔掩膜本身在 GPU 上没用了，
    下一步（连通域 / 孔洞填充）是 CPU 的。留在显存里只会白占 54 MB × 3。
    真正在 GPU 上完成的是 5400 万次的比较运算。
    """
    t = _as_tensor(arr, device)
    solid = (t > hu_tissue)
    air = (t < hu_lung)
    bone = (t > hu_bone)
    torch.cuda.synchronize() if device.type == "cuda" else None
    return (solid.detach().cpu().numpy(),
            air.detach().cpu().numpy(),
            bone.detach().cpu().numpy())


def combine_masks_gpu(body: np.ndarray, solid: np.ndarray, air: np.ndarray,
                      bone: np.ndarray, device,
                      hu_bone: float = 200.0):
    """按 CPU 版的组合逻辑在 GPU 上还原三个结构掩膜。

    CPU 版：
        lung = body & ~tissue & (arr < HU_LUNG)
        bone = body & (arr > HU_BONE)
    这里把「已经是 CPU bool 的 body」和「GPU 上的阈值结果」组合起来，
    用 GPU 做布尔矩阵运算（5400 万 × 3 次逻辑运算）。
    """
    b = torch.as_tensor(np.ascontiguousarray(body), device=device)
    s = torch.as_tensor(np.ascontiguousarray(solid), device=device)
    a = torch.as_tensor(np.ascontiguousarray(air), device=device)
    bo = torch.as_tensor(np.ascontiguousarray(bone), device=device)

    lung = b & (~s) & a
    bone_m = b & bo
    torch.cuda.synchronize() if device.type == "cuda" else None
    return (lung.detach().cpu().numpy(), bone_m.detach().cpu().numpy())


def count_voxels_gpu(mask: np.ndarray, device) -> int:
    """GPU 上数前景体素。看着小题大做，但它同时是"显存通路是否正常"的健康检查。"""
    t = torch.as_tensor(np.ascontiguousarray(mask), device=device)
    return int(t.sum().item())


def hu_stats_gpu(arr: np.ndarray, mask: np.ndarray, device) -> dict:
    """GPU 上算掩膜区域内的 HU 统计（中位/均值/分位）。

    `torch.median` 对大张量会走 sort，显存开销是 O(n)；450³ 时约 216 MB，
    12 GB 显存完全够，但这里仍只对**掩膜内的体素**做统计以省时间和显存。
    """
    t = _as_tensor(arr, device)
    m = torch.as_tensor(np.ascontiguousarray(mask), device=device)
    vals = t[m]
    if vals.numel() == 0:
        return {"n": 0}
    q = torch.quantile(vals, torch.tensor([0.05, 0.5, 0.95], device=device))
    torch.cuda.synchronize() if device.type == "cuda" else None
    return {
        "n": int(vals.numel()),
        "hu_min": round(float(vals.min()), 1),
        "hu_max": round(float(vals.max()), 1),
        "hu_mean": round(float(vals.mean()), 1),
        "hu_median": round(float(q[1]), 1),
        "hu_p05": round(float(q[0]), 1),
        "hu_p95": round(float(q[2]), 1),
    }


# --------------------------------------------------------------- 网格体积
def mesh_volume_gpu(verts: np.ndarray, faces: np.ndarray, device) -> float:
    """散度定理求闭合网格体积，在 GPU 上算。

    V = |1/6 · Σ (v0 × v1) · v2|

    这是**和 trimesh 交叉验证**用的第二条独立实现：trimesh 走的是
    它自己的 C 实现，两者在同一条网格上应该给出同一个数。
    精度必须是 float64 —— float32 在 85 万面上累加会有肉眼可见的误差。
    """
    v = torch.as_tensor(np.ascontiguousarray(verts, dtype=np.float64), device=device)
    f = torch.as_tensor(np.ascontiguousarray(faces, dtype=np.int64), device=device)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = torch.einsum("ij,ij->i", torch.cross(a, b, dim=1), c).sum() / 6.0
    torch.cuda.synchronize() if device.type == "cuda" else None
    return float(abs(vol.item()))


def voxel_volume_ml(n_voxels: int, spacing_zyx) -> float:
    """体素法体积（mL）。纯标量运算，放 CPU 就好 —— 别为了"全是 GPU"写难看代码。"""
    return float(n_voxels) * float(np.prod(spacing_zyx)) / 1000.0
