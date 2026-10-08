# -*- coding: utf-8 -*-
"""各向同性重采样的**输出网格规则** —— CPU 与 GPU 两条路径的唯一来源。

为什么必须单独抽一个模块
------------------------
这条规则原本在 `demo_01_recon.resample_isotropic()`（CPU / SimpleITK）和
`gpu.ops.resample_volume_gpu()`（GPU / grid_sample）里各写了一遍，两处都是：

    new_size = round(size · old_spacing / new_spacing)

于是两处埋着同一个坑：**输出网格的最后一个体素会落在输入体素中心张成的
物理范围之外**。

以 PCIR 数据实测为例：107 层、层距 2.5 mm、重采样到 1 mm

    round(107 × 2.5 / 1) = 268   →  最后一层索引 267 对应物理 267 mm
    →  回查输入索引 267 / 2.5 = 106.8，而输入最大索引只有 106

这时插值需要的第 107 层并不存在，只能靠"补值"顶上。SimpleITK 的补值是
`defaultPixelValue`（默认 **0.0**）—— 而 0 HU 在 CT 体系里是**水**，不是空气。
实测后果（真实数据，未修复前）：

    输出 z=267 整层 mean = 0.00，min = max = 0.0
    `arr > -500` 的体素数 = 202500 = 450 × 450

也就是整个 450×450 的切片全部被判成"实性组织"，体表掩膜顶部凭空多出一块
完整的平板，体积虚增 **202.5 mL（16977.87 → 16775.4，1.19%）**，
Marching Cubes 出来的网格顶部还多一个平盖；同时 z=266 退化成 z=265 的
重复层（连续索引 106.4 被 ITK 贴到 106）。

正确规则
--------
让输出网格严格落在输入体素中心张成的物理区间 `[0, (size-1)·old_spacing]` 内：

    new_size = floor((size - 1) · old_spacing / new_spacing) + 1

这样每个输出采样点都有**完整的插值支撑**，既不超界、也不会出现末层重复。
同一份数据：107 层 × 2.5 mm 共覆盖 265 mm，输出 **266** 层而非 268 层。
"""

from __future__ import annotations

import numpy as np

# 超出视野时的补值（HU）。CT 的 HU 体系里 −1000 是空气。
# 修复网格规则后这个补值理论上不会再被触发，留着是因为：
# 它是**语义正确**的兜底 —— 万一某个轴因为浮点舍入仍然差半个体素，
# 补空气只会让该体素被排到体外，而不是凭空造出一块"水"组织。
PAD_HU = -1000.0


def isotropic_size(size, old_spacing, new_spacing) -> np.ndarray:
    """输出网格尺寸。三个入参的轴序必须一致（都用 (x,y,z) 或都用 (z,y,x) 都行）。

    参数
    ----
    size          : 输入各轴体素数
    old_spacing   : 输入各轴体素间距 (mm)
    new_spacing   : 目标间距，标量（各向同性）或与 `size` 同长的序列

    返回
    ----
    int64 数组，每个元素 >= 1。**不使用 round()** —— 理由见模块头部。
    """
    size = np.asarray(size, dtype=np.float64)
    old_spacing = np.asarray(old_spacing, dtype=np.float64)
    new_spacing = np.asarray(new_spacing, dtype=np.float64)
    if new_spacing.ndim == 0:
        new_spacing = np.full(size.shape, float(new_spacing))
    if not (size.shape == old_spacing.shape == new_spacing.shape):
        raise ValueError(f"形状不一致: size{size.shape} spacing{old_spacing.shape} "
                         f"new_spacing{new_spacing.shape}")
    if np.any(new_spacing <= 0):
        raise ValueError("new_spacing 必须为正")

    # 体素中心张成的物理长度 = (size - 1) · old_spacing，加回第 1 个体素本身
    span = np.maximum(size - 1.0, 0.0) * old_spacing
    out = np.floor(span / new_spacing).astype(np.int64) + 1
    return np.maximum(out, 1)
