# -*- coding: utf-8 -*-
"""MedRecon3D · 分割结果目视检查

画: 中间层原始 CT / 各 mask 叠加 / 冠状面 MIP。
用来确认分割不是一团乱麻 —— 数值对不代表形状对。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from demo_01_recon import (DICOM_DIR, OUT_DIR, TARGET_SPACING,
                           load_dicom_series, resample_isotropic, segment)


# 固定 HU 显示窗（软组织窗）。
# 不能用 percentile 自适应：真实 CT 的 FOV 外 padding 值达 -3024，
# 会把动态范围拉大，导致体外空气(-1004)和软组织(-111)显示成几乎一样的灰，
# mask 叠加完全看不出来（上一版就踩了这个坑）。
DISPLAY_WIN = (-200.0, 400.0)


def window(gray: np.ndarray, win: tuple[float, float] = DISPLAY_WIN) -> np.ndarray:
    """按固定 HU 窗归一化到 [0,1]"""
    lo, hi = win
    return np.clip((gray - lo) / (hi - lo), 0, 1)


def overlay(gray: np.ndarray, mask: np.ndarray, color: tuple[float, float, float],
            alpha: float = 0.5) -> np.ndarray:
    """把 mask 以半透明颜色叠到灰度图上"""
    base = window(gray)
    rgb = np.stack([base] * 3, axis=-1)
    m = mask.astype(bool)
    for c in range(3):
        rgb[..., c] = np.where(m, rgb[..., c] * (1 - alpha) + color[c] * alpha, rgb[..., c])
    return rgb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedRecon3D · 分割结果目视检查")
    parser.add_argument("--dicom-dir", type=Path, default=DICOM_DIR)
    parser.add_argument("--out-prefix", default="demo01")
    parser.add_argument("--target-spacing", type=float, default=TARGET_SPACING)
    parser.add_argument("--slice", type=int, default=None, help="要看的层号（默认取中间层）")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img, meta = load_dicom_series(args.dicom_dir)
    img_r = resample_isotropic(img, args.target_spacing)
    arr = sitk.GetArrayFromImage(img_r)                     # (z, y, x)
    masks = segment(arr)

    z_mid = args.slice if args.slice is not None else arr.shape[0] // 2
    z_mid = int(np.clip(z_mid, 0, arr.shape[0] - 1))
    slice_ct = arr[z_mid]

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.6), dpi=95)
    fig.suptitle(f"CT {arr.shape}  spacing(z,y,x)=1mm  |  slice z={z_mid}", fontsize=12)

    # 第一行：中间层
    axes[0, 0].imshow(slice_ct, cmap="gray", vmin=DISPLAY_WIN[0], vmax=DISPLAY_WIN[1])
    axes[0, 0].set_title("original CT (soft-tissue window)")

    axes[0, 1].imshow(overlay(slice_ct, masks["lung"][z_mid], (0.25, 0.55, 1.0)))
    axes[0, 1].set_title(f"lung mask  ({int(masks['lung'].sum()):,} vox)")

    axes[0, 2].imshow(overlay(slice_ct, masks["bone"][z_mid], (1.0, 0.35, 0.15)))
    axes[0, 2].set_title(f"bone mask  ({int(masks['bone'].sum()):,} vox)")

    # 第二行：MIP
    axes[1, 0].imshow(slice_ct, cmap="gray", vmin=DISPLAY_WIN[0], vmax=DISPLAY_WIN[1])
    axes[1, 0].imshow(overlay(slice_ct, masks["body"][z_mid], (0.9, 0.75, 0.4), alpha=0.3))
    axes[1, 0].set_title(f"body mask  ({int(masks['body'].sum()):,} vox)")

    axes[1, 1].imshow(masks["lung"].max(axis=1), cmap="Blues")
    axes[1, 1].set_title("lung MIP (coronal)")

    axes[1, 2].imshow(masks["bone"].max(axis=1), cmap="Oranges")
    axes[1, 2].set_title("bone MIP (coronal)")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    out = OUT_DIR / f"{args.out_prefix}_inspect.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=95)
    plt.close(fig)
    print(f"→ {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
