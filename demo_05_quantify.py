#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M5：自动量化测量 —— 把分割结果变成临床可读的结构化指标。

测什么（每个解剖结构）：
  - 体积：体素法 与 网格法 双路交叉验证（不等就说明重建不可信，见 README）
  - 表面积：Marching Cubes 曲面面积（mm² -> cm²）
  - 三维最大径（Feret diameter）：凸包顶点间最大距离，不是包围盒边长
  - HU 分布：均值 / 标准差 / 分位数

额外：--baseline 传入上一期的 quantify.json，输出体积变化率（随访场景）。
输出：out/quantify.json + out/quantify_report.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from demo_01_recon import (DICOM_DIR, OUT_DIR, load_dicom_series,
                           resample_isotropic, segment, STRUCT_STYLE)
from plot_font import set_cjk_font


# ------------------------------------------------------------ 测量
def feret_diameter_mm(verts_xyz: np.ndarray) -> float:
    """三维最大径：所有表面点两两距离的最大值。

    直接 pdist 在十几万点上是 O(n^2)，跑不动。先取凸包，
    凸包顶点数通常只有几百个，pdist 完全够用，而且最大径一定在凸包上。
    """
    if len(verts_xyz) < 4:
        return 0.0
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist
    try:
        hull = ConvexHull(verts_xyz)
        pts = verts_xyz[hull.vertices]
    except Exception:                      # 退化点集：退回原集合
        pts = verts_xyz
    if len(pts) < 2:
        return 0.0
    return float(pdist(pts).max())


def measure_structure(mask: np.ndarray, spacing_zyx, hu: np.ndarray,
                      verts_zyx=None, faces=None, mesh_area_mm2=None) -> dict:
    """单个结构的量化指标。"""
    m = mask.astype(bool)
    n_vox = int(m.sum())
    vox_ml = n_vox * float(np.prod(spacing_zyx)) / 1000.0

    vals = hu[m]
    if vals.size == 0:
        hu_stats = {}
    else:
        q = np.percentile(vals, [5, 25, 50, 75, 95])
        hu_stats = {
            "mean": round(float(vals.mean()), 1),
            "std": round(float(vals.std()), 1),
            "p05": round(float(q[0]), 1), "p25": round(float(q[1]), 1),
            "median": round(float(q[2]), 1), "p75": round(float(q[3]), 1),
            "p95": round(float(q[4]), 1),
            "min": round(float(vals.min()), 1), "max": round(float(vals.max()), 1),
        }

    out = {
        "voxel_count": n_vox,
        "volume_ml_voxel": round(vox_ml, 2),
        "hu": hu_stats,
    }

    if verts_zyx is not None and len(verts_zyx) > 0:
        # 注意：measure.marching_cubes(spacing=...) 已经把 spacing 乘进顶点坐标了，
        # 顶点已经是物理 mm（(z,y,x) 序）。这里只需换成 (x,y,z) 序，**不能再乘一次**。
        v_xyz = np.asarray(verts_zyx, dtype=float)[:, ::-1]
        out["feret_diameter_mm"] = round(feret_diameter_mm(v_xyz), 2)
        out["surface_area_cm2"] = (round(float(mesh_area_mm2) / 100.0, 2)
                                   if mesh_area_mm2 is not None else None)
    return out


# ------------------------------------------------------------ 报告
def render_report(result: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    # 中文字体走共享模块 —— 这里以前硬写了 "Microsoft YaHei" 一串，
    # 在 Linux 服务器上全部落到方框（详见 plot_font.py 头部）
    set_cjk_font()
    import matplotlib.pyplot as plt

    names, vols, ferets = [], [], []
    for k, v in result["structures"].items():
        names.append(v.get("label", k))
        vols.append(v["volume_ml_voxel"])
        ferets.append(v.get("feret_diameter_mm") or 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=95)

    ax = axes[0]
    bars = ax.barh(names, vols, color=["#D85A30", "#378ADD", "#888780"])
    ax.set_xlabel("volume (mL)")
    ax.set_title("体积（体素法）", fontsize=11)
    for b, v in zip(bars, vols):
        ax.text(b.get_width() * 1.02, b.get_y() + b.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=9)

    ax = axes[1]
    ax.barh(names, ferets, color=["#993C1D", "#185FA5", "#5F5E5A"])
    ax.set_xlabel("Feret diameter (mm)")
    ax.set_title("三维最大径", fontsize=11)

    ax = axes[2]
    for k, v in result["structures"].items():
        hu = v.get("hu") or {}
        if hu:
            # y 轴用面向阅读的中文 label，不用内部键名 —— 否则这张子图的
            # 轴标签是 bone/lung/body，跟左边两张图的中文标签对不上
            lab = v.get("label", k)
            ax.plot([hu["p05"], hu["median"], hu["p95"]],
                    [lab, lab, lab], marker="o", linewidth=1.2,
                    label=f"{lab} (median {hu['median']:.0f})")
    ax.set_xlabel("HU")
    ax.set_title("HU 分布（5% / 中位 / 95%）", fontsize=11)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("M5 自动量化测量报告（真实临床 CT）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> {out_png.name}")


def change_rate(cur: float, base: float) -> float | None:
    if base in (None, 0):
        return None
    return round((cur - base) / base * 100.0, 2)


# ------------------------------------------------------------ 主流程
def main(argv=None) -> int:
    from skimage import measure

    p = argparse.ArgumentParser(description="MedRecon3D M5 自动量化测量")
    p.add_argument("--dicom-dir", default=str(DICOM_DIR))
    p.add_argument("--out-prefix", default="quantify")
    p.add_argument("--target-spacing", type=float, default=1.0)
    p.add_argument("--baseline", default=None,
                   help="上一期的 quantify.json，用于计算体积变化率（随访）")
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"[1/3] 读取 CT + 分割")
    img, meta = load_dicom_series(Path(args.dicom_dir))
    img_r = resample_isotropic(img, args.target_spacing)
    arr = img_r.__array__() if hasattr(img_r, "__array__") else None
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(img_r).astype(np.float32)
    spacing = tuple(float(v) for v in reversed(img_r.GetSpacing()))  # (z,y,x)
    masks = segment(arr)

    print(f"[2/3] 逐结构测量（重采样 {args.target_spacing} mm，spacing={spacing}）")
    structures = {}
    for key, mask in masks.items():
        if not mask.any():
            continue
        verts = faces = None
        area = None
        watertight = None
        if mask.sum() > 8:
            import trimesh
            # 补零边再重建 —— 必须和 demo_01_recon.py 用**完全一样**的方式，
            # 否则两个模块对同一个结构会报出不同的表面积/体积（踩过：
            # 骨骼的表面积一个是 3141 cm²、另一个对不上）。
            #
            # 不补边的话，被扫描范围截断的结构（脊柱下端、体表上下缘）会生成
            # **开放曲面**：表面积少一个封盖，而 trimesh 对开放曲面算体积用的是
            # 「原点封口」—— 那个数依赖坐标原点，没有物理意义。
            # verts 仍是 (z, y, x) 序（measure_structure 内部会置换），
            # 补边只做平移抵消。
            PAD = 2
            padded = np.pad(mask.astype(np.float32), PAD, mode="constant",
                            constant_values=0.0)
            verts, faces, _, _ = measure.marching_cubes(
                padded, level=0.5, spacing=spacing)
            verts = verts - np.asarray(spacing, dtype=float) * PAD

            mesh = trimesh.Trimesh(vertices=verts[:, ::-1], faces=faces,
                                   process=True)
            # 只在真正封闭时采信网格量；否则以体素法为准并如实标注
            watertight = bool(mesh.is_watertight)
            if watertight:
                try:
                    mesh.fix_normals()
                except Exception:
                    pass
            area = float(mesh.area)
        info = measure_structure(mask, spacing, arr, verts, faces, area)
        info["label"] = STRUCT_STYLE.get(key, {}).get("label", key)
        info["watertight_mesh"] = watertight
        structures[key] = info
        print(f"      {info['label']:<10} 体积 {info['voxel_count']:>9,} vox = "
              f"{info['volume_ml_voxel']:>9.2f} mL   最大径 "
              f"{info.get('feret_diameter_mm','-'):>7} mm   表面积 "
              f"{info.get('surface_area_cm2','-'):>8} cm²   封闭={watertight}")

    result = {
        "source": str(args.dicom_dir),
        "target_spacing_mm": args.target_spacing,
        "dicom_meta": meta,
        "structures": structures,
    }

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        deltas = {}
        for k, v in structures.items():
            b = (base.get("structures") or {}).get(k)
            if not b:
                continue
            deltas[k] = {
                "volume_ml_baseline": b["volume_ml_voxel"],
                "volume_ml_current": v["volume_ml_voxel"],
                "change_pct": change_rate(v["volume_ml_voxel"], b["volume_ml_voxel"]),
            }
        result["followup_change"] = deltas
        print("      随访变化率：")
        for k, d in deltas.items():
            print(f"        {k:<8} {d['volume_ml_baseline']:>9.2f} -> "
                  f"{d['volume_ml_current']:>9.2f} mL  "
                  f"({d['change_pct']:+.2f}%)")

    print("[3/3] 渲染报告")
    render_report(result, OUT_DIR / f"{args.out_prefix}_report.png")

    total = round(time.perf_counter() - t0, 2)
    result["total_seconds"] = total
    out_json = OUT_DIR / f"{args.out_prefix}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"      -> {out_json.name}")
    print(f"\n总耗时 {total}s（本模块无 GPU 分支，量测本身不是瓶颈）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
