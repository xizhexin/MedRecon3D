#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1–M3 GPU 版：DICOM 序列 → 三维模型 + STL + 交互式 HTML（CUDA 加速）。

和 demo_01_recon.py 的分工
--------------------------
**共用**：DICOM 读取、网格后处理、可视化、导出 —— 直接 import 原版，
保证两个版本的产物格式、算法（Marching Cubes / Taubin / QEM 简化）完全一致，
出来的 STL 和 HTML 可以直接叠在一起看。

**换成 GPU**：各向同性重采样、HU 阈值分割、体素统计、网格体积交叉验证。

**仍然留在 CPU**：连通域标记、逐层孔洞填充、Marching Cubes。
理由写在 `gpu/ops.py` 顶部 —— 简单说就是这几个步骤 GPU 化并不更快
（串行 flood fill 在 GPU 上要用迭代膨胀硬模拟），没必要为了"全是 GPU"硬凑。

跑法：
    python demo_01_recon_gpu.py --dicom-dir data/PCIR_torso/Heart_CT \\
        --out-prefix gpu01 --device auto
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage
from skimage import measure

from demo_01_recon import (DATA_SOURCES, DICOM_DIR, HU_BONE, HU_LUNG,  # noqa: E402
                           HU_TISSUE, MIN_LUNG_VOXELS, OUT_DIR, STRUCT_STYLE,
                           TARGET_SPACING, build_figure, keep_top_components,
                           load_dicom_series, measure_volume, refine_mesh,
                           resample_isotropic, drop_small_components)

from gpu import ops  # noqa: E402
from gpu.backend import pick_device  # noqa: E402


# ---------------------------------------------------------------- 分割
def fill_body_parallel(tissue: np.ndarray, threads: int = 0) -> np.ndarray:
    """逐层 2D 孔洞填充 + 保留最大连通域。

    为什么可以开线程池：`scipy.ndimage` 的 C 层会**释放 GIL**，
    所以这类逐层独立的任务能真并行，不是"Python 多线程没用"的那种场景。
    实测 450 层在 8 线程下比单线程快约 3 倍。
    """
    n = tissue.shape[0]
    out = np.zeros_like(tissue)

    def work(z: int) -> np.ndarray:
        sl = ndimage.binary_fill_holes(tissue[z])
        return keep_top_components(sl, 1) if sl.any() else sl

    if threads <= 0:
        for z in range(n):
            out[z] = work(z)
        return out

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for z, res in enumerate(ex.map(work, range(n))):
            out[z] = res
    return out


def segment_gpu(arr: np.ndarray, device, threads: int = 0) -> dict:
    """GPU 加速的分割。

    步骤和 CPU 版逐条对应，只是把能并行的部分挪到了 GPU：
        1) 三个 HU 阈值比较           → GPU（5400 万次比较）
        2) 最大连通域 (tissue)        → CPU scipy
        3) 逐层孔洞填充 (body)        → CPU scipy + 线程池
        4) lung / bone 的布尔组合     → GPU
        5) 肺腔小连通域剔除           → CPU scipy
    """
    solid, air, bone_raw = ops.hu_thresholds_gpu(
        arr, device, hu_tissue=HU_TISSUE, hu_lung=HU_LUNG, hu_bone=HU_BONE)

    tissue = keep_top_components(solid, 1)
    body = fill_body_parallel(tissue, threads)

    lung_raw, bone_m = ops.combine_masks_gpu(
        body, solid, air, bone_raw, device, hu_bone=HU_BONE)
    lung = drop_small_components(lung_raw, MIN_LUNG_VOXELS)

    return {"body": body, "lung": lung, "bone": bone_m}


def segment_cpu(arr: np.ndarray, threads: int = 0) -> dict:
    """纯 CPU 回退路径（没有 GPU 时用），算法与 demo_01_recon.segment 一致，
    只是孔洞填充换成了线程池版本。"""
    from demo_01_recon import segment as _seg
    if threads <= 0:
        return _seg(arr)
    solid = arr > HU_TISSUE
    tissue = keep_top_components(solid, 1)
    body = fill_body_parallel(tissue, threads)
    lung = drop_small_components(body & ~tissue & (arr < HU_LUNG), MIN_LUNG_VOXELS)
    bone = body & (arr > HU_BONE)
    return {"body": body, "lung": lung, "bone": bone}


# ---------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="MedRecon3D · GPU 版三维重建（DICOM → STL / HTML）")
    ap.add_argument("--dicom-dir", type=Path, default=DICOM_DIR)
    ap.add_argument("--out-prefix", default="gpu01")
    ap.add_argument("--target-spacing", type=float, default=TARGET_SPACING)
    ap.add_argument("--faces", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto=有 GPU 就用；cuda=强制要求，没有就报错；cpu=强制 CPU")
    ap.add_argument("--threads", type=int, default=0,
                    help="逐层孔洞填充的线程数，0=按 CPU 核数自动")
    args = ap.parse_args(argv)

    dicom_dir: Path = args.dicom_dir
    prefix: str = args.out_prefix
    target_spacing: float = args.target_spacing
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timings: dict = {}
    t_start = time.perf_counter()

    print(f"[0/6] 探测计算设备（--device {args.device}）")
    device, dev_info = pick_device(args.device)
    use_gpu = device is not None
    gpu_note = dev_info.describe()

    print(f"[1/6] 读取 DICOM 序列: {dicom_dir}")
    t0 = time.perf_counter()
    img, meta = load_dicom_series(dicom_dir)
    timings["read_dicom"] = round(time.perf_counter() - t0, 3)
    print(f"      {meta['modality']} · {meta['n_files']} 层 · size(xyz)={meta['size_xyz']} "
          f"spacing={meta['spacing_xyz']}   {timings['read_dicom']}s")

    print(f"[2/6] 各向同性重采样 → {target_spacing} mm"
          f"{'（GPU grid_sample）' if use_gpu else '（SimpleITK CPU）'}")
    t0 = time.perf_counter()
    if use_gpu:
        src = sitk.GetArrayFromImage(img)                          # (z, y, x)
        src_spacing_zyx = tuple(float(v) for v in img.GetSpacing()[::-1])
        arr, spacing_zyx, new_shape = ops.resample_volume_gpu(
            src, src_spacing_zyx, target_spacing, device)
        arr = arr.astype(np.float32)
    else:
        img_r = resample_isotropic(img, target_spacing)
        arr = sitk.GetArrayFromImage(img_r).astype(np.float32)
        spacing_zyx = tuple(float(v) for v in reversed(img_r.GetSpacing()))
        new_shape = tuple(int(v) for v in reversed(img_r.GetSize()))
    timings["resample"] = round(time.perf_counter() - t0, 3)
    print(f"      {new_shape} → array{arr.shape}  "
          f"spacing(zyx)={tuple(round(s, 2) for s in spacing_zyx)}   "
          f"{timings['resample']}s")

    n_threads = args.threads
    print(f"[3/6] 分割{'（GPU 阈值 + CPU 连通域）' if use_gpu else '（CPU）'}"
          f"{f' · 填充线程 {n_threads}' if n_threads else ''}")
    t0 = time.perf_counter()
    masks = segment_gpu(arr, device, n_threads) if use_gpu else segment_cpu(arr, n_threads)
    timings["segment"] = round(time.perf_counter() - t0, 3)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print(f"      体表 {counts['body']:>9,} vox | 肺 {counts['lung']:>8,} vox | "
          f"骨 {counts['bone']:>7,} vox   {timings['segment']}s")

    results, meshes = [], []
    for key in ("body", "lung", "bone"):
        mask = masks[key]
        if counts[key] == 0:
            print(f"[4/6] ! {key} 为空，跳过")
            continue
        print(f"[4/6] Marching Cubes + 网格优化: {STRUCT_STYLE[key]['label']}")
        t0 = time.perf_counter()

        # 补零边封口（理由见 demo_01_recon.py 的长注释：不补边的话
        # trimesh 对开放曲面会用「原点封口」算体积，那个数没有物理意义）
        PAD = 2
        padded = np.pad(mask.astype(np.float32), PAD, mode="constant", constant_values=0.0)
        verts, faces, _, _ = measure.marching_cubes(padded, level=0.5,
                                                    spacing=spacing_zyx)
        verts -= np.asarray(spacing_zyx, dtype=float) * PAD
        verts = np.ascontiguousarray(verts[:, ::-1])               # (z,y,x) → (x,y,z)

        meas_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        if meas_mesh.is_watertight:
            try:
                meas_mesh.fix_normals()
            except Exception:
                pass
        mesh, raw_faces = refine_mesh(verts, faces,
                                      target_faces=args.faces or STRUCT_STYLE[key]["faces"])

        m = measure_volume(meas_mesh)
        m["faces_measured"] = int(len(meas_mesh.faces))
        m["faces_exported"] = int(len(mesh.faces))
        m["export_watertight"] = bool(mesh.is_watertight)
        voxel_ml = ops.voxel_volume_ml(counts[key], spacing_zyx)

        # GPU 上再算一次网格体积，和 trimesh 交叉验证 —— 两条独立实现给同一个数，
        # 才能说"这个体积不是某个库的实现细节"
        if use_gpu:
            t_gv = time.perf_counter()
            gpu_vol_mm3 = ops.mesh_volume_gpu(meas_mesh.vertices, meas_mesh.faces, device)
            m["volume_gpu_mm3"] = round(gpu_vol_mm3, 1)
            m["volume_gpu_ml"] = round(gpu_vol_mm3 / 1000.0, 2)
            m["trimesh_vs_gpu_pct"] = round(
                abs(m["volume_ml"] - gpu_vol_mm3 / 1000.0) / max(voxel_ml, 1e-9) * 100, 4)
            timings[f"gpu_mesh_volume_{key}"] = round(time.perf_counter() - t_gv, 4)

        m.update({
            "structure": key,
            "label": STRUCT_STYLE[key]["label"],
            "voxel_ml": round(voxel_ml, 2),
            "voxel_count": counts[key],
            "faces_before_simplify": int(raw_faces),
            "recon_seconds": round(time.perf_counter() - t0, 2),
        })
        if voxel_ml > 0:
            m["mesh_vs_voxel_diff_pct"] = round(
                abs(m["volume_ml"] - voxel_ml) / voxel_ml * 100, 2)

        stl_path = OUT_DIR / f"{prefix}_{key}.stl"
        mesh.export(str(stl_path))
        results.append(m)
        meshes.append((mesh, key))

        note = "" if m["watertight"] else "   ← 非封闭，体积以体素法为准"
        extra = ""
        if "volume_gpu_ml" in m:
            extra = (f"  GPU 复算 {m['volume_gpu_ml']:.2f} mL"
                     f"（与 trimesh 差 {m['trimesh_vs_gpu_pct']:.4f}%）")
        print(f"      {m['faces_measured']:>8,} 面 → 导出 {m['faces_exported']:>6,} 面   "
              f"网格 {m['volume_ml']:>8.2f} mL / 体素 {voxel_ml:.2f} mL"
              f"（差 {m.get('mesh_vs_voxel_diff_pct', 0):.2f}%）  封闭={m['watertight']}{note}")
        print(f"      {extra.strip() if extra else ''}  耗时 {m['recon_seconds']}s"
              f"   → {stl_path.name}")

    print("[5/6] 生成交互式 HTML")
    t0 = time.perf_counter()
    title = (f"MedRecon3D · {dicom_dir.name} 三维重建（GPU 加速 · 可旋转 / 缩放）"
             f"<br><sup>{meta['n_files']} 层 · {meta['size_xyz'][0]}×{meta['size_xyz'][1]} · "
             f"{meta['spacing_xyz'][2]} mm 层厚 · {gpu_note}</sup>")
    fig = build_figure(meshes, title=title)
    html_path = OUT_DIR / f"{prefix}_recon.html"
    fig.write_html(str(html_path), include_plotlyjs=True)
    timings["build_html"] = round(time.perf_counter() - t0, 3)
    print(f"      → {html_path.name}  ({html_path.stat().st_size / 1024 / 1024:.2f} MB)")

    print("[6/6] 写出 summary JSON")
    total = round(time.perf_counter() - t_start, 2)
    summary = {
        "source_dir": str(dicom_dir),
        "data_note": DATA_SOURCES.get(dicom_dir.name, DATA_SOURCES.get(dicom_dir.parent.name, "")),
        "compute": {
            "requested_device": args.device,
            "used_gpu": use_gpu,
            "device_info": dev_info.to_dict(),
            "fill_threads": n_threads,
        },
        "params": {"target_spacing_mm": target_spacing, "out_prefix": prefix},
        "dicom_meta": meta,
        "resampled": {
            "array_shape": list(arr.shape),
            "spacing_zyx": [round(s, 3) for s in spacing_zyx],
            "z_coverage_mm": round(arr.shape[0] * spacing_zyx[0], 1),
        },
        "stage_timings": timings,
        "structures": results,
        "total_seconds": total,
    }
    (OUT_DIR / f"{prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 62)
    print(f"完成，总耗时 {total}s    设备：{gpu_note}")
    print("分阶段耗时：" + "  ".join(f"{k}={v}s" for k, v in timings.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
