#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MedRecon3D 端到端管线封装：DICOM 目录 -> 分割 -> 三维网格 -> STL / HTML / JSON。

刻意不依赖任何 Web 框架，好让 CLI、FastAPI、批处理脚本共用同一份实现，
避免"服务里跑的和论文里跑的不是一套代码"这种经典事故。
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage import measure

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from demo_01_recon import (STRUCT_STYLE, build_figure, load_dicom_series,  # noqa: E402
                           measure_volume, refine_mesh, resample_isotropic,
                           segment)

STRUCTURES = ("bone", "lung", "body")


def safe_extract(zip_path: Path, dest: Path) -> Path:
    """解压上传压缩包，带 Zip Slip 目录穿越防护。

    压缩包里可以塞 "../../windows/system32/x" 这种条目，直接 extractall
    会写到目标目录之外。上传接口必须挡这一下。
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            target = (dest / name).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"压缩包含非法路径条目: {name}")
        z.extractall(dest)
    return dest


def run_pipeline(dicom_dir: Path, out_dir: Path, target_spacing: float = 1.0,
                 structures: tuple[str, ...] = STRUCTURES,
                 make_html: bool = True, device: str = "auto",
                 threads: int = 0) -> dict:
    """完整跑一遍：读 DICOM → 分割 → Marching Cubes → 网格后处理 → 导出。

    参数 `device`：`"auto"`（有 GPU 就用）/ `"cuda"`（强制要求）/ `"cpu"`（强制 CPU）。
    管线会按设备选实现，**产物格式完全一致**（同一套 Marching Cubes、同一套网格后处理），
    所以换设备不会让下游的比对失效。
    """
    from gpu.backend import pick_device

    t0 = time.perf_counter()
    dicom_dir, out_dir = Path(dicom_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev, dev_info = pick_device(device, verbose=False)

    img, meta = load_dicom_series(dicom_dir)

    if dev is not None:
        # GPU 路径：重采样与阈值比较上卡，连通域/孔洞填充仍是 CPU（见 gpu/ops.py 的说明）
        from demo_01_recon_gpu import segment_gpu
        from gpu import ops

        src = sitk.GetArrayFromImage(img)
        src_spacing_zyx = tuple(float(v) for v in img.GetSpacing()[::-1])
        arr, spacing_zyx, _ = ops.resample_volume_gpu(
            src, src_spacing_zyx, target_spacing, dev)
        arr = arr.astype(np.float32)
        masks = segment_gpu(arr, dev, threads)
    else:
        from demo_01_recon_gpu import segment_cpu

        img_r = resample_isotropic(img, target_spacing)
        arr = sitk.GetArrayFromImage(img_r).astype(np.float32)
        spacing_zyx = tuple(float(v) for v in reversed(img_r.GetSpacing()))
        masks = segment_cpu(arr, threads) if threads > 0 else segment(arr)

    result: dict = {
        "source": str(dicom_dir),
        "dicom_meta": meta,
        "target_spacing_mm": target_spacing,
        "compute": {
            "requested_device": device,
            "used_gpu": dev is not None,
            "device_info": dev_info.to_dict(),
        },
        "grid": {
            "size_xyz": [int(arr.shape[2]), int(arr.shape[1]), int(arr.shape[0])],
            "spacing_xyz": [round(float(v), 4) for v in reversed(spacing_zyx)],
        },
        "structures": {},
    }

    meshes = []
    for key in structures:
        mask = masks.get(key)
        if mask is None or not mask.any():
            continue
        # 补两圈零边让截断结构在视野内闭合，否则 mesh.volume 是「原点封口」
        # 算出来的数，依赖坐标原点、没有物理意义（详见 demo_01_recon.py 注释）
        PAD = 2
        padded = np.pad(mask.astype(np.float32), PAD, mode="constant",
                        constant_values=0.0)
        verts, faces, _, _ = measure.marching_cubes(
            padded, level=0.5, spacing=spacing_zyx)
        verts -= np.asarray(spacing_zyx, dtype=float) * PAD
        # (z,y,x) 索引 * spacing -> (x,y,z) 物理 mm，否则导出的 STL 轴是错位的
        verts = np.ascontiguousarray(verts[:, ::-1])
        raw_faces = int(len(faces))
        # 测量用未简化网格（fast_simplification 会破坏水密性），显示/导出用简化网格
        import trimesh
        meas_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        if meas_mesh.is_watertight:
            try:
                meas_mesh.fix_normals()
            except Exception:
                pass
        mesh, _ = refine_mesh(verts, faces,
                              target_faces=STRUCT_STYLE[key]["faces"])

        stl_path = out_dir / f"{key}.stl"
        mesh.export(str(stl_path))

        m = measure_volume(meas_mesh)
        m["faces_measured"] = int(len(meas_mesh.faces))
        m["faces_exported"] = int(len(mesh.faces))
        m["export_watertight"] = bool(mesh.is_watertight)
        voxel_ml = float(mask.sum()) * float(np.prod(spacing_zyx)) / 1000.0
        diff = (abs(m["volume_ml"] - voxel_ml) / voxel_ml * 100.0) if voxel_ml else None
        m.update({
            "structure": key,
            "label": STRUCT_STYLE[key]["label"],
            "voxel_count": int(mask.sum()),
            "volume_ml_voxel": round(voxel_ml, 2),
            "volume_diff_pct": round(diff, 2) if diff is not None else None,
            "faces_before_simplify": raw_faces,
            "stl": stl_path.name,
            "stl_bytes": int(stl_path.stat().st_size),
        })
        result["structures"][key] = m
        meshes.append((mesh, key))

    if meshes and make_html:
        fig = build_figure(meshes)
        html_path = out_dir / "model.html"
        fig.write_html(str(html_path), include_plotlyjs=True)
        result["model_html"] = html_path.name

    result["total_seconds"] = round(time.perf_counter() - t0, 2)
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
