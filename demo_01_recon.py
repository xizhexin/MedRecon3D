# -*- coding: utf-8 -*-
"""MedRecon3D · Demo 01 —— 从 DICOM 到可旋转三维模型 + STL

链路:
  DICOM 序列 → HU 转换 → 各向同性重采样 → 体表/器官分割 + 后处理
            → Marching Cubes → 网格平滑/简化 → 体积测量
            → STL 导出 + 交互式 HTML

用法:
  python demo_01_recon.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
import trimesh
import plotly.graph_objects as go
from scipy import ndimage
from skimage import measure

from volume_grid import PAD_HU, isotropic_size

# ---------------------------------------------------------------- 配置
_ROOT = Path(__file__).resolve().parent

# 默认 DICOM 目录：优先本机试验数据，没有就退回仓库内的公开数据集。
# 不能只写死 Windows 绝对路径 —— 这份代码要在 Linux GPU 服务器上跑，
# `Path(r"C:\...")` 在 POSIX 下会变成一个名字里带反斜杠的**相对路径**，
# mkdir 会凭空造出 `./C:\Ai\agent\...` 这样的怪目录（踩过）。
_LOCAL_CT = Path(r"C:\Ai\agent\workbuddy\selfpro\job_analysis\dicom_lab\data\ct_series")
DICOM_DIR = _LOCAL_CT if _LOCAL_CT.exists() else (_ROOT / "data" / "PCIR_torso" / "Heart_CT")
OUT_DIR = _ROOT / "out"          # 一直是仓库相对路径，两个平台都对
TARGET_SPACING = 1.0          # 各向同性重采样目标 (mm)
TARGET_FACES = 25000          # 每个结构简化到的三角面数上限
SMOOTH_ITERS = 12             # Taubin 平滑迭代次数

# 数据来源说明（写进 summary JSON，保证数据集可追溯）
DATA_SOURCES = {
    "Heart_CT": "PCIR 人体躯干 CT（Series: CTA AORTA，GE MEDICAL SYSTEMS，43Y 男性）。"
                "License: CC0 1.0 Universal；DOI: 10.5281/zenodo.18225140",
    "PCIR_torso": "PCIR 人体躯干 CT（Series: CTA AORTA）。License: CC0 1.0；"
                  "DOI: 10.5281/zenodo.18225140",
    "ct_series": "质控体模（椭圆模体 + 两个低密度腔 + 小球测试模块），非人体 CT",
}

# 分割阈值（HU）
HU_AIR = -900          # 低于此值视为体外空气
HU_TISSUE = -500       # 高于此值视为实性组织（软组织 / 骨）
HU_LUNG = -400         # 低于此值且位于体内 → 肺腔 / 气道
HU_BONE = 200          # 高于此值且位于体内 → 骨皮质 / 松质骨
MIN_LUNG_VOXELS = 1500  # 肺腔最小连通域体积，用于剔除肠气误检


# ---------------------------------------------------------------- M1 读取
def discover_dicom_files(dicom_dir: Path, depth: int = 1) -> list[Path]:
    """发现目录下的 DICOM 文件。

    不能只按 *.dcm 匹配：真实数据集里的 DICOM 常常没有扩展名
    （如 PCIR 数据集的文件名就是 2602 / 2633 这样的裸数字）。
    所以按标准做法检查偏移 128 处的 b'DICM' magic。
    """
    found: list[Path] = []
    for p in sorted(dicom_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            with open(p, "rb") as fh:
                fh.seek(128)
                if fh.read(4) == b"DICM":
                    found.append(p)
        except OSError:
            continue
    if found:
        return found

    # 目录里没有，就往子目录找一层（数据集常是一层包装）
    if depth > 0:
        for sub in sorted(dicom_dir.iterdir()):
            if sub.is_dir():
                inner = discover_dicom_files(sub, depth - 1)
                if inner:
                    return inner
    return []


def load_dicom_series(dicom_dir: Path) -> tuple[sitk.Image, dict]:
    """读取 DICOM 序列并转换为 HU 单位。

    刻意不用 sitk.ImageSeriesReader 的自动管线：它会自动应用 RescaleSlope/Intercept，
    再手动转一次就会重复减 1024（实测过，HU 会变成 [-2048, -274]，骨头全丢）。
    这里用 pydicom 读原始 stored value，自己转 HU，语义清楚也可控。
    """
    files = discover_dicom_files(dicom_dir)
    if not files:
        raise FileNotFoundError(f"目录下没有找到 DICOM 文件: {dicom_dir}")
    dsets = [pydicom.dcmread(str(f)) for f in files]

    # 1) 层序：优先 InstanceNumber，其次 SliceLocation，都没有就按文件名
    def sort_key(pair):
        i, ds = pair
        for tag in ("InstanceNumber", "SliceLocation"):
            v = getattr(ds, tag, None)
            if v not in (None, ""):
                try:
                    return (0, float(v))
                except (TypeError, ValueError):
                    pass
        return (1, float(i))

    dsets = [ds for _, ds in sorted(enumerate(dsets), key=sort_key)]

    # 2) HU 转换: HU = slope × stored + intercept（逐层，不同层参数可能不同）
    stored = np.stack([ds.pixel_array.astype(np.float32) for ds in dsets])     # (z, y, x)
    slope = np.array([float(getattr(ds, "RescaleSlope", 1) or 1) for ds in dsets],
                     dtype=np.float32).reshape(-1, 1, 1)
    intercept = np.array([float(getattr(ds, "RescaleIntercept", 0) or 0) for ds in dsets],
                         dtype=np.float32).reshape(-1, 1, 1)
    hu = stored * slope + intercept

    # 3) 几何: PixelSpacing 是 [行间距, 列间距] = [y, x]；层间距取 SliceLocation 差的中位数
    ps = getattr(dsets[0], "PixelSpacing", [1.0, 1.0])
    row_sp, col_sp = float(ps[0]), float(ps[1])
    locs = [float(getattr(ds, "SliceLocation", 0) or 0) for ds in dsets]
    diffs = np.diff(sorted(locs))
    positive = diffs[diffs > 0]
    if positive.size:
        z_sp = float(np.median(positive))
    else:
        z_sp = float(getattr(dsets[0], "SpacingBetweenSlices", 0)
                     or getattr(dsets[0], "SliceThickness", 1.0))

    img = sitk.GetImageFromArray(hu)          # numpy (z, y, x) → SimpleITK
    img.SetSpacing((col_sp, row_sp, z_sp))    # SimpleITK 是 (x, y, z) 序

    meta = {
        "n_files": len(files),
        "modality": str(getattr(dsets[0], "Modality", "?")),
        "size_xyz": tuple(int(v) for v in img.GetSize()),
        "spacing_xyz": tuple(round(float(v), 3) for v in img.GetSpacing()),
        "rescale_slope": float(slope.ravel()[0]),
        "rescale_intercept": float(intercept.ravel()[0]),
        "hu_min": round(float(hu.min()), 1),
        "hu_max": round(float(hu.max()), 1),
    }
    return img, meta


def resample_isotropic(img: sitk.Image, target: float = TARGET_SPACING) -> sitk.Image:
    """重采样到各向同性 spacing（mm）。

    SimpleITK 的 GetSpacing/GetSize 是 (x, y, z) 序，numpy 数组是 (z, y, x) 序，
    这里只在 SimpleITK 域内计算 new_size，不涉及翻转。

    网格尺寸走 `volume_grid.isotropic_size()` 的规则（**不是 round()**），
    补值用空气而不是 0 —— 这两条都踩过坑，详见 `volume_grid.py` 的模块头部。
    简言之：`round()` 会让最后一层落在输入范围之外，SimpleITK 拿
    `defaultPixelValue=0.0` 去补，而 0 HU 是水不是空气，于是那一整层
    450×450 个体素全被 `arr > -500` 判成实性组织，体表体积虚增 202.5 mL。
    """
    old_spacing = np.array(img.GetSpacing(), dtype=float)
    old_size = np.array(img.GetSize(), dtype=float)
    new_spacing = np.array([target, target, target], dtype=float)
    new_size = isotropic_size(old_size, old_spacing, new_spacing)

    return sitk.Resample(
        img, [int(v) for v in new_size], sitk.Transform(), sitk.sitkLinear,
        img.GetOrigin(), new_spacing.tolist(), img.GetDirection(),
        float(PAD_HU), img.GetPixelID(),
    )


# ---------------------------------------------------------------- M2 分割
def keep_top_components(mask: np.ndarray, n: int) -> np.ndarray:
    """只保留最大的 n 个连通域"""
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    order = np.argsort(sizes)[::-1][:n]
    return np.isin(labeled, order + 1)


def drop_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """丢掉小于 min_voxels 的连通域。

    真实腹部 CT 里肠道气体同样满足 HU < -400，会被误判成肺，
    但这些气腔碎而小，靠体积阈值能干净去掉。
    """
    if min_voxels <= 1:
        return mask
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    keep = np.where(sizes >= min_voxels)[0] + 1
    return np.isin(labeled, keep)


def segment(arr: np.ndarray) -> dict[str, np.ndarray]:
    """基于 HU 阈值的体表 / 肺 / 骨分割。

    关键点：直接用 HU < -400 会把体外空气也算成肺，所以必须先求体表轮廓，
    再在体表内部取肺腔。逐层做 2D 填充，避免截断扫描在 z 方向"漏"出去。
    """
    # 1) 实性组织（软组织 + 骨），排除体外空气
    tissue = keep_top_components(arr > HU_TISSUE, 1)

    # 2) 体表轮廓：逐层 2D 填充内部空腔（肺腔 / 气道）
    body = np.zeros_like(tissue)
    for z in range(tissue.shape[0]):
        sl = ndimage.binary_fill_holes(tissue[z])
        body[z] = keep_top_components(sl, 1) if sl.any() else sl

    # 3) 肺腔 = 体表内 & 非实性组织 & HU 低，再按体积剔除肠气等小气腔
    lung = body & ~tissue & (arr < HU_LUNG)
    lung = drop_small_components(lung, MIN_LUNG_VOXELS)

    # 4) 骨 = 体内 & HU 高
    bone = body & (arr > HU_BONE)

    return {"body": body, "lung": lung, "bone": bone}


# ---------------------------------------------------------------- M3 重建
def refine_mesh(verts: np.ndarray, faces: np.ndarray,
                target_faces: int = TARGET_FACES,
                smooth_iters: int = SMOOTH_ITERS) -> tuple[trimesh.Trimesh, int]:
    """构建 trimesh → Taubin 平滑 → 二次误差简化

    注意: trimesh 5.x 的 mesh.simplify_quadric_decimation(n) 会把 n 当成 target_reduction
    传给 fast_simplification，直接报 "must be between 0 and 1"。
    所以这里绕过封装，直接调 fast_simplification，显式传 target_count。
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
    raw_faces = len(mesh.faces)

    if raw_faces > target_faces:
        try:
            from fast_simplification import simplify as fs_simplify
            v, f = fs_simplify(
                np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                np.ascontiguousarray(mesh.faces, dtype=np.int64),
                target_count=target_faces,
            )
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        except Exception as exc:                      # 降级：保留原始面数
            print(f"      ! 网格简化失败，保留原始 {raw_faces:,} 面: {exc}")

    # 封闭网格统一法线朝外：Marching Cubes 的绕序可能朝内（体积为负），
    # 朝内的 STL 送进切片软件会报错。开放曲面（截断结构）不动它。
    if mesh.is_watertight:
        try:
            mesh.fix_normals()
        except Exception:
            pass
    return mesh, raw_faces


def measure_volume(mesh: trimesh.Trimesh) -> dict:
    """体积测量（mm³ → mL）

    两个坑：
    1) Marching Cubes 输出的面片绕序可能朝内，mesh.volume 会是负数 —— 取绝对值，
       但要记住原符号，它能反过来说明法线方向（修复法线用 mesh.fix_normals()）。
    2) 截断扫描（体表、肋骨）生成的是开放曲面，mesh.volume 直接不可用，
       这类结构必须回退到体素法。所以 watertight 必须一起报告。
    """
    raw_vol = float(mesh.volume)
    return {
        "volume_mm3": round(abs(raw_vol), 1),
        "volume_ml": round(abs(raw_vol) / 1000.0, 2),
        "volume_raw_sign": "outward" if raw_vol >= 0 else "inward",
        "watertight": bool(mesh.is_watertight),
        "volume_reliable": bool(mesh.is_watertight),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bbox_mm": [round(float(x), 2) for x in mesh.extents],
    }


# ---------------------------------------------------------------- 可视化
STRUCT_STYLE = {
    "bone": dict(label="骨骼", color="#E8E3D9", opacity=1.0, faces=18000),
    "lung": dict(label="肺 / 气道", color="#E8846F", opacity=0.55, faces=18000),
    "body": dict(label="体表", color="#D8C6B0", opacity=0.18, faces=8000),
}


def build_figure(meshes: list[tuple[trimesh.Trimesh, str]],
                 title: str = "MedRecon3D · CT 三维重建（可旋转 / 缩放）") -> go.Figure:
    fig = go.Figure()
    for mesh, key in meshes:
        style = STRUCT_STYLE[key]
        v, f = mesh.vertices, mesh.faces
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=style["color"], name=style["label"], opacity=style["opacity"],
            flatshading=False,
            lighting=dict(ambient=0.45, diffuse=0.75, specular=0.25, roughness=0.5),
        ))
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", bgcolor="white",
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


# ---------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MedRecon3D · DICOM 序列 → 三维模型 + STL + 交互式 HTML")
    parser.add_argument("--dicom-dir", type=Path, default=DICOM_DIR, help="DICOM 序列目录")
    parser.add_argument("--out-prefix", default="demo01", help="输出文件名前缀")
    parser.add_argument("--target-spacing", type=float, default=TARGET_SPACING,
                        help="各向同性重采样目标间距 (mm)")
    parser.add_argument("--faces", type=int, default=None,
                        help="每个结构的目标三角面数（默认按结构类型取内置值）")
    args = parser.parse_args(argv)

    dicom_dir: Path = args.dicom_dir
    prefix: str = args.out_prefix
    target_spacing: float = args.target_spacing
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print(f"[1/5] 读取 DICOM 序列: {dicom_dir}")
    img, meta = load_dicom_series(dicom_dir)
    print(f"      {meta['modality']} · {meta['n_files']} 层 · size(xyz)={meta['size_xyz']} "
          f"spacing={meta['spacing_xyz']}")
    print(f"      Rescale: HU = {meta['rescale_slope']}×stored + ({meta['rescale_intercept']})  "
          f"→ HU [{meta['hu_min']}, {meta['hu_max']}]")

    print(f"[2/5] 各向同性重采样 → {target_spacing} mm")
    t0 = time.perf_counter()
    img_r = resample_isotropic(img, target_spacing)
    arr = sitk.GetArrayFromImage(img_r)                              # (z, y, x)
    spacing_zyx = tuple(float(s) for s in img_r.GetSpacing()[::-1])
    print(f"      {tuple(int(v) for v in img_r.GetSize())} → array{arr.shape}  "
          f"spacing(zyx)={tuple(round(s, 2) for s in spacing_zyx)}  耗时 {time.perf_counter() - t0:.2f}s")

    print("[3/5] 分割")
    t0 = time.perf_counter()
    masks = segment(arr)
    print(f"      体表 {int(masks['body'].sum()):>9,} vox | "
          f"肺 {int(masks['lung'].sum()):>8,} vox | "
          f"骨 {int(masks['bone'].sum()):>7,} vox   耗时 {time.perf_counter() - t0:.2f}s")

    results, meshes = [], []
    for key in ("body", "lung", "bone"):
        mask = masks[key]
        if mask.sum() == 0:
            print(f"[4/5] ! {key} 为空，跳过")
            continue
        print(f"[4/5] Marching Cubes + 网格优化: {STRUCT_STYLE[key]['label']}")
        t0 = time.perf_counter()

        # 关键：先给 mask 补零边再重建。
        # Marching Cubes 对被扫描范围截断的结构（体表、肋骨、气道）会生成**开放曲面**，
        # 而 trimesh 的 mesh.volume 对开放曲面是用「原点封口」算的 —— 结果依赖坐标原点，
        # 不是一个有意义的数字（实测同一份数据 79.76% 和 1.57% 都出现过）。
        # 补两圈零边让曲面在视野内闭合，体积才可比、可解释。
        PAD = 2
        padded = np.pad(mask.astype(np.float32), PAD, mode="constant",
                        constant_values=0.0)
        verts, faces, _, _ = measure.marching_cubes(
            padded, level=0.5, spacing=spacing_zyx)
        verts -= np.asarray(spacing_zyx, dtype=float) * PAD      # 抵消补边偏移
        # marching_cubes 的顶点是 (z,y,x) 索引乘 spacing，直接导 STL 的话三个轴是错位的。
        # 翻成 (x,y,z) 物理 mm，这样 STL 送进切片软件/后续配准才对得上。
        # （轴序置换会把面片绕向反过来，所以在 refine_mesh 里对封闭网格统一修法线）
        verts = np.ascontiguousarray(verts[:, ::-1])
        face_budget = args.faces or STRUCT_STYLE[key]["faces"]
        # 测量和显示分开用两个网格：
        #   meas_mesh —— 未简化的忠实网格，体积/水密性都基于它；
        #   mesh      —— 平滑 + 简化后的展示/导出网格。
        # 原因：fast_simplification 会破坏水密性（实测肺 watertight True -> False），
        # 拿简化后的网格报体积，等于把算法缺陷算进测量结果里。
        meas_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        if meas_mesh.is_watertight:
            try:
                meas_mesh.fix_normals()
            except Exception:
                pass
        mesh, raw_faces = refine_mesh(verts, faces, target_faces=face_budget)

        m = measure_volume(meas_mesh)
        m["faces_measured"] = int(len(meas_mesh.faces))
        m["faces_exported"] = int(len(mesh.faces))
        m["export_watertight"] = bool(mesh.is_watertight)
        # 体素法体积，用于和网格体积交叉验证（面试会问"准不准"）
        voxel_ml = float(mask.sum() * np.prod(spacing_zyx) / 1000.0)
        m.update({
            "structure": key,
            "label": STRUCT_STYLE[key]["label"],
            "voxel_ml": round(voxel_ml, 2),
            "faces_before_simplify": int(raw_faces),
            "recon_seconds": round(time.perf_counter() - t0, 2),
        })
        if voxel_ml > 0:
            m["mesh_vs_voxel_diff_pct"] = round(
                abs(m["volume_ml"] - voxel_ml) / voxel_ml * 100, 2)
        results.append(m)
        meshes.append((mesh, key))

        stl_path = OUT_DIR / f"{prefix}_{key}.stl"
        mesh.export(str(stl_path))
        note = "" if m["watertight"] else "   ← 非封闭，体积以体素法为准"
        print(f"      测量网格 {m['faces_measured']:>8,} 面（未简化）→ 导出 "
              f"{m['faces_exported']:>6,} 面   "
              f"网格体积 {m['volume_ml']:>8.2f} mL（体素法 {voxel_ml:.2f}，"
              f"差 {m.get('mesh_vs_voxel_diff_pct', 0):.2f}%）  封闭={m['watertight']}{note}")
        if not m["export_watertight"]:
            print(f"      ! 简化后的导出网格不再水密（简化算法的副作用），"
                  f"要打印请用 --faces 0 跳过简化")
        print(f"      包围盒 {m['bbox_mm']} mm   耗时 {m['recon_seconds']}s   → {stl_path.name}")

    print("[5/5] 生成交互式 HTML + 结果 JSON")
    title = (f"MedRecon3D · {dicom_dir.name} 三维重建（可旋转 / 缩放）"
             f"<br><sup>{meta['n_files']} 层 · {meta['size_xyz'][0]}×{meta['size_xyz'][1]} · "
             f"{meta['spacing_xyz'][2]} mm 层厚 · CT</sup>")
    fig = build_figure(meshes, title=title)
    html_path = OUT_DIR / f"{prefix}_recon.html"
    fig.write_html(str(html_path), include_plotlyjs=True)
    print(f"      → {html_path.name}  ({html_path.stat().st_size / 1024 / 1024:.2f} MB)")

    summary = {
        "source_dir": str(dicom_dir),
        "data_note": DATA_SOURCES.get(dicom_dir.name, DATA_SOURCES.get(dicom_dir.parent.name, "")),
        "params": {
            "target_spacing_mm": target_spacing,
            "out_prefix": prefix,
        },
        "dicom_meta": meta,
        "resampled": {
            "array_shape": list(arr.shape),
            "spacing_zyx": [round(s, 3) for s in spacing_zyx],
            "z_coverage_mm": round(arr.shape[0] * spacing_zyx[0], 1),
        },
        "structures": results,
        "total_seconds": round(time.perf_counter() - t_start, 2),
    }
    (OUT_DIR / f"{prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      → {prefix}_summary.json")

    print("-" * 62)
    print(f"完成，总耗时 {summary['total_seconds']}s")
    print(f"注意: Z 轴仅覆盖 {summary['resampled']['z_coverage_mm']} mm，"
          f"体积是该范围内的部分体积，不是全器官体积")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
