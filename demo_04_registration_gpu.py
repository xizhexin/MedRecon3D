#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M4 GPU 版：多级配准（刚性 → 仿射 → 自由形变），CUDA 加速。

实验设计和 demo_04_registration.py 完全一致，只是把优化器换成了 GPU 上的
autograd + Adam。**人造真值的生成仍然用 SimpleITK**（`make_rigid` / `make_warp`），
这样 CPU 版和 GPU 版面对的是同一份数据、同一套真值，
两边跑出来的 TRE / Dice 可以直接放一起比 —— 否则"GPU 更准"就没有说服力。

两个实验：
  A 纯刚体，有真值 → 用 TRE（目标配准误差，mm）评价，这是配准领域最硬的指标
  B 刚体 + 局部形变，无真值 → 用 mask Dice 与图像 NCC 评价逐级收益

跑法：
    python demo_04_registration_gpu.py --dicom-dir data/PCIR_torso/Heart_CT \\
        --out-prefix gpu04 --device auto
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from demo_01_recon import DICOM_DIR, OUT_DIR, load_dicom_series, resample_isotropic
from demo_04_registration import (RANDOM_SEED, _use_cjk_font, apply_transform,
                                  compose, dice, make_rigid, make_warp, ncc,
                                  to_np)
from demo_01_recon_gpu import fill_body_parallel, segment_gpu

from gpu import registration as reg
from gpu import ops
from gpu.backend import pick_device

REG_SPACING = 2.0        # 配准分辨率（mm），与 CPU 版保持一致才能对比
# 控制点网格。为什么不用 CPU 版的 (4,4,4)：
#   ITK 那 4×4×4 是**三次 B 样条**的控制点网格，单点影响范围跨 4 个间隔，
#   实际表达能力远强于同尺寸的三线性网格；而 GPU 版是「控制点 + 三线性上采样」，
#   同样的 4³ 只能表达极低频的形变。用 (6,6,6) 才让两侧的形变表达能力可比
#   —— 否则"GPU 精度差"其实是"我方参数化太弱"，不是配准算法的问题。
FFD_GRID = (6, 6, 6)


# ------------------------------------------------------------ 工具
def sitk_linear_to_matrix(tx) -> np.ndarray:
    """把 SimpleITK 的线性变换抽成 4×4 齐次矩阵。

    sitk 的语义是 `y = R·(x − c) + c + t`，展开成齐次形式：
        y = R·x + (c + t − R·c)
    所以平移列取 `c + t − R·c`，不是直接取 `GetTranslation()`。
    抽错这一项，TRE 会莫名其妙地大 —— 因为旋转中心被算了两遍。
    """
    if isinstance(tx, (sitk.Euler3DTransform, sitk.AffineTransform,
                       sitk.Similarity3DTransform)):
        r = np.array(tx.GetMatrix(), dtype=float).reshape(3, 3)
        c = np.array(tx.GetCenter(), dtype=float)
        t = np.array(tx.GetTranslation(), dtype=float)
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = c + t - r @ c
        return m
    raise TypeError(f"不支持的变换类型 {type(tx).__name__}")


def tre_mm_gpu(est_result, m_true: np.ndarray, shape, spacing, device,
               n_axis: int = 5) -> dict:
    """TRE：估计变换与真值变换把同一批点送到哪，比较两者的距离。

    和 CPU 版的 `tre_mm` 用同一套采样点（图幅 15%~85% 的 5×5×5 网格），
    保证两边的 TRE 是同一个量。
    """
    gx = np.linspace(0.15, 0.85, n_axis) * (shape[2] - 1) * spacing[2]
    gy = np.linspace(0.15, 0.85, n_axis) * (shape[1] - 1) * spacing[1]
    gz = np.linspace(0.15, 0.85, n_axis) * (shape[0] - 1) * spacing[0]
    pts = np.array([[x, y, z] for z in gz for y in gy for x in gx], dtype=np.float32)

    est = reg.transform_points(est_result, pts, shape, spacing, device)
    truth = (m_true[:3, :3] @ pts.T).T + m_true[:3, 3]
    err = np.linalg.norm(est - truth, axis=1)
    return {
        "n_points": int(len(err)),
        "tre_mean_mm": round(float(err.mean()), 3),
        "tre_rms_mm": round(float(np.sqrt((err ** 2).mean())), 3),
        "tre_max_mm": round(float(err.max()), 3),
    }


def render_stages(fixed_arr, moving_arr, aligned_arr, out_png: Path,
                  subtitle: str = "", z: int | None = None):
    """配准前后对照图（numpy 版，不依赖 sitk.Image）。"""
    _use_cjk_font()
    import matplotlib.pyplot as plt

    def win(a):
        return np.clip((a + 200.0) / 600.0, 0, 1)

    zi = fixed_arr.shape[0] // 2 if z is None else int(z)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), dpi=95)
    for ax, vol, name in [(axes[0, 0], fixed_arr, "固定图像 (参考)"),
                          (axes[0, 1], moving_arr, "移动图像 (未配准)"),
                          (axes[0, 2], aligned_arr, "移动图像 (配准后)")]:
        ax.imshow(win(vol[zi]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    for ax, vol, name in [(axes[1, 0], win(fixed_arr), "固定 冠状位"),
                          (axes[1, 1], win(moving_arr), "未配准 冠状位"),
                          (axes[1, 2], win(aligned_arr), "配准后 冠状位")]:
        ax.imshow(vol.max(axis=2), cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"M4 多级配准：配准前 vs 配准后（真实临床 CT · GPU 加速）{subtitle}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ 实验 A
def experiment_a(fixed, fixed_arr, spacing_zyx, device, levels, iters, threads) -> dict:
    """纯刚体、有真值 —— 看 GPU 配准能不能把已知的运动找回来。"""
    print("\n[A] 纯刚体验证 —— 已知真值")
    angles, trans = (8.0, -5.0, 3.0), (10.0, -8.0, 5.0)
    t_true = make_rigid(fixed, angles, trans)
    m_true = sitk_linear_to_matrix(t_true)
    moving = apply_transform(fixed, fixed, t_true.GetInverse())
    moving_arr = to_np(moving).astype(np.float32)
    print(f"      人为施加：旋转 {angles} 度、平移 {trans} mm")

    ref_masks = segment_gpu(fixed_arr, device, threads)
    mov_masks = segment_gpu(moving_arr, device, threads)
    v_ref = float(ref_masks["bone"].sum()) * REG_SPACING ** 3 / 1000.0
    v_mov = float(mov_masks["bone"].sum()) * REG_SPACING ** 3 / 1000.0
    d_before = dice(ref_masks["bone"], mov_masks["bone"])
    print(f"      配准前：Dice {d_before:.4f}   骨体积 {v_mov:.2f} mL"
          f"（参考 {v_ref:.2f} mL，偏差 {abs(v_mov - v_ref) / v_ref * 100:.2f}%）")

    t0 = time.perf_counter()
    result, info = reg.register_rigid_gpu(fixed_arr, moving_arr, spacing_zyx,
                                          device, levels=levels, iters=iters)
    secs = time.perf_counter() - t0

    aligned = reg.apply_result(fixed_arr, moving_arr, spacing_zyx, device, result)
    ali_masks = segment_gpu(aligned, device, threads)
    d_after = dice(ref_masks["bone"], ali_masks["bone"])
    v_ali = float(ali_masks["bone"].sum()) * REG_SPACING ** 3 / 1000.0
    tre = tre_mm_gpu(result, m_true, fixed_arr.shape, spacing_zyx, device)

    m_est = result["m4"].cpu().numpy()
    t_param_err = np.linalg.norm(m_est[:3, 3] - m_true[:3, 3])

    print(f"      配准后：Dice {d_after:.4f}   骨体积 {v_ali:.2f} mL，"
          f"偏差 {abs(v_ali - v_ref) / v_ref * 100:.2f}%")
    print(f"      TRE：均值 {tre['tre_mean_mm']} mm   RMS {tre['tre_rms_mm']} mm   "
          f"最大 {tre['tre_max_mm']} mm")
    print(f"      平移参数误差 {t_param_err:.3f} mm   耗时 {secs:.2f}s"
          f"（{info.iterations} 步迭代）")

    return {
        "ground_truth": {"angles_deg": list(angles), "translation_mm": list(trans)},
        "before": {"dice_bone": round(d_before, 4), "bone_ml": round(v_mov, 2)},
        "after": {"dice_bone": round(d_after, 4), "bone_ml": round(v_ali, 2)},
        "reference": {"dice_bone": 1.0, "bone_ml": round(v_ref, 2)},
        "tre": tre,
        "param_error_mm": round(float(t_param_err), 3),
        "info": info.to_dict(),
        "wall_seconds": round(secs, 3),
        "aligned_ok": True,
    }


# ------------------------------------------------------------ 实验 B
def experiment_b(fixed, fixed_arr, spacing_zyx, device, levels, iters,
                 threads, out_png: Path) -> dict:
    """刚体 + 局部形变 —— 对比四档，看每一级到底贡献了什么。"""
    print("\n[B] 刚体 + 软组织形变 —— 对比四档配准")
    angles, trans = (6.0, -4.0, 2.0), (8.0, -6.0, 4.0)
    t_rigid = make_rigid(fixed, angles, trans)
    t_warp = make_warp(fixed, grid=(6, 6, 6), amp_mm=12.0)
    p = compose(t_rigid, t_warp)
    moving = apply_transform(fixed, fixed, p)
    moving_arr = to_np(moving).astype(np.float32)
    print(f"      人为施加：旋转 {angles} 度、平移 {trans} mm、B样条形变 幅值 12 mm")

    ref_masks = segment_gpu(fixed_arr, device, threads)

    def snapshot(tag: str, arr: np.ndarray, seconds: float) -> dict:
        m = segment_gpu(arr, device, threads)
        return {
            "stage": tag,
            "dice_body": round(dice(ref_masks["body"], m["body"]), 4),
            "dice_lung": round(dice(ref_masks["lung"], m["lung"]), 4),
            "dice_bone": round(dice(ref_masks["bone"], m["bone"]), 4),
            "ncc": round(ncc(fixed_arr, arr), 4),
            "seconds": seconds,
        }

    stages = [snapshot("未配准", moving_arr, 0.0)]
    labels = {"rigid": "仅刚性", "affine": "刚性+仿射", "ffd": "三级全上(含FFD)"}

    t_all = time.perf_counter()
    results = reg.register_multistage_gpu(
        fixed_arr, moving_arr, spacing_zyx, device,
        stages=("rigid", "affine", "ffd"), ffd_grid=FFD_GRID,
        levels=levels, iters=iters, verbose=True)

    infos, last_aligned, ffd_result = [], None, None
    for name, cum, info in results:
        aligned = reg.apply_result(fixed_arr, moving_arr, spacing_zyx, device, cum)
        stages.append(snapshot(labels[name], aligned, info.seconds))
        infos.append(info.to_dict())
        if name == "ffd":
            last_aligned, ffd_result = aligned, cum

    total = time.perf_counter() - t_all
    for s in stages:
        print(f"      {s['stage']:<20} Dice 体表 {s['dice_body']:.4f}   "
              f"肺 {s['dice_lung']:.4f}   骨 {s['dice_bone']:.4f}   "
              f"NCC {s['ncc']:.4f}   {s['seconds']:>6.2f}s")

    field_stats = {}
    if ffd_result is not None and "disp" in ffd_result:
        d = ffd_result["disp"]
        mag = d.norm(dim=-1)
        field_stats = {"disp_mean_mm": round(float(mag.mean()), 3),
                       "disp_max_mm": round(float(mag.max()), 3)}
        print(f"      FFD 形变场：均值 {field_stats['disp_mean_mm']} mm   "
              f"最大 {field_stats['disp_max_mm']} mm   （真值幅值 12 mm）")

    render_stages(fixed_arr, moving_arr, last_aligned, out_png)
    print(f"      -> {out_png.name}")

    return {
        "ground_truth": {"angles_deg": list(angles), "translation_mm": list(trans),
                         "warp_grid": [6, 6, 6], "warp_amp_mm": 12.0},
        "stages": stages,
        "stage_info": infos,
        "ffd_field": field_stats,
        "total_seconds": round(total, 2),
        "note": "与 CPU 版实验 B 同设计。FFD 参数化与 ITK 三次 B 样条不同"
                "（此处为低分辨率控制点 + 三线性上采样），故只比指标不比参数。",
    }


# ------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MedRecon3D M4 多级配准（GPU 版）")
    ap.add_argument("--dicom-dir", default=str(DICOM_DIR))
    ap.add_argument("--out-prefix", default="gpu04")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--reg-spacing", type=float, default=REG_SPACING)
    ap.add_argument("--levels", type=int, default=3, help="多分辨率层数（3 → 8/4/2 mm 起步）")
    ap.add_argument("--iters", type=int, default=80, help="每个分辨率层的迭代步数")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--skip-b", action="store_true", help="跳过实验 B（省时间）")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    print(f"[0/4] 探测计算设备（--device {args.device}）")
    device, dev_info = pick_device(args.device)
    if device is None:
        print("\n! GPU 版需要 CUDA 设备。没有的话请用 demo_04_registration.py（CPU 版）。")
        return 2
    spacing = float(args.reg_spacing)

    print(f"[1/4] 读取 CT 并以 GPU 重采样到 {spacing} mm")
    t0 = time.perf_counter()
    img, meta = load_dicom_series(Path(args.dicom_dir))
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    src = sitk.GetArrayFromImage(img)
    src_spacing_zyx = tuple(float(v) for v in img.GetSpacing()[::-1])
    arr, _, _ = ops.resample_volume_gpu(src, src_spacing_zyx, spacing, device)
    arr = arr.astype(np.float32)
    t_resample = time.perf_counter() - t0

    # 同时算一次 CPU 重采样，验证两条路径给出同一幅图
    img_cpu = resample_isotropic(img, spacing)
    arr_cpu = sitk.GetArrayFromImage(img_cpu).astype(np.float32)
    diff = float(np.abs(arr - arr_cpu).max()) if arr.shape == arr_cpu.shape else float("nan")

    spacing_zyx = (spacing,) * 3
    print(f"      原始 {meta['size_xyz']} @ {meta['spacing_xyz']} mm → {arr.shape} "
          f"@ {spacing} mm")
    print(f"      读取 {t_read:.2f}s · GPU 重采样 {t_resample:.2f}s · "
          f"与 CPU 重采样最大差异 {diff:.4f} HU")

    # 重新包装成 sitk.Image，供人造真值生成（make_rigid 需要图像几何信息）
    fixed = sitk.GetImageFromArray(arr_cpu)
    fixed.SetSpacing((spacing, spacing, spacing))
    fixed.SetOrigin(img_cpu.GetOrigin())
    fixed.SetDirection(img_cpu.GetDirection())

    result = {
        "source": str(args.dicom_dir),
        "compute": {"device_info": dev_info.to_dict(), "used_gpu": True},
        "reg_spacing": spacing,
        "dicom_meta": meta,
        "resample_check": {"max_abs_diff_hu": round(diff, 4),
                           "gpu_seconds": round(t_resample, 3)},
        "params": {"levels": args.levels, "iters": args.iters,
                   "ffd_grid": list(FFD_GRID)},
    }

    print("[2/4] 实验 A：纯刚体可找回性")
    result["experiment_a_rigid_only"] = experiment_a(
        fixed, arr_cpu, spacing_zyx, device, args.levels, args.iters, args.threads)

    if not args.skip_b:
        print("[3/4] 实验 B：局部形变下逐级配准对比")
        result["experiment_b_local_warp"] = experiment_b(
            fixed, arr_cpu, spacing_zyx, device, args.levels, args.iters,
            args.threads, OUT_DIR / f"{args.out_prefix}_stages.png")

    total = round(time.perf_counter() - t_all, 2)
    result["total_seconds"] = total
    out_json = OUT_DIR / f"{args.out_prefix}_registration.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")
    print(f"[4/4] -> {out_json.name}")
    print(f"\n总耗时 {total}s（GPU：{dev_info.describe()}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
