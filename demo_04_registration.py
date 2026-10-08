#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M4：医学图像多级配准（刚性 -> 仿射 -> B 样条）。

配准的临床用途：随访 CT 对齐（比同一病灶两期的体积变化）、CT/MRI 融合。
本脚本用真实临床 CT 做两个实验，全部有真值可核，不靠"看着像"下结论。

实验 A（纯刚体，有真值）
    人为给原始 CT 施加已知刚体变换当"第二次扫描"，再用刚体配准找回来。
    评价：TRE（目标配准误差，mm）——配准估计出的变换和真值变换在同一批点上差多远，
    这是配准领域最硬的金标准指标。

实验 B（刚体 + 局部形变，用 Dice / NCC 评价）
    在刚体基础上再叠加一段已知 B 样条软组织形变（模拟呼吸/体位差异）。
    对比 未配准 / 仅刚性 / 刚性+仿射 / 三级全上 四档的 mask Dice 与图像 NCC，
    回答"配准为什么必须分三级"——刚性根本修不了局部形变。

工程上踩过的两个 ITK 坑（都写在注释里了）：
  1) ImageRegistrationMethod.Execute() 恒定返回 CompositeTransform，具体变换类型会丢；
  2) CompositeTransform 没实现 ComputeJacobianWithRespectToPosition，MI 指标算梯度时
     直接抛 "unimplemented for CompositeTransform"。
  合起来的结论：**不能把复合变换当作被优化的对象**。所以这里改成工业界常见做法——
  每级在上一级对齐后的图像上独立优化一个干净类型的变换，累积变换单独用 compose 维护。

输出：out/demo04_registration.json、out/registration_stages.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from demo_01_recon import (DICOM_DIR, OUT_DIR, load_dicom_series,
                           resample_isotropic, segment)

# 配准分辨率：1mm 全量约 7000 万体素，纯 CPU 跑不动三级配准；
# 2mm 约 680 万体素，精度足够评估刚体/仿射，也是工程上常见的做法。
REG_SPACING = 2.0
RANDOM_SEED = 20261008


# ------------------------------------------------------------ 工具
def to_np(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)          # (z, y, x)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice 系数，衡量两个 mask 的重合度。对位移比体积敏感得多。"""
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return float("nan")
    return float(2.0 * (a & b).sum() / s)


def ncc(a: np.ndarray, b: np.ndarray, thr: float = -900.0) -> float:
    """归一化互相关，配准里最常用的图像相似度指标之一。

    在"至少有一方是体内组织"的区域上算，避免大片体外空气把相关性稀释掉。
    """
    m = (a > thr) | (b > thr)
    if m.sum() < 100:
        return float("nan")
    x = a[m].astype(np.float64)
    y = b[m].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def centroid_physical(img: sitk.Image, thr: float = -900.0) -> np.ndarray:
    """体内组织的物理坐标质心（用于刚体配准初值）。"""
    a = sitk.GetArrayFromImage(img)
    idx = np.argwhere(a > thr)
    if idx.size == 0:
        c_index = [(s - 1) / 2.0 for s in img.GetSize()]
    else:
        c_index = idx.mean(axis=0)[::-1]          # (z,y,x) -> (x,y,z)
    return np.array(img.TransformContinuousIndexToPhysicalPoint(
        [float(v) for v in c_index]))


def _initial_euler(fixed: sitk.Image, moving: sitk.Image) -> sitk.Euler3DTransform:
    """手动构造刚体配准初值。

    刻意不用 sitk.CenteredTransformInitializer —— 它返回 CompositeTransform，
    类型不好接（见模块 docstring 的坑 1）。

    旋转中心刻意取**图像几何中心**，和 make_rigid 造真值时用的中心一致。
    因为同一个刚体运动在不同旋转中心下的 (R, t) 参数并不相同 ——
    如果这里用体素质心、真值用几何中心，那"平移估计 vs 真值平移"的对比就是误导，
    只有 TRE 才有意义。中心对齐了，参数本身才可以直接比。
    平移初值取两图质心之差（此时 R=I，y = x + t，正好把质心对齐）。
    """
    center = fixed.TransformContinuousIndexToPhysicalPoint(
        [(s - 1) / 2.0 for s in fixed.GetSize()])
    c_f = centroid_physical(fixed)
    c_m = centroid_physical(moving)
    tx = sitk.Euler3DTransform()
    tx.SetCenter(tuple(float(v) for v in center))
    tx.SetTranslation(tuple(float(v) for v in (c_m - c_f)))
    return tx


def make_rigid(img: sitk.Image, angles_deg, translation_mm) -> sitk.Euler3DTransform:
    """构造绕图像中心的刚体变换。"""
    tx = sitk.Euler3DTransform()
    tx.SetCenter(img.TransformContinuousIndexToPhysicalPoint(
        [(s - 1) / 2.0 for s in img.GetSize()]))
    ax, ay, az = (np.deg2rad(a) for a in angles_deg)
    tx.SetRotation(ax, ay, az)
    tx.SetTranslation(tuple(float(t) for t in translation_mm))
    return tx


def make_warp(img: sitk.Image, grid=(6, 6, 6), amp_mm=12.0, seed=RANDOM_SEED):
    """构造一段平滑的 B 样条形变，模拟软组织形变（呼吸/体位）。

    关键：ITK 的 B 样条参数是「先所有控制点的 x 分量、再所有 y、再所有 z」的顺序排列。
    直接对参数下标做正弦，会在空间上变成高频抖动，把结构揉碎。
    所以必须按控制点网格 (g+3)^3 生成三维低频位移场再展开。
    """
    bs = sitk.BSplineTransformInitializer(img, list(grid), order=3)
    total = len(bs.GetParameters())
    g = [int(v) + 3 for v in grid]
    ncp = g[0] * g[1] * g[2]
    if total != 3 * ncp:
        raise RuntimeError(f"B样条参数数不符: {total} != 3*{ncp}")

    X, Y, Z = np.meshgrid(np.linspace(0, 2 * np.pi, g[0]),
                          np.linspace(0, 2 * np.pi, g[1]),
                          np.linspace(0, 2 * np.pi, g[2]), indexing="ij")
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, size=3)
    dx = np.sin(X + ph[0]) * np.cos(Y + ph[1])
    dy = np.sin(Y + ph[0]) * np.cos(Z + ph[2])
    dz = np.sin(Z + ph[1]) * np.cos(X + ph[2])
    bs.SetParameters([float(v) for v in
                      amp_mm * np.concatenate([dx.ravel(), dy.ravel(), dz.ravel()])])
    return bs


def compose(first: sitk.Transform, second: sitk.Transform) -> sitk.CompositeTransform:
    """组合变换，语义明确：**先应用 first，再应用 second**。

    实测 SimpleITK 的 CompositeTransform([A, B]) 是「先 B 后 A」（队列反序施加），
    这里包一层避免每次靠记忆猜顺序；正确性用运行时探针断言保证。
    只用于结果重采样和评估，绝不用作被优化的对象（见模块 docstring）。
    """
    probe = (1.23, 4.56, 7.89)
    want = second.TransformPoint(first.TransformPoint(probe))
    for order in ([second, first], [first, second]):
        c = sitk.CompositeTransform(order)
        if float(np.linalg.norm(np.array(c.TransformPoint(probe))
                                - np.array(want))) < 1e-6:
            return c
    raise RuntimeError("CompositeTransform 施加顺序无法确定")


def apply_transform(moving: sitk.Image, reference: sitk.Image,
                    tx: sitk.Transform) -> sitk.Image:
    """把 moving 按 tx 重采样到 reference 的网格上。"""
    return sitk.Resample(moving, reference, tx, sitk.sitkLinear,
                         -1000.0, moving.GetPixelID())


# ------------------------------------------------------------ 配准器
def _registration(fixed, moving, shrink, sigmas, sample=0.10):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(sample, seed=RANDOM_SEED)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel(shrink)
    R.SetSmoothingSigmasPerLevel(sigmas)
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return R


def register_rigid(fixed, moving):
    """第一级：刚性（6 自由度）。先解决"整体挪了 / 转了"。"""
    t0 = time.perf_counter()
    R = _registration(fixed, moving, [4, 2, 1], [2.0, 1.0, 0.0])
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300,
        gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(_initial_euler(fixed, moving), inPlace=False)
    out = R.Execute(fixed, moving)
    return out, {
        "stage": "rigid", "dof": 6,
        "metric": round(float(R.GetMetricValue()), 6),
        "iterations": int(R.GetOptimizerIteration()),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def register_affine(fixed, moving):
    """第二级：仿射（12 自由度）。允许缩放 / 剪切，修整体形变与标定差异。

    传进来的 moving 应该已经被刚体级对齐过了（多级串联见 register_multistage）。
    """
    t0 = time.perf_counter()
    init = sitk.AffineTransform(3)
    init.SetCenter(tuple(float(v) for v in centroid_physical(fixed)))
    R = _registration(fixed, moving, [4, 2, 1], [2.0, 1.0, 0.0])
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-5, numberOfIterations=300,
        gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(init, inPlace=False)
    out = R.Execute(fixed, moving)
    return out, {
        "stage": "affine", "dof": 12,
        "metric": round(float(R.GetMetricValue()), 6),
        "iterations": int(R.GetOptimizerIteration()),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def register_bspline(fixed, moving, grid=(4, 4, 4)):
    """第三级：B 样条自由形变。修刚性/仿射都修不了的局部软组织形变。

    成本控制：B 样条有 3*(g+3)^3 个自由参数（g=4 时 1029 个），
    再套三级金字塔 + 10% 采样在 2mm 全分辨率上跑，纯 CPU 会跑到十分钟以上。
    所以这一级只用两级金字塔 [4,2]、4% 采样、函数评估上限 300 ——
    形变本身是低频平滑的，粗金字塔足够表达，实测精度不掉。
    """
    t0 = time.perf_counter()
    bs = sitk.BSplineTransformInitializer(fixed, list(grid), order=3)
    R = _registration(fixed, moving, [4, 2], [2.0, 1.0], sample=0.04)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-4,
                           numberOfIterations=60,
                           maximumNumberOfCorrections=5,
                           maximumNumberOfFunctionEvaluations=300,
                           costFunctionConvergenceFactor=1e6)
    R.SetInitialTransform(bs, inPlace=False)
    out = R.Execute(fixed, moving)
    return out, {
        "stage": "bspline", "dof": int(len(bs.GetParameters())),
        "grid": list(grid),
        "metric": round(float(R.GetMetricValue()), 6),
        "iterations": int(R.GetOptimizerIteration()),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def register_multistage(fixed: sitk.Image, moving: sitk.Image,
                        stages=("rigid", "affine", "bspline")):
    """多级串联配准。

    每级都在「上一级对齐后的图像」上独立优化一个干净类型的变换；
    累积变换单独用 compose 维护，最后拿它去重采样**原始** moving，
    这样评估时只有一次插值，不会把多次重采样的模糊算进精度里。

    返回 [(级名, 累积变换, 该级变换, 对齐图像, 该级信息), ...]
    """
    fns = {"rigid": register_rigid,
           "affine": register_affine,
           "bspline": register_bspline}
    out_list = []
    t_acc: sitk.Transform | None = None
    for name in stages:
        cur = moving if t_acc is None else apply_transform(moving, fixed, t_acc)
        tx, info = fns[name](fixed, cur)
        print(f"        · {name:<8} 完成：{info['iterations']:>4} 次迭代  "
              f"指标 {info['metric']:.4f}  {info['seconds']:>6.2f}s")
        # cur(x) = moving( t_acc(x) )，配准找到的 tx 把 cur 对到 fixed：
        # 总映射 = 先 tx 再 t_acc
        t_acc = tx if t_acc is None else compose(tx, t_acc)
        aligned = apply_transform(moving, fixed, t_acc)
        out_list.append((name, t_acc, tx, aligned, info))
    return out_list


# ------------------------------------------------------------ 评价
def tre_mm(est: sitk.Transform, truth: sitk.Transform, img: sitk.Image,
           n_axis: int = 5) -> dict:
    """TRE（目标配准误差）：在图像范围内铺一批点，比较两个变换把它们送到哪。

    只要有真值，TRE 就比任何相似度指标都硬。
    """
    size = img.GetSize()
    gx = np.linspace(0.15, 0.85, n_axis) * (size[0] - 1)
    gy = np.linspace(0.15, 0.85, n_axis) * (size[1] - 1)
    gz = np.linspace(0.15, 0.85, n_axis) * (size[2] - 1)
    pts = [img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
           for z in gz for y in gy for x in gx]
    errs = np.array([np.linalg.norm(np.array(est.TransformPoint(p))
                                    - np.array(truth.TransformPoint(p)))
                     for p in pts])
    return {
        "n_points": int(len(errs)),
        "tre_mean_mm": round(float(errs.mean()), 3),
        "tre_rms_mm": round(float(np.sqrt((errs ** 2).mean())), 3),
        "tre_max_mm": round(float(errs.max()), 3),
    }


def bspline_field_stats(tx: sitk.Transform, img: sitk.Image, n_axis: int = 9) -> dict:
    """B 样条形变场的位移统计。

    这一级学到的位移应该和真实局部形变量级相当；如果它学出一大坨位移，
    说明配准在过拟合噪声，这是必须能看出来的。
    """
    size = img.GetSize()
    gx = np.linspace(0.1, 0.9, n_axis) * (size[0] - 1)
    gy = np.linspace(0.1, 0.9, n_axis) * (size[1] - 1)
    gz = np.linspace(0.1, 0.9, n_axis) * (size[2] - 1)
    d = []
    for z in gz:
        for y in gy:
            for x in gx:
                p = img.TransformContinuousIndexToPhysicalPoint(
                    (float(x), float(y), float(z)))
                d.append(np.linalg.norm(np.array(tx.TransformPoint(p)) - np.array(p)))
    d = np.array(d)
    return {"n_points": int(len(d)),
            "disp_mean_mm": round(float(d.mean()), 3),
            "disp_max_mm": round(float(d.max()), 3)}


# ------------------------------------------------------------ 实验
def experiment_a(fixed: sitk.Image) -> dict:
    """实验 A：纯刚体变换，配准应该能把它精确找回来。"""
    print("\n[A] 纯刚体验证 —— 已知真值")
    angles, trans = (8.0, -5.0, 3.0), (10.0, -8.0, 5.0)
    T_true = make_rigid(fixed, angles, trans)
    # 用真值变换的逆生成 moving：这样配准要找回的正好就是 T_true 本身
    moving = apply_transform(fixed, fixed, T_true.GetInverse())
    print(f"      人为施加：旋转 {angles} 度、平移 {trans} mm")

    ref_bone = segment(to_np(fixed))["bone"]
    v_ref = float(ref_bone.sum()) * REG_SPACING ** 3 / 1000.0
    mov_bone = segment(to_np(moving))["bone"]
    d_before = dice(ref_bone, mov_bone)
    v_mov = float(mov_bone.sum()) * REG_SPACING ** 3 / 1000.0
    print(f"      配准前：Dice {d_before:.4f}   骨体积 {v_mov:.2f} mL"
          f"（参考 {v_ref:.2f} mL，偏差 {abs(v_mov - v_ref) / v_ref * 100:.2f}%）")

    rigid, info = register_rigid(fixed, moving)
    aligned = apply_transform(moving, fixed, rigid)
    ali_bone = segment(to_np(aligned))["bone"]
    d_after = dice(ref_bone, ali_bone)
    v_ali = float(ali_bone.sum()) * REG_SPACING ** 3 / 1000.0
    tre = tre_mm(rigid, T_true, fixed)

    est_rot = np.rad2deg(rigid.GetParameters()[:3])
    est_tr = rigid.GetParameters()[3:6]
    print(f"      配准后：Dice {d_after:.4f}   骨体积 {v_ali:.2f} mL，"
          f"偏差 {abs(v_ali - v_ref) / v_ref * 100:.2f}%")
    print(f"      TRE：均值 {tre['tre_mean_mm']} mm   最大 {tre['tre_max_mm']} mm")
    print(f"      旋转估计 {np.round(est_rot, 2)} 度（真值 {np.round(angles, 2)}）")
    print(f"      平移估计 {np.round(est_tr, 2)} mm（真值 {np.round(trans, 2)}）")

    return {
        "ground_truth": {"angles_deg": list(angles), "translation_mm": list(trans)},
        "before": {"dice_bone": round(d_before, 4), "bone_ml": round(v_mov, 2)},
        "after": {"dice_bone": round(d_after, 4), "bone_ml": round(v_ali, 2)},
        "reference": {"dice_bone": 1.0, "bone_ml": round(v_ref, 2)},
        "tre": tre,
        "estimated": {
            "angles_deg": [round(float(v), 3) for v in est_rot],
            "translation_mm": [round(float(v), 3) for v in est_tr],
            "param_error_deg": [round(float(a - b), 3) for a, b in zip(est_rot, angles)],
            "param_error_mm": [round(float(a - b), 3) for a, b in zip(est_tr, trans)],
        },
        "info": info,
    }


def experiment_b(fixed: sitk.Image, out_png: Path) -> dict:
    """实验 B：刚体 + 局部形变，逐级加码看 Dice 怎么涨。

    为什么不算 TRE：真值里含 B 样条，而含 B 样条的复合变换不可逆
    （SimpleITK 直接抛 "Unable to create inverse"），取不到真值变换就算不了 TRE。
    所以本实验只用**不依赖真值**的指标：mask Dice + 图像 NCC。
    纯刚体的 TRE 交给实验 A（那里真值是可逆的 Euler3DTransform）。
    """
    print("\n[B] 刚体 + 软组织形变 —— 对比三档配准")
    angles, trans = (6.0, -4.0, 2.0), (8.0, -6.0, 4.0)
    T_rigid_true = make_rigid(fixed, angles, trans)
    T_warp_true = make_warp(fixed, grid=(6, 6, 6), amp_mm=12.0)
    # 先整体刚体移动、再局部形变：moving(x) = fixed( T_warp( T_rigid(x) ) )
    # 单次重采样即可，全程不需要求逆
    P = compose(T_rigid_true, T_warp_true)
    moving = apply_transform(fixed, fixed, P)
    print(f"      人为施加：旋转 {angles} 度、平移 {trans} mm、B样条形变 幅值 12 mm")

    ref = to_np(fixed)
    ref_masks = segment(ref)

    def snapshot(tag: str, arr_stage: np.ndarray, seconds: float) -> dict:
        m = segment(arr_stage)
        return {
            "stage": tag,
            "dice_body": round(dice(ref_masks["body"], m["body"]), 4),
            "dice_lung": round(dice(ref_masks["lung"], m["lung"]), 4),
            "dice_bone": round(dice(ref_masks["bone"], m["bone"]), 4),
            "ncc": round(ncc(ref, arr_stage), 4),
            "seconds": seconds,
        }

    stages = [snapshot("未配准", to_np(moving), 0.0)]
    labels = {"rigid": "仅刚性", "affine": "刚性+仿射", "bspline": "三级全上(含B样条)"}
    infos, bspline_only_tx, last_aligned = [], None, None
    for name, _t_acc, tx, aligned, info in register_multistage(fixed, moving):
        stages.append(snapshot(labels[name], to_np(aligned), info["seconds"]))
        infos.append(info)
        if name == "bspline":
            bspline_only_tx, last_aligned = tx, aligned

    for s in stages:
        print(f"      {s['stage']:<20} Dice 体表 {s['dice_body']:.4f}   "
              f"肺 {s['dice_lung']:.4f}   骨 {s['dice_bone']:.4f}   "
              f"NCC {s['ncc']:.4f}   {s['seconds']:>6.2f}s")

    # B 样条那一级到底学了多少形变？（人为施加的真值幅值是 12mm）
    field = bspline_field_stats(bspline_only_tx, fixed)
    print(f"      B样条形变场：均值 {field['disp_mean_mm']} mm   "
          f"最大 {field['disp_max_mm']} mm   （真值幅值 12 mm）")

    render_stages(fixed, moving, last_aligned, out_png)

    return {
        "ground_truth": {"angles_deg": list(angles), "translation_mm": list(trans),
                         "warp_grid": [6, 6, 6], "warp_amp_mm": 12.0},
        "stages": stages,
        "bspline_field": field,
        "stage_info": infos,
        "note": "本实验不含 TRE：含 B 样条的复合变换不可逆，无法取得真值变换；"
                "只用不依赖真值的 Dice / NCC 评价。刚体 TRE 见实验 A。",
    }


def _use_cjk_font():
    """matplotlib 默认字体没有中文字形，不设的话图里中文全是方框。

    这是一份**跨平台**候选表，不是随便罗列：Windows 有微软雅黑、macOS 有苹方、
    Linux 服务器上什么都没有 —— 实测 AutoDL 的 Ubuntu 22.04 镜像里
    `fc-list | grep -i cjk` 返回 0 条，图里所有中文都渲染成空心方框，
    还刷了几百行 `Glyph xxx missing from current font` 警告。

    Linux 上装一个（约 5 MB）即可：
        apt-get install -y fonts-wqy-microhei

    字体名写错不会报错、只会静默退化成方框，所以最后一档留 `DejaVu Sans`
    （matplotlib 自带，至少拉丁字母和数字是好的），同时把候选按可用性筛选。
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
                  "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
                  "WenQuanYi Zen Hei", "Droid Sans Fallback",
                  "AR PL UMing CN", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    ordered = [c for c in candidates if c in available] or ["DejaVu Sans"]
    rcParams["font.sans-serif"] = ordered
    rcParams["axes.unicode_minus"] = False
    # 一个中文字体都没找到时明确提示，避免"静默出方框"
    if not any(c in available for c in candidates[:-1]):
        import warnings
        warnings.warn("未找到任何中文字体，图中的中文会显示为方框；"
                      "Linux 上可执行 apt-get install -y fonts-wqy-microhei")
    return ordered[0]


def render_stages(fixed, moving, aligned, out_png: Path, z: int | None = None):
    """配准前后的对照图：轴位中间层 + 冠状 MIP。"""
    _use_cjk_font()
    import matplotlib.pyplot as plt

    def win(a):
        return np.clip((a + 200.0) / 600.0, 0, 1)

    a_f, a_m, a_a = to_np(fixed), to_np(moving), to_np(aligned)
    zi = a_f.shape[0] // 2 if z is None else int(z)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), dpi=95)
    for ax, vol, name in [(axes[0, 0], a_f, "固定图像 (参考)"),
                          (axes[0, 1], a_m, "移动图像 (未配准)"),
                          (axes[0, 2], a_a, "移动图像 (配准后)")]:
        ax.imshow(win(vol[zi]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    for ax, vol, name in [(axes[1, 0], win(a_f), "固定 冠状位"),
                          (axes[1, 1], win(a_m), "未配准 冠状位"),
                          (axes[1, 2], win(a_a), "配准后 冠状位")]:
        ax.imshow(vol.max(axis=2), cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.set_title(name, fontsize=11)
        ax.axis("off")

    fig.suptitle("M4 多级配准：配准前 vs 配准后（真实临床 CT）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> {out_png.name}")


def main(argv=None) -> int:
    global REG_SPACING

    p = argparse.ArgumentParser(description="MedRecon3D M4 多级配准")
    p.add_argument("--dicom-dir", default=str(DICOM_DIR))
    p.add_argument("--out-prefix", default="demo04")
    p.add_argument("--reg-spacing", type=float, default=REG_SPACING)
    args = p.parse_args(argv)

    REG_SPACING = args.reg_spacing
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    print(f"[1/3] 读取 CT 并重采样到 {REG_SPACING} mm 各向同性（配准分辨率）")
    img, meta = load_dicom_series(Path(args.dicom_dir))
    fixed = resample_isotropic(img, REG_SPACING)
    print(f"      原始 {meta['size_xyz']} @ {meta['spacing_xyz']} mm  ->  "
          f"{fixed.GetSize()} @ {tuple(round(v, 2) for v in fixed.GetSpacing())} mm")

    result = {"source": str(args.dicom_dir), "reg_spacing": REG_SPACING,
              "dicom_meta": meta}

    print("[2/3] 实验 A：纯刚体可找回性")
    result["experiment_a_rigid_only"] = experiment_a(fixed)

    print("[3/3] 实验 B：局部形变下逐级配准对比（含渲染对照图）")
    result["experiment_b_local_warp"] = experiment_b(
        fixed, OUT_DIR / f"{args.out_prefix}_stages.png")

    total = round(time.perf_counter() - t_all, 2)
    result["total_seconds"] = total
    out_json = OUT_DIR / f"{args.out_prefix}_registration.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"      -> {out_json.name}")
    print(f"\n总耗时 {total}s（纯 CPU）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
