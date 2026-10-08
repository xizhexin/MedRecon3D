#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GPU 自检：证明 GPU 实现和 CPU 参考实现在数值上是一致的。

为什么必须写这个
----------------
"换了个后端、结果差不多"这种话在工程上是不成立的 —— 差多少、差在哪，
必须是可测量的数。没有这个自检，GPU 版跑出来的体积、TRE 就没法采信。

跑法（在 MedRecon3D/ 目录下）：
    python -m gpu.selftest

三项检查：
  1. 重采样 —— GPU `grid_sample` vs SimpleITK `ResampleImageFilter`
  2. 网格体积 —— GPU 散度定理 vs trimesh
  3. 刚体配准 —— 施加已知刚体运动，看 GPU 配准能不能找回来（TRE）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gpu.backend import probe, TORCH_AVAILABLE                      # noqa: E402
from gpu import ops, registration as reg                             # noqa: E402
from volume_grid import PAD_HU, isotropic_size                       # noqa: E402

PASS, FAIL = "  [PASS]", "  [FAIL]"
_results: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str):
    _results.append((name, ok, detail))
    print(f"{PASS if ok else FAIL} {name}: {detail}")


# ------------------------------------------------------------------ 1
def check_resample(device):
    """GPU 重采样与 SimpleITK 的差异。

    两边都按「输出索引 → 物理坐标 → 回查输入索引」的语义做，网格尺寸来自
    同一个 `volume_grid.isotropic_size()`，补值也都是 −1000 HU（空气），
    所以理论误差只剩浮点舍入。阈值 2.0 HU 是给 grid_sample 留的余量 ——
    它在归一化坐标上做 float32 运算，450³ 下会有几个 ulp 的偏移。

    ★ 这条检查**曾经 Fail 过**，而且不是精度问题：修复前最大误差 379 HU
      （动态范围的 26.8%），且全部集中在最后两层。根因是网格尺寸用了
      `round(size·old_sp/new_sp)`，让最后一层落到输入体素中心范围之外，
      而两边对超界采样的处理本来就不同（ITK 直接返回 defaultPixelValue，
      grid_sample 会拿 0 去做线性插值）。详见 `volume_grid.py` 头部。
      修完误差降到 1e-4 HU 量级 —— 所以这个阈值同时就是这个 bug 的回归测试。
    """
    import SimpleITK as sitk

    print("\n[1] 重采样：GPU grid_sample vs SimpleITK ResampleImageFilter")
    rng = np.random.default_rng(0)
    shape = (40, 64, 64)                     # (z, y, x)
    src_spacing = (2.5, 0.879, 0.879)
    # 造一段平滑数据（真实 CT 也是平滑的，纯随机噪声会放大插值差异）
    zz, yy, xx = np.meshgrid(np.linspace(0, 3, shape[0]),
                             np.linspace(0, 4, shape[1]),
                             np.linspace(0, 4, shape[2]), indexing="ij")
    arr = (np.sin(zz) * 400 + np.cos(yy) * 300 + np.sin(xx * 2) * 200 - 500
           + rng.normal(0, 5, shape)).astype(np.float32)

    target = 1.0

    # --- CPU 参考 ---
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((src_spacing[2], src_spacing[1], src_spacing[0]))   # sitk 是 (x,y,z)
    old_size = np.array(img.GetSize(), dtype=float)
    old_sp = np.array(img.GetSpacing(), dtype=float)
    new_size = isotropic_size(old_size, old_sp, target)
    ref = sitk.Resample(img, [int(v) for v in new_size], sitk.Transform(),
                        sitk.sitkLinear, img.GetOrigin(), [target] * 3,
                        img.GetDirection(), float(PAD_HU), img.GetPixelID())
    ref_arr = sitk.GetArrayFromImage(ref)

    # --- GPU ---
    gpu_arr, gpu_sp, gpu_shape = ops.resample_volume_gpu(
        arr, src_spacing, target, device)

    if gpu_arr.shape != ref_arr.shape:
        _record("重采样", False,
                f"形状不一致 GPU{gpu_arr.shape} vs CPU{ref_arr.shape}")
        return

    diff = np.abs(gpu_arr - ref_arr)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    # 相对误差以数据动态范围为单位
    rng_span = float(arr.max() - arr.min())
    rel = max_err / rng_span * 100 if rng_span else 0.0
    ok = max_err < 2.0
    _record("重采样", ok,
            f"shape {gpu_arr.shape} · 最大误差 {max_err:.6f} HU "
            f"(动态范围的 {rel:.6f}%) · 平均 {mean_err:.8f} HU")


# ------------------------------------------------------------------ 2
# ------------------------------------------------------------------ 0
def check_grid_rule():
    """输出网格规则的**独立**回归测试。

    为什么不能只靠 [1] 重采样检查：两边的参考实现都调用同一个
    `volume_grid.isotropic_size()`，万一这个规则本身被改错了，[1] 会一起错、
    照样通过。这里把期望值**硬编码**，用几何推导而不是共享函数来判断。

    真实数据的实际参数：107 层、层距 2.5 mm、目标 1 mm、面内 512×0.878906 mm。
    """
    print("\n[0] 网格规则：输出采样点必须落在输入体素中心张成的范围内")
    cases = [
        # (名字, size, old_spacing, target, 期望输出层数, 旧 round() 会给出多少)
        ("PCIR z 轴", 107, 2.5, 1.0, 266, 268),
        ("PCIR 面内", 512, 0.878906, 1.0, 450, 450),
        ("小样例 z", 40, 2.5, 1.0, 98, 100),
    ]
    ok_all = True
    for name, size, old_sp, target, want, old in cases:
        got = int(isotropic_size([size], [old_sp], [target])[0])
        # 独立判据：最后一个输出体素对应的输入索引必须 <= size-1
        last_src_idx = (got - 1) * target / old_sp
        inside = last_src_idx <= size - 1 + 1e-9
        ok = (got == want) and inside
        ok_all = ok_all and ok
        print(f"    {'✓' if ok else '✗'} {name:<10} {size}层×{old_sp}mm → {target}mm : "
              f"输出 {got} 层（期望 {want}，旧 round() 给 {old}）· "
              f"末层回查索引 {last_src_idx:.4f} ≤ {size - 1} {'成立' if inside else '超界!'}")
    _record("网格规则", ok_all, "末层不再超界；旧 round() 规则在 PCIR 上会多出 2 层垃圾切片")


def check_mesh_volume(device):
    """GPU 散度定理 vs trimesh —— 同一条网格上的两条独立实现。"""
    print("\n[2] 网格体积：GPU 散度定理 vs trimesh")
    try:
        import trimesh
    except Exception as exc:
        _record("网格体积", False, f"trimesh 不可用: {exc}")
        return

    # 用球体：解析体积已知，顺便验证绝对值
    mesh = trimesh.creation.icosphere(subdivisions=5, radius=30.0)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)

    v_cpu = float(abs(mesh.volume))
    v_gpu = ops.mesh_volume_gpu(v, f, device)
    v_true = 4.0 / 3.0 * np.pi * 30.0 ** 3

    ok = abs(v_gpu - v_cpu) / v_cpu < 1e-6
    _record("网格体积", ok,
            f"CPU {v_cpu:.3f} mm³ · GPU {v_gpu:.3f} mm³ · "
            f"解析值 {v_true:.1f} · 两者相对差 "
            f"{abs(v_gpu - v_cpu) / v_cpu * 100:.3e}% · "
            f"对解析值偏差 {abs(v_gpu - v_true) / v_true * 100:.3f}%（多面体离散误差）")


# ------------------------------------------------------------------ 3
def check_registration(device, quick=True):
    """施加已知刚体运动，看 GPU 配准能否找回 —— 用 TRE 量化。

    这是本项目配准部分最硬的一项证据：不依赖"看着对齐了"，
    而是比较「估计变换」和「真值变换」在同一批点上把点送到哪。
    """
    print("\n[3] 刚体配准：已知真值下的 TRE")
    shape = (48, 64, 64)
    spacing = (2.0, 2.0, 2.0)
    zz, yy, xx = np.meshgrid(np.arange(shape[0]) * spacing[0],
                             np.arange(shape[1]) * spacing[1],
                             np.arange(shape[2]) * spacing[2], indexing="ij")
    # 造一个带内部结构的体模（球 + 柱），光有球体的话旋转是简并的
    body = ((zz - 48) ** 2 + (yy - 64) ** 2 + (xx - 64) ** 2) < 40 ** 2
    inner = (((zz - 48) / 1.0) ** 2 + ((yy - 64) / 2.5) ** 2) < 25 ** 2
    bone = (((zz - 60) / 1.0) ** 2 + ((xx - 64) / 1.5) ** 2) < 18 ** 2
    arr = np.full(shape, -1000.0, dtype=np.float32)
    arr[body] = 40.0
    arr[inner & body] = -800.0
    arr[bone & body] = 400.0

    angles = (6.0, -4.0, 2.5)
    trans = (6.0, -5.0, 3.0)
    center = np.array([(shape[2] - 1) / 2 * spacing[2],
                       (shape[1] - 1) / 2 * spacing[1],
                       (shape[0] - 1) / 2 * spacing[0]], dtype=np.float32)

    # 真值矩阵（绕几何中心）—— 和 gpu.registration.rigid_matrix 同一套约定
    import torch
    with torch.no_grad():
        m_true = reg.rigid_matrix(
            [np.deg2rad(a) for a in angles], trans, center, device).detach().cpu().numpy()

    # 生成 moving：把 moving 的每个体素定义成 fixed 是真值变换的逆
    from numpy.linalg import inv
    m_inv = inv(m_true)
    ctx = reg.make_context(shape, spacing, arr, spacing, device)
    with torch.no_grad():
        m_t = torch.as_tensor(m_inv, dtype=torch.float32, device=device)
        moving = ctx.warp_matrix(m_t)[0, 0].cpu().numpy()

    n_iter = 40 if quick else 120
    t0 = time.perf_counter()
    result, info = reg.register_rigid_gpu(arr, moving, spacing, device,
                                          levels=3, iters=n_iter)
    secs = time.perf_counter() - t0

    # TRE：在图像范围内铺一批点
    gx = np.linspace(0.2, 0.8, 4) * (shape[2] - 1) * spacing[2]
    gy = np.linspace(0.2, 0.8, 4) * (shape[1] - 1) * spacing[1]
    gz = np.linspace(0.2, 0.8, 4) * (shape[0] - 1) * spacing[0]
    pts = np.array([[x, y, z] for z in gz for y in gy for x in gx], dtype=np.float32)

    est = reg.transform_points(result, pts, shape, spacing, device)
    truth = (m_true[:3, :3] @ pts.T).T + m_true[:3, 3]
    err = np.linalg.norm(est - truth, axis=1)

    ok = float(err.max()) < 3.0
    _record("刚体配准 TRE", ok,
            f"均值 {err.mean():.3f} mm · RMS {np.sqrt((err ** 2).mean()):.3f} mm · "
            f"最大 {err.max():.3f} mm · 迭代 {info.iterations} 次 · {secs:.2f}s")

    m_est = result["m4"].cpu().numpy()
    t_err = float(np.linalg.norm(m_est[:3, 3] - m_true[:3, 3]))

    # 旋转角误差：从两个矩阵各自反解 Rz·Ry·Rx 欧拉角再比 —— 比逐元素比矩阵直观
    def _angles_deg(m):
        r = m[:3, :3]
        ry = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
        rx = np.arctan2(r[2, 1], r[2, 2])
        rz = np.arctan2(r[1, 0], r[0, 0])
        return np.rad2deg([rx, ry, rz])

    a_est, a_true = _angles_deg(m_est), _angles_deg(m_true)
    ang_err = float(np.abs(a_est - a_true).max())
    _record("旋转角误差", ang_err < 0.5,
            f"最大 {ang_err:.4f}° · 估计 {np.round(a_est, 3).tolist()} vs "
            f"真值 {np.round(a_true, 3).tolist()}")
    _record("平移参数误差", t_err < 2.0,
            f"{t_err:.3f} mm · 估计 {np.round(m_est[:3, 3], 3).tolist()} vs "
            f"真值 {np.round(m_true[:3, 3], 3).tolist()}")
    return ok


def main() -> int:
    print("=" * 66)
    print("MedRecon3D GPU 自检")
    print("=" * 66)

    info = probe()
    print(f"\n环境：{info.describe()}")
    if info.arch_list:
        print(f"torch 编译架构：{', '.join(info.arch_list)}")
    if info.backend != "cuda":
        print(f"\n! 无可用 CUDA 设备：{info.note}")
        print("  自检需要 GPU。只想跑 CPU 管线的话直接用 demo_01_recon.py。")
        return 2

    import torch
    device = torch.device("cuda")

    check_grid_rule()
    check_resample(device)
    check_mesh_volume(device)
    check_registration(device)

    print("\n" + "=" * 66)
    n_ok = sum(1 for _, ok, _ in _results if ok)
    print(f"结果：{n_ok}/{len(_results)} 项通过")
    for name, ok, detail in _results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print("=" * 66)
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
