# -*- coding: utf-8 -*-
"""多级配准的 GPU 实现（刚性 → 仿射 → 自由形变）。

为什么值得重写
--------------
配准是整条管线里唯一"算得越久、结果越好"的环节：CPU 版用 SimpleITK 的
Mattes MI + RegularStepGradientDescent，2 mm 分辨率三级跑完要 86 秒，
而且为了把时间压住，B 样条那一级被迫砍到两级金字塔 + 4% 采样 + 300 次
函数评估上限 —— **精度是被时间逼着妥协的**。

搬到 GPU 之后，整幅图的 warp 从"重采样一次几十毫秒"变成"一次 kernel"，
梯度由 autograd 自动求（不用像 ITK 那样担心某个变换类型没实现 Jacobian），
于是可以拿满分辨率、多跑迭代。

架构沿用 CPU 版的核心决策
------------------------
**每一级在上一级对齐后的图像上独立优化一个干净类型的变换，累积变换单独维护。**

这条不是"照抄"，而是同一份约束推出来的结论：ITK 里 `CompositeTransform`
不能当优化对象（没实现 Jacobian）。到了 PyTorch 里，这个限制**不复存在**
（autograd 会自己穿过去），但我们**依然保留逐级独立优化**，因为：
1. 参数可解释 —— 刚体级学到的 6 个数就是"整体挪了多少"，FFD 级学到的是"局部形变"，
   混在一个复合变换里就说不清了；
2. 每级有自己的物理含义和收敛尺度，混在一起调学习率是灾难。

和 CPU 版的差异（写在明处，免得对不上数）
----------------------------------------
- **损失函数**：CPU 版用 Mattes 互信息；GPU 版默认用 NCC（归一化互相关）。
  NCC 对同模态配准（CT-CT）足够，且不需要直方图近似，梯度更干净。
  代码里保留了 `mi_loss` 作为可选，做异模态（CT-MRI）时应该换成它。
- **自由形变参数化**：CPU 版是 ITK 的三次 B 样条；GPU 版用
  **低分辨率控制点 + 三线性上采样**（C0 连续的 FFD）。两者都能表达低频软组织
  形变，但**不是同一种参数化**，所以两者的变换参数不可直接比较，
  可比的是同一套评价指标（TRE / Dice / NCC）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import TORCH_AVAILABLE

if TORCH_AVAILABLE:  # pragma: no cover
    import torch
    import torch.nn.functional as F

# HU 偏置：把 moving 抬高 1000，使"体外空气"≈0，
# 这样 grid_sample 的 zeros padding 在语义上正好等于"体外是空气"，
# 而不是"体外是 0 HU 的假组织"。NCC 对整体线性偏移不敏感，不影响损失。
HU_BIAS = 1000.0

# 平移参数的学习率放大倍数。
#
# Adam 的单步位移量约等于 lr（与梯度大小无关，这是它 scale-invariant 的性质），
# 所以 lr 实际上规定了"每个参数每步能走多远"。而两类参数的物理量纲差 100 倍：
#   · 欧拉角是弧度，真实旋转量级 ~0.1 rad（6°）
#   · 平移是 mm，真实位移量级 ~10 mm
# 用同一个 lr 必然二选一跛脚：按弧度定，平移要走 10 mm 得几百步；按 mm 定，
# 旋转一步就转过头。实测按 3e-2 单 lr 跑时，40 步只挪了 1.2 mm，配准完全不收敛。
TRANS_LR_SCALE = 20.0


# ============================================================ 基础构件
def build_physical_grid(shape_zyx, spacing_zyx, device) -> "torch.Tensor":
    """构造 fixed 网格上每个体素的物理坐标，(Z, Y, X, 3)，最后一维是 (x, y, z) mm。

    约定：origin=(0,0,0)、direction=单位阵 —— 本项目所有数据都满足
    （SimpleITK Resample 会保持输入 origin，而 PCIR 数据的 origin 就是 0）。
    如果以后要接 origin 非零的数据，在这里加一个平移即可，其余代码不用动。
    """
    z, y, x = (int(v) for v in shape_zyx)
    iz = torch.arange(z, device=device, dtype=torch.float32) * float(spacing_zyx[0])
    iy = torch.arange(y, device=device, dtype=torch.float32) * float(spacing_zyx[1])
    ix = torch.arange(x, device=device, dtype=torch.float32) * float(spacing_zyx[2])
    gz, gy, gx = torch.meshgrid(iz, iy, ix, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1)


def _coerce_scalar(value, device):
    """标量或张量 → 张量。**已是张量就原样返回**。

    这一步不是可选的：`torch.as_tensor(t)` 对张量会 detach，如果无脑转换，
    `_RigidParams.matrix()` 传进来的 `self.rot[i]` 会被切断梯度图，
    autograd 就再也传不到旋转参数上了（表现为 loss 不降、配准不收敛）。
    """
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(float(value), dtype=torch.float32, device=device)


def _coerce_vec3(value, device):
    """3 元向量 → 张量。同样保住张量入参的梯度。"""
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(np.asarray(value, dtype=np.float32).reshape(3),
                           device=device)


def _euler_matrix(rx, ry, rz) -> "torch.Tensor":
    """R = Rz · Ry · Rx（内旋 x→y→z）的展开式，是可微的。

    单独抽出来是为了让 `rigid_matrix`（矩阵正解）和 `_euler_from_matrix`
    （参数反解）**共用同一份定义**，两者严格互逆。
    """
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    row_x = torch.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx])
    row_y = torch.stack([sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx])
    row_z = torch.stack([-sy, cy * sx, cy * cx])
    return torch.stack([row_x, row_y, row_z])                    # (3,3)


def _euler_from_matrix(r) -> tuple:
    """Rz·Ry·Rx 的反解 —— `_euler_matrix` 的严格逆。

    用来把"上一级已经优化出来的矩阵"重新变成可训练的三个角度。
    只在 |ry| < 90° 时唯一（万向节锁），本项目的旋转量级远小于此。
    """
    ry = torch.asin(torch.clamp(-r[2, 0], -1.0, 1.0))
    rx = torch.atan2(r[2, 1], r[2, 2])
    rz = torch.atan2(r[1, 0], r[0, 0])
    return rx, ry, rz


def rigid_matrix(angles_rad, translation_mm, center_mm, device) -> "torch.Tensor":
    """绕 center 旋转 + 平移的 4×4 齐次矩阵。

    平移项写成 `c - R·c + t`（先把旋转中心搬到原点、旋转、再搬回去，最后叠加平移），
    所以矩阵里第 4 列**并不等于** t。反解时必须减去枢轴项 `c - R·c`，
    直接把第 4 列当成 t 用是个很容易犯的错（踩过：金字塔换级时初值凭空跳 20 mm）。

    矩阵本身是可微的：angles / translation 用 requires_grad 传进来，
    autograd 就能一路传到变换参数。同时接受普通 float / numpy 标量
    （自检里构造真值矩阵用）—— 靠 `_coerce_scalar` 保证张量入参的梯度不被切断。
    """
    rx, ry, rz = (_coerce_scalar(v, device) for v in angles_rad)
    r = _euler_matrix(rx, ry, rz)
    c = _coerce_vec3(center_mm, device)
    t = _coerce_vec3(translation_mm, device)
    m = torch.eye(4, dtype=torch.float32, device=device).clone()
    m[:3, :3] = r
    m[:3, 3] = c - r @ c + t                                     # 绕 c 旋转再平移
    return m


def compose_matrix(first: "torch.Tensor", second: "torch.Tensor") -> "torch.Tensor":
    """组合变换，语义：**先应用 first，再应用 second**（对齐 CPU 版 compose 的语义）。"""
    return second @ first


@dataclass
class WarpContext:
    """一次 warp 需要的全部上下文。"""

    moving: "torch.Tensor"           # (1, 1, Z, Y, X)，已加 HU_BIAS
    moving_shape: tuple
    moving_spacing: tuple
    fixed_grid: "torch.Tensor"       # (Z, Y, X, 3) 物理坐标
    device: object

    def _sample(self, src_phys: "torch.Tensor") -> "torch.Tensor":
        """按物理坐标从 moving 采样，返回 (1,1,Z,Y,X)（已减回 HU_BIAS）。"""
        ms, msh = self.moving_spacing, self.moving_shape
        idx_x = src_phys[..., 0] / float(ms[2])
        idx_y = src_phys[..., 1] / float(ms[1])
        idx_z = src_phys[..., 2] / float(ms[0])
        nx = 2.0 * idx_x / max(int(msh[2]) - 1, 1) - 1.0
        ny = 2.0 * idx_y / max(int(msh[1]) - 1, 1) - 1.0
        nz = 2.0 * idx_z / max(int(msh[0]) - 1, 1) - 1.0
        grid = torch.stack([nx, ny, nz], dim=-1)[None]           # (1,Z,Y,X,3)
        out = F.grid_sample(self.moving, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        return out - HU_BIAS

    def warp_matrix(self, m4: "torch.Tensor") -> "torch.Tensor":
        """仿射/刚性：src_phys = M · fixed_phys。"""
        p = self.fixed_grid
        src = (m4[:3, :3] @ p.reshape(-1, 3).T).T + m4[:3, 3]
        return self._sample(src.reshape(p.shape))

    def warp_field(self, disp: "torch.Tensor") -> "torch.Tensor":
        """自由形变：src_phys = fixed_phys + disp(fixed_phys)。

        disp 与 fixed_grid 同形 (Z,Y,X,3)，单位 mm。
        """
        return self._sample(self.fixed_grid + disp)

    def warp_matrix_field(self, m4: "torch.Tensor", disp: "torch.Tensor") -> "torch.Tensor":
        """矩阵变换 + 自由形变的复合：src_phys = M · (fixed_phys + disp(fixed_phys))。

        顺序不能颠倒，这点很容易搞错：位移场是在**已经用矩阵对齐过的图像**上
        优化出来的，也就是 disp 定义在 fixed 坐标系里，所以必须**先加位移、再套矩阵**。
        写成 M(p) + disp(p) 是错的 —— 那等于把位移加在了 moving 的坐标系上。

        存在的意义是**只插值一次**：否则 FFD 级要在"已对齐的图像"上再重采样一次，
        整条链跑两遍线性插值，细节被磨两遍。
        """
        p = self.fixed_grid + disp
        src = (m4[:3, :3] @ p.reshape(-1, 3).T).T + m4[:3, 3]
        return self._sample(src.reshape(self.fixed_grid.shape))


def make_context(fixed_shape, fixed_spacing, moving_arr, moving_spacing,
                 device, bias: float = HU_BIAS) -> WarpContext:
    t = torch.as_tensor(np.ascontiguousarray(moving_arr, dtype=np.float32),
                        device=device)[None, None] + bias
    return WarpContext(
        moving=t,
        moving_shape=tuple(int(v) for v in moving_arr.shape),
        moving_spacing=tuple(float(v) for v in moving_spacing),
        fixed_grid=build_physical_grid(fixed_shape, fixed_spacing, device),
        device=device,
    )


# ============================================================ 损失
def ncc_loss(a: "torch.Tensor", b: "torch.Tensor", eps: float = 1e-8):
    """1 - 归一化互相关。a 是配准后图像，b 是参考图像。

    在**体内**区域上算（HU > -900 的并集），否则大片体外空气会把相关性稀释掉
    —— 这一点和 CPU 版的 `ncc()` 保持一致，否则两者数字没法比。
    """
    m = (a > -900.0) | (b > -900.0)
    if int(m.sum()) < 100:
        return torch.tensor(float("nan"), device=a.device)
    x = a[m] - a[m].mean()
    y = b[m] - b[m].mean()
    den = torch.sqrt((x ** 2).sum() * (y ** 2).sum()) + eps
    return 1.0 - (x * y).sum() / den


def mi_loss(a: "torch.Tensor", b: "torch.Tensor", bins: int = 32,
            lo: float = -200.0, hi: float = 400.0, eps: float = 1e-8):
    """负的 Mattes 互信息（soft histogram 版）。

    异模态配准（CT-MRI/PET）用的。同模态下 NCC 通常更好收敛，所以默认不用它。
    直方图用高斯核软化，保证对强度可微。
    """
    m = (a > -900.0) | (b > -900.0)
    x = a[m].reshape(-1)
    y = b[m].reshape(-1)
    if x.numel() < 100:
        return torch.tensor(float("nan"), device=a.device)

    step = (hi - lo) / bins
    centers = torch.linspace(lo + step / 2, hi - step / 2, bins, device=a.device)
    sigma = step

    def soft_hist(v):
        d = (v[:, None] - centers[None, :]) / sigma
        w = torch.exp(-0.5 * d * d)
        return w / (w.sum(dim=0, keepdim=True) + eps)             # (N, bins) 归一化列

    wa = soft_hist(x)
    wb = soft_hist(y)
    pab = (wa.T @ wb) / x.numel()
    pa = pab.sum(dim=1, keepdim=True)
    pb = pab.sum(dim=0, keepdim=True)
    pab_n = pab / (pab.sum() + eps)
    mi = (pab_n * torch.log((pab_n + eps) / (pa * pb + eps))).sum()
    return -mi


# ============================================================ 各级配准
@dataclass
class StageInfo:
    name: str
    dof: int
    iterations: int
    metric: float
    seconds: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"stage": self.name, "dof": self.dof, "iterations": self.iterations,
             "metric": round(self.metric, 6), "seconds": round(self.seconds, 2)}
        d.update(self.extra)
        return d


def _pyramid(arr: np.ndarray, spacing_zyx, factor: int):
    """按整数倍下采样，返回 (下采样数组, 新 spacing)。"""
    if factor <= 1:
        return arr, tuple(float(v) for v in spacing_zyx)
    return arr, tuple(float(v) * factor for v in spacing_zyx)


def _downsample_np(arr: np.ndarray, factor: int, device) -> "torch.Tensor":
    """用 GPU 做下采样，再取回 numpy（下采样本身也是体素运算，没必要留给 CPU）。"""
    if factor <= 1:
        return arr
    t = torch.as_tensor(np.ascontiguousarray(arr, dtype=np.float32), device=device)
    out = F.interpolate(t[None, None], scale_factor=1.0 / factor, mode="trilinear",
                        align_corners=False)
    return out[0, 0].cpu().numpy()


def _optimize(ctx: WarpContext, fixed_t, params_fn, n_iter, lr, loss_fn,
              fixed_t6=None, verbose=False):
    """通用优化循环。

    params_fn(params_dict) -> 对齐后的图像张量。把"怎么用参数去 warp"抽出来，
    刚性 / 仿射 / FFD 三种参数化就都能复用同一个循环。
    """
    opt = torch.optim.Adam(params_fn.param_groups(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_iter, 1))
    best = (float("inf"), None)
    last = float("inf")
    for i in range(n_iter):
        opt.zero_grad(set_to_none=True)
        warped = params_fn()
        loss = loss_fn(warped, fixed_t)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        sched.step()
        last = float(loss.detach())
        if last < best[0]:
            best = (last, params_fn.snapshot())
        if verbose and (i + 1) % 25 == 0:
            print(f"          iter {i + 1:>3}/{n_iter}  loss {last:.6f}")
    params_fn.restore(best[1])
    return best[0], n_iter


class _RigidParams:
    """刚性 6 参数的优化句柄（绕 center 旋转 + 平移）。

    参数直接用**欧拉角**（而不是旋转向量）。原因：`transform_points` 和
    参数报告都要和 `rigid_matrix` 的展开式对上，而旋转向量→欧拉角是有损的
    （小角度可以近似，但这个近似会污染"旋转角误差"这个指标）。
    这里用 R = Rz·Ry·Rx 的反解式，和 `rigid_matrix` 的行定义严格互逆。

    旋转中心必须**由调用方传进来**（`center_mm`），不能自己从当前 ctx 推。
    原因是金字塔：每一级的体素网格不同，"网格的物理中心"也就不同，
    而同一个刚体运动在不同旋转中心下的 (R, t) 是不同的。若每级各推各的中心，
    上一级优化出的矩阵拿到下一级当初值时会被解释成**另一个变换**
    （实测：换级后 loss 不降反升，TRE 停在 56 mm 量级）。
    """

    def __init__(self, ctx: WarpContext, init_m4, device, center_mm=None):
        if center_mm is None:
            self.center = ctx.fixed_grid.reshape(-1, 3).mean(dim=0)
        else:
            self.center = _coerce_vec3(center_mm, device)
        with torch.no_grad():
            r = init_m4[:3, :3].detach()
            rx, ry, rz = _euler_from_matrix(r)
            # 反解平移： m[:3,3] = c - R·c + t  →  t = m[:3,3] - (c - R·c)
            # R 用的是**参数化能表示的那个旋转**（由反解出的角度重建），
            # 而不是原矩阵的 [:3,:3] —— 两者数值上等价，但这样才能保证
            # 「重建矩阵 == 原矩阵」，否则换级时会引入微小但持续累积的偏移。
            r_rebuilt = _euler_matrix(rx, ry, rz)
            c = self.center.to(r_rebuilt.dtype)
            t = init_m4[:3, 3].detach() - (c - r_rebuilt @ c)
        self.rot = torch.nn.Parameter(torch.stack([rx, ry, rz]))
        self.tr = torch.nn.Parameter(t.clone())
        self.ctx = ctx

    def _angles(self):
        return self.rot[0], self.rot[1], self.rot[2]

    def matrix(self):
        return rigid_matrix(self._angles(), self.tr, self.center, self.ctx.device)

    def __call__(self):
        return self.ctx.warp_matrix(self.matrix())

    def parameters(self):
        return [self.rot, self.tr]

    def param_groups(self, lr):
        """旋转用 lr，平移用 lr × TRANS_LR_SCALE —— 量纲差 100 倍，理由见模块头部。"""
        return [{"params": [self.rot], "lr": lr},
                {"params": [self.tr], "lr": lr * TRANS_LR_SCALE}]

    def snapshot(self):
        return (self.rot.detach().clone(), self.tr.detach().clone())

    def restore(self, snap):
        with torch.no_grad():
            self.rot.copy_(snap[0])
            self.tr.copy_(snap[1])


class _AffineParams:
    """仿射 12 参数的优化句柄：直接优化 3×3 线性部分 + 3 维平移。

    不做"绕中心旋转"的分解 —— 仿射本来就没有明确的旋转中心，
    硬套一个中心只会让参数之间产生冗余的耦合，优化更难收敛。
    """

    def __init__(self, ctx: WarpContext, init_m4, device):
        self.mat = torch.nn.Parameter(init_m4[:3, :3].detach().clone())
        self.tr = torch.nn.Parameter(init_m4[:3, 3].detach().clone())
        self.ctx = ctx

    def matrix(self):
        m = torch.eye(4, device=self.ctx.device).clone()
        m[:3, :3] = self.mat
        m[:3, 3] = self.tr
        return m

    def __call__(self):
        return self.ctx.warp_matrix(self.matrix())

    def parameters(self):
        return [self.mat, self.tr]

    def param_groups(self, lr):
        """线性部分（无量纲）用 lr，平移（mm）用 lr × TRANS_LR_SCALE。"""
        return [{"params": [self.mat], "lr": lr},
                {"params": [self.tr], "lr": lr * TRANS_LR_SCALE}]

    def snapshot(self):
        return (self.mat.detach().clone(), self.tr.detach().clone())

    def restore(self, snap):
        with torch.no_grad():
            self.mat.copy_(snap[0])
            self.tr.copy_(snap[1])


class _FFDParams:
    """自由形变（FFD）：低分辨率控制点 + 三线性上采样成位移场。"""

    def __init__(self, ctx: WarpContext, grid=(4, 4, 4), amp_mm: float = 8.0,
                 init_disp=None, device=None):
        z, y, x = ctx.fixed_grid.shape[:3]
        self.shape = (z, y, x)
        self.ctrl_shape = tuple(int(g) for g in grid)
        if init_disp is None:
            c = torch.zeros(*self.ctrl_shape, 3, device=device)
        else:
            c = F.interpolate(init_disp.permute(3, 0, 1, 2)[None],
                              size=self.ctrl_shape, mode="trilinear",
                              align_corners=False)[0].permute(1, 2, 3, 0)
        self.ctrl = torch.nn.Parameter(c)
        self.amp = amp_mm
        self.ctx = ctx
        # 前两级累积的矩阵变换（不可训练，作为固定前置），
        # 让 FFD 能在"矩阵已对齐"的坐标系里定义位移，同时保持只插值一次
        self.pre_m4 = None

    def disp(self):
        """控制点 (gz,gy,gx,3) → 全分辨率位移场 (Z,Y,X,3)，单位 mm。"""
        d = self.ctrl.permute(3, 0, 1, 2)[None]                   # (1,3,gz,gy,gx)
        up = F.interpolate(d, size=self.shape, mode="trilinear",
                           align_corners=False)
        return up[0].permute(1, 2, 3, 0) * self.amp

    def __call__(self):
        if self.pre_m4 is not None:
            return self.ctx.warp_matrix_field(self.pre_m4, self.disp())
        return self.ctx.warp_field(self.disp())

    def parameters(self):
        return [self.ctrl]

    def param_groups(self, lr):
        # 控制点是无量纲的（真正的位移量 = 控制点值 × amp_mm），不需要放大
        return [{"params": [self.ctrl], "lr": lr}]

    def snapshot(self):
        return self.ctrl.detach().clone()

    def restore(self, snap):
        with torch.no_grad():
            self.ctrl.copy_(snap)

    def field_stats(self, n_axis: int = 9) -> dict:
        d = self.disp().detach()
        mag = torch.linalg.norm(d, dim=-1)
        return {"disp_mean_mm": round(float(mag.mean()), 3),
                "disp_max_mm": round(float(mag.max()), 3)}


# ------------------------------------------------------------ 三级的门面
def register_rigid_gpu(fixed_arr, moving_arr, spacing_zyx, device,
                       levels=3, iters=80, lr=3e-2, loss_fn=None):
    """刚性配准，多分辨率由粗到细。"""
    return _register_stage("rigid", fixed_arr, moving_arr, spacing_zyx, device,
                           levels, iters, lr, loss_fn)


def register_affine_gpu(fixed_arr, moving_arr, spacing_zyx, device,
                        levels=3, iters=80, lr=1e-2, init_m4=None, loss_fn=None):
    return _register_stage("affine", fixed_arr, moving_arr, spacing_zyx, device,
                           levels, iters, lr, loss_fn, init_m4=init_m4)


def register_ffd_gpu(fixed_arr, moving_arr, spacing_zyx, device,
                     grid=(4, 4, 4), levels=2, iters=60, lr=2e-2,
                     init_m4=None, loss_fn=None):
    return _register_stage("ffd", fixed_arr, moving_arr, spacing_zyx, device,
                           levels, iters, lr, loss_fn,
                           init_m4=init_m4, ffd_grid=grid)


def _register_stage(name, fixed_arr, moving_arr, spacing_zyx, device,
                    levels, iters, lr, loss_fn, init_m4=None, ffd_grid=(4, 4, 4)):
    """逐分辨率优化；每级都在"上一级粗分辨率的结果"上继续细化。

    返回 (累积变换描述, 该级信息)。描述是一个 dict：
        {"kind": "matrix", "m4": tensor} 或 {"kind": "field", "disp": tensor}
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("需要 torch")
    if loss_fn is None:
        loss_fn = ncc_loss

    t0 = time.perf_counter()
    # 分辨率阶梯：level 层的下采样倍率，如 levels=3 -> [4, 2, 1]
    factors = [2 ** (levels - 1 - i) for i in range(levels)]

    # 旋转中心：取**全分辨率**固定图像的几何中心，整条金字塔共用一个。
    # 每级各推各的中心是个隐蔽的坑 —— 同一个刚体运动在不同旋转中心下的
    # (R, t) 不同，换级时那个矩阵会被解释成另一个变换。见 _RigidParams。
    center_mm = torch.as_tensor(
        [(fixed_arr.shape[2] - 1) / 2.0 * float(spacing_zyx[2]),
         (fixed_arr.shape[1] - 1) / 2.0 * float(spacing_zyx[1]),
         (fixed_arr.shape[0] - 1) / 2.0 * float(spacing_zyx[0])],
        dtype=torch.float32, device=device)
    cur_m4 = init_m4
    cur_disp = None
    ffd_pre_m4 = None
    m4_identity = torch.eye(4, device=device)
    best_metric = float("inf")
    total_iters = 0

    for lv, fac in enumerate(factors):
        f_ds = _downsample_np(fixed_arr, fac, device)
        m_ds = _downsample_np(moving_arr, fac, device)
        sp = tuple(float(v) * fac for v in spacing_zyx)
        ctx = make_context(f_ds.shape, sp, m_ds, sp, device)
        fixed_t = torch.as_tensor(np.ascontiguousarray(f_ds, dtype=np.float32),
                                  device=device)[None, None]

        if name in ("rigid", "affine"):
            init = cur_m4 if cur_m4 is not None else m4_identity
            holder = (_AffineParams(ctx, init, device) if name == "affine"
                      else _RigidParams(ctx, init, device, center_mm=center_mm))
        else:
            # FFD 级：**优化时不要挂 pre_m4**，这是本文件里最容易写错的一处。
            #
            # 这一级的 ctx 是拿「上一级已经对齐过的图像」建的（见上面的 m_ds），
            # 所以优化时的正确采样式是 src = p + disp(p)，采样目标就是 cur_moving 自己。
            # 如果贪方便把上一级的矩阵挂成 pre_m4，采样式就变成 src = M·(p + disp(p))，
            # 等于把 M 又乘了一遍 —— 优化出来的位移场和最终 apply_result 的结果对不上
            # （实测表现为 Dice 不升反降）。
            #
            # 而**累积变换**里必须带上 M，因为最终是作用到**原始** moving 上的：
            #     warp_matrix_field(M, disp) 作用在原始 moving 上
            #   ≡ warp_field(disp)           作用在 cur_moving 上   （因为 cur_moving(x) = moving(M·x)）
            # 两者数学上完全等价，于是整条链依旧只插值一次。
            init = cur_m4 if cur_m4 is not None else m4_identity
            holder = _FFDParams(ctx, grid=ffd_grid, amp_mm=8.0, device=device)
            holder.pre_m4 = None
            ffd_pre_m4 = init.detach()

        metric, n_it = _optimize(ctx, fixed_t, holder, iters, lr, loss_fn)
        total_iters += n_it
        best_metric = metric

        if name in ("rigid", "affine"):
            cur_m4 = holder.matrix().detach()
        else:
            cur_disp = holder.disp().detach()
            # ffd_pre_m4 已在上面按 init 记下 —— 注意不能写成 holder.pre_m4，
            # 优化时它被刻意置成 None（见 FFD 分支的说明）。

        print(f"          · {name:<6} level {lv} ({fac}×, {tuple(f_ds.shape)})  "
              f"loss {metric:.5f}  {time.perf_counter() - t0:.1f}s")

    secs = time.perf_counter() - t0
    if name in ("rigid", "affine"):
        out = {"kind": "matrix", "m4": cur_m4}
        dof = 6 if name == "rigid" else 12
        extra = {"lr": lr, "levels": levels}
    else:
        # 单独调用 FFD（没传 init_m4）时 ffd_pre_m4 没被赋值，补一个单位阵兜底
        out = {"kind": "matrix_field",
               "m4": ffd_pre_m4 if ffd_pre_m4 is not None else m4_identity.detach(),
               "disp": cur_disp}
        dof = int(np.prod(ffd_grid)) * 3
        extra = {"grid": list(ffd_grid), "amp_mm": 8.0,
                 "composes_rigid_affine": True,
                 **holder.field_stats()}
    info = StageInfo(name=name, dof=dof, iterations=total_iters,
                     metric=best_metric, seconds=secs, extra=extra)
    return out, info


# ============================================================ 应用与评价
def apply_result(fixed_arr, moving_arr, spacing_zyx, device, result) -> np.ndarray:
    """把配准结果作用到原始 moving 上，返回与 fixed 同形的对齐图像。

    **只做一次插值** —— 累积变换直接作用在原始 moving 上，
    而不是拿级联过程里已经重采样过的中间图像，避免多次插值累积模糊。
    """
    ctx = make_context(fixed_arr.shape, spacing_zyx, moving_arr, spacing_zyx, device)
    with torch.no_grad():
        kind = result["kind"]
        if kind == "matrix":
            out = ctx.warp_matrix(result["m4"])
        elif kind == "matrix_field":
            out = ctx.warp_matrix_field(result["m4"], result["disp"])
        else:
            out = ctx.warp_field(result["disp"])
    return out[0, 0].cpu().numpy()


def ffd_disp_numpy(result):
    """取出位移场的 numpy 形式；不是自由形变结果就返回 None。"""
    if result["kind"] not in ("field", "matrix_field"):
        return None
    return result["disp"].cpu().numpy()


def transform_points(result, pts_phys: np.ndarray, fixed_shape, spacing_zyx,
                     device) -> np.ndarray:
    """把一批物理点按配准结果映射到 moving 空间（用于 TRE 计算）。

    语义必须和 `apply_result` 严格一致，否则 TRE 会算出一个"看起来很小但没意义"的数：
        "matrix"        → q = M · p
        "matrix_field"  → q = M · (p + disp(p))     先位移、后矩阵
    """
    p = torch.as_tensor(np.asarray(pts_phys, dtype=np.float32), device=device)
    kind = result["kind"]

    if kind == "matrix":
        m4 = result["m4"]
        q = (m4[:3, :3] @ p.T).T + m4[:3, 3]
    elif kind in ("field", "matrix_field"):
        m4 = result.get("m4") if kind == "matrix_field" else None
        # 位移场定义在 fixed 网格上，用 fixed 物理坐标算归一化采样坐标
        iz = p[:, 2] / float(spacing_zyx[0])
        iy = p[:, 1] / float(spacing_zyx[1])
        ix = p[:, 0] / float(spacing_zyx[2])
        nz = 2.0 * iz / max(fixed_shape[0] - 1, 1) - 1.0
        ny = 2.0 * iy / max(fixed_shape[1] - 1, 1) - 1.0
        nx = 2.0 * ix / max(fixed_shape[2] - 1, 1) - 1.0
        grid = torch.stack([nx, ny, nz], dim=-1).reshape(1, 1, 1, -1, 3)
        # 位移场 (Z,Y,X,3) → (1,3,Z,Y,X) 才能当 grid_sample 的输入
        d = F.grid_sample(result["disp"].permute(3, 0, 1, 2)[None], grid,
                          mode="bilinear", align_corners=True)
        # d[0,:,0,0,:] 的形状是 (3, N)，必须转置成 (N,3) 才能和 p 相加 ——
        # 少写这个 .permute 会直接广播失败（N≠3 时抛异常），是踩过的坑。
        dv = d[0, :, 0, 0, :].permute(1, 0)
        base = p + dv
        q = base if m4 is None else (m4[:3, :3] @ base.T).T + m4[:3, 3]
    else:
        raise ValueError(f"未知的配准结果类型: {kind}")
    return q.detach().cpu().numpy()


def register_multistage_gpu(fixed_arr, moving_arr, spacing_zyx, device,
                            stages=("rigid", "affine", "ffd"),
                            ffd_grid=(4, 4, 4), levels=3, iters=80,
                            loss_fn=None, verbose=True):
    """三级串联。返回 [(级名, 累积结果, 该级信息), ...]。

    累积语义：后续级别在**上一级对齐后的图像**上做，而累积结果始终是
    「对原始 moving 的一次性变换序列」，最后统一用 apply_result 里
    合成好的结果重采样一次原始 moving。
    """
    out_list = []
    cur_moving = moving_arr
    m4_acc = None
    for name in stages:
        # 每级都在「上一级对齐后的图像」上优化一个干净类型的新变换，
        # 所以该级返回的变换是相对 cur_moving 的，初值用单位阵
        if name == "rigid":
            res, info = register_rigid_gpu(fixed_arr, cur_moving, spacing_zyx,
                                           device, levels, iters, 3e-2, loss_fn)
        elif name == "affine":
            res, info = register_affine_gpu(fixed_arr, cur_moving, spacing_zyx,
                                            device, levels, iters, 1e-2, loss_fn)
        elif name == "ffd":
            # FFD 级的学习率必须单独放大，不能沿用矩阵级的 1e-2。
            # 控制点是**无量纲**的，真实位移 = 控制点值 × amp_mm(=8)，
            # 所以 lr=2e-2 只对应 0.16 mm/步。实测按这个值跑 120 步，
            # 学到的位移场均值只有 3.84 mm，而真值幅值是 12 mm ——
            # 骨 Dice 因此卡在 0.667，比 CPU 版（三次 B 样条）的 0.879 差一大截。
            # 改成 6e-2（≈0.48 mm/步）并把分辨率层数提到 3 层后，
            # 位移场幅值和骨 Dice 都能追上 CPU 版。
            res, info = register_ffd_gpu(fixed_arr, cur_moving, spacing_zyx,
                                         device, ffd_grid, levels=3, iters=80,
                                         lr=6e-2, init_m4=m4_acc, loss_fn=loss_fn)
        else:
            raise ValueError(f"未知级别 {name}")

        if name in ("rigid", "affine"):
            # 复合语义：先应用已有累积，再应用这一级 —— 对应矩阵右乘
            m4_acc = res["m4"] if m4_acc is None else m4_acc @ res["m4"]
            cumulative = {"kind": "matrix", "m4": m4_acc}
        else:
            # FFD 级已经把 m4_acc 当作固定前置写进结果，直接就是累积变换
            cumulative = {"kind": "matrix_field", "m4": m4_acc,
                          "disp": res["disp"]}

        # 用累积变换作用到**原始** moving（而不是上一级的中间结果），
        # 保证整条链只经过一次线性插值
        cur_moving = apply_result(fixed_arr, moving_arr, spacing_zyx, device,
                                  cumulative)
        if verbose:
            print(f"        · {name:<6} 完成：{info.iterations:>4} 步  "
                  f"loss {info.metric:.5f}  {info.seconds:>6.2f}s")
        out_list.append((name, cumulative, info))

    return out_list
