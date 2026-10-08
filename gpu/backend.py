# -*- coding: utf-8 -*-
"""GPU 后端：设备探测、选择与统一计时。

设计原则
--------
1. **torch 是可选的**。没装 torch、或装了但没有可用 CUDA 设备时，
   所有调用方都应该能优雅回退到 CPU 路径，而不是抛异常。
   本项目的主要交付物是"能跑通的重建 + 配准管线"，GPU 是加速手段不是前提。
2. **设备选择只有一个入口**（`pick_device`）。不要在业务代码里散落
   `torch.cuda.is_available()` —— 那种代码在换机器时最容易出问题。
3. **硬件信息要能落盘**。跑出来的 JSON 里必须能查到"这次是拿什么卡跑的、
   torch 什么版本、显存多少"，否则复现实验时说不清楚。

一个真实的坑（写在这里省得后人再踩）
------------------------------------
AutoDL 等平台的预装镜像常常是 `torch+x-cu12x`，而 **Pascal 架构的卡
（TITAN Xp / GTX 1080Ti，sm_61）在 cu12x 的官方 wheel 里没有 kernel**：

    CUDA error: no kernel image is available for execution on the device
    NVIDIA TITAN Xp with CUDA capability sm_61 is not compatible with the
    current PyTorch installation. The current PyTorch install supports
    CUDA capabilities sm_70 sm_75 ...

`torch.cuda.is_available()` 会返回 **True**，一直到你真正做第一次运算才炸 ——
所以"能不能用"必须用一次真实运算来验证，不能只看 is_available()。
`probe()` 就是干这个的。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

# torch 是可选依赖
try:  # pragma: no cover - 环境相关
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@dataclass
class DeviceInfo:
    """一次运行的硬件/软件环境快照，直接进结果 JSON。"""

    backend: str = "cpu"                 # "cuda" / "cpu"
    torch_version: str | None = None
    torch_cuda: str | None = None        # torch 编译时的 CUDA 版本
    device_name: str | None = None
    capability: str | None = None        # 如 "6.1"
    total_memory_mb: float | None = None
    arch_list: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "torch_version": self.torch_version,
            "torch_cuda": self.torch_cuda,
            "device_name": self.device_name,
            "capability": self.capability,
            "total_memory_mb": self.total_memory_mb,
            "arch_list": self.arch_list,
            "note": self.note,
        }

    def describe(self) -> str:
        if self.backend == "cuda":
            return (f"CUDA · {self.device_name} (sm_{self.capability}) · "
                    f"torch {self.torch_version} · CUDA {self.torch_cuda} · "
                    f"{self.total_memory_mb:.0f} MB")
        return f"CPU · torch {self.torch_version or '未安装'}{(' · ' + self.note) if self.note else ''}"


def probe(device_index: int = 0) -> DeviceInfo:
    """真实跑一次 CUDA 运算来确认设备可用。

    只看 `torch.cuda.is_available()` 是不够的：Pascal 卡配 cu12x wheel 时
    它会返回 True，但第一次 kernel 调用就抛 "no kernel image is available"。
    这里用一个最小 matmul 把问题提前暴露出来。
    """
    if not TORCH_AVAILABLE:
        return DeviceInfo(backend="cpu", note="未安装 torch")

    info = DeviceInfo(
        torch_version=str(torch.__version__),
        torch_cuda=str(getattr(torch.version, "cuda", "") or "") or None,
    )

    if not torch.cuda.is_available():
        info.note = "无可用 CUDA 设备"
        return info

    try:
        info.device_name = torch.cuda.get_device_name(device_index)
        cap = torch.cuda.get_device_capability(device_index)
        info.capability = f"{cap[0]}.{cap[1]}"
        info.total_memory_mb = round(
            torch.cuda.get_device_properties(device_index).total_memory / 1024 ** 2, 1)
        info.arch_list = list(torch.cuda.get_arch_list())

        # 真实运算验证 —— 这一步才会暴露 sm 版本不匹配
        x = torch.randn(256, 256, device=f"cuda:{device_index}")
        _ = (x @ x).sum().item()
        torch.cuda.synchronize()
        info.backend = "cuda"
    except Exception as exc:                     # 设备存在但 kernel 不兼容
        info.backend = "cpu"
        info.note = f"CUDA 探测失败，回退 CPU：{type(exc).__name__}: {exc}"
    return info


def pick_device(prefer: str = "auto", verbose: bool = True) -> tuple[object, DeviceInfo]:
    """选择计算设备。

    prefer: "auto" | "cuda" | "cpu"
        auto —— 有可用 CUDA 就用，否则 CPU（默认）
        cuda —— 强制要求 CUDA，拿不到就抛异常（用于"必须 GPU 跑"的场景）
        cpu  —— 强制 CPU，用于做 CPU/GPU 对照实验

    返回 (torch.device 或 None, DeviceInfo)。返回 None 表示调用方应走纯 CPU 路径。
    """
    prefer = (prefer or "auto").lower()
    info = probe()

    if prefer == "cpu":
        info.backend = "cpu"
        info.note = "按 --device cpu 强制使用 CPU"
        if verbose:
            print(f"      设备：{info.describe()}")
        return None, info

    if info.backend == "cuda":
        if verbose:
            print(f"      设备：{info.describe()}")
            if info.arch_list:
                print(f"      torch 编译的架构：{', '.join(info.arch_list)}")
        return torch.device("cuda"), info

    # 这里 CPU 回退
    if prefer == "cuda":
        raise RuntimeError(
            f"指定了 --device cuda 但 CUDA 不可用。\n"
            f"  torch {info.torch_version} / 编译 CUDA {info.torch_cuda}\n"
            f"  设备 {info.device_name} 架构 sm_{info.capability}\n"
            f"  torch 支持的架构 {info.arch_list}\n"
            f"  原因：{info.note}"
        )
    if verbose:
        print(f"      设备：{info.describe()}")
    return None, info


@contextmanager
def timer(store: dict, key: str):
    """把一段代码的耗时写进 dict —— 用于分阶段计时，直接进结果 JSON。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round(time.perf_counter() - t0, 3)
