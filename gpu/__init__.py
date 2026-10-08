# -*- coding: utf-8 -*-
"""MedRecon3D 的 GPU 加速后端。

包结构
------
- `backend`     —— 设备探测与选择（唯一入口 `pick_device`）
- `ops`         —— 体素级 GPU 算子（重采样 / 阈值 / 统计 / 网格体积）
- `registration`—— 多级配准的 GPU 实现（本项目全局最热点）
- `quantify`    —— 量化测量的 GPU 版

用法
----
    from gpu.backend import pick_device
    device, info = pick_device("auto")     # 自动挑，拿不到 CUDA 就回退 CPU
    device, info = pick_device("cuda")     # 强制要求 CUDA，拿不到直接抛异常

约定：**没有 torch 或没有可用 GPU 时，一切都要能优雅回退到 CPU**。
GPU 是加速手段，不是项目前提 —— 换台机器就得能跑起来，这是工程底线。
"""

from .backend import (DeviceInfo, TORCH_AVAILABLE, pick_device, probe,  # noqa: F401
                      timer)

__all__ = ["DeviceInfo", "TORCH_AVAILABLE", "pick_device", "probe", "timer"]
