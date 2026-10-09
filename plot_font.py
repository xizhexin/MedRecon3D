#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""matplotlib 中文字体：**全项目唯一的一份设置**。

为什么单独抽一个模块
--------------------
原来 `demo_04_registration.py` 里写了一份跨平台探测，`demo_05_quantify.py`
里却硬写 `["Microsoft YaHei", "SimHei", "PingFang SC", ...]`。本机（Windows）
跑没问题，一上 Linux 服务器就全变成空心方框 —— 因为那个候选表里的字体
一个都不存在，而 **matplotlib 找不到字体不会报错，只会静默退化成方框**。

实测 AutoDL 的 Ubuntu 22.04 镜像 `fc-list | grep -i cjk` 返回 0 条。
`out/quantify_report.png` 就是这么出问题的。

所以凡是画中文图的地方，一律 `from plot_font import set_cjk_font`，
不要再各自写候选表。

Linux 上装字体的命令（约 5 MB）：
    apt-get install -y fonts-wqy-microhei
装完记得清 matplotlib 的字体缓存，否则新字体不会被扫描到：
    rm -rf ~/.cache/matplotlib
"""

from __future__ import annotations

# 按优先级排列。Windows / macOS / Linux 各自的常见中文字体都在里面。
CJK_CANDIDATES = [
    "Microsoft YaHei",      # Windows
    "SimHei",               # Windows
    "PingFang SC",          # macOS
    "Hiragino Sans GB",     # macOS
    "Noto Sans CJK SC",     # Linux (fonts-noto-cjk)
    "Source Han Sans SC",   # Linux (思源黑体)
    "WenQuanYi Micro Hei",  # Linux (fonts-wqy-microhei)
    "WenQuanYi Zen Hei",    # Linux (fonts-wqy-zenhei)
    "Droid Sans Fallback",  # 部分 Android / 精简镜像
    "AR PL UMing CN",       # 部分 Debian 镜像
]
# 兜底：matplotlib 自带，拉丁字母和数字一定正常（中文会是方框）
_FALLBACK = "DejaVu Sans"


def list_available_cjk() -> list:
    """当前环境里真正装有的中文字体（按候选表优先级排序）。"""
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    return [c for c in CJK_CANDIDATES if c in available]


def set_cjk_font(verbose: bool = False) -> str:
    """把 matplotlib 默认字体设成当前环境里可用的中文字体，返回生效的字体名。

    `axes.unicode_minus` 必须一起关掉：中文字体里的 U+2212 常常缺字形，
    负号会变成方框 —— 这是另一个高频坑。
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams

    ok = list_available_cjk()
    rcParams["font.sans-serif"] = ok + [_FALLBACK]
    rcParams["axes.unicode_minus"] = False

    if not ok:
        import warnings
        warnings.warn(
            "未找到任何中文字体，图里的中文会渲染成方框。"
            "Linux 上执行：apt-get install -y fonts-wqy-microhei && rm -rf ~/.cache/matplotlib",
            RuntimeWarning,
            stacklevel=2,
        )
        return _FALLBACK

    if verbose:
        print(f"      中文字体：{ok[0]}" + (f"（备选 {', '.join(ok[1:])}）" if len(ok) > 1 else ""))
    return ok[0]


if __name__ == "__main__":
    print("可用中文字体：", list_available_cjk() or "（无）")
    print("生效字体：", set_cjk_font(verbose=True))
