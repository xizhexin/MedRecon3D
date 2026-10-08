# -*- coding: utf-8 -*-
"""MedRecon3D · D1 环境体检
逐个模块在独立子进程中 import，避免 tensorboard 之类的报错污染结果。
用法: python env_check.py
"""
import subprocess
import sys

MODULES = [
    # 名称,        pip 包名,            是否必需
    ("numpy", "numpy", True),
    ("scipy", "scipy", True),
    ("matplotlib", "matplotlib", True),
    ("pydicom", "pydicom", True),
    ("SimpleITK", "SimpleITK", True),
    ("nibabel", "nibabel", True),
    ("torch", "torch", True),
    ("skimage", "scikit-image", True),   # marching_cubes
    ("trimesh", "trimesh", True),        # 网格处理 + STL 导出
    ("plotly", "plotly", True),          # 交互式三维可视化
    ("onnx", "onnx", False),
    ("onnxruntime", "onnxruntime", False),
    ("fastapi", "fastapi", False),
    ("uvicorn", "uvicorn", False),
]

SNIPPET = "import {m}; print({m}.__version__ if hasattr({m},'__version__') else 'ok')"


def probe(name: str):
    """在独立子进程里 import，返回 (ok, version)"""
    try:
        r = subprocess.run(
            [sys.executable, "-c", SNIPPET.format(m=name)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            return True, (r.stdout.strip().splitlines() or ["ok"])[-1]
        return False, ""
    except Exception:
        return False, ""


def main():
    print("解释器:", sys.executable)
    print("Python :", sys.version.split()[0])
    print("-" * 62)
    missing = []
    for name, pipname, required in MODULES:
        ok, ver = probe(name)
        flag = "OK  " if ok else ("MISS" if required else "opt ")
        print(f"[{flag}] {name:<14} {ver}")
        if not ok and required:
            missing.append(pipname)
    print("-" * 62)
    if missing:
        print("缺失必需包:", " ".join(missing))
        print("安装命令:")
        print(f'  "{sys.executable}" -m pip install {" ".join(missing)}')
        return 1
    print("必需依赖齐全，可以开工。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
