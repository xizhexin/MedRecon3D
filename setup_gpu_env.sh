#!/usr/bin/env bash
# =============================================================================
# MedRecon3D · GPU 环境一键搭建
# =============================================================================
# 适用：Ubuntu 22.04 容器（AutoDL / 自有机器 / 云主机），有 NVIDIA 卡和驱动。
# 会在 $VENV 下建一个独立 venv，不污染系统 Python。
#
# 用法：
#     bash setup_gpu_env.sh                    # 默认路径
#     VENV=/opt/my-env bash setup_gpu_env.sh   # 自定义
#     TORCH_VARIANT=cu121 bash setup_gpu_env.sh  # 非 Pascal 卡可以用更新的 CUDA
#
# 为什么默认锁 cu118 + torch 2.5.1（这段血泪史值得留着）
# -----------------------------------------------------------------------------
# 本项目第一台 GPU 机器是 **TITAN Xp（Pascal，sm_61）**，而平台预装的是
# `torch 2.8.0+cu128`，它的 wheel 只编了 sm_70 及以上：
#
#     CUDA error: no kernel image is available for execution on the device
#     NVIDIA TITAN Xp with CUDA capability sm_61 is not compatible with the
#     current PyTorch installation. The current PyTorch install supports
#     CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
#
# **最坑的一点**：`torch.cuda.is_available()` 会返回 **True** —— 直到你真正
# 做第一次运算才炸。所以环境检查必须用一次真实 matmul 来探，不能只看
# is_available()（`gpu/backend.py: probe()` 就是干这个的）。
#
# cu118 的 wheel 仍然包含 Pascal 的 kernel（arch_list 里有 sm_50/sm_60/sm_61），
# 所以 Pascal 卡走 cu118。这不是"恋旧"，是唯一能跑的选择。
#
# 另一个坑：**pip 下载器拉 799 MB 的大 wheel 会卡死**（实测两次都停在
# 64 MB 不动）。但同一个 URL 用 curl 能跑到 17 MB/s。所以这里用 curl 下
# wheel、再本地安装。顺带一提，pip 还要求 wheel 文件名符合 PEP 427
# （`{name}-{version}-{python}-{abi}-{platform}.whl`），下成 `torch.whl`
# 会被拒："not a valid wheel filename"。
# =============================================================================

set -euo pipefail

VENV="${VENV:-/root/medrecon-venv}"
BASE_PYTHON="${BASE_PYTHON:-$(command -v python3 || echo /root/miniconda3/bin/python)}"
TORCH_VARIANT="${TORCH_VARIANT:-cu118}"
TORCH_VER="${TORCH_VER:-2.5.1}"
PKG_INDEX="${PKG_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
WHEEL_MIRROR="${WHEEL_MIRROR:-https://mirrors.aliyun.com/pytorch-wheels}"
WORK="${WORK:-/root}"

echo "=============================================================="
echo " MedRecon3D GPU 环境搭建"
echo "   venv          : $VENV"
echo "   base python   : $BASE_PYTHON"
echo "   torch         : $TORCH_VER+$TORCH_VARIANT"
echo "=============================================================="

# ---------------------------------------------------------------- 0 硬件探针
echo
echo "[0/4] 硬件探针"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  ! 找不到 nvidia-smi —— 这台机器没有 NVIDIA 驱动，GPU 版跑不了。"
    echo "    CPU 管线请直接用 demo_01_recon.py（不需要这个脚本）。"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader | sed 's/^/  GPU: /'

# ---------------------------------------------------------------- 1 venv
echo
echo "[1/4] 创建 venv"
if [ ! -x "$VENV/bin/python" ]; then
    "$BASE_PYTHON" -m venv "$VENV"
fi
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
"$PY" -V | sed 's/^/  /'

# ---------------------------------------------------------------- 2 torch
echo
echo "[2/4] 安装 torch（curl 拉 wheel，避开 pip 下载器卡死的问题）"

PYTAG="$("$PY" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
WHEEL_NAME="torch-${TORCH_VER}+${TORCH_VARIANT}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
WHEEL_URL="$WHEEL_MIRROR/$TORCH_VARIANT/$WHEEL_NAME"
WHEEL_PATH="$WORK/$WHEEL_NAME"

# 先确认这个组合在镜像里真的存在（错拼一个字符就会下回一个 HTML 错误页）
echo "  探测: $WHEEL_URL"
if ! curl -sIL --max-time 30 "$WHEEL_URL" | head -1 | grep -q "200"; then
    echo "  ! 镜像里没有 $WHEEL_NAME"
    echo "    可用版本请自行查看： $WHEEL_MIRROR/$TORCH_VARIANT/"
    echo "    （manylinux 命名的 wheel 请把文件名里的 linux_x86_64 换成 manylinux_2_28_x86_64）"
    exit 1
fi

if [ ! -f "$WHEEL_PATH" ]; then
    echo "  下载中（799 MB 量级，阿里云镜像实测 ~17 MB/s）"
    curl -L --retry 5 --retry-delay 3 -C - --progress-bar \
         -o "$WHEEL_PATH" "$WHEEL_URL"
fi
ls -lh "$WHEEL_PATH" | awk '{print "  wheel: " $5}'

echo "  安装（含 CUDA 依赖，约 2-3 GB 磁盘）"
"$PIP" install --no-cache-dir "$WHEEL_PATH" \
       -i "$PKG_INDEX" --trusted-host "$(echo "$PKG_INDEX" | awk -F/ '{print $3}')"

# ---------------------------------------------------------------- 3 其余依赖
echo
echo "[3/4] 安装医学影像与网格依赖"
"$PIP" install --no-cache-dir \
    "numpy>=1.24" "scipy>=1.10" "pydicom>=2.4" "SimpleITK>=2.3" \
    "scikit-image>=0.22" "trimesh>=4.0" "fast-simplification>=0.1.7" \
    "matplotlib>=3.7" "Pillow>=10.0" "fastapi>=0.110" "uvicorn[standard]>=0.29" \
    "python-multipart>=0.0.9" "httpx>=0.27" \
    -i "$PKG_INDEX" --trusted-host "$(echo "$PKG_INDEX" | awk -F/ '{print $3}')"

# ---------------------------------------------------------------- 4 自检
echo
echo "[4/4] GPU 自检（真实运算，不只看 is_available）"
cd "$(dirname "$0")"
"$PY" -m gpu.selftest || {
    echo
    echo "  ! 自检未全部通过。常见原因："
    echo "    - 卡的架构不在 torch 编译列表里 → 换 TORCH_VARIANT"
    echo "    - 驱动太旧 → 需要 >= 520.61 才能跑 cu118"
    exit 1
}

echo
echo "=============================================================="
echo " 完成。用法："
echo "   $PY demo_01_recon_gpu.py --dicom-dir data/PCIR_torso/Heart_CT --device auto"
echo "   $PY demo_04_registration_gpu.py --dicom-dir data/PCIR_torso/Heart_CT"
echo "=============================================================="
