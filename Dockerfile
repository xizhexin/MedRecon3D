# MedRecon3D —— DICOM 序列 → 三维重建 + 量化测量 服务
# 构建（在 MedRecon3D/ 目录下执行）：
#   docker build -t medrecon3d:0.1.0 .
# 运行：
#   docker run --rm -p 8000:8000 -v medrecon_data:/data medrecon3d:0.1.0
# 用法：
#   docker run --rm -v /本地/dicom:/data/dicom \
#     medrecon3d:0.1.0 \
#     python demo_01_recon.py --dicom-dir /data/dicom --out-prefix demo01

FROM python:3.11-slim

# libgomp1 是 scikit-image / SimpleITK 的 OpenMP 运行时依赖，slim 镜像里没有
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV MEDRECON_JOBS_DIR=/data/jobs \
    MEDRECON_WORKERS=1 \
    PYTHONUNBUFFERED=1

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# 容器内必须监听 0.0.0.0 才能被端口映射访问（本机开发时用 127.0.0.1 即可）
WORKDIR /app/service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
