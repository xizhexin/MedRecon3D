#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M6：MedRecon3D HTTP 服务（FastAPI）

把整条管线包成一个可调用的服务：上传 DICOM 压缩包（或指定服务器端目录），
后台跑分割 + 三维重建，产出 STL / 三维交互模型 / 量化 JSON 供下载。

端点
    GET  /                                  简易上传页（演示用）
    GET  /health                            健康检查
    GET  /api/v1/structures                 支持重建的解剖结构
    POST /api/v1/jobs/upload                上传 DICOM zip（multipart/form-data）
    POST /api/v1/jobs/path                  指定服务器端 DICOM 目录
    GET  /api/v1/jobs                       列出全部作业
    GET  /api/v1/jobs/{job_id}              查询作业状态与结果摘要
    GET  /api/v1/jobs/{job_id}/mesh/{name}  下载 STL
    GET  /api/v1/jobs/{job_id}/model        三维交互模型 HTML
    GET  /api/v1/jobs/{job_id}/result       量化结果 JSON

启动
    cd MedRecon3D/service
    uvicorn app:app --host 127.0.0.1 --port 8000

设计取舍
    - 作业状态放进程内存 + 每个作业一个输出目录。单机演示够用；
      要上多实例得把状态挪到 Redis/数据库，作业目录挪到对象存储。
    - CPU 密集，用信号量把并发限在 1（可用 MEDRECON_WORKERS 调）。
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import (BackgroundTasks, FastAPI, File, Form, HTTPException,
                     UploadFile)
from fastapi.responses import FileResponse, HTMLResponse

from pipeline import STRUCTURES, run_pipeline, safe_extract

APP_DIR = Path(__file__).resolve().parent
JOBS_DIR = Path(os.environ.get("MEDRECON_JOBS_DIR", APP_DIR / "jobs"))
MAX_UPLOAD_MB = int(os.environ.get("MEDRECON_MAX_UPLOAD_MB", "512"))

app = FastAPI(
    title="MedRecon3D",
    version="0.1.0",
    description="DICOM 序列 → 三维重建 / 量化测量的 HTTP 服务",
)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_sem = threading.Semaphore(int(os.environ.get("MEDRECON_WORKERS", "1")))


# ------------------------------------------------------------ 作业管理
def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _create_job(kind: str, dicom_dir: Path, spacing: float) -> dict:
    job_id = uuid.uuid4().hex[:12]
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "job_id": job_id, "status": "queued", "kind": kind,
        "source": str(dicom_dir), "dicom_dir": str(dicom_dir),
        "spacing_mm": float(spacing),
        "created_at": time.time(), "started_at": None, "finished_at": None,
        "error": None, "result": None,
    }
    with _lock:
        _jobs[job_id] = rec
    return rec


def _run_job(job_id: str) -> None:
    with _lock:
        rec = _jobs.get(job_id)
    if rec is None:
        return
    with _sem:                                   # CPU 密集，限制并发
        rec["status"] = "running"
        rec["started_at"] = time.time()
        try:
            rec["result"] = run_pipeline(
                Path(rec["dicom_dir"]), _job_dir(job_id),
                target_spacing=float(rec["spacing_mm"]))
            rec["status"] = "done"
        except Exception as exc:                 # noqa: BLE001 - 作业失败要落状态
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            rec["finished_at"] = time.time()


def _public(rec: dict) -> dict:
    out = {k: rec[k] for k in ("job_id", "status", "kind", "spacing_mm",
                               "created_at", "started_at", "finished_at", "error")}
    if rec["status"] == "done" and rec["result"]:
        r = rec["result"]
        jid = rec["job_id"]
        out["summary"] = {
            "dicom_meta": r["dicom_meta"],
            "grid": r["grid"],
            "total_seconds": r["total_seconds"],
            "structures": {
                k: {"label": v["label"],
                    "volume_ml_voxel": v["volume_ml_voxel"],
                    "watertight": v["watertight"],
                    "volume_diff_pct": v["volume_diff_pct"],
                    "faces": v["faces"],
                    "stl_bytes": v["stl_bytes"]}
                for k, v in r["structures"].items()
            },
        }
        out["downloads"] = {
            "model_html": (f"/api/v1/jobs/{jid}/model" if r.get("model_html") else None),
            "result_json": f"/api/v1/jobs/{jid}/result",
            "meshes": {k: f"/api/v1/jobs/{jid}/mesh/{k}" for k in r["structures"]},
        }
    return out


def _get(job_id: str) -> dict:
    with _lock:
        rec = _jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, f"作业不存在: {job_id}")
    return rec


# ------------------------------------------------------------ 端点
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "MedRecon3D", "version": app.version,
            "jobs": len(_jobs), "jobs_dir": str(JOBS_DIR)}


@app.get("/api/v1/structures")
def structures() -> dict:
    return {"structures": list(STRUCTURES)}


@app.post("/api/v1/jobs/upload", status_code=202)
async def create_job_by_upload(background: BackgroundTasks,
                               file: UploadFile = File(...),
                               spacing_mm: float = Form(1.0)) -> dict:
    """上传 DICOM 序列压缩包（.zip）。

    只收 zip：DICOM 序列动辄上百个无扩展名文件，逐个 multipart 上传不现实。
    """
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "只接受 .zip 压缩包")

    job_id = uuid.uuid4().hex[:12]
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    zip_path = d / "upload.zip"

    size = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    with zip_path.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                fh.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"压缩包超过 {MAX_UPLOAD_MB} MB 上限")
            fh.write(chunk)

    dicom_dir = safe_extract(zip_path, d / "dicom")   # 带 Zip Slip 防护
    zip_path.unlink(missing_ok=True)

    rec = _create_job("upload", dicom_dir, spacing_mm)
    rec["upload_bytes"] = size
    background.add_task(_run_job, rec["job_id"])
    return {"job_id": rec["job_id"], "status": "queued",
            "upload_bytes": size, "status_url": f"/api/v1/jobs/{rec['job_id']}"}


@app.post("/api/v1/jobs/path", status_code=202)
def create_job_by_path(background: BackgroundTasks, dicom_dir: str,
                       spacing_mm: float = 1.0) -> dict:
    """指定服务器端已有的 DICOM 目录（避免上传大文件）。"""
    p = Path(dicom_dir)
    if not p.is_dir():
        raise HTTPException(400, f"目录不存在: {dicom_dir}")
    rec = _create_job("path", p, spacing_mm)
    background.add_task(_run_job, rec["job_id"])
    return {"job_id": rec["job_id"], "status": "queued",
            "status_url": f"/api/v1/jobs/{rec['job_id']}"}


@app.get("/api/v1/jobs")
def list_jobs() -> dict:
    with _lock:
        items = [_public(r) for r in _jobs.values()]
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return {"count": len(items), "jobs": items}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _public(_get(job_id))


def _file(job_id: str, name: str, media: str) -> FileResponse:
    rec = _get(job_id)
    if rec["status"] != "done":
        raise HTTPException(409, f"作业尚未完成（当前状态 {rec['status']}）")
    f = _job_dir(job_id) / name
    if not f.is_file():
        raise HTTPException(404, f"文件不存在: {name}")
    return FileResponse(f, media_type=media, filename=name)


@app.get("/api/v1/jobs/{job_id}/mesh/{structure}")
def get_mesh(job_id: str, structure: str) -> FileResponse:
    if structure not in STRUCTURES:
        raise HTTPException(400, f"未知结构: {structure}，可选 {list(STRUCTURES)}")
    return _file(job_id, f"{structure}.stl", "model/stl")


@app.get("/api/v1/jobs/{job_id}/model")
def get_model(job_id: str) -> FileResponse:
    return _file(job_id, "model.html", "text/html")


@app.get("/api/v1/jobs/{job_id}/result")
def get_result(job_id: str) -> FileResponse:
    return _file(job_id, "result.json", "application/json")


_PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>MedRecon3D</title>
<style>
body{font:14px/1.6 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
max-width:760px;margin:48px auto;padding:0 20px;color:#222}
h1{font-size:20px;font-weight:600;margin-bottom:4px}
p.sub{color:#666;margin-top:0}
form{border:1px solid #e3e3e3;border-radius:12px;padding:20px;margin:24px 0}
input,button{font:inherit}
button{padding:8px 18px;border:1px solid #d0d0d0;border-radius:8px;
background:#fff;cursor:pointer}
button:hover{border-color:#999}
code{background:#f5f5f5;padding:2px 6px;border-radius:4px}
pre{background:#f7f7f7;padding:14px;border-radius:8px;overflow:auto;max-height:340px}
</style>
<h1>MedRecon3D</h1>
<p class="sub">DICOM 序列 → 三维重建 + 量化测量</p>
<form id="f">
  <div style="margin-bottom:12px">
    <input type="file" name="file" accept=".zip" required>
  </div>
  <div style="margin-bottom:16px">
    重采样间距 (mm)：<input type="number" name="spacing_mm" value="1.0" step="0.5" min="0.5" style="width:80px">
  </div>
  <button type="submit">上传并重建</button>
</form>
<pre id="out">等待上传…</pre>
<script>
const f=document.getElementById('f'),out=document.getElementById('out');
f.onsubmit=async e=>{e.preventDefault();
 out.textContent='上传中…';
 const r=await fetch('/api/v1/jobs/upload',{method:'POST',body:new FormData(f)});
 if(!r.ok){out.textContent='提交失败: '+r.status+' '+(await r.text());return}
 const j=await r.json();out.textContent=JSON.stringify(j,null,2)+'\\n\\n轮询中…';
 poll(j.job_id);};
async function poll(id){
 const r=await fetch('/api/v1/jobs/'+id);const j=await r.json();
 out.textContent=JSON.stringify(j,null,2);
 if(j.status==='queued'||j.status==='running')setTimeout(()=>poll(id),1500);}
</script>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE
