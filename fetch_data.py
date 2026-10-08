# -*- coding: utf-8 -*-
"""MedRecon3D · 公开数据集下载与校验

数据集: PCIR Dataset 98890234_20010101 Series 7（人体躯干 CT）
  来源   : Zenodo
  DOI    : 10.5281/zenodo.18225140
  许可   : CC0 1.0 Universal（公有领域，无使用限制，无需注册）
  内容   : 107 层 CT，512×512，层厚 2.5 mm，Series "CTA AORTA"，GE 设备，43Y 男性
  说明   : 数据由 Patient Contributed Image Repository 贡献并去标识化

用法:
  python fetch_data.py            # 下载并解压
  python fetch_data.py --check    # 只校验已下载文件
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ZIP_PATH = DATA_DIR / "PCIR_torso_ct.zip"
EXTRACT_DIR = DATA_DIR / "PCIR_torso"

URL = ("https://zenodo.org/api/records/18225140/files/"
       "PCIR_98890234_20010101_7.zip/content")
MD5_EXPECTED = "e25e4a734cb7ffee4b0ccd96e7df358c"   # Zenodo 官方公布值
SIZE_EXPECTED = 24_658_025

# Zenodo 的 WAF 会拦陌生的 User-Agent（实测裸 UA 会返回 403 "unusual traffic"），
# 必须带完整浏览器头。
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/zip,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while blk := fh.read(chunk):
            h.update(blk)
    return h.hexdigest()


def download() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and md5_of(ZIP_PATH) == MD5_EXPECTED:
        print(f"[跳过] 已存在且校验通过: {ZIP_PATH.name}")
    else:
        print(f"[下载] {URL}")
        req = urllib.request.Request(URL, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=120) as resp, open(ZIP_PATH, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while blk := resp.read(1 << 18):
                out.write(blk)
                done += len(blk)
                if total:
                    print(f"\r       {done / 1e6:.1f}/{total / 1e6:.1f} MB "
                          f"({done / total * 100:.0f}%)", end="", flush=True)
        print()

    actual = md5_of(ZIP_PATH)
    size = ZIP_PATH.stat().st_size
    print(f"[校验] size {size:,} (期望 {SIZE_EXPECTED:,})")
    print(f"       md5  {actual}")
    print(f"       期望 {MD5_EXPECTED}")
    if actual != MD5_EXPECTED:
        print("!! 校验失败，文件可能损坏，请删除后重下")
        return 1
    print("[校验] 通过 ✓")

    if not EXTRACT_DIR.exists() or not any(EXTRACT_DIR.rglob("*")):
        print(f"[解压] → {EXTRACT_DIR}")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(EXTRACT_DIR)
    else:
        print(f"[跳过] 已解压: {EXTRACT_DIR}")

    series = EXTRACT_DIR / "Heart_CT"
    if series.is_dir():
        n = len([p for p in series.iterdir() if p.is_file()])
        print(f"[就绪] 序列目录 {series}  共 {n} 个 DICOM 文件（无扩展名，按 DICM magic 识别）")
        print(f"\n运行: python demo_01_recon.py --dicom-dir \"{series}\" --out-prefix real")
    return 0


def check() -> int:
    if not ZIP_PATH.exists():
        print(f"未找到 {ZIP_PATH}")
        return 1
    actual = md5_of(ZIP_PATH)
    ok = actual == MD5_EXPECTED
    print(f"md5 {actual}  {'✓ 通过' if ok else '✗ 不匹配 ' + MD5_EXPECTED}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="MedRecon3D · 公开数据集下载")
    parser.add_argument("--check", action="store_true", help="只校验，不下载")
    args = parser.parse_args()
    return check() if args.check else download()


if __name__ == "__main__":
    raise SystemExit(main())
