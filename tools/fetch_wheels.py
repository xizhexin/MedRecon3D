#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 PyPI 镜像直取 wheel —— 绕开 pip 的慢下载器。

为什么需要这个
--------------
实测在 AutoDL 实例上对**同一个 URL**：
    curl   → 16.4 MB/s
    pip    → 0.15 MB/s
差了整整 100 倍。装 torch 的 CUDA 运行库要下 1.7 GB，
用 pip 得等到天荒地老（更早一次实测：799 MB 的 torch wheel 卡在 64 MB 不动）。

所以流程改成：**自己解析 index、自己下载，最后 `pip install --no-index` 离线装**。

用法：
    python fetch_wheels.py --dest /root/wheels nvidia-cudnn-cu11==9.1.0.70 ...
    python fetch_wheels.py --dest /root/wheels --from-file /root/urls.txt
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

# packaging 是 pip 的依赖，正常情况下一定存在；拿不到就退化成字典序（不理想但能跑）
try:
    from packaging.version import Version as _Version
except Exception:  # pragma: no cover
    _Version = None

MIRRORS = [
    "https://mirrors.aliyun.com/pypi/simple",
    "https://pypi.tuna.tsinghua.edu.cn/simple",
]

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _filename(href: str) -> str:
    return href.split("/")[-1].split("#")[0]


_WHEEL_TAIL = re.compile(r"-([a-z0-9_.]+)-([a-z0-9_.]+)-([a-z0-9_.]+)\.whl$")


def _platform_score(name: str, python_tag: str = "cp312") -> int:
    """wheel 文件名 → 平台匹配分。分越高越优先，<=0 表示不可用。

    wheel 名格式是 `name-version-pytag-abitag-platformtags.whl`，
    pytag 可能是逗点分隔的集合（`py2.py3`、`cp312.cp313`）。

    纯 Python（`*-none-any.whl`）给 **1 分而不是 0 分**：给 0 分会被调用方的
    "丢掉不可用候选" 一并滤掉，于是在**同时存在编译版旧 wheel** 的包上
    （典型：pydantic —— v2 是纯 Python、v1 有 cp312 编译 wheel）会把 v1 选出来。
    实测被这个坑选成 pydantic 1.10.26，装上去 FastAPI 直接崩。
    """
    low = name.lower()
    if "pypy" in low:                                 # PyPy 专用轮子，不要
        return -1
    m = _WHEEL_TAIL.search(low)
    if not m:
        return 0
    pytag, _abitag, plat = m.group(1), m.group(2), m.group(3)

    # ① Python 版本必须匹配（py3 / py2 视为通用；cpXXX 必须逐个对上）
    tags = pytag.split(".")
    if not any(t == python_tag or t in ("py3", "py2") for t in tags):
        return 0

    # ② 平台匹配
    if "any" in plat.split("."):
        return 1                                      # 纯 Python，兜底但不丢弃
    if "linux" not in plat or "x86_64" not in plat:
        return 0                                      # macOS / Windows / aarch64
    if "manylinux" in plat:
        old = any(k in plat for k in ("manylinux2014", "manylinux_2_17",
                                      "manylinux_2_5", "manylinux1",
                                      "manylinux_2_12"))
        return 4 if old else 3                        # 老 glibc 门槛兼容性更广
    return 2


def _parse_version(name: str, norm: str):
    """从文件名里抠出 PEP440 版本；解析不出来返回 None。"""
    if _Version is None:
        return None
    m = re.match(rf"{re.escape(norm)}-(.+?)-", name.lower())
    if not m:
        return None
    try:
        return _Version(m.group(1))
    except Exception:
        return None


def resolve(pkg: str, ver: str | None, python_tag: str = "cp312") -> list[str]:
    """在镜像的 simple index 上找出某个包某个版本的所有候选 wheel URL。

    选择顺序（与 pip 的行为对齐）：
      1. 只保留本平台可用的候选（含纯 Python 兜底），丢掉 nyi / 其它平台；
      2. 有稳定版就丢掉预发布版（rc/dev/alpha）—— 否则 `httpx` 会选到 1.0.dev1，
         它的 release 元组 (1,0,0) 比 0.28.1 大，预发布惩罚放在次位压不住；
      3. 取**版本号最大**的那一版（不能按平台分排前面，否则旧版本有编译 wheel
         时会把新版本挤掉 —— pydantic v1/v2 就是这么中招的）；
      4. 同一版本内有多个 wheel 时，再按平台匹配分排序。
    """
    # 索引路径必须用**归一化小写名**：PyPI 索引目录大小写敏感吗？实测
    # /simple/SimpleITK/ 在阿里云镜像上直接 HTTPError，/simple/simpleitk/ 才 200。
    norm = pkg.replace("-", "_").lower()
    for mirror in MIRRORS:
        html = None
        last = "未知错误"
        for slug in dict.fromkeys((norm, pkg, pkg.replace("_", "-").lower())):
            try:
                html = _get(f"{mirror}/{slug}/").decode("utf-8", "ignore")
                break
            except Exception as exc:
                last = type(exc).__name__
                continue
        if html is None:
            print(f"    (镜像 {mirror.split('/')[2]} 不可用: {last})")
            continue

        hrefs = re.findall(r'href="([^"#]+\.whl)[^"]*"', html)
        cands: list[tuple[str, int, object]] = []
        for h in hrefs:
            base = _filename(h)
            low = base.lower()
            if not low.startswith(norm + "-"):
                continue
            if ver and not low.startswith(f"{norm}-{ver}-"):
                continue
            sc = _platform_score(base, python_tag)
            if sc <= 0:
                continue
            cands.append((h, sc, _parse_version(base, norm)))

        if not cands:
            continue

        # 只统计能解析出版本号的候选
        parsed = [c for c in cands if c[2] is not None]
        pool = parsed or cands

        if parsed:
            stable = [c for c in parsed if not c[2].is_prerelease]
            if stable:
                pool = stable
            pool.sort(key=lambda c: (c[2].release, c[1]), reverse=True)
        else:
            pool.sort(key=lambda c: c[1], reverse=True)

        top = pool[0]
        # 同一版本的全部候选（供 --from-file 或备用），按平台分排
        best_ver = top[2]
        same = [c for c in pool if best_ver is None or c[2] == best_ver]
        same.sort(key=lambda c: c[1], reverse=True)

        # index 页面里的 href 通常是 ../../packages/xx/yy/xxx.whl 这种相对路径，
        # 必须用 urljoin 对着 index 页面的 URL 做规范化；手动拼字符串会得到
        # `https://host/pypi../../packages/...` 这种 404（踩过）。
        base_url = f"{mirror}/{norm}/"
        out = []
        for h, _sc, _v in same:
            url = h if h.startswith("http") else urljoin(base_url, h)
            if url not in out:
                out.append(url)
        return out
    return []


def download(url: str, dest: Path, use_curl: bool = True) -> Path:
    """下载单个文件。

    优先用 curl —— 这不是偏好问题：同一台机器、同一个 URL，
        curl -r 0-40000000  → 16.4 MB/s
        urllib.read()       →  0.37 MB/s
    Python 标准库的 HTTP 栈在这条链路上慢 40 倍，1.7 GB 的 CUDA 库
    用 urllib 要 77 分钟，用 curl 只要 2 分钟。
    """
    name = url.split("/")[-1].split("#")[0]
    target = dest / name
    if target.exists() and target.stat().st_size > 0:
        print(f"    已有 {name}（{target.stat().st_size / 2**20:.1f} MB），跳过")
        return target

    tmp = target.with_suffix(target.suffix + ".part")
    t0 = time.time()

    if use_curl and shutil.which("curl"):
        subprocess.run(
            ["curl", "-L", "--retry", "5", "--retry-delay", "3", "-C", "-",
             "-sS", "--connect-timeout", "30", "-o", str(tmp), url],
            check=True)
        got = tmp.stat().st_size
    else:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)

    tmp.rename(target)
    dt = max(time.time() - t0, 1e-6)
    print(f"    {name}  {got / 2**20:.1f} MB  {got / dt / 2**20:.1f} MB/s")
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description="从镜像直取 wheel")
    ap.add_argument("--dest", default="/root/wheels")
    ap.add_argument("--python-tag", default="cp312")
    ap.add_argument("--no-curl", action="store_true",
                    help="强制用 urllib 下载（默认优先 curl，见 download() 的说明）")
    ap.add_argument("--from-file", help="每行一个 URL 的文件")
    ap.add_argument("packages", nargs="*", help="形如 包名 或 包名==版本")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, str | None]] = []
    for p in args.packages:
        if "==" in p:
            n, v = p.split("==", 1)
            specs.append((n.strip(), v.strip()))
        else:
            specs.append((p.strip(), None))

    urls: list[str] = []
    if args.from_file:
        urls = [l.strip() for l in Path(args.from_file).read_text().splitlines() if l.strip()]

    if specs:
        print(f"解析 {len(specs)} 个包 …")
        for name, ver in specs:
            got = resolve(name, ver, args.python_tag)
            if not got:
                print(f"  !! {name}=={ver} 未找到可用 wheel")
                continue
            print(f"  {name}=={ver} → {os.path.basename(got[0])}")
            urls.append(got[0])

    if not urls:
        print("没有要下载的东西。")
        return 1

    print(f"\n下载 {len(urls)} 个文件到 {dest}")
    ok = 0
    for u in urls:
        try:
            download(u, dest, use_curl=not args.no_curl)
            ok += 1
        except Exception as exc:
            print(f"    !! 失败 {os.path.basename(u)}: {type(exc).__name__}: {exc}")
    print(f"\n完成 {ok}/{len(urls)}")
    return 0 if ok == len(urls) else 2


if __name__ == "__main__":
    raise SystemExit(main())
