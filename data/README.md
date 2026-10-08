# 数据集不入库

体积 78 MB，且是公开数据，没必要进版本库。

获取方式：

```bash
python fetch_data.py        # 下载 + MD5 校验 + 解压到本目录
```

- 数据集：**PCIR 人体躯干 CT**
- 来源：Zenodo，DOI [`10.5281/zenodo.18225140`](https://doi.org/10.5281/zenodo.18225140)
- 许可：**CC0 1.0 Universal**（公有领域）
- MD5：`e25e4a734cb7ffee4b0ccd96e7df358c`
- 内容：107 层，512×512，层厚 2.5 mm，PixelSpacing 0.879 mm

解压后目录结构：

```
data/PCIR_torso/Heart_CT/     # 107 个无扩展名文件，pydicom 按 DICM magic 识别
```

> `fetch_data.py` 里固化了完整浏览器 User-Agent —— Zenodo 的 WAF 会拦陌生 UA，
> 裸请求返回 `403 "restricted due to unusual traffic from your network"`。
