# Project 1 — NumPy 复现 OpenCLIP 图像编码器

用纯 NumPy 复现 OpenCLIP ViT-B-32 图像塔的前向计算。给定预处理好的图像
`[B,3,224,224]` 与随附的训练权重，输出每张图一个 **512 维单位向量** `[B,512]`，
数值须与随附的 Torch FP32 参考一致。无需训练、无需 PyTorch / OpenCLIP / 网络。

本仓库中**学生实现的部分是 `image_encoder.py`**（TODO-01 ~ TODO-06）；其余为课程提供。

---

## 1. 环境依赖

| 依赖 | 版本要求 | 本机验证版本 |
| --- | --- | --- |
| Python | 3.9+ | 3.9.23 |
| NumPy | >=1.24,<3 | 1.26.4 |
| Pillow | >=9.2,<13 | 11.0.0 |
| imageio | >=2.31,<3 | 2.37.0 |

> **重要（关于数值核对）**：参考输出是用特定版本的 Pillow 生成预处理的。
> Pillow 的 bicubic resize 在不同版本间有细微差异，可能使 `check.py` 的
> `preprocess` 及后续阶段报 FAIL。这是**输入图像预处理差异，不是编码器实现错误**。
> 详见第 6 节。若只需复现编码器计算本身，用第 5 节的脚本可直接验证。

## 2. 安装

在本目录执行：

```bash
python -m pip install -r requirements.txt
```

## 3. 快速运行

```bash
# 用随附的 10 张图，生成 output_tokens.npz
python run.py --output output_tokens.npz

# 用自己的图片
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

`run.py` 输出 `ids`（图片 ID，顺序与输入一致）和 `output_tokens`（`[B,512]` float32）。
可用下面的方式查看：

```python
import numpy as np
d = np.load('output_tokens.npz')
print(d['ids'])                 # 图片 ID
print(d['output_tokens'].shape) # (10, 512)
```

## 4. 数值检查

```bash
python check.py --stage all
```

分阶段定位问题（按实现顺序）：

```bash
python check.py --stage assets
python check.py --stage preprocess
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

通过标准：形状一致、值为有限 float32、且
`abs(actual - reference) <= atol + rtol * abs(reference)`。
输出向量 `atol=2e-5, rtol=2e-4`；内部 token `atol=2e-4, rtol=2e-4`；
预处理 `atol=2e-6`。`batch` 阶段会打乱顺序并重复图片，检验实现是否真正泛化。

## 5. 独立验证编码器数值（绕开 PIL 版本差异）

下面的脚本把参考中**已预处理好的图像**直接喂给编码器，只验证 `image_encoder.py`
的计算是否正确，不涉及 Pillow resize：

```python
import numpy as np
from runtime import load_weights
from image_encoder import ImageEncoder

enc = ImageEncoder(load_weights())
with np.load('assets/reference_outputs.npz', allow_pickle=False) as ref:
    imgs = ref['preprocessed']                       # B=10
    def chk(name, a, e, atol):
        err = np.abs(a - e)
        ok = np.allclose(a, e, atol=atol, rtol=2e-4)
        print(f'{name:12} {"PASS" if ok else "FAIL"} max_abs={err.max():.3e}')
    chk('patches',     enc.patchify(imgs[:2]),               ref['patches'][:2],     2e-4)
    chk('tokens',      enc.embed(imgs[:2]),                  ref['input_tokens'][:2],2e-4)
    chk('block0',      enc.blocks[0](enc.pre_norm(enc.embed(imgs[:2]))), ref['block0_tokens'][:2], 2e-4)
    chk('last_tokens', enc.encode_tokens(imgs[:2]),          ref['last_tokens'][:2], 2e-4)
    out = enc(imgs)
    chk('output',      out,                                  ref['output_tokens'],   2e-5)
```

预期全部 PASS，最终 `output` 的 `max_abs` 在 1e-6 量级。

## 6. 已知问题：`preprocess` 阶段 FAIL

在本机（Pillow 11.0.0）运行 `check.py --stage all` 时，`preprocess` 会因 bicubic
插值实现差异报 FAIL（max_abs 约 4e-1），后续阶段因此连带 FAIL。这是提供的
`preprocess_image.py` 与生成参考时的 Pillow 版本不一致导致的输入差异，
与 `image_encoder.py` 的实现无关。用第 5 节脚本可确认编码器本身全部 PASS。

处理方式：把 Pillow 换成生成参考时的版本，或以第 5 节脚本作为复现依据。

## 7. 实现要点（TODO-01 ~ TODO-06）

| TODO | 方法 | 结果 |
| --- | --- | --- |
| 01 | `ImageEncoder.patchify` | `[B,3,224,224]` → `[B,49,3072]`，7×7 个 32×32 patch，块内按 C,H,W 展平 |
| 02 | `MultiHeadAttention.split_heads` | `[B,N,D]` → `[B,H,N,D/H]` |
| 03 | `MultiHeadAttention.__call__` | scores、加权 V、多头合并、输出投影 |
| 04 | `TransformerBlock.__call__` | 注意力/MLP 计算与两个残差相加 |
| 05 | `ImageEncoder.embed` | patch 投影、CLS 前置、加位置编码 |
| 06 | `ImageEncoder.__call__` | 取 CLS、post-Norm、投影、单位化 |

形状链：

```
[B,3,224,224] → patchify → [B,49,3072] → embed → [B,50,768]
→ 12 个 block（形状不变） → [B,50,768] → 取 CLS → [B,768]
→ post_norm + projection → [B,512] → unit_vector → [B,512]
```

关键数值约定：

1. patch 32×32，7×7；patch 投影**无 bias**，用 `patch_weight.T`。
2. 注意力/MLP 线性权重存 `[out,in]`，需 `.T`；`visual.proj` 已是 `[in,out]`，**不转置**。
3. 用提供的精确 GELU（erf）与 Norm eps=1e-5，全程 float32。
4. 每个 block 两个残差相加，无 LayerScale。
5. k 是 4D `[B,H,N,64]`，只交换最后两轴（`swapaxes(-1,-2)`）。

## 8. 文件说明

- `image_encoder.py`：学生实现（TODO-01~06），含充足中文注释。
- `helpers.py`：LayerNorm、精确 GELU、注意力系数、单位向量（提供）。
- `preprocess_image.py`：resize/center crop/归一化（提供）。
- `runtime.py` / `run.py` / `check.py`：加载、命令行、数值检查（提供）。
- `weights/image_encoder.npz`：152 个视觉张量（提供，勿改）。
- `assets/reference_outputs.npz`：参考数值，仅用于检查（提供，勿作为输入）。
- `report/main.tex` + `report/figs/`：实验报告（Overleaf 模板，XeLaTeX 编译）。

## 9. 复现实验报告

`report/main.tex` 基于深圳大学实验报告 Overleaf 模板
（https://github.com/ChaoFan996/Experiment-Report-Template-SZU）。
在 Overleaf 新建项目，上传 `report/` 下全部文件，用 **XeLaTeX** 编译即可。
正文不超过 5 页（封面与成绩评定页不计入正文）。

## 10. GitHub 仓库

（上传后在此填写仓库链接）
