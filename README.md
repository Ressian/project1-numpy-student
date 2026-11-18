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

## 2. 安装

在本目录执行：

```bash
python -m pip install -r requirements.txt
```


## 3. 实现要点（TODO-01 ~ TODO-06）

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

## 4. 文件说明

- `image_encoder.py`：学生实现（TODO-01~06），含充足中文注释。
- `helpers.py`：LayerNorm、精确 GELU、注意力系数、单位向量（提供）。
- `preprocess_image.py`：resize/center crop/归一化（提供）。
- `runtime.py` / `run.py` / `check.py`：加载、命令行、数值检查（提供）。
- `weights/image_encoder.npz`：152 个视觉张量（提供，勿改）。
- `assets/reference_outputs.npz`：参考数值，仅用于检查（提供，勿作为输入）。
- `report/main.tex` + `report/figs/`：实验报告（Overleaf 模板，XeLaTeX 编译）。

## 5. GitHub 仓库 和 overleaf 路径

仓库地址：https://github.com/Ressian/project1-numpy-student

> 按课程要求，随附的大体积二进制文件`weights/image_encoder.npz`, `assets/reference_outputs.npz`**未提交到仓库**，
> 其余代码、图片、清单、README 与报告均已上传。
> overleaf路径
