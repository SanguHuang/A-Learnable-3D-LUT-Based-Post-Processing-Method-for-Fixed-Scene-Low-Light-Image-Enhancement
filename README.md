# Reference-Constrained 3D LUT Post-Processing for Low-Light Image Enhancement

本仓库提供一种基于参考约束 3D LUT 的固定场景低光图像增强后处理方法。该方法以低光增强模型的输出图像作为输入，以正常曝光图像作为参考，通过优化 3D LUT 参数学习增强图像到参考图像之间的非线性颜色映射关系，从而进一步改善低光增强结果中的亮度偏差、颜色失真、饱和度异常和视觉风格不一致等问题。

该方法不改变前级低光增强模型的参数，可作为 Zero-DCE、HVI-CIDNet、Retinexformer 等低光增强方法之后的轻量化后处理模块。推理阶段不需要参考图像，只需使用训练得到的 3D LUT 对增强结果进行颜色映射。

## Features

* 基于参考图像约束的 3D LUT 后处理方法
* 支持固定场景低光增强结果的亮度和颜色校正
* 不依赖特定前级低光增强模型
* 提供两版 LUT 训练代码，适用于不同应用需求
* 支持生成 256³ 预计算查找表，便于边缘端快速部署
* LUT 应用过程可通过 C++ 查表方式实现，具有较好的平台通用性

## File Structure

```text
.
├── README.md
├── build_256_lut_table.py
├── fit_lut_v1.py
├── fit_lut_v2.py
├── lut3d.py
└── requirements.txt
```

## File Description

### `fit_lut_v1.py`

`fit_lut_v1.py` 是画面拟合性能更强的 LUT 训练版本。该版本更加侧重输出图像与参考图像之间的像素级一致性和亮度恢复能力，适用于以下场景：

* 前级低光增强结果质量较好；
* 图像噪声较小；
* 画面内容相对稳定；
* 主要目标是提升 PSNR、SSIM 等客观评价指标；
* 在公开数据集上进行定量实验对比。

该版本能够更充分地拟合参考图像的亮度和颜色分布，但在极端低照度、噪声较大或画面变化剧烈的场景下，可能出现局部过曝或颜色映射不稳定现象。

### `fit_lut_v2.py`

`fit_lut_v2.py` 是鲁棒性更强的保守训练版本。该版本在训练过程中加入了更多稳定性约束，对亮度提升和颜色映射更加克制，适用于以下场景：

* 前级增强结果噪声较大；
* 低照度程度较强；
* 画面中存在人员移动或物体运动；
* 对连续帧稳定性要求较高；
* 更关注视觉稳定性而不是单纯追求最高客观指标。

该版本输出结果可能相对偏暗，但能够更好地抑制高亮区域过曝、平坦区域色斑、颜色漂移以及动态画面中的映射不稳定问题。

### `lut3d.py`

`lut3d.py` 定义了可学习 3D LUT 模块及其三线性插值映射过程。训练脚本会调用该模块完成 LUT 参数优化和图像映射。

### `build_256_lut_table.py`

`build_256_lut_table.py` 用于将训练得到的 33³ LUT 预计算为 256³ 查找表。对于 8-bit RGB 图像，每个通道共有 0～255 共 256 个离散取值，因此可以提前计算所有 RGB 输入组合对应的输出值。

生成 256³ 查找表后，边缘端推理阶段不需要再执行逐像素三线性插值，只需根据输入像素的 R、G、B 数值直接查表，从而降低计算开销。

### `requirements.txt`

`requirements.txt` 记录运行代码所需的 Python 依赖包。

## Installation

建议先创建独立的 Python 环境：

```bash
conda create -n lut_post python=3.8
conda activate lut_post
```

安装依赖：

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train a LUT

使用画面拟合性能更强的版本：

```bash
python fit_lut_v1.py
```

使用鲁棒性更强的保守版本：

```bash
python fit_lut_v2.py
```

具体输入路径、输出路径、训练步数、学习率、LUT 尺寸等参数请根据脚本中的参数设置进行修改。

### 2. Build a 256³ LUT table

训练完成后，可以使用以下脚本生成 256³ 预计算查找表：

```bash
python build_256_lut_table.py
```

该查找表可进一步用于 C++ 或其他边缘端程序中，实现快速逐像素 LUT 映射。

## Recommended Version Selection

| Scenario         | Recommended script |
| ---------------- | ------------------ |
| 公开数据集定量实验        | `fit_lut_v1.py`    |
| 前级增强结果质量较好       | `fit_lut_v1.py`    |
| 追求更高 PSNR / SSIM | `fit_lut_v1.py`    |
| 极端低照度场景          | `fit_lut_v2.py`    |
| 前级增强噪声较大         | `fit_lut_v2.py`    |
| 有人或物体移动的真实场景     | `fit_lut_v2.py`    |
| 更关注连续帧稳定性        | `fit_lut_v2.py`    |

## Application Scenarios

该方法适用于固定机位或背景相对稳定的真实低光场景，例如：

* 厂房监控；
* 走廊监控；
* 室内固定摄像头；
* 夜间固定场景拍摄；
* 边缘端视觉系统中的低光增强后处理。

## Notes

1. 本方法是低光增强后的后处理模块，不包含前级低光增强网络本身。
2. 训练阶段需要正常曝光参考图像参与约束优化。
3. 推理阶段不需要参考图像，只需要使用训练得到的 LUT。
4. 若场景变化剧烈或前级增强结果噪声明显，建议优先使用 `fit_lut_v2.py`。
5. 若主要目标是公开数据集上的定量指标提升，建议优先使用 `fit_lut_v1.py`。

## Citation

If this project is helpful for your research, please cite this repository or the related paper when available.

```text
@misc{reference_constrained_3dlut_lowlight,
  title  = {Reference-Constrained 3D LUT Post-Processing for Fixed-Scene Low-Light Image Enhancement},
  author = {Huang, Yixiang},
  year   = {2026},
  note   = {GitHub repository}
}
```

