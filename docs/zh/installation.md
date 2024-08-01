---
hide:
  - navigation
---

# 安装

## 依赖

- mindspore >= 2.3
- numpy >= 1.17.0
- pyyaml >= 5.3
- openmpi 4.0.3 (分布式训练所需)

如需安装`python`相关库依赖，只需运行：

```shell
pip install -r requirements.txt
```

如需安装MindSpore，你可以通过遵循[官方指引](https://www.mindspore.cn/install)，在不同的硬件平台上获得最优的安装体验。 为了在分布式模式下运行，您还需要安装[OpenMPI](https://www.open-mpi.org/software/ompi/v4.0/)。

⚠️ 当前版本仅支持Ascend平台，GPU会在后续支持，敬请期待。


## PyPI源安装

MindYOLO 现已发布为一个`Python包`并能够通过`pip`进行安装。我们推荐您在`虚拟环境`安装使用。 打开终端，输入以下指令来安装 MindYOLO:

```shell
pip install mindyolo
```

## 源码安装 (未经测试版本)

### 通过VSC安装

```shell
pip install git+https://github.com/mindspore-lab/mindyolo.git
```

### 通过本地src安装

由于本项目处于活跃开发阶段，如果您是开发者或者贡献者，请优先选择此安装方式。

MindYOLO 可以在由 `GitHub` 克隆仓库到本地文件夹后直接使用。 这对于想使用最新版本的开发者十分方便:

```shell
git clone https://github.com/mindspore-lab/mindyolo.git
```

在克隆到本地之后，推荐您使用"可编辑"模式进行安装，这有助于解决潜在的模块导入问题。

```shell
cd mindyolo
pip install -e .
```

我们提供了一个可选的 [fast coco api](https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/fast_eval_api.py) 接口用于提升验证过程的速度。代码是以C++形式提供的，可以尝试用以下的命令进行安装 **(此操作是可选的)** :

```shell
cd mindyolo/csrc
sh build.sh
```

我们还提供了基于MindSpore [Custom自定义算子](https://www.mindspore.cn/tutorials/experts/zh-CN/master/operation/op_custom.html) 的GPU融合算子，用于提升训练过程的速度。代码采用C++和CUDA开发，位于`examples/custom_gpu_op/`路径下。您可参考示例脚本`examples/custom_gpu_op/iou_loss_fused.py`，修改`mindyolo/models/losses/iou_loss.py`的`bbox_iou`方法，在GPU训练过程中使用该特性。运行`iou_loss_fused.py`前，需要使用以下的命令，编译生成GPU融合算子运行所依赖的动态库 **(此操作并非必需)** :

```shell
bash examples/custom_gpu_op/fused_op/build.sh
```
