---
hide:
  - navigation
---

# 安装

## 依赖

- mindspore >= 2.0
- numpy >= 1.17.0
- pyyaml >= 5.3
- openmpi 4.0.3 (分布式训练所需)

为了安装`python`相关库依赖，只需运行：

```shell
pip install -r requirements.txt
```

MindSpore可以通过遵循[官方指引](https://www.mindspore.cn/install)，在不同的硬件平台上获得最优的安装体验。 为了在分布式模式下运行，您还需要安装[OpenMPI](https://www.open-mpi.org/software/ompi/v4.0/)。

⚠️ 当前版本仅支持Ascend平台，GPU会在后续支持，敬请期待。


## PyPI源安装

MindYOLO 发布为一个`Python包`并能够通过`pip`进行安装。我们推荐您在`虚拟环境`安装使用。 打开终端，输入以下指令来安装 MindYOLO:

```shell
pip install mindyolo
```

## 源码安装 (未经测试版本)

### from VSC

```shell
pip install git+https://github.com/mindspore-lab/mindyolo.git
```

### from local src

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

另外, 我们提供了一个可选的 [fast coco api](https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/fast_eval_api.py) 接口用于提升验证过程的速度。代码是以C++形式提供的，可以尝试用以下的命令进行安装 **(此操作是可选的)** :

```shell
cd mindyolo/csrc
sh build.sh
```
