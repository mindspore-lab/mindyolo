---
hide:
  - navigation
---
# MindYOLO

<p align="left">
    <a href="https://github.com/mindspore-lab/mindyolo/blob/master/README.md">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="https://github.com/mindspore-lab/mindyolo/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/mindspore-lab/mindyolo.svg">
    </a>
    <a href="https://github.com/mindspore-lab/mindyolo/pulls">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
</p>

MindYOLO是[MindSpore Lab](https://github.com/mindspore-lab)开发的AI套件，实现了最先进的YOLO系列算法，[查看支持的模型算法](MODEL_ZOO.md)。

MindYOLO使用Python语言编写，基于[MindSpore](https://mindspore.cn/)深度学习框架开发，适用于**MindSpore 1.8.1**。


<img src=".github/000000137950.jpg" />


## 新特性 

- 2023/03/30
1. 目前版本支持的模型包括YOLOv3/YOLOv5/YOLOv7的基本规格。
2. 模型可以导出为MindIR/AIR格式进行部署。
3. ⚠️ 当前版本基于GRAPH的静态Shape。后续将添加PYNATIVE的动态Shape支持，敬请期待。
4. ⚠️ 当前版本仅支持Ascend平台，GPU平台将在后续版本中支持。


## 基准和模型仓库 

查看 [MODEL ZOO](MODEL_ZOO.md).

<details open>
<summary><b>支持的算法</b></summary>

- [x] [YOLOv8](configs/yolov8)
- [x] [YOLOv7](configs/yolov7)
- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOv3](configs/yolov3)
- [x] [YOLOv4](configs/yolov4)
- [x] [YOLOX](configs/yolox)
- [ ] [YOLOv6](configs/yolov6)


</details>

## 安装

### 依赖

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- openmpi 4.0.3 (for distributed mode)

安装这些依赖，请执行以下命令
```shell
pip install -r requirements.txt
```

假定你已安装所需依赖，可以按照[官方说明](https://www.mindspore.cn/install)轻松安装MindSpore，你可以在其中选择最适合的硬件平台。要在分布式模式下运行，需要安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/)。 

## 快速入门

查看 [GETTING STARTED](GETTING_STARTED_CN.md)

## 了解 MindYOLO 的更多信息

敬请期待

## 注意
### 贡献方式

我们感谢开发者用户的所有贡献，包括提issue和PR，一起让MindYOLO变得更好。

贡献指南请参考[CONTRIBUTING.md](CONTRIBUTING.md)。


### 许可证

MindYOLO遵循[Apache License 2.0](LICENSE.md)开源协议。


### 致谢

MindYOLO是一个欢迎任何贡献和反馈的开源项目。我们希望通过提供灵活且标准化的工具包来重新实现现有方法和开发新的实时目标检测方法，从而为不断发展的研究社区服务。

### 引用

如果你觉得MindYOLO对你的项目有帮助，请考虑引用：

```latex
@misc{MindSpore Object Detection YOLO 2023,
    title={{MindSpore Object Detection YOLO}:MindSpore Object Detection YOLO Toolbox and Benchmark},
    author={MindSpore YOLO Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindyolo}},
    year={2023}
}
```
