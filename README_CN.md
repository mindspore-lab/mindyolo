# MindYOLO

<p align="left">
    <a href="https://github.com/mindspore-lab/mindyolo/blob/master/README.md">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="https://github.com/mindspore-lab/mindyolo/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/mindspore-lab/mindcv.svg">
    </a>
    <a href="https://github.com/mindspore-lab/mindyolo/pulls">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
</p>

MindYOLO是[MindSpore Lab](https://github.com/mindspore-lab)开发的AI套件，实现了最先进的YOLO系列算法，[查看支持的模型算法](MODEL_ZOO.md)。

MindYOLO使用Python语言编写，基于 [MindSpore](https://mindspore.cn/) AI框架开发。

master 分支配套 **MindSpore 2.0**。

<img src="https://raw.githubusercontent.com/mindspore-lab/mindyolo/master/.github/000000137950.jpg" />


## 新特性 

- 2023/06/15

1. 支持 YOLOv3/v4/v5/v7/v8/X 等6个模型，发布了23个模型weights，详情请参考 [MODEL ZOO](MODEL_ZOO.md)。
2. 配套 MindSpore 2.0。
3. 支持 MindSpore lite 2.0 推理。
4. 新的教程文档上线！

## 基准和模型仓库 

查看 [MODEL ZOO](MODEL_ZOO.md).

<details open markdown>
<summary><b>支持的算法</b></summary>

- [x] [YOLOv8](configs/yolov8)
- [x] [YOLOv7](configs/yolov7)
- [x] [YOLOX](configs/yolox)
- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOv4](configs/yolov4)
- [x] [YOLOv3](configs/yolov3)

</details>

## 安装

查看 [INSTALLATION](docs/zh/installation.md)

## 快速入门

查看 [GETTING STARTED](GETTING_STARTED_CN.md)

## 了解 MindYOLO 的更多信息

敬请期待

## 注意

⚠️当前版本基于GRAPH的静态Shape。后续将添加PYNATIVE的动态Shape支持，敬请期待。

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
