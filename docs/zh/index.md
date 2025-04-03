---
hide:
  - navigation
  - toc
---


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

MindYOLO基于mindspore实现了最新的YOLO系列算法。以下是mindyolo的分支与mindspore版本的对应关系：

|  mindyolo  |  mindspore  |
|    :--:    |     :--:    |
|   master   |    master   |
|   0.5    |    2.5.0    |
|    0.4     |    2.3.0/2.3.1    |
|    0.3     |    2.2.10   |
|    0.2     |    2.0      |
|    0.1     |    1.8      |

<img src="https://raw.githubusercontent.com/mindspore-lab/mindyolo/master/.github/000000137950.jpg" />

## 模型仓库和基准

详见 [模型仓库表格](modelzoo/benchmark.md)

## 支持模型列表
- [ ] YOLOv10 (欢迎开源贡献者参与开发)
- [ ] YOLOv9 (欢迎开源贡献者参与开发)
- [x] [YOLOv8](modelzoo/yolov8.md)
- [x] [YOLOv7](modelzoo/yolov7.md)
- [x] [YOLOX](modelzoo/yolox.md)
- [x] [YOLOv5](modelzoo/yolov5.md)
- [x] [YOLOv4](modelzoo/yolov4.md)
- [x] [YOLOv3](modelzoo/yolov3.md)

## 安装

详见 [安装](installation.md)

## 快速开始

详见 [快速开始](tutorials/quick_start.md)

## 说明

⚠️ 当前版本基于 [图模式静态shape](https://mindspore.cn/docs/en/r2.0/note/static_graph_syntax_support.html)开发。
动态shape将在后续支持，敬请期待。

### 参与项目

为了让mindyolo更加完善和丰富，我们欢迎包括issue和pr在内的任何开源贡献。

请参考 [参与项目](notes/contributing.md) 获取提供开源贡献的相关指导。

### 许可

MindYOLO基于 [Apache License 2.0](https://github.com/mindspore-lab/mindyolo/blob/master/LICENSE.md) 发布。

### 须知

MindYOLO 是一个开源项目，我们欢迎任何贡献和反馈。我们希望该mindyolo能够通过提供灵活且标准化的工具包来支持不断壮大的研究社区，重现现有方法，并开发自己的新实时对象检测方法。

### 引用

如果您发现该项目对您的研究有用，请考虑引用：

```latex
@misc{MindSpore Object Detection YOLO 2023,
    title={{MindSpore Object Detection YOLO}:MindSpore Object Detection YOLO Toolbox and Benchmark},
    author={MindSpore YOLO Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindyolo}},
    year={2023}
}
```
