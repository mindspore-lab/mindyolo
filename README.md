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

MindYOLO implements state-of-the-art YOLO series algorithms based on MindSpore.
The following is the corresponding `mindyolo` versions and supported `mindspore` versions.

| mindyolo |  mindspore  |
| :------: | :---------: |
|  master  |   master    |
|   0.4    | 2.3.0/2.3.1 |
|   0.3    |   2.2.10    |
|   0.2    |     2.0     |
|   0.1    |     1.8     |

<img src="https://raw.githubusercontent.com/mindspore-lab/mindyolo/master/.github/000000137950.jpg" />

## Benchmark and Model Zoo

See [Benchmark Results](benchmark_results.md).

## supported model list
- [ ] YOLOv10 (welcome to contribute)
- [ ] YOLOv9 (welcome to contribute)
- [x] [YOLOv8](configs/yolov8)
- [x] [YOLOv7](configs/yolov7)
- [x] [YOLOX](configs/yolox)
- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOv4](configs/yolov4)
- [x] [YOLOv3](configs/yolov3)

## Installation

See [INSTALLATION](docs/en/installation.md) for details.

## Getting Started

See [GETTING STARTED](GETTING_STARTED.md) for details.

## Custom dataset examples

See [examples](examples)

## Notes

⚠️ The current version is based on the [static shape of GRAPH](https://mindspore.cn/docs/en/r2.0/note/static_graph_syntax_support.html). 
The dynamic shape of verision will be supported later. Please look forward to it.

### How to Contribute

We appreciate all contributions including issues and PRs to make MindYOLO better. 

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

### License

MindYOLO is released under the [Apache License 2.0](LICENSE.md).

### Acknowledgement

MindYOLO is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could support the growing research community, reimplement existing methods, and develop their own new real-time object detection methods by providing a flexible and standardized toolkit.

### Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{MindSpore Object Detection YOLO 2023,
    title={{MindSpore Object Detection YOLO}:MindSpore Object Detection YOLO Toolbox and Benchmark},
    author={MindSpore YOLO Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindyolo}},
    year={2023}
}
```
