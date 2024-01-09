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

MindYOLO is [MindSpore Lab](https://github.com/mindspore-lab)'s software toolbox that implements state-of-the-art YOLO series algorithms, [support list and benchmark](MODEL_ZOO.md). It is written in Python and powered by the [MindSpore](https://mindspore.cn/) AI framework.

The master branch supporting **MindSpore 2.2.10**.

<img src="https://raw.githubusercontent.com/mindspore-lab/mindyolo/master/.github/000000137950.jpg" />


## What is New

- 2023/09/05

1. Add YOLOv8-X segment model.
2. Dataset pipeline reconstruction(current supports seg/detect tasks).
3. Add IoU custom operators example on GPU.
4. Add distribute eval function.
5. Add fast coco eval api.
6. Tutorials and [Docs](https://mindspore-lab.github.io/mindyolo/) update(e.q. [Write a new model](https://mindspore-lab.github.io/mindyolo/zh/how_to_guides/write_a_new_model/), [Train Process Tutorial](https://mindspore-lab.github.io/mindyolo/zh/tutorials/quick_start/), ...).

## Benchmark and Model Zoo

See [MODEL ZOO](MODEL_ZOO.md).

<details open markdown>
<summary><b>Supported Algorithms</b></summary>

- [x] [YOLOv8](configs/yolov8)
- [x] [YOLOv7](configs/yolov7)
- [x] [YOLOX](configs/yolox)
- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOv4](configs/yolov4)
- [x] [YOLOv3](configs/yolov3)

</details>

## Installation

See [INSTALLATION](docs/en/installation.md) for details.

## Getting Started

See [GETTING STARTED](GETTING_STARTED.md) for details.

## Learn More about MindYOLO

To be supplemented.

## Notes

⚠️ The current version is based on the [static shape of GRAPH](https://www.mindspore.cn/docs/en/r2.2/note/static_graph_syntax_support.html). The dynamic shape of the PYNATIVE will be supported later. Please look forward to it.

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
