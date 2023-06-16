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

The master branch supporting **MindSpore 2.0**.

<img src="https://raw.githubusercontent.com/mindspore-lab/mindyolo/master/.github/000000137950.jpg" />


## What is New

- 2023/06/15

1. Support YOLOv3/v4/v5/X/v7/v8 6 models and release 23 corresponding weights, see [MODEL ZOO](MODEL_ZOO.md) for details.
2. Support MindSpore 2.0.
3. Support deployment on MindSpore lite 2.0.
4. New online documents are available!

## Benchmark and Model Zoo

See [MODEL ZOO](MODEL_ZOO.md).

<details open>
<summary><b>Supported Algorithms</b></summary>

- [x] [YOLOv8](configs/yolov8)
- [x] [YOLOv7](configs/yolov7)
- [x] [YOLOX](configs/yolox)
- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOv4](configs/yolov4)
- [x] [YOLOv3](configs/yolov3)

</details>

## Installation

### Dependency

- mindspore >= 2.0
- numpy >= 1.17.0
- pyyaml >= 5.3
- openmpi 4.0.3 (for distributed mode)

To install the dependency, please run

```shell
pip install -r requirements.txt
```

MindSpore can be easily installed by following the official [instructions](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.

⚠️ The current version only supports the Ascend platform, and the GPU platform will be supported later.

## Getting Started

See [GETTING STARTED](GETTING_STARTED.md)

## Learn More about MindYOLO

To be supplemented.

## Notes

⚠️ The current version is based on the static shape of GRAPH. The dynamic shape of the PYNATIVE will be supported later. Please look forward to it.

### How to Contribute

We appreciate all contributions including issues and PRs to make MindYOLO better. 

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

### License

MindYOLO is released under the [Apache License 2.0](LICENSE.md).

### Acknowledgement

MindYOLO is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new realtime object detection methods.

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
