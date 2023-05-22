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

MindYOLO is [MindSpore Lab](https://github.com/mindspore-lab)'s software system that implements state-of-the-art YOLO series algorithms, [support list and benchmark](MODEL_ZOO.md). It is written in Python and powered by the [MindSpore](https://mindspore.cn/) deep learning framework.

The master branch works with **MindSpore 1.8.1**.

<img src=".github/000000137950.jpg" />


## What is New 
- 2023/03/30
1. Currently, the models supported by the first release include the basic specifications of YOLOv3/YOLOv5/YOLOv7;
2. Models can be exported to MindIR/AIR format for deployment.
3. ⚠️ The current version is based on the static shape of GRAPH. The dynamic shape of the PYNATIVE will be added later. Please look forward to it.
4. ⚠️ The current version only supports the Ascend platform, and the GPU platform will support it later.

## Benchmark and Model Zoo

Seed [MODEL ZOO](MODEL_ZOO.md).

<details open>
<summary><b>Supported Algorithms</b></summary>

- [x] [YOLOv8](configs/yolov8)
- [x] [YOLOv7](configs/yolov7)
- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOv3](configs/yolov3)
- [x] [YOLOv4](configs/yolov4)
- [x] [YOLOX](configs/yolox)
- [ ] [YOLOv6](configs/yolov6)


</details>

## Installation

### Dependency

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- openmpi 4.0.3 (for distributed mode)

To install the dependency, please run
```shell
pip install -r requirements.txt
```

MindSpore can be easily installed by following the official [instructions](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.

The following instructions assume the desired dependency is fulfilled.


## Getting Started

See [GETTING STARTED](GETTING_STARTED.md)

## Learn More about MindYOLO

To be supplemented.

## Notes
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
