---
hide:
  - navigation
  - toc
---

# Installation

## 1. Install MindYOLO as a package
Please run:
```shell
pip install mindyolo -i https://pypi.tuna.tsinghua.edu.cn/simple
```
or
```shell
pip install git+https://github.com/mindspore-lab/mindyolo.git
```

## 2. Install MindYOLO by source code
**Note to select either this step or step 1 to install MindYOLO**

Please run:
```shell
git clone https://github.com/mindspore-lab/mindyolo.git
cd mindyolo
pip install -r requirements.txt
```

It is suggested to build fast coco api extension for better performance in evaluating coco data. **Note that this behavior is optional**:
```shell
cd $project_root/mindyolo/csrc
sh build.sh
```

## 3. Install MindSpore

MindSpore can be easily installed by following the official [instructions](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.

⚠️ The current version only supports the Ascend platform, and the GPU platform will be supported later.
