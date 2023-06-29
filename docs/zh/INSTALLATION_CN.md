---
hide:
  - navigation
  - toc
---

# 安装

## 1. 将MindYOLO作为依赖包安装
运行:
```shell
pip install mindyolo -i https://pypi.tuna.tsinghua.edu.cn/simple
```
或
```shell
pip install git+https://github.com/mindspore-lab/mindyolo.git
```

## 2. 将MindYOLO以源码安装
**请注意在该步骤和步骤1之间选择其一**

运行:
```shell
git clone https://github.com/mindspore-lab/mindyolo.git
cd mindyolo
pip install -r requirements.txt
```

推荐编译快速coco api扩展模块来加速coco数据评估过程，**请注意该编译过程是可选的**：
```shell
cd $project_root/mindyolo/csrc
sh build.sh
```

## 3. 安装MindSpore

按照[官方说明](https://www.mindspore.cn/install)轻松安装MindSpore，你可以在其中选择最适合的硬件平台。要在分布式模式下运行，需要安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/)。 

⚠️ 当前版本仅支持Ascend平台，GPU平台将在后续版本中支持。
