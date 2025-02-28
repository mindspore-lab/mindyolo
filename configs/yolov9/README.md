# YOLOv9

## Abstract
 Today’s deep learning methods focus on how to design
 the most appropriate objective functions so that the pre
diction results of the model can be closest to the ground
 truth. Meanwhile, an appropriate architecture that can
 facilitate acquisition of enough information for prediction
 has to be designed. Existing methods ignore a fact that
 when input data undergoes layer-by-layer feature extrac
tion and spatial transformation, large amount of informa
tion will be lost. This paper will delve into the important is
sues of data loss when data is transmitted through deep net
works, namely information bottleneck and reversible func
tions. We proposed the concept of programmable gradi
ent information (PGI) to cope with the various changes
 required by deep networks to achieve multiple objectives.
 PGI can provide complete input information for the tar
get task to calculate objective function, so that reliable
 gradient information can be obtained to update network
 weights. In addition, a new lightweight network architec
ture– Generalized Efficient Layer Aggregation Network
 (GELAN), based on gradient path planning is designed.
 GELAN’s architecture confirms that PGI has gained su
perior results on lightweight models. We verified the pro
posed GELAN and PGI on MS COCO dataset based ob
ject detection. The results show that GELAN only uses
 conventional convolution operators to achieve better pa
rameter utilization than the state-of-the-art methods devel
oped based on depth-wise convolution. PGI can be used
 for variety of models from lightweight to large. It can be
 used to obtain complete information, so that train-from
scratch models can achieve better results than state-of-the
art models pre-trained using large datasets.

<div align=center>
<img width="416" alt="35e6a70183e7389f4bef728c88db90b" src="https://github.com/user-attachments/assets/f78a9032-b032-44ca-9ef8-b624750603d4">
</div>

## Requirements

| mindspore | ascend driver | firmware     | cann toolkit/kernel |
| :-------: | :-----------: | :----------: |:-------------------:|
|   2.5.0   |    24.1.0     | 7.5.0.3.220  |     8.0.0.beta1     |

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Training

<details open>
<summary><b>View More</b></summary>

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple Ascend devices
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov9_log python train.py --config ./configs/yolov9/yolov9-t.yaml --device_target Ascend --is_parallel True
```

**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/docs/en/r2.5.0/model_train/parallel/msrun_launcher.html).

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/Ascend device
python train.py --config ./configs/yolov9/yolov9-t.yaml --device_target Ascend
```

</details>

### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov9/yolov9-t.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt --ms_amp_level O2
```

## Performance


### Detection


Experiments are tested on Ascend 910* with mindspore 2.5.0 graph mode.

|  model name  |  scale  | cards  | batch size | resolution |  jit level  | graph compile | ms/step | img/s  |  map  |          recipe              |                                                       weight                                                       |
|  :--------:  |  :---:  |  :---: |   :---:    |   :---:    |    :---:    |     :---:     |  :---:  |  :---: |:-----:|          :---:               |:------------------------------------------------------------------------------------------------------------------:|
|    YOLOv9    |    T    |    8   |     16     |  640x640   |     O2      |    1316.4s    | 350 | 365.71 | 37.3% |    [yaml](./yolov9-t.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9t_500e_MAP373-c0ee5cbc.ckpt) |
|    YOLOv9    |    S    |    8   |     16     |  640x640   |     O2      |    1337.1s    | 377 | 339.52 | 46.3% |    [yaml](./yolov9-s.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9s_500e_MAP463-b3cb691d.ckpt) |
|    YOLOv9    |    M    |    8   |     16     |  640x640   |     O2      |    897.32s    | 499 | 256.51 | 51.4% |    [yaml](./yolov9-m.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9m_500e_MAP514-86aa8761.ckpt) |
|    YOLOv9    |    C    |    8   |     16     |  640x640   |     O2      |    1017.9s    | 627 | 204.15 | 52.6% |    [yaml](./yolov9-c.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9c_500e_MAP526-ff7bdf68.ckpt) |
|    YOLOv9    |    E    |    8   |     16     |  640x640   |     O2      |    1927.8s    | 826 | 154.96 | 55.1% |    [yaml](./yolov9-e.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9e_500e_MAP551-6b55c121.ckpt) |




### Notes

- map: Accuracy reported on the validation set.
- We refer to the official [YOLOV9](https://github.com/WongKinYiu/yolov9) to reproduce the P5 series model.

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. YOLOv9: Learning What You Want to Learn
 Using Programmable Gradient Information. arXiv preprint arXiv:2402.13616v2, 2024.
