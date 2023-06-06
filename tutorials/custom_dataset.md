# 数据集格式介绍

适用于MindYOLO的数据集格式具有如下形式：
```
            ROOT_DIR
                ├── val.txt
                ├── train.txt
                ├── annotations
                │        └── instances_val2017.json
                ├── images
                │     ├── train
                │     │     ├── 00000001.jpg
                │     │     └── 00000002.jpg
                │     └── val
                │          ├── 00006563.jpg
                │          └── 00006564.jpg
                └── labels
                      └── train
                            ├── 00000001.txt
                            └── 00000002.txt
```

其中train.txt文件每行对应单张图片的相对路径，例如：
```
./images/train/00000000.jpg
./images/train/00000001.jpg
./images/train/00000002.jpg
./images/train/00000003.jpg
./images/train/00000004.jpg
./images/train/00000005.jpg
```
train文件夹下的txt文件为相应图片的标注信息，通常每行有5列，分别对应类别id以及标注框归一化之后的中心点坐标xy和宽高wh，例如：
```
62 0.417040 0.206280 0.403600 0.412560
62 0.818810 0.197933 0.174740 0.189680
39 0.684540 0.277773 0.086240 0.358960
0 0.620220 0.725853 0.751680 0.525840
63 0.197190 0.364053 0.394380 0.669653
39 0.932330 0.226240 0.034820 0.076640
```

instances_val.json为coco格式的验证集标注，可直接调用coco api用于map的计算。

使用MindYOLO套件完成自定义数据集finetune的实际案例可参考[README.md](https://github.com/mindspore-lab/mindyolo/blob/master/examples/finetune_SHWD/README.md)