## Getting Started with MindYOLO

This document provides a brief intro of the usage of builtin command-line tools in MindYOLO.

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from
  [model zoo](MODEL_ZOO.md),
  for example, `./configs/yolov7/yolov7.yaml`.
2. Download the corresponding pretrain checkpoint from the link in the README of each model.
3. We provide `infer.py` that is able to demo builtin configs. Run it with:

```
python infer.py \
  --config ./configs/yolov7/yolov7.yaml \
  --device_target=Ascend \
  --weight=MODEL.WEIGHTS \
  --image_path /PATH/TO/IMAGE.jpg
```

The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for inference.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `infer.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run on cpu, modify device_target to CPU.
* To save outputs to a directory (for images), use --output_img.

### Training & Evaluation in Command Line

We provide two scripts in "train.py", "test.py" and "tools/run_**.sh",
that are made to train all the configs provided in MindYOLO. You may want to
use it as a reference to write your own training script.

Compared to 'train.py', 'test.py', 'run_train.sh' and 'run_test.sh' can provide simpler 
parallel training and input parameters.

To train a model with "train.py" / "run_train.sh", first
setup the corresponding datasets following
[configs/data/README.md](configs/data/README.md),
then run:

```
bash ./tools/run_train.sh [CONFIG_PATH] [DEVICE_TRAGET] [DEVICE_NUM] [DEVICE_ID|RANK_TABLE_FILE|CUDA_VISIBLE_DEVICES]
e.g.:
bash ./tools/run_train.sh ./configs/yolov7/yolov7.yaml Ascend 8 ./rank_table.json
bash ./tools/run_train.sh ./configs/yolov7/yolov7.yaml GPU 8 0,1,2,3,4,5,6,7

OR

mpirun --allow-run-as-root -n 8 python train.py
e.g.:
mpirun --allow-run-as-root -n 8 python train.py \
   --config ./configs/yolov7/yolov7.yaml \
   --device_target GPU \
   --is_parallel True > log.txt 2>&1 &
tail -f ./log.txt
```

The configs are made for 8 NPU/GPU training.
To train on 1 NPU/GPU, you may need to change some parameters, e.g.:

```
python train.py \
    --config ./configs/yolov7/yolov7.yaml \
    --device_target Ascend \
    --per_batch_size 16
```

To evaluate a model's performance, use

```
bash run_test.sh [CONFIG_PATH] [DEVICE_TRAGET] [DEVICE_ID|CUDA_VISIBLE_DEVICES] [WEIGHT]
e.g.:
bash run_test.sh ./configs/yolov7/yolov7.yaml Ascend 0 MODEL.WEIGHTS
bash run_test.sh ./configs/yolov7/yolov7.yaml GPU 0 MODEL.WEIGHTS

OR

python test.py \
    --config ./configs/yolov7/yolov7.yaml \
    --device_target Ascend \
    --weight=MODEL.WEIGHTS
```

For more options, see `train.py -h` or `test.py -h`.


### Use MindYOLO APIs in Your Code

To be supplemented.
