import argparse
import ast
import contextlib
import json
import os
import time
from typing import Union

import yaml
from pathlib import Path
import numpy as np
from mindspore.communication import init, get_rank, get_group_size
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mindspore as ms
from mindspore import Tensor, context, nn, ParallelMode

from mindyolo.data import COCO80_TO_COCO91_CLASS, COCODataset, create_loader
from mindyolo.models.model_factory import create_model
from mindyolo.utils import logger, get_logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh
from mindyolo.utils.utils import set_seed, get_broadcast_datetime, Synchronizer


def get_parser_test(parents=None):
    parser = argparse.ArgumentParser(description="Test", parents=[parents] if parents else [])
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="train mode, graph/pynative")
    parser.add_argument("--ms_amp_level", type=str, default="O0", help="amp level, O0/O1/O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument("--weight", type=str, default="yolov7_300.ckpt", help="model.ckpt path(s)")
    parser.add_argument("--per_batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )
    parser.add_argument("--rect", type=ast.literal_eval, default=False, help="rectangular training")
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="IOU threshold for NMS")
    parser.add_argument(
        "--conf_free", type=ast.literal_eval, default=False, help="Whether the prediction result include conf"
    )
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="save dir")
    parser.add_argument("--save_dir", type=str, default="./runs_test", help="save dir")

    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument("--data_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--ckpt_url", type=str, default="", help="ModelArts: obs path to checkpoint folder")
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument(
        "--data_dir", type=str, default="/cache/data/", help="ModelArts: local device path to dataset folder"
    )
    parser.add_argument("--is_parallel", type=ast.literal_eval, default=False, help="Distribute test or not")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/cache/pretrain_ckpt/",
        help="ModelArts: local device path to checkpoint folder",
    )
    return parser


def set_default_test(args):
    # Set Context
    context.set_context(mode=args.ms_mode, device_target=args.device_target, max_call_depth=2000)
    if args.device_target == "Ascend":
        context.set_context(device_id=int(os.getenv("DEVICE_ID", 0)))
    elif args.device_target == "GPU" and args.ms_enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)
    # Set Parallel
    if args.is_parallel:
        init()
        args.rank, args.rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(device_num=args.rank_size, parallel_mode=parallel_mode)
    else:
        args.rank, args.rank_size = 0, 1
    # Set Data
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # number of classes
    args.data.names = ["item"] if args.single_cls and len(args.names) != 1 else args.data.names  # class names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )
    # Directories and Save run settings
    time = get_broadcast_datetime(rank_size=args.rank_size)
    args.save_dir = os.path.join(
        args.save_dir, f'{time[0]:04d}.{time[1]:02d}.{time[2]:02d}-{time[3]:02d}.{time[4]:02d}.{time[5]:02d}')
    os.makedirs(args.save_dir, exist_ok=True)
    if args.rank % args.rank_size == 0:
        with open(os.path.join(args.save_dir, "cfg.yaml"), "w") as f:
            yaml.dump(vars(args), f, sort_keys=False)
    # Set Logger
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=args.rank, device_per_servers=args.rank_size)
    logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))
    # Modelarts: Copy data, from the s3 bucket to the computing node; Reset dataset dir.
    if args.enable_modelarts:
        from mindyolo.utils.modelarts import sync_data

        os.makedirs(args.data_dir, exist_ok=True)
        sync_data(args.data_url, args.data_dir)
        sync_data(args.save_dir, args.train_url)
        if args.ckpt_url:
            sync_data(args.ckpt_url, args.ckpt_dir)  # pretrain ckpt
        # args.data.dataset_dir = os.path.join(args.data_dir, args.data.dataset_dir)
        args.data.val_set = os.path.join(args.data_dir, args.data.val_set)
        args.data.test_set = os.path.join(args.data_dir, args.data.test_set)
        args.weight = args.ckpt_dir if args.ckpt_dir else ""


def test(
    network: nn.Cell,
    dataloader: ms.dataset.Dataset,
    anno_json_path: str,
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
    conf_free: bool = False,
    nms_time_limit: float = -1.0,
    is_coco_dataset: bool = True,
    imgIds: list = [],
    per_batch_size: int = -1,
    rank: int = 0,
    rank_size: int = 1,
    save_dir: str = '',
    synchronizer: Synchronizer = None,
    cur_epoch: Union[str, int] = 0,  # to distinguish saving directory from different epoch in eval while run mode
):
    steps_per_epoch = dataloader.get_dataset_size()
    loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    coco91class = COCO80_TO_COCO91_CLASS

    sample_num = 0
    infer_times = 0.0
    nms_times = 0.0
    result_dicts = []

    for i, data in enumerate(loader):
        imgs, _, paths, ori_shape, pad, hw_scale = (
            data["image"],
            data["labels"],
            data["img_files"],
            data["hw_ori"],
            data["pad"],
            data["hw_scale"],
        )
        nb, _, height, width = imgs.shape
        imgs = Tensor(imgs, ms.float32)

        # Run infer
        _t = time.time()
        out, _ = network(imgs)  # inference and training outputs
        infer_times += time.time() - _t

        # Run NMS
        t = time.time()
        out = out.asnumpy()
        out = non_max_suppression(
            out,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            conf_free=conf_free,
            multi_label=True,
            time_limit=nms_time_limit,
        )
        nms_times += time.time() - t

        # Statistics pred
        for si, pred in enumerate(out):
            path = Path(str(paths[si]))
            sample_num += 1
            if len(pred) == 0:
                continue

            # Predictions
            predn = np.copy(pred)
            scale_coords(
                imgs[si].shape[1:], predn[:, :4], ori_shape[si], ratio=hw_scale[si], pad=pad[si]
            )  # native-space pred

            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                result_dicts.append(
                    {
                        "image_id": image_id,
                        "category_id": coco91class[int(p[5])] if is_coco_dataset else int(p[5]),
                        "bbox": [round(x, 3) for x in b],
                        "score": round(p[4], 5),
                    }
                )
        logger.info(f"Sample {steps_per_epoch}/{i + 1}, time cost: {(time.time() - _t) * 1000:.2f} ms.")

    # save and load result file for distributed case
    if rank_size > 1:
        # save result to file
        # each epoch has a unique directory in eval while run mode
        infer_dir = os.path.join(save_dir, 'infer', str(cur_epoch))
        os.makedirs(infer_dir, exist_ok=True)
        infer_path = os.path.join(infer_dir, f'det_result_rank{rank}_{rank_size}.json')
        with open(infer_path, 'w') as f:
            json.dump(result_dicts, f)
        # synchronize
        assert synchronizer is not None
        synchronizer()

        # load file to result_dicts
        f_names = os.listdir(infer_dir)
        f_paths = [os.path.join(infer_dir, f) for f in f_names]
        logger.info(f"Loading {len(f_names)} eval file from directory {infer_dir}: {sorted(f_names)}.")
        assert len(f_names) == rank_size, f'number of eval file({len(f_names)}) should be equal to rank size({rank_size})'
        result_dicts = []
        for path in f_paths:
            with open(path, 'r') as fp:
                result_dicts += json.load(fp)

    # Compute mAP
    if not result_dicts:
        logger.warning(f'Got 0 bbox after NMS, skip computing map')
        map, map50 = 0.0, 0.0
    else:
        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            with contextlib.redirect_stdout(get_logger()):  # redirect stdout to logger
                anno = COCO(anno_json_path)  # init annotations api
                pred = anno.loadRes(result_dicts)  # init predictions api
                eval = COCOeval(anno, pred, "bbox")
                if is_coco_dataset:
                    eval.params.imgIds = imgIds
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            logger.error(f"pycocotools unable to run: {e}")
            raise e

    t = tuple(x / sample_num * 1E3 for x in (infer_times, nms_times, infer_times + nms_times)) + \
        (height, width, per_batch_size)  # tuple
    logger.info(f'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;' % t)

    return map, map50


def main(args):
    # Init
    s_time = time.time()
    set_seed(args.seed)
    set_default_test(args)
    logger.info(f"parse_args:\n{args}")

    # Create Network
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=False,
        checkpoint_path=args.weight,
    )
    network.set_train(False)
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)

    # Create Dataloader
    dataset_path = args.data.val_set
    is_coco_dataset = "coco" in args.data.dataset_name
    dataset = COCODataset(
        dataset_path=dataset_path,
        img_size=args.img_size,
        transforms_dict=args.data.test_transforms,
        is_training=False,
        augment=False,
        rect=args.rect,
        single_cls=args.single_cls,
        batch_size=args.per_batch_size,
        stride=max(args.network.stride),
    )
    dataloader = create_loader(
        dataset=dataset,
        batch_collate_fn=dataset.test_collate_fn,
        dataset_column_names=dataset.dataset_column_names,
        batch_size=args.per_batch_size,
        epoch_size=1,
        rank=args.rank,
        rank_size=args.rank_size,
        shuffle=False,
        drop_remainder=False,
        num_parallel_workers=args.data.num_parallel_workers,
        python_multiprocessing=True,
    )

    # Run test
    test(
        network=network,
        dataloader=dataloader,
        anno_json_path=os.path.join(
            args.data.val_set[: -len(args.data.val_set.split("/")[-1])], "annotations/instances_val2017.json"
        ),
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        conf_free=args.conf_free,
        nms_time_limit=args.nms_time_limit,
        is_coco_dataset=is_coco_dataset,
        imgIds=None if not is_coco_dataset else dataset.imgIds,
        per_batch_size=args.per_batch_size,
        rank=args.rank,
        rank_size=args.rank_size,
        save_dir=args.save_dir,
        synchronizer=Synchronizer(args.rank_size) if args.rank_size > 1 else None,
    )
    e_time = time.time()
    logger.info(f"Testing completed, cost {e_time - s_time:.2f}s.")


if __name__ == "__main__":
    parser = get_parser_test()
    args = parse_args(parser)
    main(args)
