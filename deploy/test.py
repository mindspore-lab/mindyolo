"""MindYolo Evaluation Script of COCO dataset"""

import argparse
import ast
import os
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mindyolo.data import COCO80_TO_COCO91_CLASS, COCODataset, create_loader
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh


def test(args):
    if args.task == "detect":
        return test_detect(args)
    else:
        raise NotImplementedError


def test_detect(args):
    # Create Network
    if args.model_type == "MindX":
        from infer_engine.mindx import MindXModel
        network = MindXModel(args.model_path)
    elif args.model_type == "Lite":
        from infer_engine.lite import LiteModel
        network = LiteModel(args.model_path)
    else:
        raise TypeError("the type only supposed MindX/Lite")

    dataset = COCODataset(
        dataset_path=args.val_set,
        img_size=args.img_size,
        transforms_dict=args.test_transforms,
        is_training=False,
        augment=False,
        rect=args.rect,
        single_cls=args.single_cls,
        batch_size=args.batch_size,
        stride=max(args.network.stride),
    )
    dataloader = create_loader(
        dataset=dataset,
        batch_collate_fn=dataset.test_collate_fn,
        column_names_getitem=dataset.column_names_getitem,
        column_names_collate=dataset.column_names_collate,
        batch_size=args.batch_size,
        epoch_size=1,
        rank=0,
        rank_size=1,
        shuffle=False,
        drop_remainder=False,
        num_parallel_workers=2,
        python_multiprocessing=True,
    )

    loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    dataset_dir = args.val_set[: -len(args.val_set.split("/")[-1])]
    anno_json_path = os.path.join(dataset_dir, "annotations/instances_val2017.json")
    coco91class = COCO80_TO_COCO91_CLASS

    step_num = dataloader.get_dataset_size()
    sample_num = 0
    infer_times = 0.0
    nms_times = 0.0
    result_dicts = []
    for i, data in enumerate(loader):
        imgs, paths, ori_shape, pad, hw_scale = (
            data["images"],
            data["img_files"],
            data["hw_ori"],
            data["pad"],
            data["hw_scale"],
        )
        nb, _, height, width = imgs.shape
        # Run infer
        _t = time.time()
        out = network.infer(imgs)[0]  # inference and training outputs
        infer_times += time.time() - _t
        # print(f"Sample {step_num}/{i + 1}, network time cost: {(time.time() - _t) * 1000:.2f} ms.")

        # Run NMS
        t = time.time()
        out = non_max_suppression(
            out,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            multi_label=True,
            time_limit=args.nms_time_limit,
        )
        nms_times += time.time() - t
        # print(f"Sample {step_num}/{i + 1}, nms time cost: {(time.time() - t) * 1000:.2f} ms.")

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
                        "category_id": coco91class[int(p[5])],
                        "bbox": [round(x, 3) for x in b],
                        "score": round(p[4], 5),
                    }
                )
        print(f"Sample {step_num}/{i + 1}, time cost: {(time.time() - _t) * 1000:.2f} ms.")

        # Compute mAP
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        anno = COCO(anno_json_path)  # init annotations api
        pred = anno.loadRes(result_dicts)  # init predictions api
        eval = COCOeval(anno, pred, "bbox")

        eval.params.imgIds = [int(Path(im_file).stem) for im_file in dataset.img_files]
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    except Exception as e:
        logger.warning(f"pycocotools unable to run: {e}")
        raise e

    t = tuple(x / sample_num * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (
        height,
        width,
        args.batch_size,
    )  # tuple
    # logger.info(f'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;' % t)
    print(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)

    return map, map50


def get_parser_test(parents=None):
    parser = argparse.ArgumentParser(description="Test", parents=[parents] if parents else [])

    parser.add_argument("--task", type=str, default="detect", choices=["detect"])
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--rect", type=ast.literal_eval, default=False, help="rectangular training")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")

    parser.add_argument("--model_type", type=str, default="Lite", help="model type MindX/Lite")
    parser.add_argument("--model_path", type=str, default="./models/yolov5s.om", help="model weight path")

    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--val_set", type=str, default="./coco/val2017.txt")
    parser.add_argument(
        "--test_transforms",
        type=list,
        default=[
            {"func_name": "letterbox", "scaleup": False},
            {"func_name": "label_norm", "xyxy2xywh_": True},
            {"func_name": "label_pad", "padding_size": 160, "padding_value": -1},
            {"func_name": "image_norm", "scale": 255.0},
            {"func_name": "image_transpose", "bgr2rgb": True, "hwc2chw": True},
        ],
    )
    parser.add_argument("--conf_thres", type=float, default=0.001)
    parser.add_argument("--iou_thres", type=float, default=0.65)
    parser.add_argument(
        "--conf_free", type=ast.literal_eval, default=False, help="Whether the prediction result include conf"
    )
    parser.add_argument("--nms_time_limit", type=float, default=20.0)

    return parser


if __name__ == "__main__":
    parser = get_parser_test()
    args = parse_args(parser)
    test(args)
