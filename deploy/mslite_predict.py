"""yolo prediction example script"""

import argparse
import ast
import os
import sys
import time
import cv2
import numpy as np
import yaml
from datetime import datetime

import mindspore_lite as mslite

from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh
from mindyolo.utils.utils import draw_result, set_seed


def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="Infer", parents=[parents] if parents else [])
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--mindir_path", type=str, help="mindir path")
    parser.add_argument("--result_folder", type=str, default="./log_result", help="predicted results folder")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="IOU threshold for NMS")
    parser.add_argument(
        "--conf_free", type=ast.literal_eval, default=False, help="Whether the prediction result include conf"
    )
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    parser.add_argument("--image_path", type=str, help="path to image")
    parser.add_argument("--save_result", type=ast.literal_eval, default=True, help="whether save the inference result")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )

    return parser

def set_default_infer(args):
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # number of classes
    args.data.names = ["item"] if args.single_cls and len(args.names) != 1 else args.data.names  # class names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )
    # Directories and Save run settings
    args.result_folder = os.path.join(args.result_folder, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    os.makedirs(args.result_folder, exist_ok=True)
    with open(os.path.join(args.result_folder, "cfg.yaml"), "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)
    # Set Logger
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO")
    logger.setup_logging_file(log_dir=os.path.join(args.result_folder, "logs"))


def detect(
    mindir_path: str,
    img: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
    conf_free: bool = False,
    nms_time_limit: float = 20.0,
    img_size: int = 640,
    is_coco_dataset: bool = True,
):
    # Resize
    h_ori, w_ori = img.shape[:2]  # orig hw
    r = img_size / max(h_ori, w_ori)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        if isinstance(img_size, int):
            new_shape = (img_size, img_size)
        dh, dw = (new_shape[0] - h) / 2, (new_shape[1] - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

    # Transpose Norm
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    img = np.array(img[None], np.float32)
    img = np.ascontiguousarray(img)
    # Run infer
    _t = time.time()
    # init mslite model to predict
    context = mslite.Context()
    context.target = ["Ascend"]
    model = mslite.Model()
    logger.info('mslite model init...')
    model.build_from_file(mindir_path,mslite.ModelType.MINDIR,context)
    inputs = model.get_inputs()
    model.resize(inputs,[list(img.shape)])
    inputs[0].set_data_from_numpy(img)
    
    outputs = model.predict(inputs)
    outputs = [output.get_data_to_numpy().copy() for output in outputs]
    out = outputs[0]
    infer_times = time.time() - _t

    # Run NMS
    logger.info('perform nms...')
    t = time.time()
    out = non_max_suppression(
        out,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        conf_free=conf_free,
        multi_label=True,
        time_limit=nms_time_limit,
    )
    nms_times = time.time() - t
    result_dict = {"category_id": [], "bbox": [], "score": []}
    total_category_ids, total_bboxes, total_scores = [], [], []

    for si, pred in enumerate(out):
        if len(pred) == 0:
            continue
        # Predictions
        predn = np.copy(pred)
        scale_coords(img.shape[2:], predn[:, :4], (h_ori, w_ori))  # native-space pred , ratio=hw_scale, pad=hw_pad
        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        category_ids, bboxes, scores = [], [], []
        for p, b in zip(pred.tolist(), box.tolist()):
            category_ids.append(COCO80_TO_COCO91_CLASS[int(p[5])] if is_coco_dataset else int(p[5]))
            bboxes.append([round(x, 3) for x in b])
            scores.append(round(p[4], 5))

        total_category_ids.extend(category_ids)
        total_bboxes.extend(bboxes)
        total_scores.extend(scores)

    result_dict["category_id"].extend(total_category_ids)
    result_dict["bbox"].extend(total_bboxes)
    result_dict["score"].extend(total_scores)

    t = tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)  # tuple
    logger.info(f"Predict result is: {result_dict}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info(f"Detect a image success.")

    return result_dict

def infer(args):
    # Init
    set_seed(args.seed)
    set_default_infer(args)

    # Load Image
    if isinstance(args.image_path, str) and os.path.isfile(args.image_path):
        img = cv2.imread(args.image_path)
    else:
        raise ValueError("Detect: input image file not available.")
    # referred from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py#L74
    is_coco_dataset = (
            isinstance(args.data.val_set, str)
            and "coco" in args.data.val_set
            and (args.data.val_set.endswith(f"{os.sep}val2017.txt") or args.data.val_set.endswith(f"{os.sep}test-dev2017.txt"))
    )  # is COCO
    # Detect
    result_dict = detect(
        mindir_path=args.mindir_path,
        img=img,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        conf_free=args.conf_free,
        nms_time_limit=args.nms_time_limit,
        img_size=args.img_size,
        is_coco_dataset=is_coco_dataset,
        )
    if args.save_result:
        save_path = os.path.join(args.result_folder, "detect_results")
        draw_result(args.image_path, result_dict, args.data.names, save_path=save_path)
    else:        
        raise NotImplementedError

    logger.info("predict completed.")
if __name__ == "__main__":
    parser = get_parser_infer()
    args = parse_args(parser)
    infer(args)
