"""MindYolo predict Script. Support evaluation of one image file"""

import os
import yaml
import ast
import argparse
import time
import cv2
import numpy as np
from datetime import datetime

from mindspore import nn, context

from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh
from mindyolo.utils.utils import set_seed, draw_result


def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description='Infer', parents=[parents] if parents else [])
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--ms_mode', type=int, default=0, help='train mode, graph/pynative')
    parser.add_argument('--ms_amp_level', type=str, default='O0', help='amp level, O0/O1/O2')
    parser.add_argument('--ms_enable_graph_kernel', type=ast.literal_eval, default=False,
                        help='use enable_graph_kernel or not')
    parser.add_argument('--weight', type=str, default='yolov7_300.ckpt', help='model.ckpt path(s)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--seed', type=int, default=2, help='set global seed')
    
    parser.add_argument('--model_type', type=str, default="MindX", help='model type MindX/Lite')
    parser.add_argument('--model_path', type=str, default="./models/yolov5s.om", help='model weight path')

    parser.add_argument('--save_dir', type=str, default='./runs_infer', help='save dir')
    parser.add_argument('--log_level', type=str, default='INFO', help='save dir')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--nms_time_limit', type=float, default=60.0, help='time limit for NMS')

    parser.add_argument("--image_path", type=str, help="path to image")
    parser.add_argument("--save_result", type=ast.literal_eval, default=True, help="whether save the inference result")
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False,
                        help='train multi-class data as single-class')

    return parser


def set_default_infer(args):
    # Set Context
    context.set_context(mode=args.ms_mode, device_target=args.device_target, max_call_depth=2000)
    if args.device_target == "Ascend":
        context.set_context(device_id=int(os.getenv('DEVICE_ID', 0)))
    elif args.device_target == "GPU" and args.ms_enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)
    args.rank, args.rank_size = 0, 1
    # Set Data
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # number of classes
    args.data.names = ['item'] if args.single_cls and len(args.names) != 1 else args.data.names  # class names
    assert len(args.data.names) == args.data.nc, '%g names found for nc=%g dataset in %s' % \
                                                 (len(args.data.names), args.data.nc, args.config)
    # Directories and Save run settings
    args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.rank % args.rank_size == 0:
        with open(os.path.join(args.save_dir, "cfg.yaml"), 'w') as f:
            yaml.dump(vars(args), f, sort_keys=False)
    # Set Logger
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=args.rank, device_per_servers=args.rank_size)
    logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))


def detect(
        network: nn.Cell,
        head: nn.Cell,
        img: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.65,
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
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border

    # Transpose Norm
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.
    img = np.array(img[None], np.float32)
    img = np.ascontiguousarray(img)
    # Run infer
    _t = time.time()
    out = network.infer(img)  # inference and training outputs
    out, _ = head(out)
    infer_times = time.time() - _t

    # Run NMS
    t = time.time()
    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres,
                              multi_label=True, time_limit=nms_time_limit)
    nms_times = time.time() - t
    result_dict = {'category_id': [], 'bbox': [], 'score': []}
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

    result_dict['category_id'].extend(total_category_ids)
    result_dict['bbox'].extend(total_bboxes)
    result_dict['score'].extend(total_scores)

    t = tuple(x * 1E3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)  # tuple
    logger.info(f"Predict result is: {result_dict}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info(f"Detect a image success.")

    return result_dict


def Head(nc=80, anchor=(), stride=()):
    no = nc + 5
    nl = len(anchor)
    na = len(anchor[0]) // 2
    anchor_grid = np.array(anchor).reshape(nl, 1, -1, 1, 1, 2)

    def forward(x):
        z = ()
        outs = ()
        for i in range(len(x)):
            out = x[i]
            bs, _, ny, nx = out.shape
            out = out.reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2)
            outs += (out,)

            xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
            grid = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2))
            y = 1 / (1 + np.exp(-out))
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]
            z += (y.reshape(bs, -1, no),)
        return np.concatenate(z, 1), outs

    return forward

def infer(args):
    # Init
    set_seed(args.seed)
    set_default_infer(args)

    # Create Network
    if args.model_type == "MindX":
        from infer_engine.mindx import MindXModel
        network = MindXModel(args.model_path)
    elif args.model_type == "Lite":
        from infer_engine.lite import LiteModel
        network = LiteModel(args.model_path)
    else:
        raise TypeError("the type only supposed MindX/Lite")
    head = Head(nc=80, anchor=args.network.anchors, stride=args.network.stride)
    
    # Load Image
    if isinstance(args.image_path, str) and os.path.isfile(args.image_path):
        import cv2
        img = cv2.imread(args.image_path)
    else:
        raise ValueError("Detect: input image file not available.")

    # Detect
    is_coco_dataset = ('coco' in args.data.dataset_name)
    result_dict = detect(
        network=network,
        head=head,
        img=img,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        nms_time_limit=args.nms_time_limit,
        img_size=args.img_size,
        is_coco_dataset=is_coco_dataset
    )
    if args.save_result:
        save_path = os.path.join(args.save_dir, 'detect_results')
        draw_result(args.image_path, result_dict, args.data.names, save_path=save_path)

    logger.info('Infer completed.')


if __name__ == '__main__':
    parser = get_parser_infer()
    args = parse_args(parser)
    infer(args)
