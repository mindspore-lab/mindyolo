import argparse
import ast
import os
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from infer_engine import *
from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.data import COCODataset, create_loader
from mindyolo.utils import logger
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh


def Detect(nc=80, anchor=(), stride=()):
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


def infer(cfg):
    # Create Network
    if cfg.model_type is "MindX":
        network = MindXModel("./models/yolov5s.om")
    elif cfg.model_type is "Lite":
        network = LiteModel(cfg.model_path)
    else:
        raise TypeError("the type only supposed MindX/Lite")
    detect = Detect(nc=80, anchor=cfg.anchors, stride=cfg.stride)

    dataset = COCODataset(
        dataset_path=cfg.val_set,
        img_size=cfg.img_size,
        transforms_dict=cfg.test_transforms,
        is_training=False, augment=False, rect=cfg.rect, single_cls=cfg.single_cls,
        batch_size=cfg.batch_size, stride=max(cfg.stride),
    )
    dataloader = create_loader(
        dataset=dataset,
        batch_collate_fn=dataset.test_collate_fn,
        dataset_column_names=dataset.dataset_column_names,
        batch_size=cfg.batch_size,
        epoch_size=1, rank=0, rank_size=1, shuffle=False, drop_remainder=False,
        num_parallel_workers=cfg.num_parallel_workers,
        python_multiprocessing=True
    )

    loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    dataset_dir = cfg.val_set[:-len(cfg.val_set.split('/')[-1])]
    anno_json_path = os.path.join(dataset_dir, 'annotations/instances_val2017.json')
    coco91class = COCO80_TO_COCO91_CLASS

    step_num = dataloader.get_dataset_size()
    sample_num = 0
    infer_times = 0.
    nms_times = 0.
    result_dicts = []
    for i, data in enumerate(loader):
        imgs, _, paths, ori_shape, pad, hw_scale = data['image'], data['labels'], data['img_files'], \
            data['hw_ori'], data['pad'], data['hw_scale']
        nb, _, height, width = imgs.shape

        # Run infer
        _t = time.time()
        out = network.infer(imgs)  # inference and training outputs
        out, _ = detect(out)
        infer_times += time.time() - _t

        # Run NMS
        t = time.time()
        out = non_max_suppression(out, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres,
                                  multi_label=True, time_limit=cfg.nms_time_limit)
        nms_times += time.time() - t

        # Statistics pred
        for si, pred in enumerate(out):
            path = Path(str(paths[si]))
            sample_num += 1
            if len(pred) == 0:
                continue

            # Predictions
            predn = np.copy(pred)
            scale_coords(imgs[si].shape[1:], predn[:, :4], ori_shape[si], ratio=hw_scale[si],
                         pad=pad[si])  # native-space pred

            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                result_dicts.append({'image_id': image_id,
                                     'category_id': coco91class[int(p[5])],
                                     'bbox': [round(x, 3) for x in b],
                                     'score': round(p[4], 5)})
        print(f"Sample {step_num}/{i + 1}, time cost: {(time.time() - _t) * 1000:.2f} ms.")

        # Compute mAP
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        anno = COCO(anno_json_path)  # init annotations api
        pred = anno.loadRes(result_dicts)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')

        eval.params.imgIds = [int(Path(im_file).stem) for im_file in dataset.img_files]
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    except Exception as e:
        logger.warning(f'pycocotools unable to run: {e}')
        raise e

    t = tuple(x / sample_num * 1E3 for x in (infer_times, nms_times, infer_times + nms_times)) + \
        (height, width, cfg.batch_size)  # tuple
    logger.info(f'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;' % t)

    return map, map50


def parse_args():
    parser = argparse.ArgumentParser(description='Config', add_help=False)

    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False,
                        help='train multi-class data as single-class')
    parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')

    parser.add_argument('--stride', type=list, default=[8, 16, 32])
    parser.add_argument('--anchors', type=list,
                        default=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]])
    parser.add_argument('--model_type', type=str, default="MindX", help='model type MindX/Lite')
    parser.add_argument('--model_path', type=str, default="./models/yolov5s.om", help='model weight path')

    parser.add_argument('--nc', type=int, default=80)
    parser.add_argument('--val_set', type=str, default='./coco/val2017.txt')
    parser.add_argument('--test_transforms', type=list, default=[{'func_name': 'letterbox', 'scaleup': False},
                                                                 {'func_name': 'label_norm', 'xyxy2xywh_': True},
                                                                 {'func_name': 'label_pad', 'padding_size': 160,
                                                                  'padding_value': -1},
                                                                 {'func_name': 'image_norm', 'scale': 255.},
                                                                 {'func_name': 'image_transpose', 'bgr2rgb': True,
                                                                  'hwc2chw': True}])
    parser.add_argument('--conf_thres', type=float, default=0.001)
    parser.add_argument('--iou_thres', type=float, default=0.05)
    parser.add_argument('--nms_time_limit', type=float, default=20.0)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    infer(args)
