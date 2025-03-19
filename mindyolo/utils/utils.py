import os
import random
import yaml
import cv2
from datetime import datetime
import numpy as np

import mindspore as ms
from mindspore import ops, Tensor, nn
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore import ParallelMode

from mindyolo.utils import logger


def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


def set_default(args):
    # Set Context
    ms.set_context(mode=args.ms_mode)
    ms.set_recursion_limit(args.max_call_depth)
    if args.ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O2"})
    if args.device_target == "Ascend":
        ms.set_device("Ascend", int(os.getenv("DEVICE_ID", 0)))
    # Set Parallel
    if args.is_parallel:
        init()
        args.rank, args.rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
        ms.set_auto_parallel_context(device_num=args.rank_size, parallel_mode=parallel_mode, gradients_mean=True)
    else:
        args.rank, args.rank_size = 0, 1
    # Set Default
    args.total_batch_size = args.per_batch_size * args.rank_size
    args.sync_bn = args.sync_bn and ms.get_context("device_target") == "Ascend" and args.rank_size > 1
    args.accumulate = max(1, np.round(args.nbs / args.total_batch_size)) if args.auto_accumulate else args.accumulate
    # optimizer
    args.optimizer.warmup_epochs = args.optimizer.get("warmup_epochs", 0)
    args.optimizer.min_warmup_step = args.optimizer.get("min_warmup_step", 0)
    args.optimizer.epochs = args.epochs
    args.optimizer.nbs = args.nbs
    args.optimizer.accumulate = args.accumulate
    args.optimizer.total_batch_size = args.total_batch_size
    # data
    cv2.setNumThreads(args.opencv_threads_num)  # Set the number of threads for opencv.
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

    # callback
    args.callback = args.get('callback', [])

    # Set Logger
    logger.setup_logging(
        logger_name="MindYOLO", log_level=args.log_level, rank_id=args.rank, device_per_servers=args.rank_size
    )
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
        args.data.train_set = os.path.join(args.data_dir, args.data.train_set)
        args.data.val_set = os.path.join(args.data_dir, args.data.val_set)
        args.data.test_set = os.path.join(args.data_dir, args.data.test_set)
        args.weight = args.ckpt_dir if args.ckpt_dir else ""
        args.ema_weight = os.path.join(args.ckpt_dir, args.ema_weight) if args.ema_weight else ""


def drop_inconsistent_shape_parameters(model, param_dict):
    updated_param_dict = dict()

    # TODO: hard code
    param_dict = {k.replace('ema.', ''): v for k, v in param_dict.items()}

    for param in model.get_parameters():
        name = param.name
        if name in param_dict:
            if param_dict[name].shape == param.shape:
                updated_param_dict[name] = param_dict[name]
            else:
                logger.warning(
                    f"Dropping checkpoint parameter `{name}` with shape `{param_dict[name].shape}`, "
                    f"which is inconsistent with cell shape `{param.shape}`"
                )
        else:
            logger.warning(f"Cannot find checkpoint parameter `{name}`.")
    return updated_param_dict


def load_pretrain(network, weight, ema=None, ema_weight=None, strict=True):
    if weight.endswith(".ckpt"):
        param_dict = ms.load_checkpoint(weight)
        if not strict:
            param_dict = drop_inconsistent_shape_parameters(network, param_dict)
        ms.load_param_into_net(network, param_dict)
        logger.info(f'Pretrain model load from "{weight}" success.')
    if ema:
        if ema_weight.endswith(".ckpt"):
            param_dict_ema = ms.load_checkpoint(ema_weight)
            if not strict:
                param_dict_ema = drop_inconsistent_shape_parameters(ema.ema, param_dict_ema)
            ms.load_param_into_net(ema.ema, param_dict_ema)
            logger.info(f'Ema pretrain model load from "{ema_weight}" success.')
        else:
            ema.clone_from_model()
            logger.info("ema_weight not exist, default pretrain weight is currently used.")


def freeze_layers(network, freeze=[]):
    if len(freeze) > 0:
        freeze = [f"model.{x}." for x in freeze]  # parameter names to freeze (full or partial)
        for n, p in network.parameters_and_names():
            if any(x in n for x in freeze):
                logger.info("freezing %s" % n)
                p.requires_grad = False


def draw_result(img_path, result_dict, data_names, is_coco_dataset=True, save_path="./detect_results"):
    import random
    import cv2
    from mindyolo.data import COCO80_TO_COCO91_CLASS

    os.makedirs(save_path, exist_ok=True)
    save_result_path = os.path.join(save_path, img_path.split("/")[-1])
    im = cv2.imread(img_path)
    category_id, bbox, score = result_dict["category_id"], result_dict["bbox"], result_dict["score"]
    seg = result_dict.get("segmentation", None)
    mask = None if seg is None else np.zeros_like(im, dtype=np.float32)
    for i in range(len(bbox)):
        # draw box
        x_l, y_t, w, h = bbox[i][:]
        x_r, y_b = x_l + w, y_t + h
        x_l, y_t, x_r, y_b = int(x_l), int(y_t), int(x_r), int(y_b)
        _color = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(im, (x_l, y_t), (x_r, y_b), tuple(_color), 2)
        if seg:
            _color_seg = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
            mask += seg[i][:, :, None] * _color_seg[None, None, :]

        # draw label
        if is_coco_dataset:
            class_name_index = COCO80_TO_COCO91_CLASS.index(category_id[i])
        else:
            class_name_index = category_id[i]
        class_name = data_names[class_name_index]  # args.data.names[class_name_index]
        text = f"{class_name}: {score[i]}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(im, (x_l, y_t - text_h - baseline), (x_l + text_w, y_t), tuple(_color), -1)
        cv2.putText(im, text, (x_l, y_t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # save results
    if seg:
        im = (0.7 * im + 0.3 * mask).astype(np.uint8)
    cv2.imwrite(save_result_path, im)


def get_broadcast_datetime(rank_size=1, root_rank=0):
    time = datetime.now()
    time_list = [time.year, time.month, time.day, time.hour, time.minute, time.second, time.microsecond]
    if rank_size <=1:
        return time_list

    # only broadcast in distribution mode
    x = broadcast((Tensor(time_list, dtype=ms.int32),), root_rank)
    x = x[0].asnumpy().tolist()
    return x

@ms.jit
def broadcast(x, root_rank):
    return ops.Broadcast(root_rank=root_rank)(x)

class AllReduce(nn.Cell):
    """
    a wrapper class to make ops.AllReduce become a Cell. This is a workaround for sync_wait
    """
    def __init__(self):
        super(AllReduce, self).__init__()
        self.all_reduce = ops.AllReduce(op=ops.ReduceOp.SUM)

    def construct(self, x):
        return self.all_reduce(x)


class Synchronizer:
    def __init__(self, rank_size=1):
        # this init method should be run only once
        self.all_reduce = AllReduce()
        self.rank_size = rank_size

    def __call__(self):
        if self.rank_size <= 1:
            return
        sync = Tensor(np.array([1]).astype(np.int32))
        sync = self.all_reduce(sync)
        sync = sync.asnumpy()[0]
        if sync != self.rank_size:
            raise ValueError(f'Sync value {sync} is not equal to rank size {self.rank_size}.'
                             f' There might be wrong with devices')
