"""MindYolo Export Script. Transform MindSpore weight format"""

import argparse
import ast
import os
import sys
import numpy as np

import mindspore as ms
from mindspore import Tensor, context, export

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mindyolo.models import create_model
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.utils import set_seed


def get_parser_export(parents=None):
    parser = argparse.ArgumentParser(description="Export", parents=[parents] if parents else [])
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="train mode, graph/pynative")
    parser.add_argument("--weight", type=str, default="yolov7_300.ckpt", help="model.ckpt path(s)")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--per_batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="save dir")
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    parser.add_argument("--file_format", type=str, default="MINDIR", help="treat as single-class dataset")
    parser.add_argument("--save_dir", type=str, default="./export", help="save dir")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )

    return parser


def set_default_export(args):
    # Set Context
    context.set_context(mode=args.ms_mode)
    ms.set_recursion_limit(2000)
    if args.device_target == "Ascend":
        ms.set_device("Ascend", int(os.getenv("DEVICE_ID", 0)))
    args.rank, args.rank_size = 0, 1
    # Set Data
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # number of classes
    args.data.names = ["item"] if args.single_cls and len(args.names) != 1 else args.data.names  # class names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )
    # Set Logger
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=args.rank, device_per_servers=args.rank_size)
    logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))


def export_weight(args):
    # Init
    set_seed(args.seed)
    set_default_export(args)

    # Create Network
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        checkpoint_path=args.weight,
    )
    if args.file_format == "ONNX":
        from mindyolo.models.layers.activation import SiLU
        for name, cell in network.name_cells().items():
            if isinstance(cell, SiLU):
                cell.fused_op = False

    network.set_train(False)
    # Export
    input_arr = Tensor(np.ones([args.per_batch_size, 3, args.img_size, args.img_size]), ms.float32)
    file_name = os.path.basename(args.config)[:-5]  # delete ".yaml"
    export(network, input_arr, file_name=file_name, file_format=args.file_format)

    logger.info("Export completed.")

if __name__ == "__main__":
    parser = get_parser_export()
    args = parse_args(parser)
    export_weight(args)
