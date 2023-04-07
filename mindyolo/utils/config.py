import os
import ast
import yaml
import argparse
import collections
from copy import deepcopy
from pprint import pprint, pformat

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

__all__ = ['parse_args']


def parse_args(mode='train'):
    parser_config, parser = create_parser(mode)
    args_config, remaining = parser_config.parse_known_args()

    # Do we have a config file to parse?
    if args_config.config:
        cfg, _, _ = load_config(args_config.config)
        cfg = Config(cfg)
        parser.set_defaults(**cfg)
        parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    print("parse_args:")
    pprint(args)
    print("Please check the above information for the configurations", flush=True)

    return args


def create_parser(mode='train'):
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    parser_config = argparse.ArgumentParser(description='Config', add_help=False)
    parser_config.add_argument('-c', '--config', type=str, default='',
                               help='YAML config file specifying default arguments.')

    # The main parser. It inherits the --config argument for better help information.
    if mode == 'train':
        parser = get_args_train(parents=parser_config)
    elif mode in ["test", "eval", "val"]:
        parser = get_args_test(parents=parser_config)
    elif mode in ['infer', 'detect']:
        parser = get_args_infer(parents=parser_config)
    elif mode in ["export", "310"]:
        parser = get_args_310(parents=parser_config)
    else:
        raise NotImplementedError

    return parser_config, parser


def get_args_train(parents=None):
    parser = argparse.ArgumentParser(description='Train', parents=[parents] if parents else [])
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--save_dir', type=str, default='./runs', help='save dir')
    parser.add_argument('--device_per_servers', type=int, default=8, help='device number on a server')
    parser.add_argument('--log_level', type=str, default='INFO', help='save dir')
    parser.add_argument('--is_parallel', type=ast.literal_eval, default=False, help='Distribute train or not')
    parser.add_argument('--ms_mode', type=int, default=0, help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    parser.add_argument('--ms_amp_level', type=str, default='O0', help='amp level, O0/O1/O2/O3')
    parser.add_argument('--keep_loss_fp32', type=ast.literal_eval, default=True, help='Whether to maintain loss using fp32/O0-level calculation')
    parser.add_argument('--ms_loss_scaler', type=str, default='static', help='train loss scaler, static/dynamic/none')
    parser.add_argument('--ms_loss_scaler_value', type=float, default=1024.0, help='static loss scale value')
    parser.add_argument('--ms_grad_sens', type=float, default=1024.0, help='gard sens')
    parser.add_argument('--ms_jit', type=ast.literal_eval, default=True, help='use jit or not')
    parser.add_argument('--ms_enable_graph_kernel', type=ast.literal_eval, default=False, help='use enable_graph_kernel or not')
    parser.add_argument('--overflow_still_update', type=ast.literal_eval, default=True, help='overflow still update')
    parser.add_argument('--ema', type=ast.literal_eval, default=True, help='ema')
    parser.add_argument('--weight', type=str, default='', help='initial weight path')
    parser.add_argument('--ema_weight', type=str, default='', help='initial ema weight path')
    parser.add_argument('--freeze', type=list, default=[],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--epochs', type=int, default=300, help="total train epochs")
    parser.add_argument('--per_batch_size', type=int, default=32, help='per batch size for each device')
    parser.add_argument('--img_size', type=list, default=640, help='train image sizes')
    parser.add_argument('--nbs', type=list, default=64, help='nbs')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='grad accumulate step, recommended when batch-size is less than 64')
    parser.add_argument('--auto_accumulate', type=ast.literal_eval, default=False, help='auto accumulate')
    parser.add_argument('--log_interval', type=int, default=100, help='log interval')
    parser.add_argument('--multi_scale', type=ast.literal_eval, default=False, help='vary img-size +/- 50%')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False,
                        help='train multi-class data as single-class')
    parser.add_argument('--sync_bn', type=ast.literal_eval, default=False,
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--keep_checkpoint_max', type=int, default=100)
    parser.add_argument('--run_eval', type=ast.literal_eval, default=False,
                        help='Whether to run eval during training')
    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--nms_time_limit', type=float, default=20.0, help='time limit for NMS')
    parser.add_argument('--recompute', type=ast.literal_eval, default=False, help='Recompute')
    parser.add_argument('--recompute_layers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2, help='set global seed')

    # args for ModelArts
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False, help='enable modelarts')
    parser.add_argument('--data_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--ckpt_url', type=str, default='', help='ModelArts: obs path to checkpoint folder')
    parser.add_argument('--train_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--data_dir', type=str, default='/cache/data/',
                        help='ModelArts: local device path to dataset folder')
    parser.add_argument('--ckpt_dir', type=str, default='/cache/pretrain_ckpt/',
                        help='ModelArts: local device path to checkpoint folder')
    return parser


def get_args_test(parents=None):
    parser = argparse.ArgumentParser(description='Test', parents=[parents] if parents else [])
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--ms_mode', type=int, default=0, help='train mode, graph/pynative')
    parser.add_argument('--ms_amp_level', type=str, default='O0', help='amp level, O0/O1/O2')
    parser.add_argument('--ms_jit', type=ast.literal_eval, default=True, help='use jit or not')
    parser.add_argument('--ms_enable_graph_kernel', type=ast.literal_eval, default=False,
                        help='use enable_graph_kernel or not')
    parser.add_argument('--weight', type=str, default='yolov7_300.ckpt', help='model.ckpt path(s)')
    parser.add_argument('--per_batch_size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False,
                        help='train multi-class data as single-class')
    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--nms_time_limit', type=float, default=60.0, help='time limit for NMS')
    parser.add_argument('--recompute', type=ast.literal_eval, default=False, help='Recompute')
    parser.add_argument('--recompute_layers', type=int, default=0)
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--seed', type=int, default=2, help='set global seed')

    # TODO: delete unused parameter in test
    parser.add_argument('--log_level', type=str, default='INFO', help='save dir')
    parser.add_argument('--nbs', type=list, default=64, help='nbs')
    parser.add_argument('--is_parallel', type=ast.literal_eval, default=False, help='Distribute eval or not')
    parser.add_argument('--sync_bn', type=ast.literal_eval, default=False,
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='grad accumulate step, recommended when batch-size is less than 64')
    parser.add_argument('--auto_accumulate', type=ast.literal_eval, default=False, help='auto accumulate')
    parser.add_argument('--epochs', type=int, default=1, help="total epochs")
    parser.add_argument('--save_dir', type=str, default='./runs_test', help='save dir')
    parser.add_argument('--ema', type=ast.literal_eval, default=False, help='ema')
    parser.add_argument('--ema_weight', type=str, default='', help='initial ema weight path')
    parser.add_argument('--run_eval', type=ast.literal_eval, default=False,
                        help='Whether to run eval during training')
    parser.add_argument('--multi_scale', type=ast.literal_eval, default=False, help='vary img-size +/- 50%')
    parser.add_argument('--overflow_still_update', type=ast.literal_eval, default=False, help='overflow still update')
    parser.add_argument('--log_interval', type=int, default=100, help='log interval')
    parser.add_argument('--freeze', type=list, default=[],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')

    # args for ModelArts
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False, help='enable modelarts')
    parser.add_argument('--data_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--ckpt_url', type=str, default='', help='ModelArts: obs path to checkpoint folder')
    parser.add_argument('--train_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--data_dir', type=str, default='/cache/data/',
                        help='ModelArts: local device path to dataset folder')
    parser.add_argument('--ckpt_dir', type=str, default='/cache/pretrain_ckpt/',
                        help='ModelArts: local device path to checkpoint folder')
    return parser


def get_args_infer(parents=None):
    parser = argparse.ArgumentParser(description='Infer', parents=[parents] if parents else [])
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--ms_mode', type=int, default=0, help='train mode, graph/pynative')
    parser.add_argument('--ms_amp_level', type=str, default='O0', help='amp level, O0/O1/O2')
    parser.add_argument('--ms_jit', type=ast.literal_eval, default=True, help='use jit or not')
    parser.add_argument('--ms_enable_graph_kernel', type=ast.literal_eval, default=False,
                        help='use enable_graph_kernel or not')
    parser.add_argument('--weight', type=str, default='yolov7_300.ckpt', help='model.ckpt path(s)')
    parser.add_argument('--per_batch_size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--recompute', type=ast.literal_eval, default=False, help='Recompute')
    parser.add_argument('--recompute_layers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2, help='set global seed')
    parser.add_argument("--image_path", type=str, help="path to image")

    # TODO: delete unused parameter in infer
    parser.add_argument('--nbs', type=list, default=64, help='nbs')
    parser.add_argument('--is_parallel', type=ast.literal_eval, default=False, help='Distribute eval or not')
    parser.add_argument('--sync_bn', type=ast.literal_eval, default=False,
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='grad accumulate step, recommended when batch-size is less than 64')
    parser.add_argument('--auto_accumulate', type=ast.literal_eval, default=False, help='auto accumulate')
    parser.add_argument('--epochs', type=int, default=1, help="total epochs")
    parser.add_argument('--save_dir', type=str, default='./runs_test', help='save dir')
    parser.add_argument('--ema', type=ast.literal_eval, default=False, help='ema')
    parser.add_argument('--ema_weight', type=str, default='', help='initial ema weight path')
    parser.add_argument('--run_eval', type=ast.literal_eval, default=False,
                        help='Whether to run eval during training')
    parser.add_argument('--multi_scale', type=ast.literal_eval, default=False, help='vary img-size +/- 50%')
    parser.add_argument('--overflow_still_update', type=ast.literal_eval, default=False, help='overflow still update')
    parser.add_argument('--log_interval', type=int, default=100, help='log interval')
    parser.add_argument('--freeze', type=list, default=[],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--log_level', type=str, default='INFO', help='save dir')
    # test
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--nms_time_limit', type=float, default=60.0, help='time limit for NMS')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False,
                        help='train multi-class data as single-class')

    # args for ModelArts
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False, help='enable modelarts')
    parser.add_argument('--data_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--ckpt_url', type=str, default='', help='ModelArts: obs path to checkpoint folder')
    parser.add_argument('--train_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--data_dir', type=str, default='/cache/data/',
                        help='ModelArts: local device path to dataset folder')
    parser.add_argument('--ckpt_dir', type=str, default='/cache/pretrain_ckpt/',
                        help='ModelArts: local device path to checkpoint folder')
    return parser


def get_args_310(parents=None):
    parser = argparse.ArgumentParser(description='Export', parents=[parents] if parents else [])

    # export
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--log_level', type=str, default='INFO', help='save dir')
    parser.add_argument('--ms_mode', type=int, default=0, help='train mode, graph/pynative')
    parser.add_argument('--ms_amp_level', type=str, default='O0', help='amp level, O0/O1/O2')
    parser.add_argument('--weight', type=str, default='yolov7_300.ckpt', help='model.ckpt path')
    parser.add_argument('--per_batch_size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False,
                        help='train multi-class data as single-class')
    parser.add_argument('--file_format', type=str, default='MINDIR', help='treat as single-class dataset')
    parser.add_argument('--seed', type=int, default=2, help='set global seed')

    # preprocess
    parser.add_argument('--output_path', type=str, default='./', help='output preprocess data path')

    # postprocess
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--result_path', type=str, default='./result_Files', help='path to 310 infer result floder')

    return parser


def load_config(file_path):
    BASE = "__BASE__"
    assert os.path.splitext(file_path)[-1] in [".yaml", ".yml"], f"[{file_path}] not yaml format."
    cfg_default, cfg_helper, cfg_choices = _parse_yaml(file_path)

    # NOTE: cfgs outside have higher priority than cfgs in _BASE_
    if BASE in cfg_default:
        all_base_cfg_default = {}
        all_base_cfg_helper = {}
        all_base_cfg_choices = {}
        base_yamls = list(cfg_default[BASE])
        for base_yaml in base_yamls:
            if base_yaml.startswith("~"):
                base_yaml = os.path.expanduser(base_yaml)
            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            base_cfg_default, base_cfg_helper, base_cfg_choices = load_config(base_yaml)
            all_base_cfg_default = _merge_config(base_cfg_default, all_base_cfg_default)
            all_base_cfg_helper = _merge_config(base_cfg_helper, all_base_cfg_helper)
            all_base_cfg_choices = _merge_config(base_cfg_choices, all_base_cfg_choices)

        del cfg_default[BASE]
        return _merge_config(cfg_default, all_base_cfg_default), \
               _merge_config(cfg_helper, all_base_cfg_helper), \
               _merge_config(cfg_choices, all_base_cfg_choices)

    return cfg_default, cfg_helper, cfg_choices


def _parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg = cfgs[0]
                cfg_helper = {}
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def _merge_config(config, base):
    """Merge config"""
    new = deepcopy(base)
    for k, v in config.items():
        if (k in new and isinstance(new[k], dict) and
                isinstance(config[k], collectionsAbc.Mapping)):
            new[k] = _merge_config(config[k], new[k])
        else:
            new[k] = config[k]
    return new


class Config(dict):
    """
    Configuration namespace. Convert dictionary to members.
    """
    def __init__(self, cfg_dict):
        super(Config, self).__init__()
        for k, v in cfg_dict.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()
