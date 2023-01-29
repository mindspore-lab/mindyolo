import os
import yaml
import argparse
import collections
from copy import deepcopy

from .args import get_args_train, get_args_test, get_args_310

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

def parse_args(mode='train'):
    parser_config, parser = create_parser(mode)
    args_config, args_main = parser_config.parse_known_args()
    cfg = load_config(args_config.config)
    parser.set_defaults(**cfg)
    parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(args_main)
    return args

def create_parser(mode='train'):
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    parser_config = argparse.ArgumentParser(description='Config', add_help=False)
    parser_config.add_argument('--config', type=str, default='./config/yolov7/net/yolov7.yaml',
                               help='YAML config file specifying default arguments.')

    # The main parser. It inherits the --config argument for better help information.
    if mode == 'train':
        parser = get_args_train(parser_config)
    elif mode in ["test", "eval", "val"]:
        parser = get_args_test(parser_config)
    elif mode in ["export", "310"]:
        parser = get_args_310(parser_config)
    else:
        raise NotImplementedError

    return parser_config, parser

def load_config(file_path):
    BASE = "__BASE__"
    assert os.path.splitext(file_path)[-1] in [".yaml", ".yml"], f"[{file_path}] not yaml format."
    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)

    # NOTE: cfgs outside have higher priority than cfgs in _BASE_
    if BASE in file_cfg:
        all_base_cfg = {}
        base_yamls = list(file_cfg[BASE])
        for base_yaml in base_yamls:
            if base_yaml.startswith("~"):
                base_yaml = os.path.expanduser(base_yaml)
            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            base_cfg = load_config(base_yaml)
            all_base_cfg = merge_config(base_cfg, all_base_cfg)

        del file_cfg[BASE]
        return merge_config(file_cfg, all_base_cfg)

    return file_cfg

def merge_config(config, base):
    """Merge config"""
    new = deepcopy(base)
    for k, v in config.items():
        if (k in new and isinstance(new[k], dict) and
                isinstance(config[k], collectionsAbc.Mapping)):
            new[k] = merge_config(config[k], new[k])
        else:
            new[k] = config[k]
    return new
