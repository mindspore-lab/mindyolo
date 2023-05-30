import argparse
import collections
import os
from copy import deepcopy
import yaml

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

__all__ = ["parse_args"]


def parse_args(parser):
    parser_config = argparse.ArgumentParser(description="Config", add_help=False)
    parser_config.add_argument(
        "-c", "--config", type=str, default="", help="YAML config file specifying default arguments."
    )

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

    return Config(vars(args))


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
            if not base_yaml.startswith("/"):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            base_cfg_default, base_cfg_helper, base_cfg_choices = load_config(base_yaml)
            all_base_cfg_default = _merge_config(base_cfg_default, all_base_cfg_default)
            all_base_cfg_helper = _merge_config(base_cfg_helper, all_base_cfg_helper)
            all_base_cfg_choices = _merge_config(base_cfg_choices, all_base_cfg_choices)

        del cfg_default[BASE]
        return (
            _merge_config(cfg_default, all_base_cfg_default),
            _merge_config(cfg_helper, all_base_cfg_helper),
            _merge_config(cfg_choices, all_base_cfg_choices),
        )

    return cfg_default, cfg_helper, cfg_choices


def _parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, "r") as fin:
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
        if k in new and isinstance(new[k], dict) and isinstance(config[k], collectionsAbc.Mapping):
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
        self.__dict__.update({name: value})

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __str__(self):
        return config_format_func(self)

    def __repr__(self):
        return self.__str__()


def config_format_func(config, prefix=""):
    """
    Args:
        config: dict-like object
    Returns:
        formatted str
    """
    msg = ""
    if prefix:
        prefix += "."

    for k, v in config.__dict__.items():
        if isinstance(v, Config):
            msg += config_format_func(v, prefix=str(k))
        else:
            msg += format(prefix + str(k), "<40") + format(str(v), "<") + "\n"
    return msg
