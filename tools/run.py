import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindyolo.engine.enginer import Enginer
from mindyolo.utils.config import parse_config


def run():
    cfg = parse_config()
    enginer = Enginer(cfg)
    enginer.run(cfg.input_img)


if __name__ == '__main__':
    run()
