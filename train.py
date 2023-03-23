# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindyolo.engine.enginer import Enginer
from mindyolo.utils.config import parse_args


if __name__ == '__main__':
    args = parse_args('train')
    enginer = Enginer(args, 'train')
    enginer.train()
