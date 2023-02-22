from mindyolo.engine.enginer import Enginer
from mindyolo.utils.config import parse_config

def main():
    cfg = parse_config()
    enginer = Enginer(cfg, task='train')
    enginer.train()


if __name__ == '__main__':
    main()
