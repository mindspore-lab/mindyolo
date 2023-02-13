import argparse
import mindspore as ms
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

def fuse_checkpoint(base_name, start, num):
    """
    Fusion of continuous weights obtained during training.

    Examples:
        # Fusing weights between 'yolov7_291.ckpt' to 'yolov7_300.ckpt'
        fuse_checkpoint("./path_to/yolov7", 291, 10)
    """

    new_par_dict = {}
    for i in range(start, start + num):
        ckpt_file = base_name + f'_{i}.ckpt'
        par_dict = ms.load_checkpoint(ckpt_file)
        for k in par_dict:
            if k in new_par_dict:
                new_par_dict[k] += par_dict[k].asnumpy()
            else:
                new_par_dict[k] = par_dict[k].asnumpy()

    new_params_list = []
    for k in new_par_dict:
        _param_dict = {'name': k, 'data': Tensor(new_par_dict[k] / num)}
        new_params_list.append(_param_dict)

    ms_ckpt = f"{base_name}_fuse_{start}to{start + num - 1}.ckpt"
    save_checkpoint(new_params_list, ms_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='checkpoint_fuse.py')
    parser.add_argument('--num', type=int, default=10, help='fuse checkpoint num')
    parser.add_argument('--start', type=int, default=291, help='Distribute train or not')
    parser.add_argument('--base_name', type=str, default='./yolov7', help='source checkpoint file base')
    opt = parser.parse_args()
    fuse_checkpoint(opt.base_name, opt.start, opt.num)
