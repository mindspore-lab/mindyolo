import mindspore as ms


def convert_weight(ori_weight, new_weight):
    new_ckpt = []
    param_dict = ms.load_checkpoint(ori_weight)
    for k, v in param_dict.items():
        if '77' in k:
            continue
        new_ckpt.append({'name': k, 'data': v})
    ms.save_checkpoint(new_ckpt, new_weight)


if __name__ == '__main__':
    convert_weight('./yolov7-tiny_300e_mAP375-d8972c94.ckpt', './yolov7-tiny_pretrain.ckpt')
