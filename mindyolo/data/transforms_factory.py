import transforms


def create_transforms(transforms_list):
    transforms_ops = []
    for t in transforms_list:
        for k, v in t.items():
            op_cls = getattr(transforms, k)
            f = op_cls(**v)

            transforms_ops.append(f)
    return transforms_ops