from . import transforms

__all__ = ['create_transforms', 'create_per_batch_map']


def create_transforms(transforms_list):
    """
    Convert the list in the yaml file into a set of data enhancement lists
    the list in yaml file maybe like this:
        single_img_transforms:
            - RandomFlip: {prob: 0.5}
            - RandomHSV: {}
            - LetterBox: {target_size: 640}
    """
    transforms_ops = []
    for t in transforms_list:
        for k, v in t.items():
            op_cls = getattr(transforms, k)
            f = op_cls(**v)

            transforms_ops.append(f)
    return transforms_ops


def create_per_batch_map(transform_dict):
    """
    Convert the dict in the yaml file into a callable class
    the dict in yaml file maybe like this:
        batch_imgs_transform:
            BatchRandomResize: {target_size: [[601, 701], [672, 772]], keep_ratio: True, random_range: True}
    """
    for k, v in transform_dict.items():
        op_cls = getattr(transforms, k)
        f = op_cls(**v)

        return f
