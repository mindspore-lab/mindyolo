from . import transforms

__all__ = ['create_transform']


def create_transform(transform):
    """
    Convert the list in the yaml file into a set of data enhancement lists
    the list in yaml file maybe like this:
        single_img_transforms:
            - RandomFlip: {prob: 0.5}
            - RandomHSV: {}
            - LetterBox: {target_size: 640}
    """
    for k, v in transform.items():
        op_cls = getattr(transforms, k)
        f = op_cls(**v)

    return f
