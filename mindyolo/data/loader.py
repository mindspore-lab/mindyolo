import cv2

import mindspore.dataset as de

from dataset import COCODataset
from transforms_factory import create_transforms, create_per_batch_map
from general import normalize_shape, normalize_shape_with_poly


def create_dataloader(config):
    data_config = config.data
    if config.task == 'train':
        image_dir = data_config.train_img_dir
        anno_path = data_config.train_anno_path
    if config.task == 'val':
        image_dir = data_config.val_img_dir
        anno_path = data_config.val_anno_path

    multi_imgs_transforms = getattr(data_config, 'multi_imgs_transforms', None)
    dataset = COCODataset(dataset_dir=data_config.dataset_dir, image_dir=image_dir, anno_path=anno_path, multi_imgs_transforms=multi_imgs_transforms)
    dataset_column_names = ['image', 'w', 'h', 'gt_bbox', 'gt_class']
    ds = de.GeneratorDataset(dataset, column_names=dataset_column_names)

    single_img_transforms = getattr(data_config, 'single_img_transforms', None)
    if single_img_transforms:
        single_img_transforms = create_transforms(single_img_transforms)
        ds = ds.map(operations=single_img_transforms, input_columns=dataset_column_names)

    per_batch_map = getattr(data_config, 'batch_imgs_transform', None)
    if per_batch_map:
        per_batch_map = create_per_batch_map(per_batch_map)
    else:
        per_batch_map = normalize_shape

    ds = ds.batch(config.per_batch_size, input_columns=dataset_column_names, per_batch_map=per_batch_map)

    return ds


if __name__ == '__main__':
    from general import show_img_with_bbox, show_img_with_poly
    import sys
    sys.path.append('../')
    from utils.config import parse_config

    config = parse_config()
    ds = create_dataloader(config)
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
    print('done')
    for i, data in enumerate(data_loader):
        img = show_img_with_bbox(data, config.data.names)
        # img = show_img_with_poly(data)
        cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        cv2.imshow('img', img)
        cv2.waitKey(0)