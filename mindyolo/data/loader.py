import cv2
import numpy as np

import mindspore.dataset as de

from dataset import COCODataset
from transforms_factory import create_transforms
from general import show_img_with_bbox


def normalize_shape(gt_bboxes, gt_classes, batch_info):
    most_boxes_per_img = 0
    for gt_class in gt_classes:
        most_boxes_per_img = max(most_boxes_per_img, gt_class.shape[0])

    for i, (gt_bbox, gt_class) in enumerate(zip(gt_bboxes, gt_classes)):
        nL = gt_class.shape[0]
        gt_bboxes[i] = np.full((most_boxes_per_img, 4), -1, dtype=np.float32)
        gt_bboxes[i][:nL, :] = gt_bbox[:nL, :]
        gt_classes[i] = np.full((most_boxes_per_img, 1), -1, dtype=np.int32)
        gt_classes[i][:nL, :] = gt_class[:nL, :]

    return gt_bboxes, gt_classes


def create_dataloader(config):
    data_config = config.Data
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

    single_img_transforms = create_transforms(data_config.single_img_transforms)

    ds = ds.map(operations=single_img_transforms, input_columns=dataset_column_names)
    ds = ds.batch(config.per_batch_size, input_columns=['gt_bbox', 'gt_class'], per_batch_map=normalize_shape)

    return ds


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from utils.config import parse_config

    config = parse_config()
    ds = create_dataloader(config)
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
    print('done')
    for i, data in enumerate(data_loader):
        img = show_img_with_bbox(data, config.Data.names)
        cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        cv2.imshow('img', img)
        cv2.waitKey(0)

