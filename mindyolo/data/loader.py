import cv2

import mindspore.dataset as de

from .transforms_factory import create_transforms, create_per_batch_map
from .dataset import COCODataset
from .general import normalize_shape

__all__ = ['create_dataloader']


def create_dataloader(data_config, task, per_batch_size, rank=0, rank_size=1, shuffle=True, drop_remainder=False):
    if task == 'train':
        image_dir = data_config.train_img_dir
        anno_path = data_config.train_anno_path
        trans_config = getattr(data_config, 'train_transforms', data_config)
    elif task in ('val', 'eval'):
        image_dir = data_config.val_img_dir
        anno_path = data_config.val_anno_path
        trans_config = getattr(data_config, 'eval_transforms', data_config)
    elif task == "test":
        image_dir = data_config.test_img_dir
        anno_path = data_config.test_anno_path
        trans_config = getattr(data_config, 'eval_transforms', data_config)
        trans_config = getattr(data_config, 'test_transforms', trans_config)
    else:
        raise NotImplementedError

    num_parallel_worker = getattr(data_config, 'num_parallel_worker', 4)

    multi_imgs_transforms = getattr(trans_config, 'multi_imgs_transforms', None)
    dataset = COCODataset(dataset_dir=data_config.dataset_dir,
                          image_dir=image_dir,
                          anno_path=anno_path,
                          multi_imgs_transforms=multi_imgs_transforms,
                          allow_empty=True,
                          detection_require_poly=False
                          )
    dataset_column_names = ['image', 'im_file',
                            'ori_shape', 'pad', 'ratio',
                            'gt_bbox', 'gt_class']

    if rank_size > 1:
        ds = de.GeneratorDataset(dataset, column_names=dataset_column_names,
                                 num_parallel_workers=num_parallel_worker, shuffle=shuffle,
                                 num_shards=rank_size, shard_id=rank)
    else:
        ds = de.GeneratorDataset(dataset, column_names=dataset_column_names,
                                 num_parallel_workers=num_parallel_worker, shuffle=shuffle)

    single_img_transforms = getattr(trans_config, 'single_img_transforms', None)
    if single_img_transforms:
        single_img_transforms = create_transforms(single_img_transforms)
        ds = ds.map(operations=single_img_transforms, input_columns=dataset_column_names, num_parallel_workers=num_parallel_worker)

    per_batch_map = getattr(trans_config, 'batch_imgs_transform', None)
    if per_batch_map:
        per_batch_map = create_per_batch_map(per_batch_map)
    else:
        per_batch_map = normalize_shape

    batch_column = dataset_column_names + ['batch_idx',]
    ds = ds.batch(per_batch_size,
                  input_columns=dataset_column_names,
                  output_columns=batch_column,
                  per_batch_map=per_batch_map,
                  num_parallel_workers=num_parallel_worker,
                  drop_remainder=drop_remainder
                  )
    ds = ds.repeat(1)

    return ds, dataset


if __name__ == '__main__':
    from .general import show_img_with_bbox, show_img_with_poly
    from mindyolo.utils.config import parse_config

    config = parse_config()
    ds = create_dataloader(data_config=config.data,
                           task='train',
                           per_batch_size=config.per_batch_size)
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
    print('done')
    for i, data in enumerate(data_loader):
        img = show_img_with_bbox(data, config.data.names)
        # img = show_img_with_poly(data)
        cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        cv2.imshow('img', img)
        cv2.waitKey(0)