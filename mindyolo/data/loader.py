import mindspore.dataset as de

from .transforms_factory import create_transform
from .dataset import COCODataset
from .general import normalize_shape

__all__ = ['create_dataloader']


def create_dataloader(data_config, task, per_batch_size, rank=0, rank_size=1, shuffle=True, drop_remainder=False):
    consider_poly = False

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

    num_parallel_worker = getattr(data_config, 'num_parallel_workers', 4)

    multi_imgs_transforms = getattr(trans_config, 'multi_imgs_transforms', [])
    single_img_transforms = getattr(trans_config, 'single_img_transforms', [])
    multi_imgs_transforms_num = len(multi_imgs_transforms)

    transforms_dict_list = multi_imgs_transforms + single_img_transforms

    transforms_name_list = []
    for transform in transforms_dict_list:
        transforms_name_list.extend(transform.keys())

    last_trans_need_poly = -1
    if 'SimpleCopyPaste' in transforms_name_list:
        last_trans_need_poly = transforms_name_list.index('SimpleCopyPaste')
        consider_poly = True
    else:
        if 'PasteIn' in transforms_name_list:
            last_trans_need_poly = max(last_trans_need_poly, transforms_name_list.index('PasteIn'))
        if 'Mosaic' in transforms_name_list:
            mosaic_index = transforms_name_list.index('Mosaic')
            mosaic = transforms_dict_list[mosaic_index]
            if mosaic['Mosaic']['copy_paste_prob']:
                last_trans_need_poly = max(last_trans_need_poly, mosaic_index)

    transforms_list = []
    for i, transform_name in enumerate(transforms_name_list):
        if i <= last_trans_need_poly:
            transforms_dict_list[i][transform_name]['consider_poly'] = True
        transform = create_transform(transforms_dict_list[i])
        transforms_list.append(transform)

    multi_imgs_transforms = transforms_list[:multi_imgs_transforms_num]
    if single_img_transforms:
        single_img_transforms = transforms_list[multi_imgs_transforms_num:]

    dataset = COCODataset(dataset_dir=data_config.dataset_dir,
                          image_dir=image_dir,
                          anno_path=anno_path,
                          multi_imgs_transforms=multi_imgs_transforms,
                          allow_empty=True,
                          consider_poly=consider_poly
                          )
    dataset_column_names = ['image', 'im_file', 'im_id',
                            'ori_shape', 'pad', 'ratio',
                            'gt_bbox', 'gt_class']
    if consider_poly:
        dataset_column_names.append('gt_poly')

    if rank_size > 1:
        ds = de.GeneratorDataset(dataset, column_names=dataset_column_names,
                                 num_parallel_workers=num_parallel_worker, shuffle=shuffle,
                                 num_shards=rank_size, shard_id=rank)
    else:
        ds = de.GeneratorDataset(dataset, column_names=dataset_column_names,
                                 num_parallel_workers=num_parallel_worker, shuffle=shuffle)

    if task != 'train':
        map_columns = ['image', 'pad', 'ratio', 'gt_bbox', 'gt_class']
    else:
        map_columns = ['image', 'gt_bbox', 'gt_class']
    if consider_poly:
        map_columns.append('gt_poly')

    if single_img_transforms:
        ds = ds.map(operations=single_img_transforms,
                    input_columns=map_columns,
                    num_parallel_workers=num_parallel_worker)

    ds = ds.project(['image', 'im_file', 'im_id',
                     'ori_shape', 'pad', 'ratio',
                     'gt_bbox', 'gt_class'])

    per_batch_map = getattr(trans_config, 'batch_imgs_transform', None)
    if per_batch_map:
        per_batch_map = create_transform(per_batch_map)
    else:
        per_batch_map = normalize_shape

    input_columns = ['image', 'gt_bbox', 'gt_class']
    batch_column = input_columns + ['batch_idx',]
    ds = ds.batch(per_batch_size,
                  input_columns=input_columns,
                  output_columns=batch_column,
                  per_batch_map=per_batch_map,
                  num_parallel_workers=num_parallel_worker,
                  drop_remainder=drop_remainder
                  )
    ds = ds.repeat(1)

    return ds, dataset
