import cv2

from mindyolo.data.dataset import COCODataset
from mindyolo.data.loader import create_loader
from mindyolo.utils.poly import show_img_with_bbox
from mindyolo.utils.config import _parse_yaml, Config

if __name__ == "__main__":
    cfg, cfg_helper, cfg_choices = _parse_yaml('../configs/coco.yaml')
    cfg = Config(cfg)

    transforms_dict = [
        {'func_name': 'resample_segments'},
        {'func_name': 'mosaic', 'prob': 1.0},
        {'func_name': 'random_perspective', 'prob': 1.0, 'translate': 0.1, 'scale': 0.9},
        {'func_name': 'mixup', 'prob': 0.1, 'alpha': 8.0, 'beta': 8.0, 'pre_transform': [
            {'func_name': 'resample_segments'},
            {'func_name': 'mosaic', 'prob': 1.0},
            {'func_name': 'random_perspective', 'prob': 1.0, 'translate': 0.1, 'scale': 0.9},
        ]},
        {'func_name': 'hsv_augment', 'prob': 1.0, 'hgain': 0.015, 'sgain': 0.7, 'vgain': 0.4},
        {'func_name': 'label_norm', 'xyxy2xywh_': True},
        {'func_name': 'albumentations'},
        {'func_name': 'fliplr', 'prob': 0.5},
        {'func_name': 'label_pad', 'padding_size': 160, 'padding_value': -1},
        {'func_name': 'image_norm', 'scale': 255.},
        {'func_name': 'image_transpose', 'bgr2rgb': True, 'hwc2chw': True}
    ]

    dataset = COCODataset(
        dataset_path=cfg.data.train_set,
        transforms_dict=transforms_dict,
        is_training=True
    )
    dataloader = create_loader(
        dataset=dataset,
        batch_collate_fn=dataset.train_collate_fn,
        column_names_getitem=dataset.column_names_getitem,
        column_names_collate=dataset.column_names_collate,
        batch_size=4,
        epoch_size=1,
        shuffle=False,
        drop_remainder=True,
        python_multiprocessing=True,
    )
    data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)

    for i, data in enumerate(data_loader):
        img = show_img_with_bbox(data, cfg.data.names)
        cv2.namedWindow("img", cv2.WINDOW_FREERATIO)
        cv2.imshow("img", img)
        cv2.waitKey(0)
