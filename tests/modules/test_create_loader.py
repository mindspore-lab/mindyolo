import sys

sys.path.append(".")

import os
import pytest
from tqdm import tqdm
import zipfile
import urllib.request

import mindspore as ms

from mindyolo.data import COCODataset, create_loader



@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("drop_remainder", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_create_loader(mode, drop_remainder, shuffle, batch_size):
    dataset_url = (
        "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    )
    if not os.path.exists('./coco128'):
        with open('./coco128.zip', "wb") as f:
            request = urllib.request.Request(dataset_url)
            with urllib.request.urlopen(request) as response:
                with tqdm(total=response.length, unit="B") as pbar:
                    for chunk in iter(lambda: response.read(1024), b""):
                        if not chunk:
                            break
                        pbar.update(1024)
                        f.write(chunk)
        compression_mode = zipfile.ZIP_STORED
        with zipfile.ZipFile('./coco128.zip', "r", compression=compression_mode) as zip_file:
            zip_file.extractall()

    dataset_path = './coco128'
    ms.set_context(mode=mode)
    transforms_dict = [
        {'func_name': 'resample_segments'},
        {'func_name': 'mosaic', 'prob': 1.0},
        {'func_name': 'random_perspective', 'prob': 1.0, 'translate': 0.1, 'scale': 0.9},
        {'func_name': 'mixup', 'prob': 0.1, 'alpha': 8.0, 'beta': 8.0, 'pre_transform': [
            {'func_name': 'resample_segments'},
            { 'func_name': 'mosaic', 'prob': 1.0 },
            { 'func_name': 'random_perspective', 'prob': 1.0, 'translate': 0.1, 'scale': 0.9},
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
        dataset_path=dataset_path,
        transforms_dict=transforms_dict,
        img_size=640,
        is_training=True,
        augment=True,
        batch_size=batch_size,
        stride=64,
    )
    dataloader = create_loader(
        dataset=dataset,
        batch_collate_fn=dataset.train_collate_fn,
        column_names_getitem=dataset.column_names_getitem,
        column_names_collate=dataset.column_names_collate,
        batch_size=batch_size,
        epoch_size=1,
        shuffle=shuffle,
        drop_remainder=drop_remainder,
        num_parallel_workers=1,
        python_multiprocessing=True,
    )

    out_batch_size = dataloader.get_batch_size()
    out_shapes = dataloader.output_shapes()[0]
    assert out_batch_size == batch_size
    assert out_shapes == [batch_size, 3, 640, 640]

    for data in dataset:
        assert data is not None


if __name__ == '__main__':
    test_create_loader()
