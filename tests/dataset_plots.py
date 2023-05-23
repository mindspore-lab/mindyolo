import cv2

from mindyolo.data.dataset import COCODataset
from mindyolo.data.loader import create_loader
from mindyolo.data.poly import show_img_with_bbox
from mindyolo.utils.config import parse_args

if __name__ == "__main__":
    cfg = parse_args("eval")

    dataset = COCODataset(
        dataset_path=cfg.data.train_set,
        img_size=cfg.img_size,
        transforms_dict=cfg.data.test_transforms,
        is_training=False,
        rect=False,
        batch_size=cfg.per_batch_size * 2,
        stride=max(cfg.network.stride),
    )
    dataloader = create_loader(
        dataset=dataset,
        batch_collate_fn=dataset.test_collate_fn,
        dataset_column_names=dataset.dataset_column_names,
        batch_size=cfg.per_batch_size * 2,
        epoch_size=1,
        rank=0,
        rank_size=1,
        shuffle=False,
        drop_remainder=False,
        num_parallel_workers=cfg.data.num_parallel_workers,
        python_multiprocessing=True,
    )
    data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)

    for i, data in enumerate(data_loader):
        img = show_img_with_bbox(data, cfg.data.names)
        # img = show_img_with_poly(data)
        cv2.namedWindow("img", cv2.WINDOW_FREERATIO)
        cv2.imshow("img", img)
        cv2.waitKey(0)
