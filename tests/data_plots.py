import cv2

from mindyolo.data.loader import create_dataloader
from mindyolo.data.general import show_img_with_bbox, show_img_with_poly
from mindyolo.utils.config import parse_config

if __name__ == '__main__':
    config = parse_config()
    ds, dataset = create_dataloader(data_config=config.data,
                                    task=config.task,
                                    per_batch_size=config.per_batch_size,)
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
    print('done')

    for i, data in enumerate(data_loader):
        img = show_img_with_bbox(data, config.data.names)
        # img = show_img_with_poly(data)
        cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        cv2.imshow('img', img)
        cv2.waitKey(0)
