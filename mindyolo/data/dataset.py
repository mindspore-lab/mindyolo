import os
import sys
import cv2
import copy
import numpy as np

import transforms
from general import resample_polys

sys.path.append('../')
from utils import logger
logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=0, device_per_servers=8)
logger.setup_logging_file(log_dir="./logs")


class COCODataset:
    """
    Load the COCO dataset, parse the labels of each image to form a list of dictionaries,
    apply multi_images fusion data enhancement in __getitem__()
    COCO dataset download URL: http://cocodataset.org

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): annotation file path.
        for example:
            COCO_ROOT
                ├── annotations
                │     └── instances_train2017.json
                └── train2017
                      ├── 000000000001.jpg
                      └── 000000000002.jpg
            dataset_dir (str): ./COCO_ROOT
            image_dir (str): ./train2017
            anno_path (str): ./annotations/instances_train2017.json

        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth.
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total
            record's, if empty_ratio is out of [0. ,1.), do not sample the
            records and use all the empty entries. 1. as default
        multi_imgs_transforms (list): A list of multi_images data enhancements
            that apply data enhancements on data set objects in order.
        is_segmentaion (bool): whether the task is segmentaion.
            False as default
        detection_require_poly (bool): whether the gt_poly is required for detection data enhancement.
            False as default
    """

    def __init__(self,
                 dataset_dir='',
                 image_dir='',
                 anno_path='',
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 multi_imgs_transforms=None,
                 is_segmentaion=False,
                 detection_require_poly=False
                 ):
        self.dataset_dir = dataset_dir
        self.anno_path = anno_path
        self.image_dir = image_dir
        self.sample_num = sample_num
        self.load_image_only = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        self.muliti_imgs_transforms = multi_imgs_transforms
        self.is_segmentaion = is_segmentaion
        self.detection_require_poly = detection_require_poly
        self.parse_dataset()

    def __len__(self, ):
        return len(self.imgs_records)

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, idx):
        n = len(self.imgs_records)
        record_out = copy.deepcopy(self.imgs_records[idx])

        # multi_imgs_transforms
        if self.muliti_imgs_transforms:
            for t in self.muliti_imgs_transforms:
                for k, v in t.items():
                    op_cls = getattr(transforms, k) # get class name of the operator
                    f = op_cls(**v) # Instantiate the class object

                    # get the other images
                    if k == 'Mosaic':
                        records_outs = [record_out, ] + [copy.deepcopy(self.imgs_records[np.random.randint(n)]) for _ in range(3)] # 3 additional image
                    elif k == 'PasteIn':
                        records_outs = [record_out, ] + [copy.deepcopy(self.imgs_records[np.random.randint(n)]) for _ in range(120)]
                    elif k == 'MixUp':
                        records_outs = [record_out, ] + [copy.deepcopy(self.imgs_records[np.random.randint(n)]) for _ in range(7)]
                    elif k == 'SimpleCopyPaste':
                        records_outs = [record_out, ]
                    # apply the multi_images data enhancements in turn
                    for record_out in records_outs:
                        if 'image' not in record_out:
                            img_path = record_out['im_file']
                            record_out['image'] = cv2.imread(img_path)  # BGR

                    record_out = f(records_outs)
        else:
            img_path = record_out['im_file']
            record_out['image'] = cv2.imread(img_path)  # BGR

        if self.detection_require_poly:
            return record_out['image'], record_out['w'], record_out['h'], record_out['gt_bbox'], record_out['gt_class'], record_out['gt_poly']
        else:
            return record_out['image'], record_out['w'], record_out['h'], record_out['gt_bbox'], record_out['gt_class']

    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = np.random.sample(records, sample_num)
        return records

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        empty_records = []
        ct = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])
            im_path = os.path.join(image_dir, im_fname) if image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(im_w, im_h, img_id))
                continue

            img_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            }

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            continue

                    x1, y1, box_w, box_h = inst['bbox']
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 and box_w > eps and box_h > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox

                has_segmentation = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']

                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(box['segmentation']).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation'][0]
                            if self.detection_require_poly:
                                gt_poly[i] = resample_polys(gt_poly[i])
                        has_segmentation = True

                if has_segmentation and not any(gt_poly) and not self.allow_empty:
                    continue

                gt_poly = [np.array(x, dtype=np.float32).reshape(-1, 2) for x in gt_poly]

                gt_rec = {
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_poly': gt_poly,
                }

                for k, v in gt_rec.items():
                    img_rec[k] = v

            # logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(
            #     im_path, img_id, im_h, im_w))
            if is_empty:
                empty_records.append(img_rec)
            else:
                records.append(img_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        # logger.debug('{} samples in file {}'.format(ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.imgs_records = records


if __name__ == '__main__':
    from general import show_img_with_bbox, show_img_with_poly
    from mindspore import context
    import sys
    sys.path.append('../')
    from utils.config import parse_config

    context.set_context(mode=context.PYNATIVE_MODE, pynative_synchronize=True)
    config = parse_config()
    data_config = config.Data
    image_dir = data_config.train_img_dir
    anno_path = data_config.train_anno_path
    multi_imgs_transforms = getattr(data_config, 'multi_imgs_transforms', None)

    dataset = COCODataset(dataset_dir=data_config.dataset_dir, image_dir=image_dir, anno_path=anno_path,
                          multi_imgs_transforms=multi_imgs_transforms)
    print('done')
    data = dataset[0]
    img = show_img_with_bbox(data, config.Data.names)
    # img = show_img_with_poly(data)
    cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
    cv2.imshow('img', img)
    cv2.waitKey(0)