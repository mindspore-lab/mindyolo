import os
import cv2
import copy
import numpy as np

import transforms


class COCODataset:
    """
    Load detection dataset.
    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): annotation file path.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth.
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total
            record's, if empty_ratio is out of [0. ,1.), do not sample the
            records and use all the empty entries. 1. as default
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 muliti_imgs_transforms=None,
                 most_boxes_per_img=160):
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.sample_num = sample_num
        self.load_image_only = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        self.muliti_imgs_transforms = muliti_imgs_transforms
        self.most_boxes_per_img = most_boxes_per_img
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
                    op_cls = getattr(transforms, k)
                    f = op_cls(**v)
                    if k == 'Mosaic':
                        records_outs = [record_out,] + [copy.deepcopy(self.imgs_records[np.random.randint(n)]) for _ in range(3)] # 3 additional image
                    for record_out in records_outs:
                        if 'image' not in record_out:
                            img_path = record_out['im_file']
                            record_out['image'] = cv2.imread(img_path)  # BGR
                    record_out = f(records_outs)
        else:
            img_path = record_out['im_file']
            record_out['image'] = cv2.imread(img_path)  # BGR
        real_gt_bbox = record_out['gt_bbox']
        real_gt_class = record_out['gt_class']
        nL = real_gt_bbox.shape[0]
        gt_bbox = np.full((self.most_boxes_per_img, 4), -1, dtype=np.float32)
        gt_bbox[:min(nL, self.most_boxes_per_img), :] = real_gt_bbox[:min(nL, self.most_boxes_per_img), :]
        gt_class = np.full((self.most_boxes_per_img, 1), -1, dtype=np.int32)
        gt_class[:min(nL, self.most_boxes_per_img), :] = real_gt_class[:min(nL, self.most_boxes_per_img), :]
        return record_out['image'], record_out['h'], record_out['w'], gt_class, gt_bbox

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
            # logger.warning('Annotation file: {} does not contains ground truth '
            #                'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])
            im_path = os.path.join(image_dir, im_fname) if image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                # logger.warning('Illegal image file: {}, and it will be '
                #                'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                # logger.warning('Illegal width: {} or height: {} in annotation, '
                #                'and im_id: {} will be ignored'.format(
                #     im_w, im_h, img_id))
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

                    x, y, box_w, box_h = inst['bbox']
                    eps = 1e-5
                    if inst['area'] > 0 and box_w > eps and box_h > eps:
                        norm_clean_bbox = [
                            (x + box_w/2) * 1.0 / im_w, (y + box_h/2) * 1.0 / im_h,
                            box_w * 1.0 / im_w, box_h * 1.0 / im_h
                        ]
                        inst['clean_bbox'] = [
                            round(float(x), 6) for x in norm_clean_bbox
                        ]
                        bboxes.append(inst)
                    # else:
                    #     logger.warning(
                    #         'Found an invalid bbox in annotations: im_id: {}, '
                    #         'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                    #             img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox

                has_segmentation = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']

                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(box['segmentation']).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation'][0]
                        has_segmentation = True

                if has_segmentation and not any(gt_poly) and not self.allow_empty:
                    continue

                gt_poly = [np.array(x, dtype=np.float32).reshape(-1, 2) for x in gt_poly]
                for x in gt_poly:
                    x[:, 0] /= im_w
                    x[:, 1] /= im_h

                gt_rec = {
                    'is_crowd': is_crowd,
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