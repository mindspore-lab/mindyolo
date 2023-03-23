import os
import cv2
import random
import numpy as np

from mindyolo.utils import logger
from .copypaste import copy_paste
from .perspective import random_perspective

__all__ = ["COCODataset"]


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
        transforms (list): A list of multi_images data enhancements
            that apply data enhancements on data set objects in order.
    """

    def __init__(self,
                 dataset_dir='',
                 image_dir='',
                 anno_path='',
                 img_size=640,
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=True,
                 transforms_dict=None,
                 is_training=True,
                 rect=False,
                 batch_size=32,
                 stride=32,
                 pad=0.0
                 ):

        # 1. Set Dir
        self.dataset_dir = dataset_dir
        self.anno_path = anno_path
        self.image_dir = image_dir
        self.img_size = img_size
        self.sample_num = sample_num
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.transforms_dict = transforms_dict
        self.is_training = is_training
        self.rect = rect
        self.load_image_only = False
        if is_training:
            self.dataset_column_names = ['image', 'labels', 'img_files']
        else:
            self.dataset_column_names = ['image', 'labels', 'img_files', 'hw_ori', 'hw_scale', 'pad']
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        # 2. COCO Init
        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        ct = 0
        nf, nm, ne = 0, 0, 0
        cat_ids = coco.getCatIds()
        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })
        if 'annotations' not in coco.dataset:
            raise ValueError('Annotation file: {} does not contains ground truth '
                             'and load image information only.'.format(anno_path))

        # 3. Parse Dataset
        self.img_files, self.labels, self.segments, self.img_shapes = [], [], [], []
        for img_id in img_ids:

            # img file
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_path = os.path.join(image_dir, im_fname) if image_dir else im_fname
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue
            nf += 1

            # img size
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])
            if im_w < 10 or im_h < 10:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(im_w, im_h, img_id))
                nm += 1
                continue

            # classes/labels/segment
            ins_anno_ids = coco.getAnnIds(
                imgIds=[img_id], iscrowd=None if self.load_crowd else False)
            instances = coco.loadAnns(ins_anno_ids)

            bboxes, classes, labels, segments = [], [], [], []

            has_segmentation = False
            for i, inst in enumerate(instances):
                # check gt bbox
                if inst.get('ignore', False):
                    continue
                if 'bbox' not in inst.keys():
                    continue
                else:
                    if not any(np.array(inst['bbox'])):
                        continue

                x1, y1, box_w, box_h = inst['bbox']
                xc, yc = x1 + box_w / 2, y1 + box_h / 2
                eps = 1e-5
                if inst['area'] > 0 and box_w > eps and box_h > eps:
                    if 'segmentation' in inst and inst['iscrowd'] == 1:
                        segment = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    elif 'segmentation' in inst and inst['segmentation']:
                        if not np.array(inst['segmentation']).size > 0 and not self.allow_empty:
                            logger.warning(
                                'Found an invalid segment in annotations, drop: im_id: {}, '
                                'area: {} x1: {}, y1: {}, w: {}, h: {}.'.format(
                                    img_id, float(inst['area']), x1, y1, box_w, box_h))
                            continue
                        else:
                            segment = inst['segmentation'][0]

                        has_segmentation = True
                    else:
                        logger.warning(
                            'Found an invalid segment in annotations, drop: im_id: {}, '
                            'area: {} x1: {}, y1: {}, w: {}, h: {}.'.format(
                                img_id, float(inst['area']), x1, y1, box_w, box_h))
                        continue

                    bboxes.append(np.array([xc / im_w, yc / im_h, box_w / im_w, box_h / im_h], np.float32))  # box, xywh
                    catid = inst['category_id']
                    classes.append(np.array([self.catid2clsid[catid],], np.int32))

                    _segment = np.array(segment, np.float32).reshape((-1, 2))
                    _segment[:, 0] /= im_w
                    _segment[:, 1] /= im_h
                    segments.append(_segment)

                else:
                    logger.warning(
                        'Found an invalid bbox in annotations, drop: im_id: {}, '
                        'area: {} x1: {}, y1: {}, w: {}, h: {}.'.format(
                            img_id, float(inst['area']), x1, y1, box_w, box_h))

            if has_segmentation and not segments and not self.allow_empty:
                continue

            num_bbox = len(bboxes)
            if num_bbox == 0:
                ne += 1
                if not self.allow_empty:
                    continue
                bboxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0, 1), dtype=np.int32)
                labels = np.concatenate((classes.reshape(-1, 1), bboxes), 1)
                segments = []
            else:
                # list -> numpy
                bboxes = np.stack(bboxes, 0)
                classes = np.stack(classes, 0)
                labels = np.concatenate((classes.reshape(-1, 1), bboxes), 1)
                # segments = [np.array(x, dtype=np.float32).reshape(-1, 2) for x in segments]
                segments = segments
                assert len(segments) == len(labels)

            self.img_files.append(im_path)
            self.img_shapes.append(np.array([im_w, im_h])) # (width, height)
            self.labels.append(labels)
            self.segments.append(segments)

            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break

        n = len(self.labels)
        self.imgs, self.img_hw_ori = [None, ] * n, [None, ] * n
        self.indices = range(n)

        # 4. Set Rectangular Eval/Train
        if self.rect:
            # Sort by aspect ratio
            s = np.array(self.img_shapes)  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.img_shapes = [self.img_shapes[i] for i in irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
            nb = bi[-1] + 1  # number of batches
            self.batch = bi  # batch index of image
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        assert ct > 0, 'Not found any coco record in %s' % (anno_path)
        logger.info(f"COCO Scanning images and labels... {nf} found, {nm} missing, {ne} empty, {ct} used.")

    def __getitem__(self, index):

        index, image, labels, segment, hw_ori, hw_scale, pad = index, None, None, None, None, None, None
        for _i, ori_trans in enumerate(self.transforms_dict):
            _trans = ori_trans.copy()
            func_name, prob = _trans.pop('func_name'), _trans.pop('prob', 1.0)
            if random.random() < prob:
                if func_name == 'mosaic':
                    image, labels = self.mosaic(index, **_trans)
                elif func_name == 'letterbox':
                    image, hw_ori = self.load_image(index)
                    labels = self.labels[index].copy()
                    new_shape = self.img_size if not self.rect else self.batch_shapes[self.batch[index]]
                    image, labels, hw_ori, hw_scale, pad = self.letterbox(image, labels, hw_ori, new_shape, **_trans)
                else:
                    image, labels = getattr(self, func_name)(image, labels, **_trans)

        image = np.ascontiguousarray(image)

        if self.is_training:
            return image, labels, self.img_files[index]
        else:
            return image, labels, self.img_files[index], hw_ori, hw_scale, pad

    def __len__(self):
        return len(self.img_files)

    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = self.imgs[index]
        if img is None:  # not cached
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, 'Image Not Found ' + path
            h_ori, w_ori = img.shape[:2]  # orig hw
            r = self.img_size / max(h_ori, w_ori)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)

            return img, np.array([h_ori, w_ori])  # img, hw_original
        else:
            return self.imgs[index], self.img_hw_ori[index]  # img, hw_original

    def load_samples(self, index):
        # loads images in a 4-mosaic
        labels4, segments4 = [], []
        s = self.img_size
        mosaic_border = [-s // 2, -s // 2]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _ = self.load_image(index)
            (h, w) = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # Augment
        sample_labels, sample_images, sample_masks = self._sample_segments(img4, labels4, segments4, probability=0.5)

        return sample_labels, sample_images, sample_masks

    def mosaic(self,
               index,
               mosaic9_prob=0.,
               copy_paste_prob=0.0,
               degrees=0.0,
               translate=0.2,
               scale=0.9,
               shear=0.0,
               perspective=0.0):

        assert mosaic9_prob >= 0. and mosaic9_prob <= 1.
        if random.random() < (1 - mosaic9_prob):
            return self.mosaic4(index, copy_paste_prob, degrees, translate, scale, shear, perspective)
        else:
            return self.mosaic9(index, copy_paste_prob, degrees, translate, scale, shear, perspective)

    def mosaic4(self, index, copy_paste_prob=0.0, degrees=0.0, translate=0.2, scale=0.9, shear=0.0, perspective=0.0):
        # loads images in a 4-mosaic
        labels4, segments4 = [], []
        s = self.img_size
        mosaic_border = [-s // 2, -s // 2]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _ = self.load_image(index)
            (h, w) = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, probability=copy_paste_prob)
        img4, labels4 = random_perspective(img4, labels4, segments4,
                                           degrees=degrees,
                                           translate=translate,
                                           scale=scale,
                                           shear=shear,
                                           perspective=perspective,
                                           border=mosaic_border)  # border to remove

        return img4, labels4

    def mosaic9(self, index, copy_paste_prob=0.0, degrees=0.0, translate=0.2, scale=0.9, shear=0.0, perspective=0.0):
        # loads images in a 9-mosaic

        labels9, segments9 = [], []
        s = self.img_size
        mosaic_border = [-s // 2, -s // 2]
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _ = self.load_image(index)
            (h, w) = img.shape[:2]

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = [int(random.uniform(0, s)) for _ in mosaic_border]  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # Augment
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, probability=copy_paste_prob)
        img9, labels9 = random_perspective(img9, labels9, segments9,
                                           degrees=degrees,
                                           translate=translate,
                                           scale=scale,
                                           shear=shear,
                                           perspective=perspective,
                                           border=mosaic_border)  # border to remove

        return img9, labels9

    def mixup(self, image, labels, alpha=8.0, beta=8.0, needed_mosaic=True):
        if needed_mosaic:
            mosaic_trans = None
            for _trans in self.transforms_dict:
                if _trans['func_name'] == 'mosaic':
                    mosaic_trans = _trans.copy()
                    break
            assert mosaic_trans is not None, "Mixup needed mosaic bug 'mosaic' not in transforms_dict"
            _, _ = mosaic_trans.pop('func_name'), mosaic_trans.pop('prob', 1.0)
            image2, labels2 = self.mosaic(random.randint(0, len(self.labels) - 1), **mosaic_trans)
        else:
            index2 = random.randint(0, len(self.labels) - 1)
            image2, _ = self.load_image(index2)
            labels2 = self.labels[index2]

        r = np.random.beta(alpha, beta)  # mixup ratio, alpha=beta=8.0
        image = (image * r + image2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
        return image, labels

    def pastein(self, image, labels, num_sample=30):

        # load sample
        sample_labels, sample_images, sample_masks = [], [], []
        while len(sample_labels) < num_sample:
            sample_labels_, sample_images_, sample_masks_ = \
                self.load_samples(random.randint(0, len(self.labels) - 1))
            sample_labels += sample_labels_
            sample_images += sample_images_
            sample_masks += sample_masks_
            if len(sample_labels) == 0:
                break

        # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
        h, w = image.shape[:2]

        # create random masks
        scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
        for s in scales:
            if random.random() < 0.2:
                continue
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            if len(labels):
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            else:
                ioa = np.zeros(1)

            if (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin + 20) and (
                    ymax > ymin + 20):  # allow 30% obscuration of existing labels
                sel_ind = random.randint(0, len(sample_labels) - 1)
                hs, ws, cs = sample_images[sel_ind].shape
                r_scale = min((ymax - ymin) / hs, (xmax - xmin) / ws)
                r_w = int(ws * r_scale)
                r_h = int(hs * r_scale)

                if (r_w > 10) and (r_h > 10):
                    r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                    r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                    temp_crop = image[ymin:ymin + r_h, xmin:xmin + r_w]
                    m_ind = r_mask > 0
                    if m_ind.astype(np.int).sum() > 60:
                        temp_crop[m_ind] = r_image[m_ind]
                        box = np.array([xmin, ymin, xmin + r_w, ymin + r_h], dtype=np.float32)
                        if len(labels):
                            labels = np.concatenate((labels, [[sample_labels[sel_ind], *box]]), 0)
                        else:
                            labels = np.array([[sample_labels[sel_ind], *box]])

                        image[ymin:ymin + r_h, xmin:xmin + r_w] = temp_crop # Modify on the original image

        return image, labels

    def hsv_augment(self, image, labels, hgain=0.5, sgain=0.5, vgain=0.5):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)  # Modify on the original image
        return image, labels

    def fliplr(self, image, labels):
        # flip left-right
        image = np.fliplr(image)
        if len(labels):
            labels[:, 1] = 1 - labels[:, 1]
        return image, labels

    def flipud(self, image, labels):
        # flip up-down
        image = np.flipud(image)
        if len(labels):
            labels[:, 2] = 1 - labels[:, 2]
        return image, labels

    def letterbox(self, image, labels, hw_ori, new_shape, scaleup=False, color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[:2]  # current shape [height, width]
        h, w = shape[:]
        h0, w0 = hw_ori
        hw_scale = np.array([h / h0, w / w0])
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        hw_pad = np.array([dh, dw])

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        # convert labels
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], r * w, r * h, padw=hw_pad[1], padh=hw_pad[0])

        return image, labels, hw_ori, hw_scale, hw_pad

    def label_norm(self, image, labels, xyxy2xywh_=True):
        if len(labels) == 0:
            return image, labels

        if xyxy2xywh_:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh

        labels[:, [2, 4]] /= image.shape[0]  # normalized height 0-1
        labels[:, [1, 3]] /= image.shape[1]  # normalized width 0-1

        return image, labels

    def label_pad(self, image, labels, padding_size=160, padding_value=-1):
        # create fixed label, avoid dynamic shape problem.
        labels_out = np.full((padding_size, 6), padding_value, dtype=np.float32)
        nL = len(labels)
        if nL:
            labels_out[:min(nL, padding_size), 0:1] = 0.
            labels_out[:min(nL, padding_size), 1:] = labels[:min(nL, padding_size), :]
        return image, labels_out

    def image_norm(self, image, labels, scale=255.):
        image = image.astype(np.float32, copy=False)
        image /= scale
        return image, labels

    def image_transpose(self, image, labels, bgr2rgb=True, hwc2chw=True):
        if bgr2rgb:
            image = image[:, :, ::-1]
        if hwc2chw:
            image = image.transpose(2, 0, 1)
        return image, labels

    def _sample_segments(self, img, labels, segments, probability=0.5):
        # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
        n = len(segments)
        sample_labels = []
        sample_images = []
        sample_masks = []
        if probability and n:
            h, w, c = img.shape  # height, width, channels
            for j in random.sample(range(n), k=round(probability * n)):
                l, s = labels[j], segments[j]
                box = l[1].astype(int).clip(0, w - 1), l[2].astype(int).clip(0, h - 1), l[3].astype(int).clip(0, w - 1), \
                      l[
                          4].astype(int).clip(0, h - 1)

                if (box[2] <= box[0]) or (box[3] <= box[1]):
                    continue

                sample_labels.append(l[0])

                mask = np.zeros(img.shape, np.uint8)

                cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
                sample_masks.append(mask[box[1]:box[3], box[0]:box[2], :])

                result = cv2.bitwise_and(src1=img, src2=mask)
                i = result > 0  # pixels to replace
                mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
                sample_images.append(mask[box[1]:box[3], box[0]:box[2], :])

        return sample_labels, sample_images, sample_masks

    @staticmethod
    def train_collate_fn(imgs, labels, path, batch_info):
        for i, l in enumerate(labels):
            l[:, 0] = i  # add target image index for build_targets()
        return np.stack(imgs, 0), np.stack(labels, 0), path

    @staticmethod
    def test_collate_fn(imgs, labels, path, hw_ori, hw_scale, pad, batch_info):
        for i, l in enumerate(labels):
            l[:, 0] = i  # add target image index for build_targets()
        return np.stack(imgs, 0), np.stack(labels, 0), path, \
               np.stack(hw_ori, 0), np.stack(hw_scale, 0), np.stack(pad, 0)


def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y
