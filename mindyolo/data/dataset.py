import os
import cv2
import math
import hashlib
import random
import glob
import numpy as np
from pathlib import Path
from PIL import ExifTags, Image
from tqdm import tqdm
from copy import deepcopy

from mindyolo.utils import logger
from mindyolo.data.albumentations import Albumentations
from mindyolo.data.utils import xywhn2xyxy, xyxy2xywh, xyn2xy, segment2box, segments2boxes, \
    box_candidates, polygons2masks, polygons2masks_overlap, bbox_ioa

__all__ = ["COCODataset"]


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


class COCODataset:
    """
    Load the COCO dataset (yolo format coco labels)

    Args:
        dataset_path (str): dataset label directory for dataset.
        for example:
            COCO_ROOT
                ├── train2017.txt
                ├── annotations
                │     └── instances_train2017.json
                ├── images
                │     └── train2017
                │             ├── 000000000001.jpg
                │             └── 000000000002.jpg
                └── labels
                      └── train2017
                              ├── 000000000001.txt
                              └── 000000000002.txt
            dataset_path (str): ./coco/train2017.txt
        transforms (list): A list of images data enhancements
            that apply data enhancements on data set objects in order.
    """

    def __init__(
        self,
        dataset_path="",
        img_size=640,
        transforms_dict=None,
        is_training=False,
        augment=False,
        rect=False,
        single_cls=False,
        batch_size=32,
        stride=32,
        num_cls=80,
        pad=0.0,
        return_segments=False,  # for segment
        return_keypoints=False, # for keypoint
        nkpt=0,                 # for keypoint
        ndim=0                  # for keypoint
    ):
        # acceptable image suffixes
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
        self.cache_version = 0.2

        self.return_segments = return_segments
        self.return_keypoints = return_keypoints
        assert not (return_segments and return_keypoints), 'Can not return both segments and keypoints.'

        self.path = dataset_path
        self.img_size = img_size
        self.augment = augment
        self.rect = rect
        self.stride = stride
        self.num_cls = num_cls
        self.nkpt = nkpt
        self.ndim = ndim
        self.transforms_dict = transforms_dict
        self.is_training = is_training

        # set column names
        self.column_names_getitem = ['im_file', 'cls', 'bboxes', 'segments', 'keypoints', 'bbox_format', 'segment_format', 
                                     'img', 'ori_shape', 'hw_scale', 'hw_pad'] if self.is_training else ['samples']
        if self.is_training:
            self.column_names_collate = ['images', 'labels']
            if self.return_segments:
                self.column_names_collate = ['images', 'labels', 'masks']
            elif self.return_keypoints:
                self.column_names_collate = ['images', 'labels', 'keypoints']
        else:
            self.column_names_collate = ["images", "img_files", "hw_ori", "hw_scale", "pad"]

        try:
            f = []  # image files
            for p in self.path if isinstance(self.path, list) else [self.path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.is_file():  # file
                    with open(p, "r") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                else:
                    raise Exception(f"{p} does not exist")
            self.img_files = sorted([x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in self.img_formats])
            assert self.img_files, f"No images found"
        except Exception as e:
            raise Exception(f"Error loading data from {self.path}: {e}\n")

        # Check cache
        self.label_files = self._img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache.npy")  # cached labels
        if cache_path.is_file():
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            if cache["version"] == self.cache_version \
                    and cache["hash"] == self._get_hash(self.label_files + self.img_files):
                logger.info(f"Dataset Cache file hash/version check success.")
                logger.info(f"Load dataset cache from [{cache_path}] success.")
            else:
                logger.info(f"Dataset cache file hash/version check fail.")
                logger.info(f"Datset caching now...")
                cache, exists = self.cache_labels(cache_path), False  # cache
                logger.info(f"Dataset caching success.")
        else:
            logger.info(f"No dataset cache available, caching now...")
            cache, exists = self.cache_labels(cache_path), False  # cache
            logger.info(f"Dataset caching success.")

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f"No labels in {cache_path}. Can not train without labels."

        # Read cache
        cache.pop("hash")  # remove hash
        cache.pop("version")  # remove version
        self.labels = cache['labels']
        self.img_files = [lb['im_file'] for lb in self.labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in self.labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            print(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in self.labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels.')

        if single_cls:
            for x in self.labels:
                x['cls'][:, 0] = 0

        n = len(self.labels)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int_)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_hw_ori, self.indices = [None] * n, [None] * n, range(n)
        # Buffer thread for mosaic images
        self.buffer = []
        self.max_buffer_length = min((n, batch_size * 8, 1000)) if self.augment else 0

        # Rectangular Train/Test
        if self.rect:
            # Sort by aspect ratio
            s = self.img_shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.img_shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int_) * stride

        self.imgIds = [int(Path(im_file).stem) for im_file in self.img_files]

    def cache_labels(self, path=Path("./labels.cache.npy")):
        # Cache dataset labels, check images and read shapes
        x = {'labels': []}  # dict
        nm, nf, ne, nc, segments, keypoints = 0, 0, 0, 0, [], None  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc="Scanning images", total=len(self.img_files))
        if self.return_keypoints and (self.nkpt <= 0 or self.ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = self._exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
                assert im.format.lower() in self.img_formats, f"invalid image format {im.format}"

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, "r") as f:
                        lb = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 6 for x in lb]) and (not self.return_keypoints):  # is segment
                            classes = np.array([x[0] for x in lb], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                            lb = np.concatenate(
                                (classes.reshape(-1, 1), segments2boxes(segments)), 1
                            )  # (cls, xywh)
                        lb = np.array(lb, dtype=np.float32)
                    nl = len(lb)
                    if nl:
                        if self.return_keypoints:
                            assert lb.shape[1] == (5 + self.nkpt * self.ndim), \
                                f'labels require {(5 + self.nkpt * self.ndim)} columns each'
                            assert (lb[:, 5::self.ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                            assert (lb[:, 6::self.ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        else:
                            assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                            assert (lb[:, 1:] <= 1).all(), \
                                f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                            assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                        # All labels
                        max_cls = int(lb[:, 0].max())  # max label count
                        assert max_cls <= self.num_cls, \
                            f'Label class {max_cls} exceeds dataset class count {self.num_cls}. ' \
                            f'Possible class labels are 0-{self.num_cls - 1}'
                        _, j = np.unique(lb, axis=0, return_index=True)
                        if len(j) < nl:  # duplicate row check
                            lb = lb[j]  # remove duplicates
                            if segments:
                                segments = [segments[x] for x in i]
                            print(f'WARNING ⚠️ {im_file}: {nl - len(j)} duplicate labels removed')
                    else:
                        ne += 1  # label empty
                        lb = np.zeros((0, (5 + self.nkpt * self.ndim)), dtype=np.float32) \
                            if self.return_keypoints else np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    lb = np.zeros((0, (5 + self.nkpt * self.ndim)), dtype=np.float32) \
                        if self.return_keypoints else np.zeros((0, 5), dtype=np.float32)
                if self.return_keypoints:
                    keypoints = lb[:, 5:].reshape(-1, self.nkpt, self.ndim)
                    if self.ndim == 2:
                        kpt_mask = np.ones(keypoints.shape[:2], dtype=np.float32)
                        kpt_mask = np.where(keypoints[..., 0] < 0, 0.0, kpt_mask)
                        kpt_mask = np.where(keypoints[..., 1] < 0, 0.0, kpt_mask)
                        keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
                lb = lb[:, :5]
                x['labels'].append(
                    dict(
                        im_file=im_file,
                        cls=lb[:, 0:1],     # (n, 1)
                        bboxes=lb[:, 1:],   # (n, 4)
                        segments=segments,  # list of (mi, 2)
                        keypoints=keypoints,
                        bbox_format='xywhn',
                        segment_format='polygon'
                    )
                )
            except Exception as e:
                nc += 1
                print(f"WARNING: Ignoring corrupted image and/or label {im_file}: {e}")

            pbar.desc = f"Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f"WARNING: No labels found in {path}.")

        x["hash"] = self._get_hash(self.label_files + self.img_files)
        x["results"] = nf, nm, ne, nc, len(self.img_files)
        x["version"] = self.cache_version  # cache version
        np.save(path, x)  # save for next time
        logger.info(f"New cache created: {path}")
        return x

    def __getitem__(self, index):
        sample = self.get_sample(index)

        for _i, ori_trans in enumerate(self.transforms_dict):
            _trans = ori_trans.copy()
            func_name, prob = _trans.pop("func_name"), _trans.pop("prob", 1.0)
            if func_name == 'copy_paste':
                sample = self.copy_paste(sample, prob)
            elif random.random() < prob:
                if func_name == "albumentations" and getattr(self, "albumentations", None) is None:
                    self.albumentations = Albumentations(size=self.img_size, **_trans)
                if func_name == "letterbox":
                    new_shape = self.img_size if not self.rect else self.batch_shapes[self.batch[index]]
                    sample = self.letterbox(sample, new_shape, **_trans)
                else:
                    sample = getattr(self, func_name)(sample, **_trans)

        sample['img'] = np.ascontiguousarray(sample['img'])
        if self.is_training:
            train_sample = []
            for col_name in self.column_names_getitem:
                if sample.get(col_name) is None:
                    train_sample.append(np.nan)
                else:
                    train_sample.append(sample.get(col_name, np.nan))
            return tuple(train_sample)
        return sample

    def __len__(self):
        return len(self.img_files)

    def get_sample(self, index):
        """Get and return label information from the dataset."""
        sample = deepcopy(self.labels[index])
        if self.imgs is None:
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, "Image Not Found " + path
            h_ori, w_ori = img.shape[:2]  # orig hw
            r = self.img_size / max(h_ori, w_ori)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)

            sample['img'], sample['ori_shape'] = img, np.array([h_ori, w_ori])  # img, hw_original
            if self.augment:
                self.imgs[index], self.img_hw_ori[index] = img, np.array([h_ori, w_ori]) # img, hw_original
                self.buffer.append(index)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.imgs[j], self.img_hw_ori[j] = None, np.array([None, None])
        else:
            sample['img'], sample['ori_shape'] = self.imgs[index], self.img_hw_ori[index]  # img, hw_original

        return sample

    def mosaic(
        self,
        sample,
        mosaic9_prob=0.0,
        post_transform=None,
    ):
        segment_format = sample['segment_format']
        bbox_format = sample['bbox_format']
        assert segment_format == 'polygon', f'The segment format should be polygon, but got {segment_format}'
        assert bbox_format == 'xywhn', f'The bbox format should be xywhn, but got {bbox_format}'

        mosaic9_prob = min(1.0, max(mosaic9_prob, 0.0))
        if random.random() < (1 - mosaic9_prob):
            sample = self._mosaic4(sample)
        else:
            sample = self._mosaic9(sample)

        if post_transform:
            for _i, ori_trans in enumerate(post_transform):
                _trans = ori_trans.copy()
                func_name, prob = _trans.pop("func_name"), _trans.pop("prob", 1.0)
                sample = getattr(self, func_name)(sample, **_trans)

        return sample

    def _mosaic4(self, sample):
        # loads images in a 4-mosaic
        classes4, bboxes4, segments4 = [], [], []
        mosaic_samples = [sample, ]
        indices = random.choices(self.buffer, k=3)  # 3 additional image indices

        segments_is_list = isinstance(sample['segments'], list)
        if segments_is_list:
            mosaic_samples += [self.get_sample(i) for i in indices]
        else:
            mosaic_samples += [self.resample_segments(self.get_sample(i)) for i in indices]

        s = self.img_size
        mosaic_border = [-s // 2, -s // 2]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y

        for i, mosaic_sample in enumerate(mosaic_samples):
            # Load image
            img = mosaic_sample['img']
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

            # box and cls
            cls, bboxes = mosaic_sample['cls'], mosaic_sample['bboxes']
            assert mosaic_sample['bbox_format'] == 'xywhn'
            bboxes = xywhn2xyxy(bboxes, w, h, padw, padh)  # normalized xywh to pixel xyxy format
            classes4.append(cls)
            bboxes4.append(bboxes)

            # seg
            assert mosaic_sample['segment_format'] == 'polygon'
            segments = mosaic_sample['segments']
            if segments_is_list:
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
                segments4.extend(segments)
            else:
                segments = xyn2xy(segments, w, h, padw, padh)
                segments4.append(segments)

        classes4 = np.concatenate(classes4, 0)
        bboxes4 = np.concatenate(bboxes4, 0)
        bboxes4 = bboxes4.clip(0, 2 * s)

        if segments_is_list:
            for x in segments4:
                np.clip(x, 0, 2 * s, out=x)
        else:
            segments4 = np.concatenate(segments4, 0)
            segments4 = segments4.clip(0, 2 * s)

        sample['img'] = img4
        sample['cls'] = classes4
        sample['bboxes'] = bboxes4
        sample['bbox_format'] = 'ltrb'
        sample['segments'] = segments4
        sample['mosaic_border'] = mosaic_border

        return sample

    def _mosaic9(self, sample):
        # loads images in a 9-mosaic
        classes9, bboxes9, segments9 = [], [], []
        mosaic_samples = [sample, ]
        indices = random.choices(self.buffer, k=8)  # 8 additional image indices

        segments_is_list = isinstance(sample['segments'], list)
        if segments_is_list:
            mosaic_samples += [self.get_sample(i) for i in indices]
        else:
            mosaic_samples += [self.resample_segments(self.get_sample(i)) for i in indices]
        s = self.img_size
        mosaic_border = [-s // 2, -s // 2]

        for i, mosaic_sample in enumerate(mosaic_samples):
            # Load image
            img = mosaic_sample['img']
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

            # box and cls
            assert mosaic_sample['bbox_format'] == 'xywhn'
            cls, bboxes = mosaic_sample['cls'], mosaic_sample['bboxes']
            bboxes = xywhn2xyxy(bboxes, w, h, padx, pady)  # normalized xywh to pixel xyxy format
            classes9.append(cls)
            bboxes9.append(bboxes)

            # seg
            assert mosaic_sample['segment_format'] == 'polygon'
            segments = mosaic_sample['segments']
            if segments_is_list:
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
                segments9.extend(segments)
            else:
                segments = xyn2xy(segments, w, h, padx, pady)
                segments9.append(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = [int(random.uniform(0, s)) for _ in mosaic_border]  # mosaic center x, y
        img9 = img9[yc: yc + 2 * s, xc: xc + 2 * s]

        # Concat/clip labels
        classes9 = np.concatenate(classes9, 0)
        bboxes9 = np.concatenate(bboxes9, 0)
        bboxes9[:, [0, 2]] -= xc
        bboxes9[:, [1, 3]] -= yc
        bboxes9 = bboxes9.clip(0, 2 * s)

        if segments_is_list:
            c = np.array([xc, yc])  # centers
            segments9 = [x - c for x in segments9]
            for x in segments9:
                np.clip(x, 0, 2 * s, out=x)
        else:
            segments9 = np.concatenate(segments9, 0)
            segments9[..., 0] -= xc
            segments9[..., 1] -= yc
            segments9 = segments9.clip(0, 2 * s)

        sample['img'] = img9
        sample['cls'] = classes9
        sample['bboxes'] = bboxes9
        sample['bbox_format'] = 'ltrb'
        sample['segments'] = segments9
        sample['mosaic_border'] = mosaic_border

        return sample

    def resample_segments(self, sample, n=1000):
        segment_format = sample['segment_format']
        assert segment_format == 'polygon', f'The segment format is should be polygon, but got {segment_format}'

        segments = sample['segments']
        if len(segments) > 0:
            # Up-sample an (n,2) segment
            for i, s in enumerate(segments):
                s = np.concatenate((s, s[0:1, :]), axis=0)
                x = np.linspace(0, len(s) - 1, n)
                xp = np.arange(len(s))
                segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
            segments = np.stack(segments, axis=0)
        else:
            segments = np.zeros((0, 1000, 2), dtype=np.float32)
        sample['segments'] = segments
        return sample

    def copy_paste(self, sample, probability=0.5):
        # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
        bbox_format, segment_format = sample['bbox_format'], sample['segment_format']
        assert bbox_format == 'ltrb', f'The bbox format should be ltrb, but got {bbox_format}'
        assert segment_format == 'polygon', f'The segment format should be polygon, but got {segment_format}'

        img = sample['img']
        cls = sample['cls']
        bboxes = sample['bboxes']
        segments = sample['segments']

        n = len(segments)
        if probability and n:
            h, w, _ = img.shape  # height, width, channels
            im_new = np.zeros(img.shape, np.uint8)
            for j in random.sample(range(n), k=round(probability * n)):
                c, l, s = cls[j], bboxes[j], segments[j]
                box = w - l[2], l[1], w - l[0], l[3]
                ioa = bbox_ioa(box, bboxes)  # intersection over area
                if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                    cls = np.concatenate((cls, [c]), 0)
                    bboxes = np.concatenate((bboxes, [box]), 0)
                    if isinstance(segments, list):
                        segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                    else:
                        segments = np.concatenate((segments, [np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1)]), 0)
                    cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

            result = cv2.bitwise_and(src1=img, src2=im_new)
            result = cv2.flip(result, 1)  # augment segments (flip left-right)
            i = result > 0  # pixels to replace
            img[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

        sample['img'] = img
        sample['cls'] = cls
        sample['bboxes'] = bboxes
        sample['segments'] = segments

        return sample

    def random_perspective(
            self, sample, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0)
    ):
        bbox_format, segment_format = sample['bbox_format'], sample['segment_format']
        assert bbox_format == 'ltrb', f'The bbox format should be ltrb, but got {bbox_format}'
        assert segment_format == 'polygon', f'The segment format should be polygon, but got {segment_format}'

        img = sample['img']
        cls = sample['cls']
        targets = sample['bboxes']
        segments = sample['segments']
        assert isinstance(segments, np.ndarray), f"segments type expect numpy.ndarray, but got {type(segments)}; " \
                                                 f"maybe you should resample_segments before that."

        border = sample.pop('mosaic_border', border)
        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            use_segments = len(segments)
            new_bboxes = np.zeros((n, 4))
            if use_segments:  # warp segments
                point_num = segments[0].shape[0]
                new_segments = np.zeros((n, point_num, 2))
                for i, segment in enumerate(segments):
                    xy = np.ones((len(segment), 3))
                    xy[:, :2] = segment
                    xy = xy @ M.T  # transform
                    xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                    # clip
                    new_segments[i] = xy
                    new_bboxes[i] = segment2box(xy, width, height)

            else:  # warp boxes
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new_bboxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # clip
                new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, width)
                new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=targets.T * s, box2=new_bboxes.T, area_thr=0.01 if use_segments else 0.10)

            cls = cls[i]
            targets = new_bboxes[i]
            sample['cls'] = cls
            sample['bboxes'] = targets
            if use_segments:
                sample['segments'] = new_segments[i]

        sample['img'] = img

        return sample

    def mixup(self, sample, alpha: 32.0, beta: 32.0, pre_transform=None):
        bbox_format, segment_format = sample['bbox_format'], sample['segment_format']
        assert bbox_format == 'ltrb', f'The bbox format should be ltrb, but got {bbox_format}'
        assert segment_format == 'polygon', f'The segment format should be polygon, but got {segment_format}'

        index = random.choices(self.indices, k=1)[0]
        sample2 = self.get_sample(index)
        if pre_transform:
            for _i, ori_trans in enumerate(pre_transform):
                _trans = ori_trans.copy()
                func_name, prob = _trans.pop("func_name"), _trans.pop("prob", 1.0)
                if func_name == 'copy_paste':
                    sample2 = self.copy_paste(sample2, prob)
                elif random.random() < prob:
                    if func_name == "albumentations" and getattr(self, "albumentations", None) is None:
                        self.albumentations = Albumentations(size=self.img_size, **_trans)
                    sample2 = getattr(self, func_name)(sample2, **_trans)

        assert isinstance(sample['segments'], np.ndarray), \
            f"MixUp: sample segments type expect numpy.ndarray, but got {type(sample['segments'])}; " \
            f"maybe you should resample_segments before that."
        assert isinstance(sample2['segments'], np.ndarray), \
            f"MixUp: sample2 segments type expect numpy.ndarray, but got {type(sample2['segments'])}; " \
            f"maybe you should add resample_segments in pre_transform."

        image, image2 = sample['img'], sample2['img']
        r = np.random.beta(alpha, beta)  # mixup ratio, alpha=beta=8.0
        image = (image * r + image2 * (1 - r)).astype(np.uint8)

        sample['img'] = image
        sample['cls'] = np.concatenate((sample['cls'], sample2['cls']), 0)
        sample['bboxes'] = np.concatenate((sample['bboxes'], sample2['bboxes']), 0)
        sample['segments'] = np.concatenate((sample['segments'], sample2['segments']), 0)
        return sample

    def pastein(self, sample, num_sample=30):
        bbox_format = sample['bbox_format']
        assert bbox_format == 'ltrb', f'The bbox format should be ltrb, but got {bbox_format}'
        assert not self.return_segments, "pastein currently does not support seg data."
        assert not self.return_keypoints, "pastein currently does not support keypoint data."
        sample.pop('segments', None)
        sample.pop('keypoints', None)

        image = sample['img']
        cls = sample['cls']
        bboxes = sample['bboxes']
        # load sample
        sample_labels, sample_images, sample_masks = [], [], []
        while len(sample_labels) < num_sample:
            sample_labels_, sample_images_, sample_masks_ = self._pastin_load_samples()
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
            if len(bboxes):
                ioa = bbox_ioa(box, bboxes)  # intersection over area
            else:
                ioa = np.zeros(1)

            if (
                    (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin + 20) and (ymax > ymin + 20)
            ):  # allow 30% obscuration of existing labels
                sel_ind = random.randint(0, len(sample_labels) - 1)
                hs, ws, cs = sample_images[sel_ind].shape
                r_scale = min((ymax - ymin) / hs, (xmax - xmin) / ws)
                r_w = int(ws * r_scale)
                r_h = int(hs * r_scale)

                if (r_w > 10) and (r_h > 10):
                    r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                    r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                    temp_crop = image[ymin: ymin + r_h, xmin: xmin + r_w]
                    m_ind = r_mask > 0
                    if m_ind.astype(np.int_).sum() > 60:
                        temp_crop[m_ind] = r_image[m_ind]
                        box = np.array([xmin, ymin, xmin + r_w, ymin + r_h], dtype=np.float32)
                        if len(bboxes):
                            cls = np.concatenate((cls, [[sample_labels[sel_ind]]]), 0)
                            bboxes = np.concatenate((bboxes, [box]), 0)
                        else:
                            cls = np.array([[sample_labels[sel_ind]]])
                            bboxes = np.array([box])

                        image[ymin: ymin + r_h, xmin: xmin + r_w] = temp_crop  # Modify on the original image

        sample['img'] = image
        sample['bboxes'] = bboxes
        sample['cls'] = cls
        return sample

    def _pastin_load_samples(self):
        # loads images in a 4-mosaic
        classes4, bboxes4, segments4 = [], [], []
        mosaic_samples = []
        indices = random.choices(self.indices, k=4)  # 3 additional image indices
        mosaic_samples += [self.get_sample(i) for i in indices]
        s = self.img_size
        mosaic_border = [-s // 2, -s // 2]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y

        for i, sample in enumerate(mosaic_samples):
            # Load image
            img = sample['img']
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
            cls, bboxes = sample['cls'], sample['bboxes']
            bboxes = xywhn2xyxy(bboxes, w, h, padw, padh)  # normalized xywh to pixel xyxy format

            classes4.append(cls)
            bboxes4.append(bboxes)

            segments = sample['segments']
            segments_is_list = isinstance(segments, list)
            if segments_is_list:
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
                segments4.extend(segments)
            else:
                segments = xyn2xy(segments, w, h, padw, padh)
                segments4.append(segments)

        # Concat/clip labels
        classes4 = np.concatenate(classes4, 0)
        bboxes4 = np.concatenate(bboxes4, 0)
        bboxes4 = bboxes4.clip(0, 2 * s)

        if segments_is_list:
            for x in segments4:
                np.clip(x, 0, 2 * s, out=x)
        else:
            segments4 = np.concatenate(segments4, 0)
            segments4 = segments4.clip(0, 2 * s)

        # Augment
        sample_labels, sample_images, sample_masks = \
            self._pastin_sample_segments(img4, classes4, bboxes4, segments4, probability=0.5)

        return sample_labels, sample_images, sample_masks

    def _pastin_sample_segments(self, img, classes, bboxes, segments, probability=0.5):
        # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
        n = len(segments)
        sample_labels = []
        sample_images = []
        sample_masks = []
        if probability and n:
            h, w, c = img.shape  # height, width, channels
            for j in random.sample(range(n), k=round(probability * n)):
                cls, l, s = classes[j], bboxes[j], segments[j]
                box = (
                    l[0].astype(int).clip(0, w - 1),
                    l[1].astype(int).clip(0, h - 1),
                    l[2].astype(int).clip(0, w - 1),
                    l[3].astype(int).clip(0, h - 1),
                )

                if (box[2] <= box[0]) or (box[3] <= box[1]):
                    continue

                sample_labels.append(cls[0])

                mask = np.zeros(img.shape, np.uint8)

                cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
                sample_masks.append(mask[box[1]: box[3], box[0]: box[2], :])

                result = cv2.bitwise_and(src1=img, src2=mask)
                i = result > 0  # pixels to replace
                mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
                sample_images.append(mask[box[1]: box[3], box[0]: box[2], :])

        return sample_labels, sample_images, sample_masks

    def hsv_augment(self, sample, hgain=0.5, sgain=0.5, vgain=0.5):
        image = sample['img']
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)  # Modify on the original image

        sample['img'] = image
        return sample

    def fliplr(self, sample):
        # flip image left-right
        image = sample['img']
        image = np.fliplr(image)
        sample['img'] = image

        # flip box
        _, w = image.shape[:2]
        bboxes, bbox_format = sample['bboxes'], sample['bbox_format']
        if bbox_format == "ltrb":
            if len(bboxes):
                x1 = bboxes[:, 0].copy()
                x2 = bboxes[:, 2].copy()
                bboxes[:, 0] = w - x2
                bboxes[:, 2] = w - x1
        elif bbox_format == "xywhn":
            if len(bboxes):
                bboxes[:, 0] = 1 - bboxes[:, 0]
        else:
            raise NotImplementedError
        sample['bboxes'] = bboxes

        # flip seg
        if self.return_segments:
            segment_format, segments = sample['segment_format'], sample['segments']
            assert segment_format == 'polygon', \
                f'FlipLR: The segment format should be polygon, but got {segment_format}'
            assert isinstance(segments, np.ndarray), \
                f"FlipLR: segments type expect numpy.ndarray, but got {type(segments)}; " \
                f"maybe you should resample_segments before that."

            if len(segments):
                segments[..., 0] = w - segments[..., 0]

            sample['segments'] = segments

        return sample

    def letterbox(self, sample, new_shape=None, xywhn2xyxy_=True, scaleup=False, only_image=False, color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        if sample['bbox_format'] == 'ltrb':
            xywhn2xyxy_ = False

        if not new_shape:
            new_shape = self.img_size

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        image = sample['img']
        shape = image.shape[:2]  # current shape [height, width]

        h, w = shape[:]
        ori_shape = sample['ori_shape']
        h0, w0 = ori_shape
        hw_scale = np.array([h / h0, w / w0])
        sample['hw_scale'] = hw_scale

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

        if shape != new_shape:
            if shape[::-1] != new_unpad:  # resize
                image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
            sample['hw_pad'] = hw_pad
        else:
            sample['hw_pad'] = np.array([0., 0.])
        bboxes = sample['bboxes']
        if not only_image:
            # convert bboxes
            if len(bboxes):
                if xywhn2xyxy_:
                    bboxes = xywhn2xyxy(bboxes, r * w, r * h, padw=dw, padh=dh)
                else:
                    bboxes *= r
                    bboxes[:, [0, 2]] += dw
                    bboxes[:, [1, 3]] += dh
                sample['bboxes'] = bboxes
            sample['bbox_format'] = 'ltrb'

            # convert segments
            if 'segments' in sample:
                segments, segment_format = sample['segments'], sample['segment_format']
                assert segment_format == 'polygon', f'The segment format should be polygon, but got {segment_format}'

                if len(segments):
                    if isinstance(segments, np.ndarray):
                        if xywhn2xyxy_:
                            segments[..., 0] *= w
                            segments[..., 1] *= h
                        else:
                            segments *= r
                        segments[..., 0] += dw
                        segments[..., 1] += dh
                    elif isinstance(segments, list):
                        for segment in segments:
                            if xywhn2xyxy_:
                                segment[..., 0] *= w
                                segment[..., 1] *= h
                            else:
                                segment *= r
                            segment[..., 0] += dw
                            segment[..., 1] += dh
                    sample['segments'] = segments

        sample['img'] = image
        return sample

    def label_norm(self, sample, xyxy2xywh_=True):
        bbox_format = sample['bbox_format']
        if bbox_format == "xywhn":
            return sample

        bboxes = sample['bboxes']
        if len(bboxes) == 0:
            sample['bbox_format'] = 'xywhn'
            return sample

        if xyxy2xywh_:
            bboxes = xyxy2xywh(bboxes)  # convert xyxy to xywh
        height, width = sample['img'].shape[:2]
        bboxes[:, [1, 3]] /= height  # normalized height 0-1
        bboxes[:, [0, 2]] /= width  # normalized width 0-1
        sample['bboxes'] = bboxes
        sample['bbox_format'] = 'xywhn'

        return sample

    def label_pad(self, sample, padding_size=160, padding_value=-1):
        # create fixed label, avoid dynamic shape problem.
        bbox_format = sample['bbox_format']
        assert bbox_format == 'xywhn', f'The bbox format should be xywhn, but got {bbox_format}'

        cls, bboxes = sample['cls'], sample['bboxes']
        cls_pad = np.full((padding_size, 1), padding_value, dtype=np.float32)
        bboxes_pad = np.full((padding_size, 4), padding_value, dtype=np.float32)
        nL = len(bboxes)
        if nL:
            cls_pad[:min(nL, padding_size)] = cls[:min(nL, padding_size)]
            bboxes_pad[:min(nL, padding_size)] = bboxes[:min(nL, padding_size)]
        sample['cls'] = cls_pad
        sample['bboxes'] = bboxes_pad

        if "segments" in sample:
            if sample['segment_format'] == "mask":
                segments = sample['segments']
                assert isinstance(segments, np.ndarray), \
                    f"Label Pad: segments type expect numpy.ndarray, but got {type(segments)}; " \
                    f"maybe you should resample_segments before that."
                assert nL == segments.shape[0], f"Label Pad: segments len not equal bboxes"
                h, w = segments.shape[1:]
                segments_pad = np.full((padding_size, h, w), padding_value, dtype=np.float32)
                segments_pad[:min(nL, padding_size)] = segments[:min(nL, padding_size)]
                sample['segments'] = segments_pad

        return sample

    def image_norm(self, sample, scale=255.0):
        image = sample['img']
        image = image.astype(np.float32, copy=False)
        image /= scale
        sample['img'] = image
        return sample

    def image_transpose(self, sample, bgr2rgb=True, hwc2chw=True):
        image = sample['img']
        if bgr2rgb:
            image = image[:, :, ::-1]
        if hwc2chw:
            image = image.transpose(2, 0, 1)
        sample['img'] = image
        return sample

    def segment_poly2mask(self, sample, mask_overlap, mask_ratio):
        """convert polygon points to bitmap."""
        segments, segment_format = sample['segments'], sample['segment_format']
        assert segment_format == 'polygon', f'The segment format should be polygon, but got {segment_format}'
        assert isinstance(segments, np.ndarray), \
            f"Segment Poly2Mask: segments type expect numpy.ndarray, but got {type(segments)}; " \
            f"maybe you should resample_segments before that."

        h, w = sample['img'].shape[:2]
        if mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=mask_ratio)
            sample['cls'] = sample['cls'][sorted_idx]
            sample['bboxes'] = sample['bboxes'][sorted_idx]
            sample['segments'] = masks  # (h/mask_ratio, w/mask_ratio)
            sample['segment_format'] = 'overlap'
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=mask_ratio)
            sample['segments'] = masks
            sample['segment_format'] = 'mask'

        return sample

    def _img2label_paths(self, img_paths):
        # Define label paths as a function of image paths
        sa, sb = os.sep + "images" + os.sep, os.sep + "labels" + os.sep  # /images/, /labels/ substrings
        return ["txt".join(x.replace(sa, sb, 1).rsplit(x.split(".")[-1], 1)) for x in img_paths]

    def _get_hash(self, paths):
        # Returns a single hash value of a list of paths (files or dirs)
        size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
        h = hashlib.md5(str(size).encode())  # hash sizes
        h.update("".join(paths).encode())  # hash paths
        return h.hexdigest()  # return hash

    def _exif_size(self, img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        try:
            rotation = dict(img._getexif().items())[orientation]
            if rotation == 6:  # rotation 270
                s = (s[1], s[0])
            elif rotation == 8:  # rotation 90
                s = (s[1], s[0])
        except:
            pass

        return s

    def train_collate_fn(self, im_file, cls, bboxes, segments, keypoints, bbox_format, 
                         segment_format, img, ori_shape, hw_scale, hw_pad, batch_info):
        labels = []
        for i, (c, b) in enumerate(zip(cls, bboxes)):
            labels.append(np.concatenate((np.full_like(c, i), c, b), axis=-1))
        return_items = [np.stack(img, 0), np.stack(labels, 0)]
        if self.return_segments:
            return_items.append(np.stack(segments, 0))
        if self.return_keypoints:
            return_items.append(np.stack(keypoints, 0))
        
        return tuple(return_items)

    def test_collate_fn(self, batch_samples, batch_info):
        imgs = [sample.pop('img') for sample in batch_samples]
        path = [sample.pop('im_file') for sample in batch_samples]
        hw_ori = [sample.pop('ori_shape') for sample in batch_samples]
        hw_scale = [sample.pop('hw_scale') for sample in batch_samples]
        pad = [sample.pop('hw_pad') for sample in batch_samples]
        return (
            np.stack(imgs, 0),
            path,
            np.stack(hw_ori, 0),
            np.stack(hw_scale, 0),
            np.stack(pad, 0),
        )
