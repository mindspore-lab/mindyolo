import os

import cv2
import math
from pathlib import Path
import numpy as np
from PIL import ExifTags, Image
from tqdm import tqdm
import hashlib
import random
import glob

from mindyolo.utils import logger

from .albumentations import _check_version, _colorstr
from .copypaste import copy_paste
from .perspective import _resample_segments, _segment2box, _box_candidates

__all__ = ["COCODatasetSeg"]


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


class COCODatasetSeg:
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
        pad=0.0,
    ):
        self.cache_version = 0.1
        self.path = dataset_path

        # acceptable image suffixes
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
        self.help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'

        self.img_size = img_size
        self.augment = augment
        self.rect = rect
        self.stride = stride
        self.transforms_dict = transforms_dict
        self.is_training = is_training
        if is_training:
            self.dataset_column_names = ["image", "labels", "segments", "img_files"]
        else:
            self.dataset_column_names = ["image", "labels", "img_files", "hw_ori", "hw_scale", "pad"]

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
            raise Exception(f"Error loading data from {self.path}: {e}\nSee {self.help_url}")

        # Check cache
        self.label_files = self._img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache.npy")  # cached labels
        if cache_path.is_file():
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            if cache["version"] == self.cache_version and cache["hash"] == self._get_hash(
                self.label_files + self.img_files
            ):
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
        assert nf > 0 or not augment, f"No labels in {cache_path}. Can not train without labels. See {self.help_url}"

        # Read cache
        cache.pop("hash")  # remove hash
        cache.pop("version")  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.img_shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = self._img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(labels)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int_)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_hw_ori, self.indices = [None, ] * n, [None, ] * n, range(n)

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

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        self.imgIds = [int(Path(im_file).stem) for im_file in self.img_files]

    def cache_labels(self, path=Path("./labels.cache")):
        # Get orientation exif tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc="Scanning images", total=len(self.img_files))
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
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate(
                                (classes.reshape(-1, 1), self._segments2boxes(segments)), 1
                            )  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, "labels require 5 columns each"
                        assert (l >= 0).all(), "negative labels"
                        assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels"
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], "duplicate labels"
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f"WARNING: Ignoring corrupted image and/or label {im_file}: {e}")

            pbar.desc = f"Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f"WARNING: No labels found in {path}. See {self.help_url}")

        x["hash"] = self._get_hash(self.label_files + self.img_files)
        x["results"] = nf, nm, ne, nc, i + 1
        x["version"] = self.cache_version  # cache version
        np.save(path, x)  # save for next time
        logger.info(f"New cache created: {path}")
        return x

    def __getitem__(self, index):
        index, image, labels, segments, hw_ori, hw_scale, pad = index, None, None, None, None, None, None
        for _i, ori_trans in enumerate(self.transforms_dict):
            _trans = ori_trans.copy()
            func_name, prob = _trans.pop("func_name"), _trans.pop("prob", 1.0)
            if func_name == 'copy_paste':
                image, labels, segments = self.copy_paste(image, labels, segments, prob)
            elif random.random() < prob:
                if func_name == "mosaic":
                    image, labels, segments = self.mosaic(index)
                elif func_name == "letterbox":
                    if image is None:
                        image, hw_ori = self.load_image(index)
                        labels = self.labels[index].copy()
                        segments = self.segments[index].copy()
                        if len(segments) > 0:
                            segments = _resample_segments(segments)
                            segments = np.stack(segments, axis=0)
                        else:
                            segments = np.zeros((0, 1000, 2), dtype=np.float32)

                    new_shape = self.img_size
                    image, labels, segments = self.letterbox(image, labels, segments, new_shape, **_trans)
                elif func_name == "albumentations":
                    if getattr(self, "albumentations", None) is None:
                        self.albumentations = Albumentations(size=self.img_size)
                    image, labels, segments = self.albumentations(image, labels, segments, **_trans)
                else:
                    image, labels, segments = getattr(self, func_name)(image, labels, segments, **_trans)

        image = np.ascontiguousarray(image)

        if self.is_training:
            return image, labels, segments, self.img_files[index]
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
            assert img is not None, "Image Not Found " + path
            h_ori, w_ori = img.shape[:2]  # orig hw
            r = self.img_size / max(h_ori, w_ori)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)

            return img, np.array([h_ori, w_ori])  # img, hw_original
        else:
            return self.imgs[index], self.img_hw_ori[index]  # img, hw_original

    def mosaic(self, index):
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
            if len(segments) > 0:
                segments = _resample_segments(segments)
                segments = np.stack(segments, axis=0)
            else:
                segments = np.zeros((0, 1000, 2), dtype=np.float32)

            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments[..., 0] *= w
                segments[..., 1] *= h
                segments[..., 0] += padw
                segments[..., 1] += padh

            labels4.append(labels)
            segments4.append(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        segments4 = np.concatenate(segments4, 0)

        labels4[:, 1:] = labels4[:, 1:].clip(0, 2 * s)
        segments4 = segments4.clip(0, 2 * s)

        return img4, labels4, segments4

    def copy_paste(self, img, labels, segments, prob):
        # Augment
        img, labels, segments = copy_paste(img, labels, segments, probability=prob)
        return img, labels, segments

    def random_perspective(
            self, img, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
    ):
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
        # s = random.uniform(1 - scale, 1.1 + scale)
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
            if use_segments:  # warp segments
                n, num = segments.shape[:2]
                xy = np.ones((n * num, 3), dtype=segments.dtype)
                segments = segments.reshape(-1, 2)
                xy[:, :2] = segments
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3]
                segments = xy.reshape(n, -1, 2)
                new_bboxes = np.stack([_segment2box(xy, width, height) for xy in segments], 0)
                new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, width)
                new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, height)
                segments[..., 0] = segments[..., 0].clip(0, width)
                segments[..., 1] = segments[..., 1].clip(0, height)
            else:  # warp boxes
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n,
                                                                                    8)  # perspective rescale or affine

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new_bboxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # clip
                new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, width)
                new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, height)

            # filter candidates
            i = _box_candidates(box1=targets[:, 1:5].T * s, box2=new_bboxes.T, area_thr=0.01 if use_segments else 0.10)
            targets = targets[i]
            targets[:, 1:5] = new_bboxes[i]
            segments = segments[i]

        return img, targets, segments

    def mixup(self, image, labels, segments, needed_mosaic=True):
        if needed_mosaic:
            for i in range(len(self.transforms_dict)):
                if self.transforms_dict[i]["func_name"] == "mosaic":
                    while self.transforms_dict[i]['func_name'] != 'mixup':
                        _trans = self.transforms_dict[i].copy()
                        func_name, prob = _trans.pop("func_name"), _trans.pop("prob", 1.0)
                        if func_name == 'copy_paste':
                            image2, labels2, segments2 = self.copy_paste(image2, labels2, segments2, prob)
                        elif random.random() < prob:
                            if func_name == "mosaic":
                                image2, labels2, segments2 = self.mosaic(random.randint(0, len(self.labels) - 1))
                            elif func_name == "letterbox":
                                new_shape = self.img_size
                                image2, labels2, segments2 = self.letterbox(image2, labels2, segments2, new_shape, **_trans)
                            else:
                                image2, labels2, segments2 = getattr(self, func_name)(image2, labels2, segments2, **_trans)
                        i += 1
                    break
        else:
            index2 = random.randint(0, len(self.labels) - 1)
            image2, _ = self.load_image(index2)
            labels2 = self.labels[index2]
            segments2 = self.segments[index2]

        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=8.0
        image = (image * r + image2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
        segments = np.concatenate((segments, segments2), 0)
        return image, labels, segments

    def hsv_augment(self, image, labels, segments, hgain=0.5, sgain=0.5, vgain=0.5):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)  # Modify on the original image
        return image, labels, segments

    def fliplr(self, image, labels, segments):
        # flip left-right
        image = np.fliplr(image)
        h, w = image.shape[:2]
        if len(labels):
            x1 = labels[:, 1].copy()
            x2 = labels[:, 3].copy()
            labels[:, 0] = w - x2
            labels[:, 2] = w - x1
        if len(segments):
            segments[..., 0] = w - segments[..., 0]
        return image, labels, segments

    def flipud(self, image, labels):
        # flip up-down
        image = np.flipud(image)
        if len(labels):
            labels[:, 2] = 1 - labels[:, 2]
        return image, labels

    def letterbox(self, image, labels, segments, new_shape, xywhn2xyxy_=True, scaleup=False, color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[:2]  # current shape [height, width]
        h, w = shape[:]
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

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        # convert labels
        if labels.size:
            if xywhn2xyxy_:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], r * w, r * h, padw=dw, padh=dh)
                segments[..., 0] *= w
                segments[..., 1] *= h
            else:
                labels[:, 1:] *= r
                labels[:, [1, 3]] += dw
                labels[:, [2, 4]] += dh
                segments *= r
            segments[..., 0] += dw
            segments[..., 1] += dh

        return image, labels, segments

    def label_norm(self, image, labels, segments, xyxy2xywh_=True):
        if len(labels) == 0:
            return image, labels, segments

        if xyxy2xywh_:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh

        labels[:, [2, 4]] /= image.shape[0]  # normalized height 0-1
        labels[:, [1, 3]] /= image.shape[1]  # normalized width 0-1

        return image, labels, segments

    def label_pad(self, image, labels, segments, padding_size=160, padding_value=-1):
        # create fixed label, avoid dynamic shape problem.
        labels_out = np.full((padding_size, 6), padding_value, dtype=np.float32)
        nL = len(labels)
        if nL:
            labels_out[: min(nL, padding_size), 0:1] = 0.0
            labels_out[: min(nL, padding_size), 1:] = labels[: min(nL, padding_size), :]

        return image, labels_out, segments

    def image_norm(self, image, labels, segments, scale=255.0):
        image = image.astype(np.float32, copy=False)
        image /= scale
        return image, labels, segments

    def image_transpose(self, image, labels, segments, bgr2rgb=True, hwc2chw=True):
        if bgr2rgb:
            image = image[:, :, ::-1]
        if hwc2chw:
            image = image.transpose(2, 0, 1)
        return image, labels, segments

    def format_segments(self, image, labels, segments, mask_overlap, mask_ratio):
        """convert polygon points to bitmap."""
        h, w = image.shape[:2]
        if mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=mask_ratio)
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            labels = labels[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=mask_ratio)

        return image, labels, masks

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

    def _segments2boxes(self, segments):
        # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
        boxes = []
        for s in segments:
            x, y = s.T  # segment xy
            boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
        return xyxy2xywh(np.array(boxes))  # cls, xywh

    @staticmethod
    def train_collate_fn(imgs, labels, segments, path, batch_info):
        for i, l in enumerate(labels):
            l[:, 0] = i  # add target image index for build_targets()
        return np.stack(imgs, 0), np.stack(labels, 0), np.stack(segments, 0), path

    @staticmethod
    def test_collate_fn(imgs, labels, path, hw_ori, hw_scale, pad, batch_info):
        for i, l in enumerate(labels):
            l[:, 0] = i  # add target image index for build_targets()
        return (
            np.stack(imgs, 0),
            np.stack(labels, 0),
            path,
            np.stack(hw_ori, 0),
            np.stack(hw_scale, 0),
            np.stack(pad, 0),
        )


def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
        np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
    ).clip(0)

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


class Albumentations:
    # Implement Albumentations augmentation https://github.com/ultralytics/yolov5
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        self.transform = None
        prefix = _colorstr("albumentations: ")
        try:
            import albumentations as A

            _check_version(A.__version__, "1.0.3", hard=True)  # version requirement
            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            print(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p), flush=True)
            print("[INFO] albumentations load success", flush=True)
        except ImportError:  # package not installed, skip
            pass
            print("[WARNING] package not installed, albumentations load failed", flush=True)
        except Exception as e:
            print(f"{prefix}{e}", flush=True)
            print("[WARNING] albumentations load failed", flush=True)

    def __call__(self, im, labels, segments, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels, segments


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M], N is number of polygons, M is number of points (M % 2 = 0)
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(imgsz, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): [N, M], N is the number of polygons, M is the number of points(Be divided by 2).
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask

