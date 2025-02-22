import time
import cv2
import numpy as np

import mindspore as ms
from mindspore import ops, Tensor

__all__ = ["non_max_suppression", "scale_coords", "xyxy2xywh", "xywh2xyxy"]


def non_max_suppression(
    prediction,
    mask_coefficient=None,
    conf_thres=0.25,
    iou_thres=0.45,
    conf_free=False,
    classes=None,
    agnostic=False,
    multi_label=False,
    time_limit=20.0,
    need_nms=True,
):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Args:
        prediction (ndarray): Prediction. If conf_free is False, prediction on (bs, N, 5+nc) ndarray each point,
            the last dimension meaning [center_x, center_y, width, height, conf, cls0, ...]; If conf_free is True,
            prediction on (bs, N, 4+nc) ndarray each point, the last dimension meaning [center_x, center_y, width, height, cls0, ...].
        conf_free (bool): Whether the prediction result include conf.

    Returns:
         list of detections, on (n,6) ndarray per image, the last dimension meaning [xyxy, conf, cls].
    """

    if not conf_free:
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
    else:
        nc = prediction.shape[2] - 4  # number of classes
        xc = prediction[..., 4:].max(-1) > conf_thres  # candidates
        prediction = np.concatenate(
            (prediction[..., :4], prediction[..., 4:].max(-1, keepdims=True), prediction[..., 4:]), axis=-1
        )
    
    max_det = 300  # maximum number of detections per image
    if not need_nms: # end-to-end model
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == np.array(classes)).any(1)] for pred in output]
        return output

    nm = 0
    if mask_coefficient is not None:
        assert mask_coefficient.shape[:2] == prediction.shape[:2], \
            f"mask_coefficient shape {mask_coefficient.shape[:2]} and " \
            f"prediction.shape {prediction.shape[:2]} are not equal."
        nm = mask_coefficient.shape[2]
        prediction = np.concatenate((prediction, mask_coefficient), axis=-1)

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = time_limit if time_limit > 0 else 1e3  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6+nm))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Scale class with conf
        if not conf_free:
            if nc == 1:
                x[:, 5:5+nc] = x[:, 4:5]  # signle cls no need to multiplicate.
            else:
                x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:5+nc] > conf_thres).nonzero()
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1) if nm == 0 else \
                np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32), x[i, -nm:]), 1)
        else:  # best class only
            conf = x[:, 5:5+nc].max(1, keepdims=True)  # get maximum conf
            j = np.argmax(x[:, 5:5+nc], axis=1,keepdims=True)  # get maximum index
            x = np.concatenate((box, conf, j.astype(np.float32)), 1)[conf.flatten() > conf_thres] if nm == 0 else \
                np.concatenate((box, conf, j.astype(np.float32), x[:, -nm:]), 1)[conf.flatten() > conf_thres]


        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[-max_nms:]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = _nms(boxes, scores, iou_thres)  # NMS for per sample

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = _box_iou(boxes[i], boxes) > iou_thres  # iou matrix # (N, M)
            weights = iou * scores[None]  # box weights
            # (N, M) @ (M, 4) / (N, 1)
            x[i, :4] = np.matmul(weights, x[:, :4]) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(
                f"WARNING: Batch NMS time limit {time_limit}s exceeded, this batch "
                f"process {xi + 1}/{prediction.shape[0]} sample."
            )
            break  # time limit exceeded

    return output


def scale_coords(img1_shape, coords, img0_shape, ratio=None, pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape

    if ratio is None:  # calculate from img0_shape
        ratio = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # ratio  = old / new
    else:
        ratio = ratio[0]

    if pad is None:
        padh, padw = (img1_shape[0] - img0_shape[0] * ratio) / 2, (img1_shape[1] - img0_shape[1] * ratio) / 2
    else:
        padh, padw = pad[:]

    coords[:, [0, 2]] -= padw  # x padding
    coords[:, [1, 3]] -= padh  # y padding
    coords[:, [0, 2]] /= ratio  # x rescale
    coords[:, [1, 3]] /= ratio  # y rescale
    coords = _clip_coords(coords, img0_shape)
    return coords


def _clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img_shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img_shape[0])  # y1, y2
    return boxes


def _nms(xyxys, scores, threshold):
    """Calculate NMS"""
    s_time = time.time()
    x1 = xyxys[:, 0]
    y1 = xyxys[:, 1]
    x2 = xyxys[:, 2]
    y2 = xyxys[:, 3]
    scores = scores
    # areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    reserved_boxes = []
    while order.size > 0:
        i = order[0]
        reserved_boxes.append(i)
        max_x1 = np.maximum(x1[i], x1[order[1:]])
        max_y1 = np.maximum(y1[i], y1[order[1:]])
        min_x2 = np.minimum(x2[i], x2[order[1:]])
        min_y2 = np.minimum(y2[i], y2[order[1:]])

        # intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
        # intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
        intersect_w = np.maximum(0.0, min_x2 - max_x1)
        intersect_h = np.maximum(0.0, min_y2 - max_y1)
        intersect_area = intersect_w * intersect_h

        ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area + 1e-6)
        indexes = np.where(ovr <= threshold)[0]
        order = order[indexes + 1]
    return np.array(reserved_boxes)


def _box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 ([N, 4])
        box2 ([M, 4])
    Returns:
        iou ([N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0, None).prod(2)
    )
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


#------------------------for segment------------------------

def scale_image(masks, img0_shape, pad=None):
    """
    Takes a mask, and resizes it to the original image size
    Args:
      masks (numpy.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
      img0_shape (tuple): the original image shape
      ratio_pad (tuple): the ratio of the padding to the original image.
    Returns:
      masks (numpy.ndarray): The masks that are being returned.
    """

    # Rescale coordinates (xyxy) from img1_shape to img0_shape
    img1_shape = masks.shape
    if (np.array(img1_shape[:2]) == np.array(img0_shape[:2])).all():
        return masks

    if pad is None:
        ratio = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # ratio  = old / new
        pad = (img1_shape[0] - img0_shape[0] * ratio) / 2, (img1_shape[1] - img0_shape[1] * ratio) / 2

    top, left = int(pad[0]), int(pad[1])  # y, x
    bottom, right = int(img1_shape[0] - pad[0]), int(img1_shape[1] - pad[1])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, dsize=(img0_shape[1], img0_shape[0]), interpolation=cv2.INTER_LINEAR)
    # masks = ops.interpolate(Tensor(masks, dtype=ms.float32)[None], shape, mode='bilinear', align_corners=False)[0].asnumpy()  # CHW
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box
    Args:
      masks (numpy.ndarray): [h, w, n] array of masks
      boxes (numpy.ndarray): [n, 4] array of bbox coordinates in relative point form
    Returns:
      (numpy.ndarray): The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
    quality but is slower.
    Args:
      protos (numpy.ndarray): [mask_dim, mask_h, mask_w]
      masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms
      bboxes (numpy.ndarray): [n, 4], n is number of masks after nms
      shape (tuple): the size of the input image (h,w)
    Returns:
      (numpy.ndarray): The upsampled masks.
    """
    assert len(shape) == 2, f"The length of the shape is {len(shape)}, expected to be 2."
    c, mh, mw = protos.shape  # CHW
    masks = sigmoid((np.matmul(masks_in, protos.reshape(c, -1)))).reshape(-1, mh, mw)

    # interpolate bilinear
    # (n, mh, mw) -> (mh, mw, n) -> (*shape, n) -> (n, *shape)
    # masks = cv2.resize(masks.transpose(1, 2, 0), dsize=shape, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    masks = ops.interpolate(Tensor(masks, dtype=ms.float32)[None], shape, mode='bilinear', align_corners=False)[0].asnumpy()  # CHW

    masks = crop_mask(masks, bboxes)  # CHW
    return masks > 0.5


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (numpy.ndarray): A array of shape [mask_dim, mask_h, mask_w].
        masks_in (numpy.ndarray): A array of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (numpy.ndarray): A array of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (numpy.ndarray): A binary mask array of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """

    assert len(shape) == 2, f"The length of the shape is {len(shape)}, expected to be 2."
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = sigmoid(np.matmul(masks_in, protos.view(c, -1))).reshape(-1, mh, mw)  # CHW

    downsampled_bboxes = np.copy(bboxes)
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        # masks = cv2.resize(masks.transpose(1, 2, 0), dsize=shape, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        masks = ops.interpolate(Tensor(masks, dtype=ms.float32)[None], shape, mode='bilinear', align_corners=False)[0].asnumpy()  # CHW
    return masks > 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#----------------------------------------------------------
