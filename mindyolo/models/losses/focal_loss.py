import mindspore as ms
from mindspore import nn, ops


def smooth_BCE(eps=0.1):
    """
    Return positive, negative label smoothing BCE targets,
    https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Cell):
    """
    Focal Loss for Dense Object Detection, https://arxiv.org/pdf/1708.02002v2.pdf

    Args:
        bce_weight (Tensor, optional): A rescaling weight applied to the loss of each batch element for BCEWithLogitsLoss.
            If not None, it can be broadcast to a tensor with shape of `logits`,
            data type must be float16 or float32. Default: None.
        bce_pos_weight (Tensor, optional): A weight of positive examples for BCEWithLogitsLoss. Must be a vector with length equal to the
            number of classes. If not None, it must be broadcast to a tensor with shape of `logits`, data type
            must be float16 or float32. Default: None.
        gamma: A modulating factor (1 − pt)^gamma to the cross entropy loss, with tunable focusing. Default: 1.5
        alpha: An alpha-balanced variant of the focal loss. Default: 0.25
        reduction (str): Type of reduction to be applied to loss. The optional values are 'mean', 'sum', and 'none'.
            If 'none', do not perform reduction. Default: 'mean'.
    """

    def __init__(self, bce_weight=None, bce_pos_weight=None, gamma=1.5, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(weight=bce_weight, pos_weight=bce_pos_weight, reduction="none")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction  # default mean
        assert self.loss_fcn.reduction == "none"  # required to apply FL to each element

    def construct(self, pred, true, mask=None):
        ori_dtype = pred.dtype
        loss = self.loss_fcn(pred.astype(ms.float32), true.astype(ms.float32))

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = ops.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if mask is not None:
            loss *= mask

        if self.reduction == "mean":
            if mask is not None:
                return (loss.sum() / mask.astype(loss.dtype).sum().clip(1, None)).astype(ori_dtype)
            return loss.mean().astype(ori_dtype)
        elif self.reduction == "sum":
            return loss.sum().astype(ori_dtype)
        else:  # 'none'
            return loss.astype(ori_dtype)


class BCEWithLogitsLoss(nn.Cell):
    def __init__(self, bce_weight=None, bce_pos_weight=None, reduction="mean"):
        """
        Adds sigmoid activation function to input logits, and uses the given logits to compute binary cross entropy
        between the logits and the labels.

        Args:
            bce_weight (Tensor, optional): A rescaling weight applied to the loss of each batch element.
                If not None, it can be broadcast to a tensor with shape of `logits`,
                data type must be float16 or float32. Default: None.
            bce_pos_weight (Tensor, optional): A weight of positive examples. Must be a vector with length equal to the
                number of classes. If not None, it must be broadcast to a tensor with shape of `logits`, data type
                must be float16 or float32. Default: None.
            reduction (str): Type of reduction to be applied to loss. The optional values are 'mean', 'sum', and 'none'.
                If 'none', do not perform reduction. Default: 'mean'.
        """

        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(weight=bce_weight, pos_weight=bce_pos_weight, reduction="none")
        self.reduction = reduction  # default mean
        assert self.loss_fcn.reduction == "none"  # required to apply FL to each element

    def construct(self, pred, true, mask=None):
        ori_dtype = pred.dtype
        loss = self.loss_fcn(pred.astype(ms.float32), true.astype(ms.float32))

        if mask is not None:
            loss *= mask

        if self.reduction == "mean":
            if mask is not None:
                return (loss.sum() / mask.astype(loss.dtype).sum().clip(1, None)).astype(ori_dtype)
            return loss.mean().astype(ori_dtype)
        elif self.reduction == "sum":
            return loss.sum().astype(ori_dtype)
        else:  # 'none'
            return loss.astype(ori_dtype)
