import mindspore as ms
from mindspore import nn, ops

@ops.constexpr
def shape_prod(shape):
    size = 1
    for i in shape:
        size *= i
    return size


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
        self.reduction = reduction # default mean
        assert self.loss_fcn.reduction == 'none'  # required to apply FL to each element

    def construct(self, pred, true, mask=None):
        ori_dtype = pred.dtype
        loss = self.loss_fcn(pred.astype(ms.float32), true.astype(ms.float32))

        if mask is not None:
            loss *= mask

        if self.reduction == 'mean':
            if mask is not None:
                mask_repeat = shape_prod(loss.shape) / shape_prod(mask.shape)
                return (loss.sum() / (mask.astype(loss.dtype).sum() * mask_repeat).clip(1, None)).astype(ori_dtype)
            return loss.mean().astype(ori_dtype)
        elif self.reduction == 'sum':
            return loss.sum().astype(ori_dtype)
        else:  # 'none'
            return loss.astype(ori_dtype)
