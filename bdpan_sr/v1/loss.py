import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class L1Loss():
    """L1 (mean absolute error, MAE) loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.

    """
    def __init__(self, reduction='mean', loss_weight=1.0):
        # when loss weight less than zero return None
        if loss_weight <= 0:
            return None
        self._l1_loss = nn.L1Loss(reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction

    def __call__(self, pred, target, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * self._l1_loss(pred, target)
