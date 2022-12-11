import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from math import exp


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


class CharbonnierLoss():
    """Charbonnier Loss (L1).

    Args:
        eps (float): Default: 1e-12.

    """
    def __init__(self, eps=1e-3, reduction='mean'):
        self.eps = eps
        self.reduction = reduction

    def __call__(self, pred, target, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        if self.reduction == 'sum':
            out = paddle.sum(paddle.sqrt((pred - target)**2 + self.eps))
        elif self.reduction == 'mean':
            out = paddle.mean(paddle.sqrt((pred - target)**2 + self.eps))
        else:
            raise NotImplementedError('CharbonnierLoss %s not implemented' %
                                      self.reduction)
        return out


class MSELoss():
    """MSE (L2) loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.

    """
    def __init__(self, reduction='mean', loss_weight=1.0):
        # when loss weight less than zero return None
        if loss_weight <= 0:
            return None
        self._l2_loss = nn.MSELoss(reduction)
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
        return self.loss_weight * self._l2_loss(pred, target)


def gaussian(window_size, sigma):
    gauss = paddle.to_tensor([
        exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = paddle.to_tensor(paddle.expand(
        _2D_window, (channel, 1, window_size, window_size)),
        stop_gradient=False)
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(paddle.nn.Layer):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.shape

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            tt = img1.dtype
            window = paddle.to_tensor(window, dtype=tt)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel,
                     self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class SRLoss(paddle.nn.Layer):

    def __init__(self, step_per_epoch, max_epoch=50):
        super(SRLoss, self).__init__()
        self.mse = MSELoss()
        self.ssim = SSIM()
        self.step_per_epoch = step_per_epoch
        self.max_epoch = max_epoch

    def forward(self, img1, img2, g_step=None):
        r = float(g_step) / (self.max_epoch * self.step_per_epoch) if g_step is not None else 2.
        if r > 1.:
            loss = self.mse(img1, img2)
        else:
            loss = r * self.mse(img1, img2) + (1 - r) * (1. - self.ssim(img1, img2))
        return loss


class SR2StageLoss(nn.Layer):

    def __init__(self, step_per_epoch, epoch):
        super(SR2StageLoss, self).__init__()
        self.step_per_epoch = step_per_epoch
        self.epoch = epoch
        self.l1loss = L1Loss()
        self.ssim = SSIM()
        self.c_loss = CharbonnierLoss()

    def forward(self, img1, img2, g_step=None):
        cur_epoch = g_step // self.step_per_epoch if g_step is not None else 0
        if cur_epoch < self.epoch:
            loss = self.l1loss(img1, img2)
        else:
            loss = self.c_loss(img1, img2) + 0.5 * (1. - self.ssim(img1, img2))
        return loss
