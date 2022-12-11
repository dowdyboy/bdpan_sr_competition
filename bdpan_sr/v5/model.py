import functools
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# Basic DW Convolution
# class Conv2D(nn.Layer):
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  padding_mode='zeros',
#                  weight_attr=None,
#                  bias_attr=None,
#                  data_format="NCHW"):
#         super(Conv2D, self).__init__()
#         self.dw_conv = nn.Conv2D(in_channels, in_channels, kernel_size, stride,
#                                  padding, dilation, in_channels, padding_mode, weight_attr,
#                                  bias_attr, data_format)
#         self.pw_conv = nn.Conv2D(in_channels, out_channels, 1, 1, 0)
#
#     def forward(self, x):
#         return self.pw_conv(self.dw_conv(x))

# class Conv2D(nn.Layer):
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  padding_mode='zeros',
#                  weight_attr=None,
#                  bias_attr=None,
#                  data_format="NCHW"):
#         super(Conv2D, self).__init__()
#         self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride,
#                                  padding, dilation, groups, padding_mode, weight_attr,
#                                  bias_attr, data_format)
#         self.res = in_channels == out_channels
#
#     def forward(self, x):
#         if self.res:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)


Conv2D = nn.Conv2D
# Conv2DPlus = nn.Conv2D
class Conv2DPlus(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(Conv2DPlus, self).__init__()
        # self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride,
        #                          padding, dilation, groups, padding_mode, weight_attr,
        #                          bias_attr, data_format)
        # self.res = in_channels == out_channels
        assert in_channels == out_channels
        nf = in_channels
        gc = nf // 3
        self.conv1 = nn.Conv2D(nf, gc, 3, 1, 1, bias_attr=bias_attr)
        self.conv2 = nn.Conv2D(nf + gc, gc, 3, 1, 1, bias_attr=bias_attr)
        self.conv3 = nn.Conv2D(nf + 2 * gc, nf, 3, 1, 1, bias_attr=bias_attr)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(paddle.concat((x, x1), 1)))
        x3 = self.conv3(paddle.concat((x, x1, x2), 1))
        return x + x3


def make_multi_blocks(func, num_layers):
    """Make layers by stacking the same blocks.

    Args:
        func (nn.Layer): nn.Layer class for basic block.
        num_layers (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    Blocks = nn.Sequential()
    for i in range(num_layers):
        Blocks.add_sublayer('block%d' % i, func())
    return Blocks


# 对输入特征图进行像素注意力操作
class PA(nn.Layer):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = Conv2D(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = x * y

        return out

    def init_weights(self):
        from .init import reset_parameters
        def reset_func(m):
            if hasattr(m, 'weight') and (not isinstance(
                    m, (nn.BatchNorm, nn.BatchNorm2D))):
                reset_parameters(m)
        self.apply(reset_func)


# 对输入特征图进行分离式像素注意力卷积
class PAConv(nn.Layer):
    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = Conv2D(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = Conv2D(nf,
                            nf,
                            kernel_size=k_size,
                            padding=(k_size - 1) // 2,
                            bias_attr=False)  # 3x3 convolution
        self.k4 = Conv2D(nf,
                            nf,
                            kernel_size=k_size,
                            padding=(k_size - 1) // 2,
                            bias_attr=False)  # 3x3 convolution
        self.init_weights()

    def forward(self, x):

        # 使用110卷积生成注意力参数
        y = self.k2(x)
        y = self.sigmoid(y)

        # 特征图经过311卷积配合注意力参数
        out = self.k3(x) * y
        out = self.k4(out)

        return out

    def init_weights(self):
        from .init import reset_parameters
        def reset_func(m):
            if hasattr(m, 'weight') and (not isinstance(
                    m, (nn.BatchNorm, nn.BatchNorm2D))):
                reset_parameters(m)
        self.apply(reset_func)


class SCPA(nn.Layer):
    """
    SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
    """
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction

        self.conv1_a = Conv2DPlus(nf,
                                 group_width,
                                 kernel_size=1,
                                 bias_attr=False)
        self.conv1_b = Conv2DPlus(nf,
                                 group_width,
                                 kernel_size=1,
                                 bias_attr=False)

        self.k1 = nn.Sequential(
            Conv2D(group_width,
                      group_width,
                      kernel_size=3,
                      stride=stride,
                      padding=dilation,
                      dilation=dilation,
                      bias_attr=False))

        self.PAConv = PAConv(group_width)

        self.conv3 = Conv2D(group_width * 2 if reduction == 1 else group_width * reduction,
                               nf,
                               kernel_size=1,
                               bias_attr=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.init_weights()


    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)  # 通过110卷积生成通道数减半的两个特征图
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)  # 一个进入普通311卷积、一个进入PAConv
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        # 将两个特征图拼接后通过110卷积, 加上残差
        out = self.conv3(paddle.concat([out_a, out_b], axis=1))
        out += residual

        return out

    def init_weights(self):
        from .init import reset_parameters
        def reset_func(m):
            if hasattr(m, 'weight') and (not isinstance(
                    m, (nn.BatchNorm, nn.BatchNorm2D))):
                reset_parameters(m)
        self.apply(reset_func)


class PANPlus(nn.Layer):
    def __init__(self, in_nc, out_nc, nf, unf, nb):
        super(PANPlus, self).__init__()
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=1)

        ### first convolution
        self.conv_first = Conv2D(in_nc, nf, 3, 1, 1)

        ### main blocks
        self.SCPA_trunk = make_multi_blocks(SCPA_block_f, nb)
        self.trunk_conv = Conv2D(nf, nf, 3, 1, 1)

        #### upsampling
        self.upconv1 = Conv2D(nf, unf, 3, 1, 1)
        self.att1 = PA(unf)
        self.HRconv1 = Conv2D(unf, unf, 3, 1, 1)

        self.upconv2 = Conv2D(unf, unf, 3, 1, 1)
        self.att2 = PA(unf)
        self.HRconv2 = Conv2D(unf, unf, 3, 1, 1)

        self.conv_last_x2 = Conv2D(unf, out_nc, 3, 1, 1)
        self.conv_last_x4 = Conv2D(unf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.init_weights()

    def forward(self, x):

        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk  # 初始特征图和经过backbone处理的特征图相加

        fea = self.upconv1(
            F.interpolate(fea, scale_factor=2, mode='nearest'))  # 放大后卷积
        fea = fea + self.lrelu(self.att1(fea))  # 再进入有像素注意力再激活
        fea = fea + self.lrelu(self.HRconv1(fea))  # 最后再接一个311卷积、激活
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out_x2 = self.conv_last_x2(fea) + x

        fea = self.upconv2(
            F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = fea + self.lrelu(self.att2(fea))
        fea = fea + self.lrelu(self.HRconv2(fea))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out_x4 = self.conv_last_x4(fea) + x + \
                 F.interpolate(out_x2, scale_factor=2, mode='bilinear', align_corners=False)

        return out_x2, out_x4

    def init_weights(self):
        from .init import reset_parameters
        def reset_func(m):
            if hasattr(m, 'weight') and (not isinstance(
                    m, (nn.BatchNorm, nn.BatchNorm2D))):
                reset_parameters(m)
        self.apply(reset_func)

