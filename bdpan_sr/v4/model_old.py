import functools

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import ConvNormActivation


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    channels_per_group = num_channels // groups

    # reshape
    x = paddle.reshape(
        x, shape=[batch_size, groups, channels_per_group, height, width])

    # transpose
    x = paddle.transpose(x, perm=[0, 2, 1, 3, 4])

    # flatten
    x = paddle.reshape(x, shape=[batch_size, num_channels, height, width])
    return x


class InvertedResidualDS(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bias=None,
                 norm_layer=None,
                 activation_layer=functools.partial(nn.LeakyReLU, negative_slope=0.2)):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = ConvNormActivation(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=bias,
            norm_layer=norm_layer,
            activation_layer=None)
        self._conv_linear_1 = ConvNormActivation(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
            norm_layer=norm_layer,
            activation_layer=activation_layer)
        # branch2
        self._conv_pw_2 = ConvNormActivation(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
            norm_layer=norm_layer,
            activation_layer=activation_layer)
        self._conv_dw_2 = ConvNormActivation(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            bias=bias,
            norm_layer=norm_layer,
            activation_layer=None)
        self._conv_linear_2 = ConvNormActivation(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
            norm_layer=norm_layer,
            activation_layer=activation_layer)

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._conv_linear_2(x2)
        out = paddle.concat([x1, x2], axis=1)
        return channel_shuffle(out, 2)


class InvertedResidual(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bias=None,
                 norm_layer=None,
                 activation_layer=functools.partial(nn.LeakyReLU, negative_slope=0.2)):
        super(InvertedResidual, self).__init__()
        self._conv_pw_left = ConvNormActivation(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
            norm_layer=norm_layer,
            activation_layer=activation_layer)
        self._conv_pw = ConvNormActivation(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
            norm_layer=norm_layer,
            activation_layer=activation_layer)
        self._conv_dw = ConvNormActivation(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            bias=bias,
            norm_layer=norm_layer,
            activation_layer=None)
        self._conv_linear = ConvNormActivation(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
            norm_layer=norm_layer,
            activation_layer=activation_layer)

    def forward(self, inputs):
        x1, x2 = paddle.split(
            inputs,
            num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            axis=1)
        x1 = self._conv_pw_left(x1)
        x2 = self._conv_pw(x2)
        x2 = self._conv_dw(x2)
        x2 = self._conv_linear(x2)
        out = paddle.concat([x1, x2], axis=1)
        return channel_shuffle(out, 2)


class ResidualDenseBlock_4C(nn.Layer):
    def __init__(self, nf=64, gc=32, ):
        super(ResidualDenseBlock_4C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        # self.conv1 = nn.Conv2D(nf, gc, 3, 1, 1, )
        # self.conv2 = nn.Conv2D(nf + gc, gc, 3, 1, 1, )
        # self.conv3 = nn.Conv2D(nf + 2 * gc, gc, 3, 1, 1, )
        # self.conv4 = nn.Conv2D(nf + 3 * gc, gc, 3, 1, 1, bias_attr=bias)
        # self.conv5 = nn.Conv2D(nf + 4 * gc, nf, 3, 1, 1, bias_attr=bias)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = InvertedResidual(nf, gc, 1, )
        self.conv2 = InvertedResidual(nf + gc, gc, 1, )
        self.conv3 = InvertedResidual(nf + 2 * gc, gc, 1, )
        self.conv_f = InvertedResidualDS(nf + 3 * gc, nf, 1, activation_layer=None)

    def forward(self, x):
        # x1 = self.lrelu(self.conv1(x))
        # x2 = self.lrelu(self.conv2(paddle.concat((x, x1), 1)))
        # x3 = self.lrelu(self.conv3(paddle.concat((x, x1, x2), 1)))
        # x4 = self.lrelu(self.conv4(paddle.concat((x, x1, x2, x3), 1)))
        # x5 = self.conv5(paddle.concat((x, x1, x2, x3, x4), 1))
        # return x5 * 0.2 + x
        x1 = self.conv1(x)
        x2 = self.conv2(paddle.concat((x, x1), 1))
        x3 = self.conv3(paddle.concat((x, x1, x2), 1))
        x4 = self.conv_f(paddle.concat((x, x1, x2, x3), 1))
        return x4 * 0.2 + x


# 对输入特征图进行分离式像素注意力卷积
class PAConv(nn.Layer):
    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2D(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        # self.k3 = nn.Conv2D(nf,
        #                     nf,
        #                     kernel_size=k_size,
        #                     padding=(k_size - 1) // 2,
        #                     bias_attr=False)  # 3x3 convolution
        # self.k4 = nn.Conv2D(nf,
        #                     nf,
        #                     kernel_size=k_size,
        #                     padding=(k_size - 1) // 2,
        #                     bias_attr=False)  # 3x3 convolution
        self.k3 = InvertedResidual(nf, nf, 1, bias=False, activation_layer=None)
        self.k4 = InvertedResidual(nf, nf, 1, bias=False, activation_layer=None)

    def forward(self, x):
        # 使用110卷积生成注意力参数
        y = self.k2(x)
        y = self.sigmoid(y)
        # 特征图经过311卷积配合注意力参数
        out = self.k3(x) * y
        out = self.k4(out)
        return out


class RDBPA(nn.Layer):
    
    def __init__(self, nf=64, reduction=2, with_pa=False):
        super(RDBPA, self).__init__()
        self.with_pa = with_pa
        if with_pa:
            group_width = nf // reduction
            self.conv1_a = nn.Conv2D(nf,
                                     group_width,
                                     kernel_size=1,
                                     bias_attr=False)
            self.conv1_b = nn.Conv2D(nf,
                                     group_width,
                                     kernel_size=1,
                                     bias_attr=False)
            self.rdb = ResidualDenseBlock_4C(group_width, group_width // 2)
            self.pa = PAConv(group_width)
            self.conv3 = nn.Conv2D(group_width * reduction,
                                   nf,
                                   kernel_size=1,
                                   bias_attr=False)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.rdb = ResidualDenseBlock_4C(nf, nf // 2)

    def forward(self, x):
        if self.with_pa:
            residual = x
            out_a = self.conv1_a(x)  # 通过110卷积生成通道数减半的两个特征图
            out_b = self.conv1_b(x)
            out_a = self.lrelu(out_a)
            out_b = self.lrelu(out_b)

            out_a = self.rdb(out_a)
            out_b = self.pa(out_b)
            out_a = self.lrelu(out_a)
            out_b = self.lrelu(out_b)

            out = self.conv3(paddle.concat([out_a, out_b], axis=1))
            out += residual
        else:
            out = self.rdb(x)
        return out


class MultiRDBPA(nn.Layer):
    
    def __init__(self, nf):
        super(MultiRDBPA, self).__init__()
        self.rdb1 = RDBPA(nf=nf, with_pa=False)
        self.rdb2 = RDBPA(nf=nf, with_pa=False)
        self.rdb3 = RDBPA(nf=nf, with_pa=True)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class RDBPANet(nn.Layer):
    
    def __init__(self, in_nc, out_nc, nf, nb,):
        super(RDBPANet, self).__init__()
        RDBPA_block_f = functools.partial(MultiRDBPA, nf=nf)
        self.conv_first = nn.Conv2D(in_nc, nf, 3, 1, 1, bias_attr=True)
        self.rdbpa_trunk = make_layer(RDBPA_block_f, nb)
        self.trunk_conv = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.up_conv_x2 = InvertedResidual(nf, nf, 1, )
        self.up_conv_x4 = InvertedResidual(nf, nf, 1, )
        self.last_x2 = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)
        self.last_x4 = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.rdbpa_trunk(fea))
        fea = fea + trunk

        fea = self.up_conv_x2(F.interpolate(fea, scale_factor=2, mode='nearest'))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out_x2 = self.last_x2(fea) + x

        fea = self.up_conv_x4(F.interpolate(fea, scale_factor=2, mode='nearest'))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out_x4 = self.last_x4(fea) + x

        return out_x2, out_x4

