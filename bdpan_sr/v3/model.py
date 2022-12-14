import functools
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def pixel_unshuffle(x, scale):
    """ Pixel unshuffle function.

    Args:
        x (paddle.Tensor): Input feature.
        scale (int): Downsample ratio.

    Returns:
        paddle.Tensor: the pixel unshuffled feature.
    """
    b, c, h, w = x.shape
    out_channel = c * (scale**2)
    assert h % scale == 0 and w % scale == 0
    hh = h // scale
    ww = w // scale
    x_reshaped = x.reshape([b, c, hh, scale, ww, scale])
    return x_reshaped.transpose([0, 1, 3, 5, 2,
                                 4]).reshape([b, out_channel, hh, ww])


class ResidualDenseBlock_5C(nn.Layer):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2D(nf, gc, 3, 1, 1, bias_attr=bias)
        self.conv2 = nn.Conv2D(nf + gc, gc, 3, 1, 1, bias_attr=bias)
        self.conv3 = nn.Conv2D(nf + 2 * gc, gc, 3, 1, 1, bias_attr=bias)
        self.conv4 = nn.Conv2D(nf + 3 * gc, gc, 3, 1, 1, bias_attr=bias)
        self.conv5 = nn.Conv2D(nf + 4 * gc, nf, 3, 1, 1, bias_attr=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(paddle.concat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(paddle.concat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(paddle.concat((x, x1, x2, x3), 1)))
        x5 = self.conv5(paddle.concat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Layer):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class RRDBNet(nn.Layer):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, is_init_weight=False):
        super(RRDBNet, self).__init__()

        self.scale = scale
        if scale == 2:
            in_nc = in_nc * 4
        elif scale == 1:
            in_nc = in_nc * 16

        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2D(in_nc, nf, 3, 1, 1, bias_attr=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)

        #### upsampling
        self.upconv1 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.upconv2 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)

        ## last
        self.HRconv_x2 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv_last_x2 = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)
        self.HRconv_x4 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv_last_x4 = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        if is_init_weight:
            self.init_weights()

    def forward(self, x):
        if self.scale == 2:  # ????????????4?????????????????????????????????????????????????????????????????????
            fea = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            fea = pixel_unshuffle(x, scale=4)
        else:
            fea = x

        fea = self.conv_first(fea)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(
            self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out_x2 = self.conv_last_x2(self.lrelu(self.HRconv_x2(fea)))
        fea = self.lrelu(
            self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out_x4 = self.conv_last_x4(self.lrelu(self.HRconv_x4(fea)))

        return out_x2, out_x4

    def init_weights(self):
        from .init import reset_parameters
        def reset_func(m):
            if hasattr(m, 'weight') and (not isinstance(
                    m, (nn.BatchNorm, nn.BatchNorm2D))):
                reset_parameters(m)
        self.apply(reset_func)

