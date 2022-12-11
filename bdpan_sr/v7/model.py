import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .nafnet_arch import NAFNet


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Args:
        nets (network list): a list of networks
        requires_grad (bool): whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.trainable = requires_grad


class NAFModel(nn.Layer):
    
    def __init__(self, naf_pretrain='checkpoint/pretrain/v7_model.pdparams', frozen_naf=False, ):
        super(NAFModel, self).__init__()
        self.naf_net = NAFNet(img_channel=3, width=32,
                              middle_blk_num=1,
                              enc_blk_nums=[1, 1, 1, 10], dec_blk_nums=[1, 1, 1, 1])
        self.naf_net.load_dict(paddle.load(naf_pretrain))
        print(f'loaded naf net : {naf_pretrain}')
        if frozen_naf:
            set_requires_grad(self.naf_net, False)

    def forward(self, x):
        return self.naf_net.forward_final_feat(x)


# 对输入特征图进行像素注意力操作
class PA(nn.Layer):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2D(nf, nf, 1)
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


class SRHeadModel(nn.Layer):

    def __init__(self, num_feat=32, up_num_feat=32, alpha=0.5):
        super(SRHeadModel, self).__init__()
        self.alpha = alpha
        self.deep_feat_trans = nn.Conv2D(num_feat, up_num_feat, 3, 1, 1) if num_feat != up_num_feat else nn.Identity()
        self.shallow_feat_trans = nn.Conv2D(num_feat, up_num_feat, 3, 1, 1) if num_feat != up_num_feat else nn.Identity()
        self.att1 = PA(up_num_feat)
        self.HRconv1 = nn.Conv2D(up_num_feat, up_num_feat, 3, 1, 1)
        self.att2 = PA(up_num_feat)
        self.HRconv2 = nn.Conv2D(up_num_feat, up_num_feat, 3, 1, 1)
        self.conv_last_x2 = nn.Conv2D(up_num_feat, 3, 3, 1, 1)
        self.conv_last_x4 = nn.Conv2D(up_num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.init_weights()

    def forward(self, x, deep_feat, shallow_feat):
        deep_feat = self.deep_feat_trans(deep_feat)
        shallow_feat = self.shallow_feat_trans(shallow_feat)
        fea = self.alpha * deep_feat + (1. - self.alpha) * shallow_feat

        fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = fea + self.lrelu(self.att1(fea))
        fea = fea + self.lrelu(self.HRconv1(fea))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out_x2 = self.conv_last_x2(fea) + x

        fea = F.interpolate(fea, scale_factor=2, mode='nearest')
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
