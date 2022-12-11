import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from itertools import repeat
import collections.abc
import math

from .init import constant_


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


# 将输入（b，h，w，c）转换为（窗口数量，窗口大小，窗口大小，c）
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, window_size, window_size, C])
    return windows


# 上面的逆过程
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([B, H, W, -1])
    return x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    initializer = nn.initializer.TruncatedNormal(mean=mean, std=std)
    initializer(tensor)


class DropPath(nn.Layer):
    """DropPath class"""
    def __init__(self, drop_prob=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0 or self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor() # mask  生成随机的0、1，用于决定是否dropout
        output = inputs.divide(keep_prob) * random_tensor # divide is to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchMerging(nn.Layer):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape([B, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape([B, -1, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2D(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2D(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2D(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class PatchEmbed(nn.Layer):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if self.patch_size[0] > 1:
            self.down_conv = nn.Conv2D(embed_dim, embed_dim, patch_size[0], patch_size[0], 0)
        else:
            self.down_conv = nn.Identity()
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.down_conv(x)
        x = x.flatten(2).transpose([0, 2, 1])  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, tuple(self.patches_resolution)


class PatchUnEmbed(nn.Layer):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if self.patch_size[0] > 1:
            self.up_conv = UpsampleOneStep(self.patch_size[0], embed_dim, embed_dim)
        else:
            self.up_conv = nn.Identity()
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, x_size):
        B, HW, C = x.shape
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose([0, 2, 1]).reshape([B, self.embed_dim, x_size[0], x_size[1]])  # B Ph*Pw C
        x = self.up_conv(x)
        return x


class WindowAttention(nn.Layer):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 生成相对坐标
        # define a parameter table of relative position bias
        self.relative_position_bias_table = paddle.create_parameter(
            [(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads], dtype="float32")  # 2*Wh-1 * 2*Ww-1, nH
        constant_(self.relative_position_bias_table)
        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww  # 这里计算出的是每个坐标点到其余坐标点每个维度上的距离
        relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 为了使得每一个位置索引都是唯一的
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww  # 生成了位置的索引，用以访问上面的可学习参数
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape([B_, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])  # 【3，B，num_head, N, dim】
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose([0, 1, 3, 2]))  # b,nh,n,n

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape([
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:  # 加掩码用于调整softmax后的概率，原始mask为nW，N
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N]) + mask.unsqueeze(1).unsqueeze(0)  # mask调整后为1，nW，1，N
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Layer):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # self.drop_path = nn.Identity()
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = paddle.zeros([1, H, W, 1])  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                # print(h, w,"wh")
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.reshape([-1, self.window_size * self.window_size])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        ones = paddle.ones_like(attn_mask) * (-100.0)
        zero = paddle.zeros_like(attn_mask)
        attn_mask = paddle.where(attn_mask != 0, ones, attn_mask)
        attn_mask = paddle.where(attn_mask == 0, zero, attn_mask)

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape([B, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size))

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape([B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Layer):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0, attn_drop=0,
                 drop_path=0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.LayerList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            # if self.use_checkpoint:
            #     x = checkpoint.checkpoint(blk, x, x_size)
            # else:
            x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RSTB(nn.Layer):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0, attn_drop=0,
                 drop_path=0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2D(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2D(dim, dim // 4, 3, 1, 1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2D(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2D(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        out, _ = self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size)))
        return out + x


class SwinIRPlus(nn.Layer):

    def __init__(self, img_size=64, patch_size=1, in_channel=3, out_channel=3,
                 embed_dim=96, up_dim=48, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 drop_rate=0, attn_drop_rate=0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=False,
                 use_checkpoint=False, resi_connection='1conv', ):
        super(SwinIRPlus, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.embed_dim = embed_dim
        self.up_dim = up_dim
        self.depths = depths
        self.num_layer = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        self.conv_first = nn.Conv2D(in_channel, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.num_patch = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        if self.ape:
            self.absolute_pos_embed = paddle.create_parameter([1, self.num_patch, embed_dim])
            constant_(self.absolute_pos_embed, 0.)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layer):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(self.patches_resolution[0],
                                           self.patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2D(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2D(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, ),
                                                 nn.Conv2D(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, ),
                                                 nn.Conv2D(embed_dim // 4, embed_dim, 3, 1, 1))

        # up sample
        self.up_conv_x2 = nn.Conv2D(embed_dim, up_dim, 3, 1, 1)
        self.hrconv_x2 = nn.Sequential(
            nn.Conv2D(up_dim, up_dim, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2D(up_dim, up_dim, 3, 1, 1),
        )
        self.last_conv_x2 = nn.Conv2D(up_dim, out_channel, 3, 1, 1)
        self.up_conv_x4 = nn.Conv2D(up_dim, up_dim, 3, 1, 1)
        self.hrconv_x4 = nn.Sequential(
            nn.Conv2D(up_dim, up_dim, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2D(up_dim, up_dim, 3, 1, 1),
        )
        self.last_conv_x4 = nn.Conv2D(up_dim, out_channel, 3, 1, 1)
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        from .init import _calculate_fan_in_and_fan_out, kaiming_uniform_, uniform_
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_(m.bias, 0)
            constant_(m.weight, 1.0)
        elif hasattr(m, 'weight') and (not isinstance(
                m, (nn.BatchNorm, nn.BatchNorm2D))):
            kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                uniform_(m.bias, -bound, bound)

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        # x_size = (x.shape[2], x.shape[3])
        # x = self.patch_embed(x)
        x, x_size = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        x = self.check_image_size(x)
        # shallow feature extraction
        feat = self.conv_first(x)
        # deep feature extraction
        feat = self.conv_after_body(self.forward_features(feat) + feat)
        # up sample
        feat = self.up_conv_x2(F.interpolate(feat, scale_factor=2, mode='nearest'))
        feat = self.hrconv_x2(feat) + feat
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out_x2 = self.last_conv_x2(feat) + x
        feat = self.up_conv_x4(F.interpolate(feat, scale_factor=2, mode='nearest'))
        feat = self.hrconv_x4(feat) + feat
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out_x4 = self.last_conv_x4(feat) + x + \
                 F.interpolate(out_x2, scale_factor=2, mode='bilinear', align_corners=False)
        return out_x2, out_x4

