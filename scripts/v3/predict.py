import os
import sys
import glob
import cv2
from PIL import Image
import numpy as np
from paddle.io import DataLoader
import paddle

from dowdyboy_lib.assist import calc_time_used
from bdpan_sr.v3.dataset import SRTestDataset
from bdpan_sr.v3.model import RRDBNet

assert len(sys.argv) == 4

src_image_dir = sys.argv[1]
save_x2_dir = sys.argv[2]
save_x4_dir = sys.argv[3]


def to_img_arr(x, un_norm):
    y = un_norm(x)
    y = y.numpy().transpose(1, 2, 0)
    y = np.clip(y, 0., 255.).astype(np.uint8)
    return y


def build_data():
    test_dataset = SRTestDataset(src_image_dir, use_normal=True)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=0, drop_last=False)
    return test_loader, test_dataset


def build_model():
    model = RRDBNet(
        in_nc=3,
        out_nc=3,
        nf=32,
        gc=16,
        # nb=23,
        nb=12,
        scale=4,
        is_init_weight=True,
    )
    return model


def test_step(model, bat, dataset):
    bat_im, bat_filepath = bat
    _, _, ori_h, ori_w = bat_im.shape
    pad = 6
    step = 500
    pre_m = None
    pre_pad_right = 0
    pre_pad_bottom = 0
    if ori_h < step + 2 * pad or ori_w < step + 2 * pad:
        if ori_h < step + 2 * pad:
            pre_pad_bottom = step + 2 * pad - ori_h
        if ori_w < step + 2 * pad:
            pre_pad_right = step + 2 * pad - ori_w
        pre_m = paddle.nn.Pad2D((0, pre_pad_right, 0, pre_pad_bottom), )
        bat_im = pre_m(bat_im)
    m = paddle.nn.Pad2D(pad, mode='reflect')
    bat_im = m(bat_im)
    _, _, h, w = bat_im.shape
    res_x2 = paddle.zeros((bat_im.shape[0], bat_im.shape[1], h * 2, w * 2))
    res_x4 = paddle.zeros((bat_im.shape[0], bat_im.shape[1], h * 4, w * 4))

    all_count = 0

    for i in range(0, h, step):
        for j in range(0, w, step):
            all_count += 1
            if h - i < step + 2 * pad:
                i = h - (step + 2 * pad)
            if w - j < step + 2 * pad:
                j = w - (step + 2 * pad)
            clip = bat_im[:, :, i:i + step + 2 * pad, j:j + step + 2 * pad]
            pred_x2, pred_x4 = model(clip)
            res_x2[:, :, 2*i + 2*pad:2*i + 2*step + 2*pad, 2*j + 2*pad:2*j + 2*step + 2*pad] = pred_x2[:, :, 2*pad:-2*pad, 2*pad:-2*pad]
            res_x4[:, :, 4*i + 4*pad:4*i + 4*step + 4*pad, 4*j + 4*pad:4*j + 4*step + 4*pad] = pred_x4[:, :, 4*pad:-4*pad, 4*pad:-4*pad]
            # res[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = restore_pred[:, :, pad:-pad, pad:-pad]

    # print(f'forward count : {all_count}')
    res_x2 = res_x2[:, :, 2*pad:-2*pad, 2*pad:-2*pad]
    res_x4 = res_x4[:, :, 4*pad:-4*pad, 4*pad:-4*pad]
    if pre_m is not None:
        res_x2 = res_x2[:, :, :ori_h * 2, :ori_w * 2]
        res_x4 = res_x4[:, :, :ori_h * 4, :ori_w * 4]
    res_x2 = to_img_arr(res_x2[0], dataset.un_normalize)
    res_x4 = to_img_arr(res_x4[0], dataset.un_normalize)
    Image.fromarray(res_x2).save(
        os.path.join(save_x2_dir, os.path.basename(bat_filepath[0]))
    )
    Image.fromarray(res_x4).save(
        os.path.join(save_x4_dir, os.path.basename(bat_filepath[0]))
    )


def main():
    test_loader, test_dataset = build_data()
    # model_chk_path = 'checkpoint/v3/best_epoch_93/model_0.pd'
    # model_chk_path = 'checkpoint/v3/best_epoch_173/model_0.pd'
    model_chk_path = 'checkpoint/v3/best_epoch_303/model_0.pd'
    model = build_model()
    model.load_dict(paddle.load(model_chk_path))
    model.eval()

    for bat in test_loader:
        with paddle.no_grad():
            test_step(model, bat, test_dataset)


if __name__ == '__main__':
    if not os.path.exists(save_x2_dir):
        os.makedirs(save_x2_dir)
    if not os.path.exists(save_x4_dir):
        os.makedirs(save_x4_dir)
    main()
