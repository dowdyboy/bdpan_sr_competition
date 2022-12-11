import os
import sys
import glob
import cv2
from PIL import Image
import numpy as np
from paddle.io import DataLoader
import paddle
import paddle.nn.functional as F

from dowdyboy_lib.assist import calc_time_used
from bdpan_sr.v1.dataset import SRTestDataset
from bdpan_sr.v1.model import RRDBNet

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
        nf=64,
        nb=23,
        scale=4,
        is_init_weight=True,
    )
    return model


def test_step(model, bat, dataset):
    bat_im, bat_filepath = bat
    bat_im_x2 = F.interpolate(bat_im, scale_factor=2, mode='bilinear')
    bat_im_x4 = F.interpolate(bat_im, scale_factor=4, mode='bilinear')
    ori_im_x2 = to_img_arr(bat_im_x2[0], dataset.un_normalize)
    ori_im_x4 = to_img_arr(bat_im_x4[0], dataset.un_normalize)

    pad = 6
    m = paddle.nn.Pad2D(pad, mode='reflect')
    bat_im = m(bat_im)
    _, _, h, w = bat_im.shape
    step = 500
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
    res_x2 = to_img_arr(res_x2[0], dataset.un_normalize)
    res_x4 = to_img_arr(res_x4[0], dataset.un_normalize)
    res_x2 = (res_x2.astype(np.float) + ori_im_x2.astype(np.float)) / 2.
    res_x4 = (res_x4.astype(np.float) + ori_im_x4.astype(np.float)) / 2.
    res_x2 = res_x2.astype(np.uint8)
    res_x4 = res_x4.astype(np.uint8)
    Image.fromarray(res_x2).save(
        os.path.join(save_x2_dir, os.path.basename(bat_filepath[0]))
    )
    Image.fromarray(res_x4).save(
        os.path.join(save_x4_dir, os.path.basename(bat_filepath[0]))
    )


def main():
    test_loader, test_dataset = build_data()
    # model_chk_path = 'checkpoint/v1/best_epoch_122/model_0.pd'
    model_chk_path = 'checkpoint/v1/best_epoch_185/model_0.pd'
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
