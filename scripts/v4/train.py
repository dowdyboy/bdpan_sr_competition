import paddle
from paddle.io import DataLoader
import os
import argparse
from PIL import Image
import numpy as np

from dowdyboy_lib.paddle.trainer import Trainer, TrainerConfig

from bdpan_sr.v4.optim import CosineAnnealingRestartLR
from bdpan_sr.v4.model import PANPlus
from bdpan_sr.v4.dataset import SRDataset
from bdpan_sr.v4.loss import SRLoss
from bdpan_sr.v4.psnr_ssim import calculate_psnr, calculate_ssim


parser = argparse.ArgumentParser(description='sr v4')
# data config
parser.add_argument('--train-data-dir', type=str, required=True, help='train data dir')
parser.add_argument('--val-data-dir', type=str, required=True, help='val data dir')
parser.add_argument('--num-workers', type=int, default=8, help='num workers')
parser.add_argument('--img-size', type=int, default=128, help='input img size')
# optimizer config
parser.add_argument('--lr', type=float, default=0.0001, help='lr')
parser.add_argument('--use-scheduler', default=False, action='store_true', help='use schedule')
parser.add_argument('--use-warmup', default=False, action='store_true', help='use warmup')
parser.add_argument('--weight-decay', type=float, default=0., help='model weight decay')
# train config
parser.add_argument('--epoch', type=int, default=10, help='epoch num')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--out-dir', type=str, default='./output', help='out dir')
parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
parser.add_argument('--last-epoch', type=int, default=-1, help='last epoch')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--log-interval', type=int, default=500, help='log process')
parser.add_argument('--save-val-count', type=int, default=50, help='log process')
parser.add_argument('--sync-bn', default=False, action='store_true', help='sync_bn')
parser.add_argument('--device', default=None, type=str, help='device')

args = parser.parse_args()


def to_img_arr(x, un_norm):
    y = un_norm((x, x, x))[0]
    y = y.numpy().transpose(1, 2, 0)
    y = np.clip(y, 0., 255.).astype(np.uint8)
    return y


def build_data():
    train_dataset = SRDataset(
        root_dir=args.train_data_dir,
        patch_size=args.img_size,
        is_val=False,
        hflip_p=0.5,
        vflip_p=0.5,
        transpose_p=0.5,
        use_normal=True,
        use_cache=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_dataset = SRDataset(
        root_dir=args.val_data_dir,
        patch_size=args.img_size,
        is_val=True,
        use_normal=True,
        use_cache=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)
    return train_loader, train_dataset, val_loader, val_dataset


def build_model():
    model = PANPlus(3, 3, 64, 32, 21)
    return model


# paddle.nn.Layer | list[paddle.nn.Layer]
def build_optimizer(model):
    lr = args.lr
    lr_scheduler = None
    if args.use_scheduler:
        # lr = paddle.optimizer.lr.CosineAnnealingDecay(lr, args.epoch, last_epoch=args.last_epoch, verbose=True)
        lr = CosineAnnealingRestartLR(
            lr,
            periods=[250000, 250000, 250000, 250000],
            restart_weights=[1, 1, 1, 1],
            eta_min=1e-7,
            last_epoch=args.last_epoch,
        )
        lr_scheduler = lr
    if args.use_warmup:
        lr = paddle.optimizer.lr.LinearWarmup(lr, 10, args.lr * 0.1, args.lr, last_epoch=args.last_epoch, verbose=True)
        lr_scheduler = lr
    optimizer = paddle.optimizer.Adam(
        lr,
        parameters=[{
            'params': m.parameters()
        } for m in model] if isinstance(model, list) else model.parameters(),
        weight_decay=args.weight_decay,
        beta1=0.9,
        beta2=0.99,
    )
    return optimizer, lr_scheduler


def train_step(trainer: Trainer, bat, bat_idx, global_step):
    # [model, loss_func] = trainer.get_models()
    [model] = trainer.get_models()
    [loss_func] = trainer.get_components()
    _, [lr_scheduler] = trainer.get_optimizers()

    bat_x, bat_x2, bat_x4 = bat
    pred_x2, pred_x4 = model(bat_x)
    loss_x2 = loss_func(pred_x2, bat_x2, global_step)
    loss_x4 = loss_func(pred_x4, bat_x4, global_step)
    loss = 0.4 * loss_x2 + 0.6 * loss_x4

    trainer.log({
        'train_loss': loss.item(),
        'train_loss_x2': loss_x2.item(),
        'train_loss_x4': loss_x4.item(),
    }, global_step)
    trainer.set_records({
        'train_loss': loss.item(),
    })
    if global_step % args.log_interval == 0:
        trainer.print(f'global step: {global_step}, loss: {loss.item()}')

    trainer.step(lr_scheduler=lr_scheduler)
    return loss


def val_step(trainer: Trainer, bat, bat_idx, global_step):
    # [model, loss_func] = trainer.get_models()
    [model] = trainer.get_models()
    [loss_func] = trainer.get_components()
    un_normalize = trainer.val_dataloader.dataset.un_normalize

    bat_x, bat_x2, bat_x4 = bat

    _, _, h, w = bat_x.shape
    _, _, h_x2, w_x2 = bat_x2.shape
    _, _, h_x4, w_x4 = bat_x4.shape
    rh, rw = h, w
    rh_x2, rw_x2 = h_x2, w_x2
    rh_x4, rw_x4 = h_x4, w_x4
    step = args.img_size
    pad_h = step - h if h < step else 0
    pad_w = step - w if w < step else 0
    m = paddle.nn.Pad2D((0, pad_w, 0, pad_h))
    m_x2 = paddle.nn.Pad2D((0, pad_w * 2, 0, pad_h * 2))
    m_x4 = paddle.nn.Pad2D((0, pad_w * 4, 0, pad_h * 4))
    bat_x = m(bat_x)
    bat_x2 = m_x2(bat_x2)
    bat_x4 = m_x4(bat_x4)
    _, _, h, w = bat_x.shape
    _, _, h_x2, w_x2 = bat_x2.shape
    _, _, h_x4, w_x4 = bat_x4.shape
    res_x2 = paddle.zeros_like(bat_x2)
    res_x4 = paddle.zeros_like(bat_x4)
    loss_list = []
    for i in range(0, h, step):
        for j in range(0, w, step):
            if h - i < step:
                i = h - step
            if w - j < step:
                j = w - step
            clip_x = bat_x[:, :, i:i+step, j:j+step]
            clip_x2 = bat_x2[:, :, i*2:i*2+step*2, j*2:j*2+step*2]
            clip_x4 = bat_x4[:, :, i*4:i*4+step*4, j*4:j*4+step*4]
            pred_x2, pred_x4 = model(clip_x)
            loss_x2 = loss_func(pred_x2, clip_x2)
            loss_x4 = loss_func(pred_x4, clip_x4)
            loss = 0.4 * loss_x2 + 0.6 * loss_x4
            loss_list.append(loss.item())
            res_x2[:, :, i*2:i*2+step*2, j*2:j*2+step*2] = pred_x2
            res_x4[:, :, i*4:i*4+step*4, j*4:j*4+step*4] = pred_x4
            # break
        # break
    loss = sum(loss_list) / len(loss_list)
    res_x2 = res_x2[:, :, :rh_x2, :rw_x2]
    res_x4 = res_x4[:, :, :rh_x4, :rw_x4]
    bat_x2 = bat_x2[:, :, :rh_x2, :rw_x2]
    bat_x4 = bat_x4[:, :, :rh_x4, :rw_x4]

    pred_im_x2 = to_img_arr(res_x2[0], un_normalize)
    pred_im_x4 = to_img_arr(res_x4[0], un_normalize)
    gt_im_x2 = to_img_arr(bat_x2[0], un_normalize)
    gt_im_x4 = to_img_arr(bat_x4[0], un_normalize)

    psnr_x2 = float(calculate_psnr(pred_im_x2, gt_im_x2, crop_border=4, test_y_channel=True, ))
    psnr_x4 = float(calculate_psnr(pred_im_x4, gt_im_x4, crop_border=4, test_y_channel=True, ))
    ssim_x2 = float(calculate_ssim(pred_im_x2, gt_im_x2, crop_border=4, test_y_channel=True, ))
    ssim_x4 = float(calculate_ssim(pred_im_x4, gt_im_x4, crop_border=4, test_y_channel=True, ))
    mean_psnr = (0.4 * psnr_x2 + 0.6 * psnr_x4)
    mean_ssim = (0.4 * ssim_x2 + 0.6 * ssim_x4)
    psnr_ssim = mean_psnr / 15. + mean_ssim

    trainer.log({
        'val_loss': loss,
        'val_psnr_x2': psnr_x2,
        'val_ssim_x2': ssim_x2,
        'val_psnr_x4': psnr_x4,
        'val_ssim_x4': ssim_x4,
        'val_psnr_ssim': psnr_ssim,
    }, global_step)
    trainer.set_bar_state({
        'val_psnr_x2': psnr_x2,
        'val_ssim_x2': ssim_x2,
        'val_psnr_x4': psnr_x4,
        'val_ssim_x4': ssim_x4,
    })
    trainer.set_records({
        'psnr_ssim': psnr_ssim,
        'mean_psnr': mean_psnr,
        'mean_ssim': mean_ssim,
        'val_loss': loss,
    })
    Image.fromarray(pred_im_x2).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_pred_x2.png'))
    Image.fromarray(pred_im_x4).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_pred_x4.png'))
    Image.fromarray(gt_im_x2).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_gt_x2.png'))
    Image.fromarray(gt_im_x4).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_gt_x4.png'))

    return loss


def on_epoch_end(trainer: Trainer, ep):
    [optimizer], [lr_scheduler] = trainer.get_optimizers()
    rec = trainer.get_records()
    psnr_ssim = paddle.mean(rec['psnr_ssim']).item()
    mean_psnr = paddle.mean(rec['mean_psnr']).item()
    mean_ssim = paddle.mean(rec['mean_ssim']).item()
    val_loss = paddle.mean(rec['val_loss']).item()
    trainer.log({
        'ep_psnr_ssim': psnr_ssim,
        'ep_val_loss': val_loss,
        'ep_lr': optimizer.get_lr(),
    }, ep)
    trainer.print(f'loss : {val_loss}, mean_psnr : {mean_psnr}, mean_ssim : {mean_ssim}, '
                  f'psnr_ssim : {psnr_ssim}, lr : {optimizer.get_lr()}')


def main():
    cfg = TrainerConfig(
        epoch=args.epoch,
        out_dir=args.out_dir,
        mixed_precision='no',
        multi_gpu=False,
        device=args.device,
        save_interval=5,
        save_best=True,
        save_best_type='max',
        save_best_rec='psnr_ssim',
        seed=args.seed,
        auto_optimize=True,
        auto_schedule=False,
        auto_free=False,
        sync_bn=args.sync_bn,
    )
    trainer = Trainer(cfg)
    trainer.print(args)

    train_loader, train_dataset, val_loader, val_dataset = build_data()
    trainer.print(f'train size: {len(train_dataset)}, val size: {len(val_dataset)}')

    model = build_model()
    loss_func = SRLoss(step_per_epoch=3000 // args.batch_size, max_epoch=10)
    # optimizer, lr_scheduler = build_optimizer([model, loss_func])
    optimizer, lr_scheduler = build_optimizer(model)

    trainer.set_train_dataloader(train_loader)
    trainer.set_val_dataloader(val_loader)
    # trainer.set_models([model, loss_func])
    trainer.set_model(model)
    trainer.set_components([loss_func])
    trainer.set_optimizer(optimizer, lr_scheduler=lr_scheduler)

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        trainer.print(f'load checkpoint from {args.resume}')

    trainer.fit(
        train_step=train_step,
        val_step=val_step,
        on_epoch_end=on_epoch_end,
    )


if __name__ == '__main__':
    main()
