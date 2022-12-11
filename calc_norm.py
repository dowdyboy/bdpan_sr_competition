import os
import cv2
import numpy as np


root_path = r'F:\BaiduNetdiskDownload\bdpan_sr\train_full'       #图片保存路径

def compute(root_path):
    img_count = 0
    for sub_dir_name in os.listdir(root_path):
        sub_dir_path = os.path.join(root_path, sub_dir_name, 'x')
        file_names = os.listdir(sub_dir_path)
        per_image_Rmean = []
        per_image_Gmean = []
        per_image_Bmean = []
        for file_name in file_names:
            img = cv2.imread(os.path.join(sub_dir_path, file_name), 1)
            per_image_Bmean.append(np.mean(img[:, :, 0]))
            per_image_Gmean.append(np.mean(img[:, :, 1]))
            per_image_Rmean.append(np.mean(img[:, :, 2]))
            img_count += 1
    R_mean = np.mean(per_image_Rmean)/255
    G_mean = np.mean(per_image_Gmean)/255
    B_mean = np.mean(per_image_Bmean)/255
    stdR = np.std(per_image_Rmean)/255
    stdG = np.std(per_image_Gmean)/255
    stdB = np.std(per_image_Bmean)/255
    print(f'img_count: {img_count}')
    return R_mean, G_mean, B_mean, stdR, stdG, stdB


if __name__ == '__main__':
    R_mean, G_mean, B_mean, stdR, stdG, stdB = compute(root_path)
    print("R_mean= ", R_mean, "G_mean= ", G_mean, "B_mean=", B_mean, "stdR = ", stdR, "stdG = ", stdG, "stdB =", stdB)
