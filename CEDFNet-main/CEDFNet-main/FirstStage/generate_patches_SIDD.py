from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse
#产生训练数据，下面的参数分别是完整的原始图片目录，生成的patch的目录，每个patch的大小，一张图分为多少个patch，使用多少个cpu核心进行处理
parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='./LSUI', type=str, help='Directory for full resolution images')   #原始图片的目录
parser.add_argument('--tar_dir', default='./patches/train',type=str, help='Directory for image patches') #生成的patch的目录
parser.add_argument('--ps', default=200, type=int, help='Image Patch Size')  #改成了224*224的patch
parser.add_argument('--num_patches', default=10, type=int, help='Number of patches per image')   #每张照片分为多少个patchs、
parser.add_argument('--num_cores', default=10, type=int, help='Number of CPU Cores')   #我改成了使用10个cpu核心

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores
#分别代表了处理后生成的噪声图片的目录和真实GroundTruth图片的目录
noisy_patchDir = os.path.join("/data2/CarnegieBin_data/Kashing/dataset/UIEB/diffusion input/train", 'input')
clean_patchDir = os.path.join("/data2/CarnegieBin_data/Kashing/dataset/UIEB/diffusion input/train", 'groundtruth')

Val_noisy_patchDir = os.path.join('/data2/CarnegieBin_data/Kashing/dataset/UIEB/Diffusion_input/val', 'input')  #建立验证集的目录
Val_clean_patchDir = os.path.join('/data2/CarnegieBin_data/Kashing/dataset/UIEB/Diffusion_input/val', 'groundtruth')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

#os.makedirs(noisy_patchDir)
#os.makedirs(clean_patchDir)
#os.makedirs(Val_noisy_patchDir)
#os.makedirs(Val_clean_patchDir)


noisy_files=natsorted(glob(os.path.join("/data2/CarnegieBin_data/Kashing/dataset/UIEB/train", 'input', '*.[pb][nm][gp]')))
clean_files=natsorted(glob(os.path.join("/data2/CarnegieBin_data/Kashing/dataset/UIEB/train", 'groundtruth', '*.[pb][nm][gp]')))

val_noisy_files=natsorted(glob(os.path.join("/data2/CarnegieBin_data/Kashing/dataset/UIEB/val/input", '*.png')))
val_clean_files=natsorted(glob(os.path.join("/data2/CarnegieBin_data/Kashing/dataset/UIEB/val/groundtruth", '*.png')))
def save_files(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)
    str = noisy_file.split("/")[-1].split(".")[0]
    H = noisy_img.shape[0]   #高
    W = noisy_img.shape[1]   #宽
    PS=256

    if H < 256 and W < 256:
        noisy_img = cv2.resize(noisy_img, (256, 256))
        clean_img = cv2.resize(clean_img, (256, 256))
        H, W = 256, 256
    elif H < 256:
        noisy_img = cv2.resize(noisy_img, (W, 256))
        clean_img = cv2.resize(clean_img, (W, 256))
        H = 256
    elif W < 256:
        noisy_img = cv2.resize(noisy_img, (256, H))
        clean_img = cv2.resize(clean_img, (256, H))
        W = 256

    for j in range(5):  # 对每一张图片进行切割，随机切割为NUM_PATCHES个patch
        if H != 256:
            rr = np.random.randint(0, H - 256)
        else:
            rr = 0
        if W != 256:
            cc = np.random.randint(0, W - 256)
        else:
            cc = 0
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]  # 切片得到对应的patch，最后一个:表示选中所有的颜色通道
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]  # 切片得到对应的GroundTruth

        cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(str, j + 1)), noisy_patch)
        cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(str, j + 1)), clean_patch)



def save_files2(i):
    PS=256
    noisy_file, clean_file = val_noisy_files[i], val_clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)
    str = noisy_file.split("/")[-1].split(".")[0]
    H = noisy_img.shape[0]   #高
    W = noisy_img.shape[1]   #宽
    PS = 256

    if H < 256 and W < 256:
        noisy_img = cv2.resize(noisy_img, (256, 256))
        clean_img = cv2.resize(clean_img, (256, 256))
        H, W = 256, 256
    elif H < 256:
        noisy_img = cv2.resize(noisy_img, (W, 256))
        clean_img = cv2.resize(clean_img, (W, 256))
        H = 256
    elif W < 256:
        noisy_img = cv2.resize(noisy_img, (256, H))
        clean_img = cv2.resize(clean_img, (256, H))
        W = 256

    for j in range(1):  # 对每一张图片进行切割，随机切割为NUM_PATCHES个patch
        if H != 256:
            rr = np.random.randint(0, H - 256)
        else:
            rr = 0
        if W != 256:
            cc = np.random.randint(0, W - 256)
        else:
            cc = 0
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :] #切片得到对应的patch，最后一个:表示选中所有的颜色通道
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :] #切片得到对应的GroundTruth

        cv2.imwrite(os.path.join(Val_noisy_patchDir, '{}_{}.png'.format(str,j+1)), noisy_patch)
        cv2.imwrite(os.path.join(Val_clean_patchDir, '{}_{}.png'.format(str,j+1)), clean_patch)

#Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(noisy_files))))
Parallel(n_jobs=NUM_CORES)(delayed(save_files2)(i) for i in tqdm(range(len(val_clean_files))))
