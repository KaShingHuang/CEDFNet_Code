import numpy as np
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
# from tensorboardX import SummaryWriter
import os
import time
from pytorch_msssim import ssim
import torch.nn.functional as F
from openpyxl import load_workbook
from openpyxl import Workbook
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/underwater.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="5")
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    print(len(val_set))

    # model
    diffusion = Model.create_model(opt)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    print('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    avg_psnr = 0
    ssim_value = 0
    fid_score=0

    # 加载现有的Excel文件---------------------------------------------------------
    wb = Workbook()
    ws = wb.active
    '''
    #获取去噪过程的图像
    for _,  val_data in enumerate(tqdm(val_loader)):
        idx += 1
        filename, val_data = val_data[1], val_data[0]
        if(filename[0]!="1152_1"):
            continue
        diffusion.feed_data(val_data)
        start = time.time()
        diffusion.test(continous=True)
        end = time.time()
        print('Execution time:', (end - start), 'seconds')
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            Metrics.save_img(
                sr_img, '{}/{}_{}_sr_process.png'.format("/data2/CarnegieBin_data/Kashing/EnhanceUf/lunwen", current_step, idx))
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format("/data2/CarnegieBin_data/Kashing/EnhanceUf/lunwen", current_step, idx))
            # for i in range(len(visuals['SR'])):
            #     Metrics.save_img(
            #         Metrics.tensor2img(visuals['SR'][i]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, str(i)))

        Metrics.save_img(
            hr_img, '{}/{}_{}_hr.png'.format("/data2/CarnegieBin_data/Kashing/EnhanceUf/lunwen", current_step, idx))
        Metrics.save_img(
            fake_img, '{}/{}_{}_inf.png'.format("/data2/CarnegieBin_data/Kashing/EnhanceUf/lunwen", current_step, idx))
            '''
    for _, val_data in enumerate(tqdm(val_loader)):
        filename,val_data= val_data[1],val_data[0]
        idx += 1
        diffusion.feed_data(val_data)
        start = time.time()
        diffusion.test(continous=False)
        end = time.time()
        print('Execution time:', (end - start), 'seconds')
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img

            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8

            for i in range(len(visuals['SR'])):
                Metrics.save_img(Metrics.tensor2img(visuals['SR'][i]),
                                 '{}/{}_{}_sr_{}.png'.format("/data2/CarnegieBin_data/Kashing/EnhanceUf/lunwen", current_step, idx, str(i)))
            
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8


           # for i in range(len(visuals['SR'])):
           #      Metrics.save_img(Metrics.tensor2img(visuals['SR'][i]), '{}/{}.png'.format("/data2/CarnegieBin_data/Kashing/EnhanceUf/moretestpatch/U45_restoration", filename[0]))


            avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)
            tensor1 = torch.tensor(sr_img).float() / 255.0
            tensor2 = torch.tensor(hr_img).float() / 255.0

            # 调整张量的形状为(C, H, W)
            tensor1 = tensor1.permute(2, 0, 1).unsqueeze(0)
            tensor2 = tensor2.permute(2, 0, 1).unsqueeze(0)


            # 计算SSIM并累积结果
           # ssim_value += ssim(tensor1, tensor2, data_range=1.0, size_average=True).item()
            #保存psnr的结果到xlsx文件
            new_data = ("{}.png".format(filename[0]), Metrics.calculate_psnr(sr_img, hr_img))
            ws.append(new_data)


    avg_psnr = avg_psnr / idx
    avg_ssim = ssim_value / idx

    print(avg_psnr)
    print(avg_ssim)
   # wb.save("/data2/CarnegieBin_data/Kashing/EnhanceUf/moretestpatch/input_mymethod.xlsx")







