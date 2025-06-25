import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tqdm import tqdm
import os
import time
from warmup_scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/underwater.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'], help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-batch_size', type=int, default=64)  #96


    # parse configs
    args = parser.parse_args()   #解析命令行参数把结果存在args里面
    opt = Logger.parse(args)    # 把args的参数变为字典的形式
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)    #把字典的形式变为NoneDict的形式，主要是当键不存在的时候，会返回None而不是抛出异常
    opt["datasets"]["train"]["batch_size"]=args.batch_size
    print(opt)
    # logging，记录日志的，
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)   #日志记录器，名为train,记录train时候的INFO以上的日志到log中，
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    #tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])



    # dataset，创建一个数据集
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')
    print(len(val_set))
    print(len(train_set))


    # model，在这里建立模型
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(diffusion.optG, 2000 , eta_min=0.000001)
    scheduler = GradualWarmupScheduler(diffusion.optG, multiplier=1, total_epoch=10,after_scheduler=scheduler_cosine)
    scheduler.step()

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    for param_group in diffusion.optG.param_groups:
        print(f'Learning rate: {param_group["lr"]}')


    n_iter = 1000000

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    best_avg_psnr=29   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!恢复训练的时候这里要记得改
    printAndVal=len(train_set)//opt["datasets"]["train"]["batch_size"]//2
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])      #默认的phase是train
    if opt['phase'] == 'train':
        while current_epoch < n_iter:
            start_time = time.time()
            current_epoch += 1
            current_step=1
            for param_group in diffusion.optG.param_groups:
                logger.info('{}轮当前的学习率是{}'.format(current_epoch,param_group["lr"]))
            for _, train_data in enumerate(tqdm(train_loader), 0):
                current_step += 1
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()    #训练diffusion

                if current_step % printAndVal == 0:       #每隔多少次打印一次当前的信息
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)

                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    logger.info(message)

                # validation                                             #每隔多少次验证一次
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - start_time
            logger.info('该轮的时间是{}'.format(epoch_time))
            scheduler.step()# save model
            if current_epoch % 50== 0:                                #每100轮执行一次验证，好比原来的设置每50000步执行一次
                avg_psnr = 0.0
                idx = 0
                result_path = '{}/{}'.format(opt['path']
                                             ['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['val'], schedule_phase='val')
                for _, val_data in enumerate(tqdm(val_loader), 0):
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    avg_psnr += Metrics.batch_PSNR(visuals['SR'].double(), visuals['HR'].double())
                avg_psnr = avg_psnr / len(val_set)
                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['train'], schedule_phase='train')
                # log

                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    current_epoch, current_step, avg_psnr))


                if (avg_psnr > best_avg_psnr):
                    best_avg_psnr = avg_psnr
                    logger.info('Get best PSNR,Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step, best_avg_psnr)
                else:
                    diffusion.save_network(current_epoch, current_step)
                '''
                else:
                    diffusion.save_network(current_epoch, current_step)
                '''
        print("最好结果是{}".format(best_avg_psnr))
        logger.info('End of training.')
































