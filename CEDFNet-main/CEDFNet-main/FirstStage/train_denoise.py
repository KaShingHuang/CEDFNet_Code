import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
print(sys.path)
print(dir_name)
import argparse
import options
######### parser ###########，方便看当前在做什么
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
print(opt)

import utils
from dataset.dataset_denoise import *
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu    #这里要看实际还有多少个gpu，已经设置成了0，1
import torch
torch.backends.cudnn.benchmark = True  #可以提高训练的性能
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from losses import CharbonnierLoss
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler


######### Logs dir ########### log的目录是  ../logs/stage1/LSUI
log_dir = os.path.join(opt.save_dir, opt.dataset, opt.arch)
if not os.path.exists(log_dir):  #log目录不存在就新建
    print("不存在")
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt')  # 这里不知道有没有创建，要debug，下面的Model open那里会进行创建
print("Now time is : ",datetime.datetime.now().isoformat())
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(model_dir) #和模型目录

# ######### Set Seeds ###########，确保随机数生成是固定的
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:    #把当前opt的参数和生成的模型细节写入logname，logname若不存在会自动进行创建
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########，优化器设置
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ########### 
model_restoration = torch.nn.DataParallel (model_restoration) 
model_restoration.cuda() 
     

######### Scheduler ###########
step = 100
print("Using StepLR,step={}!".format(step))
scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
scheduler.step()

######### Resume ###########   #可以加载之前阶段的训练好的权重，有时候可能会断开，可以继续训练
if opt.resume: 
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    lr = utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')


######### Loss ###########
criterion = CharbonnierLoss().cuda()   #损失函数

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True,
        num_workers=12, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False,
        num_workers=10, pin_memory=True, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### validation ###########，没搞懂为什么一开始就做验证？？？？？？？？？？？？？？？？？？？？？？？？？？？
'''
with torch.no_grad():
    model_restoration.eval()
    psnr_dataset = []
    psnr_model_init = []
    for ii, data_val in enumerate((val_loader), 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            restored = torch.clamp(restored,0,1)  
        psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
        psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
    psnr_dataset = sum(psnr_dataset)/len_valset
    psnr_model_init = sum(psnr_model_init)/len_valset
    print('Input & GT (PSNR) -->%.4f dB'%(psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB'%(psnr_model_init))
'''


######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 20
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)//2
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    for i, data in enumerate(tqdm(train_loader), 0):    #tqdm是进度条
        # zero_grad
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()

        if epoch>5:
            target, input_ = utils.MixUp_AUG().aug(target, input_)     #进行数据混合增强
        with torch.cuda.amp.autocast():  #实现自动混合精度
            restored = model_restoration(input_)
            loss = criterion(restored, target)
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss +=loss.item()

        #### Evaluation ####，每个epoch过了1/4的数据就进行一次验证

        if (i+1)%eval_now==0 and i>0:
            starttime = time.time()
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_)
                    restored = torch.clamp(restored, 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb) / len_valset
                endtime = time.time()
                print("Validation Time: ", endtime - starttime)

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))

                print(
                    "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (
                        epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
                with open(logname, 'a') as f:
                    f.write(
                        "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')
                model_restoration.train()
                torch.cuda.empty_cache()

    scheduler.step()
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')


print("Now time is : ",datetime.datetime.now().isoformat())
