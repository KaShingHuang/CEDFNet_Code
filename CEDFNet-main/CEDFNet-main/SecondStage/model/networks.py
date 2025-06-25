import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    model_opt = opt['model']
    from .ddpm_trans_modules import diffusion, unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:   #这里opt中默认的是24
        model_opt['unet']['norm_groups']=32
    if model_opt['which_model_G'] == 'trans_div':                  #默认不执行
        model = unet_backup.DiT(depth=12, in_channels=6, hidden_size=384, patch_size=4, num_heads=6, input_size=128)
    else:
        model = unet.UNet(                                       #这里是论文里面Unet的设置
            in_channel=model_opt['unet']['in_channel'],       #6
            out_channel=model_opt['unet']['out_channel'],     #3
            norm_groups=model_opt['unet']['norm_groups'],          #24
            inner_channel=model_opt['unet']['inner_channel'],       #48
            channel_mults=model_opt['unet']['channel_multiplier'],    #[1, 2, 4, 8,8]
            attn_res=model_opt['unet']['attn_res'],           #[16]
            res_blocks=model_opt['unet']['res_blocks'],   #2
            dropout=model_opt['unet']['dropout'],              #0.2
            image_size=model_opt['diffusion']['image_size']     #256
        )
    netG = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train']
    )
    # if opt['phase'] == 'train':
    #     # init_weights(netG, init_type='kaiming', scale=0.1)
    #     init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG   #这里是一个整体的模型包含unet和diffusion
