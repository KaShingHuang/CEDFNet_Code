'''
这个py文件是用来生成颜色均衡之后的图像的
'''




import numpy as np
import os,sys
import argparse
from tqdm import tqdm
import torch

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))


from dataset.dataset_denoise import *
import utils
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='Image denoising evaluation on SIDD')
parser.add_argument('--input_dir', default='/data2/CarnegieBin_data/Kashing/EnhanceUf/moretestpatch/input_0',
    type=str, help='Directory of validation images')   #验证输入数据的目录，不确定对不对
parser.add_argument('--result_dir', default='/data2/CarnegieBin_data/Kashing/EnhanceUf/moretestpatch/input',
    type=str, help='Directory for results')   #验证结果的目录
parser.add_argument('--weights', default='/data2/CarnegieBin_data/Kashing/Uf/model_best.pth',
    type=str, help='Path to weights')     #训练好的模型权重的目录
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')  #gpu
parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')   #模型名字
parser.add_argument('--batch_size', default=48, type=int, help='Batch size for dataloader')  #batch_size
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory') #是否保存图片
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus



# if args.save_images:
result_dir_img = "/data2/CarnegieBin_data/Kashing/EnhanceUf"    #保存增强图像的目录



model_restoration= utils.get_arch(args)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

noisy_files = natsorted(glob(os.path.join("/data2/CarnegieBin_data/Kashing/dataset/U45/input", '*.png')))           #要处理的图像存在的目录

for file in tqdm(noisy_files):             #file.split("/")[-1]
    FileName= file.split("/")[-1]
    input_array = torch.from_numpy(np.array(Image.open(file))/255.).permute(2,0,1)
    input_array=input_array.unsqueeze(0).float().cuda()
    print(input_array.shape)
    restoration_result=model_restoration(input_array)
    restoration_result=torch.clamp(restoration_result,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
    # 保存图像
    save_file = os.path.join(result_dir_img, "{}".format(FileName))
    utils.save_img(save_file, img_as_ubyte(restoration_result))












