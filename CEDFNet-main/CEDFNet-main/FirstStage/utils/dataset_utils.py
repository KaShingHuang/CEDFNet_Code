import torch
import os

### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


### mix two images
class MixUp_AUG:   #混合数据增强，将两个样本按照一定的比例混合，生成新的样本
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))   #初始化一个beta分布

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)   #batchsize
        indices = torch.randperm(bs)    #包含0-bs-1大小的随机排序
        rgb_gt2 = rgb_gt[indices]              #随机打乱顺序之后的groundtruth和nosiyinput
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()  #引入beta噪声

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy
