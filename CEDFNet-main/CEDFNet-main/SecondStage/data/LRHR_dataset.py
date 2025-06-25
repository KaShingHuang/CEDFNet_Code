from PIL import Image
from torch.utils.data import Dataset
import data.util as Util



class LRHRDataset2(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1,
                 need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = -1
        self.need_LR = need_LR
        self.split = split

        if datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/{}'.format(dataroot, "input"))
            self.hr_path = Util.get_paths_from_images(
                '{}/{}'.format(dataroot, "groundtruth"))
            self.Trans_path = Util.get_paths_from_images(
                '{}/{}'.format(dataroot, "Transmission"))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        img_HR = Image.open(self.hr_path[index]).convert("RGB")
        img_SR = Image.open(self.sr_path[index]).convert("RGB")
        img_Trans = Image.open(self.Trans_path[index]).convert("RGB")
        img_style = Image.open(self.sr_path[index]).convert("RGB")
        if self.need_LR:
            img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR, img_style] = Util.transform_augment(
                [img_LR, img_SR, img_HR, img_style], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'style': img_style, 'Index': index}
        else:
            [img_SR, img_HR, img_Trans, img_style] = Util.transform_augment(
                [img_SR, img_HR, img_Trans, img_style], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Trans': img_Trans, 'style': img_style, 'Index': index}   #,self.hr_path[index].split('/')[-1].split('.')[0]

