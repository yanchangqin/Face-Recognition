from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image

class DATA(Dataset):
    def __init__(self,path):
        self.path = path
        self.box = []
        self.box.extend(open(os.path.join(path,'positive.txt')).readlines())
        self.box.extend(open(os.path.join(path, 'negative.txt')).readlines())
        self.box.extend(open(os.path.join(path, 'part.txt')).readlines())

    def __len__(self):
        return len(self.box)

    def __getitem__(self, index):
        str = self.box[index].split()
        # print(str)
        img_path = os.path.join(self.path,str[0])
        cond =torch.Tensor([int(str[1])])
        offset = torch.Tensor([float(str[2]),float(str[3]),float(str[4]),float(str[4])])
        img_data = torch.Tensor(np.array(Image.open(img_path)) / 255.-0.5)
        return img_data,cond,offset


#
# dat = DATA(r'F:\MTCNN\test2\celeba\12')
# dat.__getitem__(45)
