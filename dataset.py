import math
import torch
import PIL.ImageDraw as draw
import os
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
from torch.utils.data import Dataset


path = r'F:\ycq\centerloss\face'
# label_path = r'F:\yolo\smaple\small_much\label.txt'

change_tensor = transforms.Compose([
    transforms.ToTensor()
]
)

class MyDataset(Dataset):

    def __init__(self):

        self.box = []
        for _dir in os.listdir(path):
            img_path = os.path.join(path, _dir)
            for name in os.listdir(img_path):
                label = _dir
                self.box.append(label)
        # print(self.box)

    def __len__(self):
        # print(len(self.box))
        return len(self.box)


    def __getitem__(self, index):
        label_data = int(self.box[index])
        # print(label_data)
        img_path =os.path.join(path,str(label_data))
        # print(img_path)
        img_box =[]
        for name in os.listdir(img_path):
            img_box.append(name)
        # print(img_box)
        picture = img_box[index % 21]
        # print(picture)
        im = image.open(os.path.join(img_path, picture))
        im = im.convert('RGB')
        # print()
        img = convert_to_squre(im)
        array = np.array(img)
        array = change_tensor(array)
        # print(label_)
        # array = torch.Tensor([array])
        # print('000')
        return array,label_data


def convert_to_squre(im):
    w,h = im.size
    side = np.maximum(w,h)
    ig =image.new('RGB',(96,96),(128,128,128))
    # ig.show()
    a = int(96 * w / h)
    b1 = int((96 - a) / 2)
    img = im.resize((a, 96))
    ig.paste(img, (0,0))
    return ig
# mydataset = MyDataset()
# mydataset.__getitem__(25)
# mydataset.__len__()
# print()