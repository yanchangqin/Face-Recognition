import os
from dataset import convert_to_squre
import PIL.Image as image
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from net import MainNet

save_path = r'F:\ycq\arcloss\face_arc\2000'
# save_path_validata = r'F:\ycq\centerloss\face_validate'
save_path_validata = r'.\face_validate'
# save_path_test =r'F:\ycq\centerloss\face_test'
save_path_test =r'.\face_test'
save_param = r'param_net.pt'


change_tensor = transforms.Compose([
    transforms.ToTensor()
]
)
use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
net = MainNet().to(device)
net.load_state_dict(torch.load(os.path.join(save_path,save_param)))
def detector():
    net.eval()
    total_target = []
    total_feature = []
    features_test = []
    for name in os.listdir(save_path_validata):
        img = image.open(os.path.join(save_path_validata, name))
        im = img.convert('RGB')
        img = convert_to_squre(im)
        array = np.array(img)
        array = change_tensor(array)
        array = array.unsqueeze(0)
        array=array.to(device)
        label =torch.Tensor([int(name[0])])
        label = label.to(device)
        features,output = net(array)
        total_feature.append(features)
        total_target.append(label)

    total_feature = torch.cat(total_feature, dim=0)
    # print(total_feature.size())
    total_target = torch.cat(total_target)

    for name in os.listdir(save_path_test):
        img = image.open(os.path.join(save_path_test,name))
        # img.show()
        im = img.convert('RGB')
        img = convert_to_squre(im)
        array = np.array(img)
        array = change_tensor(array)
        array =array.unsqueeze(0)
        array = array.to(device)
        features, output = net(array)
        features_test.append(features)
    features_test = torch.cat(features_test,dim=0)
    total_feature = F.normalize(total_feature)
    features_test = F.normalize(features_test).t()

    cosa = torch.matmul(total_feature,features_test)
    print(cosa)
    list_name = ['刘涛','殷桃','吴亦凡','黄晓明']
    mask = torch.gt(cosa,0.7)
    feature_finally = cosa[mask]

    if feature_finally.size(0) ==0:
        print('谁都不是！')
        exit()
    feature_finally = feature_finally[torch.argmax(feature_finally)]
    p_mask = cosa==feature_finally
    p_mask = p_mask.long()
    num = p_mask.argmax().cpu().numpy()
    person = list_name[num]

    print('刘涛====>',cosa[0].item())
    print('殷桃====>', cosa[1].item())
    print('吴亦凡====>', cosa[2].item())
    print('黄晓明====>', cosa[3].item())
    print('这是：',person)
    # target = total_target[mask]
    # person = list_name[max(target)]
    # print(person)
    # # print(cosa,total_target)
    # print(cosa)
    # print(target)
detector()