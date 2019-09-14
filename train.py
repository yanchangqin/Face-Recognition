import torch
import torch.nn as nn
import os
import PIL.Image as image
from net import MainNet
from LOSS import CenterLoss
# from LOSS_1 import CenterLoss
# from CenterLoss import CenterLoss
from ArcLoss import Arc_Loss
import torch.optim as optim
import numpy  as np
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as F
from dataset import MyDataset,convert_to_squre


# save_path = r'F:\ycq\centerloss\face_all'
save_path =r'F:\ycq\arcloss\face_arc'
save_path_test = r'F:\ycq\centerloss\face_test'
save_param = r'param_net.pt'


change_tensor = transforms.Compose([
    transforms.ToTensor()
]
)
use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
# net = Net().to(device)
net = MainNet().to(device)
nllloss = nn.NLLLoss().to(device)
centerloss = CenterLoss(5,256).to(device)
arcloss = Arc_Loss(256,5).to(device)
net.load_state_dict(torch.load(os.path.join(save_path,save_param)))
optimizer1 = optim.Adam(net.parameters())
optimizer2 = optim.Adam(centerloss.parameters())
mydata = MyDataset()
dataloder = data.DataLoader(dataset=mydata,batch_size=10,shuffle=True)
# optimizer4nn = optim.SGD(net.parameters(),lr=0.00001,momentum=0.9, weight_decay=0.0005)
# sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)


def trainer(epoch):
    print("training Epoch: {}".format(epoch))
    net.train()

    for j,(array,label) in enumerate(dataloder):
        # print(label)
        # print(label)
        array=array.to(device)
        label = label.to(device)
        features,output = net(array)
        # print(output)
        # print(label)
        # print(features.size())

        # loss1 =centerloss(features,label)
        # print('loss1',loss1.item())
        # print('out',out.size())
        # loss = F.cross_entropy(out,y.cpu())
        out = arcloss(features)
        # print('loss2',loss2.item())
        out = torch.log(out)
        loss2 = nllloss(out,label)
        # loss = loss1+loss2
        loss =  loss2

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        loss.backward()

        optimizer1.step()
        optimizer2.step()
        print('epoch', epoch, 'loss:', loss.item())
        if epoch%100==0:
            if not os.path.exists(os.path.join(save_path, str(epoch))):
                os.makedirs(os.path.join(save_path, str(epoch)))
            torch.save(net.state_dict(), os.path.join(save_path, str(epoch), save_param))


for epoch in range(2000):
    trainer(epoch + 1)

