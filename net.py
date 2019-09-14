import torch
import torch.nn as nn
import torch.nn.functional as f

class P_Net(nn.Module):
    def __init__(self):
        super(P_Net, self).__init__()
        #x--->12*12
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3), #10*10*10
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),#步长2   #5*5*10

            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=3),##3*3*16
            nn.PReLU(),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),#1*1*32
            nn.PReLU(),

            # nn.Conv2d(in_channels=32,out_channels=5,kernel_size=1),#1*1*5
        )
        self.layer2_1 = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride=1)
        self.layer2_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)
    def forward(self, x):
        y = self.layer1(x)
        conf = torch.sigmoid(self.layer2_1(y))
        offset = self.layer2_2(y)

        return conf,offset

class R_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3),#22*22*28
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),#11*11*28

            nn.Conv2d(in_channels=28,out_channels=48,kernel_size=3),#10*10*48
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),#4*4*48

            nn.Conv2d(in_channels=48,out_channels=64,kernel_size=2),#3*3*64
            nn.PReLU(),

            # nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),#1*1*128
            # nn.PReLU(),
        )
        self.layer2_2 = nn.Sequential(
            nn.Linear(3*3*64,128),
            nn.PReLU(),
        )
        self.layer3_1 = nn.Linear(in_features=128,out_features=1)
        self.layer3_2 = nn.Linear(in_features=128,out_features=4)
    def forward(self, x):
        fcn1 = self.layer2(x)
        fcn1 = fcn1.view(fcn1.size(0),-1)
        fcn1 = self.layer2_2(fcn1)
        conf = torch.sigmoid(self.layer3_1(fcn1))
        # conf = f.sigmoid(self.layer3_1(fcn1))
        offset = self.layer3_2(fcn1)

        return conf,offset

class O_Net(nn.Module):
    def __init__(self):
        super().__init__()
        #x--->48*48
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3),#46*46*32
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),#23*23*32

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),#21*21*64
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),#10*10*64

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3),#8*8*64
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),# 4*4*64

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),  # 3*3*128
            nn.PReLU(),

            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),  # 1*1*256
            # nn.ReLU(),
        )
        self.layer4_1 = nn.Sequential(
            nn.Linear(in_features=3*3*128, out_features=256),
            nn.PReLU(),
        )
        self.layer4_2 = nn.Linear(256,1)
        self.layer4_3 = nn.Linear(256,4)
    def forward(self, x):
        fcn1 = self.layer4(x)
        fcn1 = fcn1.view(fcn1.size(0),-1)
        fcn1= self.layer4_1(fcn1)
        conf = torch.sigmoid(self.layer4_2(fcn1))
        offset = self.layer4_3(fcn1)
        return conf,offset