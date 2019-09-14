import torch
import torch.nn as nn
import torch.nn.functional as F

class Arc_Loss(nn.Module):
    def __init__(self,feature_num,cls_num,m =0.1,s=5):
        super().__init__()
        self.s = s
        self.m = m
        self.W=nn.Parameter(torch.randn(feature_num,cls_num))
        # print('vvvv',self.W)

    def forward(self, feature):
        _w = F.normalize(self.W,dim=0)
        # print(_w.size())
        _x = F.normalize(feature,dim=1)
        # print(_x.size())
        cosa = torch.matmul(_x,_w)
        # # print(cosa.size())
        a = torch.acos(cosa)
        # print(torch.mean(cosa))

        top = torch.exp(torch.cos(a+self.m)*self.s)
        _top = torch.exp(torch.cos(a)*self.s)
        bottom = torch.sum(_top,dim=1,keepdim=True)

        # sina = torch.sqrt(1-torch.pow(cosa,2))
        # cosm = torch.cos(torch.tensor(self.m)).cuda()
        # sinm = torch.cos(torch.tensor(self.m)).cuda()
        # cosa_m =cosa*cosm-sina*sinm
        # top =torch.exp(cosa_m*self.s)
        # _top =torch.exp(cosa*self.s)
        # bottom =torch.sum(_top,dim=1,keepdim=True)

        return (top / (bottom - _top + top))+1e-10
# arc = Arc_Loss(256,5)



