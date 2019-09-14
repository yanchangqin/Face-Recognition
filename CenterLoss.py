#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 11:13
# @Author  : YangZhenHui
# @File    : CenterLoss.py


import torch
import torch.nn as nn
from torch.autograd.function import Function

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim

        self.size_average = size_average

    def forward(self, feat,label ):
        batch_size = feat.size(0)

        feat = feat.view(batch_size, -1)
        # print(feat.size(1))
        # print(self.feat_dim)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        # print('loss',loss)
        return loss


class CenterlossFunc(Function):
    @staticmethod#无需实例化也可以调用该方法，当然也可以实例化以后调用
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        #根据label选择中心点，统计次数
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        # print('grad_output',grad_output)
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        # print('counts',counts)
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())
        # print(grad_centers)

        # print('label',label)
        #根据标签统计了各个分类的数量+1
        counts = counts.scatter_add_(0, label.long(), ones)
        # print(label.unsqueeze(1).expand(feature.size()))
        # print('diff',diff)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        # print(grad_centers)
        grad_centers = grad_centers/counts.view(-1, 1)
        # print(- grad_output * diff / batch_size, None, grad_centers / batch_size, None)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


def main(test_cuda=False):
    print('-'*80)
    device = torch.device("cuda" if test_cuda else "cpu")
    ct = CenterLoss(10,2,size_average=True).to(device)
    y = torch.Tensor([0,0,2,1,4,4,8,7,3,5]).to(device)
    feat = torch.zeros(10,2).to(device).requires_grad_()
    # print (list(ct.parameters()))
    # print (ct.centers.grad)
    out = ct(y,feat)
    # print(out.item())
    out.backward()
    # print(ct.centers.grad)
    # print(feat.grad)

if __name__ == '__main__':
    #manual_seed,每次得到的随机数都相同
    torch.manual_seed(999)
    main(test_cuda=False)
    if torch.cuda.is_available():
        main(test_cuda=True)

