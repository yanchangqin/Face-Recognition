import torch
import torch.nn.functional as F
import torch.nn as nn

class UpsampleLayer(torch.nn.Module):

    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)


class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.sub_module(x)


class DownsamplingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


class MainNet(torch.nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()

        self.layer1 = torch.nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            ConvolutionalLayer(32, 64, 3, 2, 1),

            ResidualLayer(64),
            DownsamplingLayer(64, 128),

            ResidualLayer(128),
            ResidualLayer(128),
            DownsamplingLayer(128, 256),

            ResidualLayer(256),
            ResidualLayer(256),

            DownsamplingLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),

            DownsamplingLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),

        )
        self.layer2 =nn.Sequential(
            nn.Linear(1024*3*3,512),
            nn.BatchNorm1d(512),

            nn.Linear(512,256)
        )
        self.layer3 =nn.Sequential(
            nn.Linear(256,5)
        )

    def forward(self, x):
        y1 = self.layer1(x)
        y1 = y1.view(-1, 1024 * 3 * 3)
        feature = self.layer2(y1)
        # print(feature.size())
        output = self.layer3(feature)
        return feature,F.log_softmax(output)

