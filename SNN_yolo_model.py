import torch
import torch.nn as nn
import snntorch as snn
import torch.optim as optim

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class snnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(snnBlock, self).__init__()
        self.A1 = torch.rand(1)
        self.B1 = torch.rand(1)
        self.TH1 = torch.rand(1)
        self.TH2 = torch.rand(1)
        self.men1 = []
        self.men2 = []
        self.ALPHA = snn.Alpha(alpha=self.A1,beta=self.B1,threshold=self.TH1,learn_alpha=True,learn_beta=True,learn_threshold=True)
        self.conv = snn.SConv2dLSTM(in_channels, out_channels,kernel_size=3,threshold=self.TH2,learn_threshold=True)
        self.bn = snn.BatchNormTT2d(out_channels)
    def forward(self, x):
        if len(self.men1)==0:
            men1 = self.ALPHA.init_alpha()
            men2 = self.conv.init_sconv2dlstm()
        else:
            men1 = self.men1[-1]
            men2 = self.men2[-1]
        x, men1 = self.ALPHA(x, men1)
        x, men2 = self.conv(x, men2)
        x = self.bn(x)
        self.men1.append(men1)
        self.men2.append(men2)
        return x

class Yolov1(nn.Module):
    def __init__(self, architecture_config ,in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darkSNNnet = snnBlock(in_channels, 64)
        self.smallconv = nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.smallconv2 = nn.Conv2d(128,30,kernel_size=3,stride=1,padding=1)
        self.ad_mp = nn.AdaptiveAvgPool2d((7,7))
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30*7*7, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 7*7*(3+2*5)),
        )
    def forward(self, x):
        x=x.transpose(0,1)#[steps,batch,channel,h,w]
        for step in range(x[0]):
            x0 = x[step]
            x0 = self.darkSNNnet(x0)
            x0 = self.smallconv(x0)
            x0 = self.smallconv2(x0)
        x = self.ad_mp(x0)#[batch,channel,h,w]
        x = self.fcs(torch.flatten(x, start_dim=1))
        return x