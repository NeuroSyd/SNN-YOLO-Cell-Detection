import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import torch.optim as optim

spike_grad1 = surrogate.fast_sigmoid(slope=25)
IN_CHANNELS = 2
TIMESTEPS = 4
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

    
class CNNBlockV3(nn.Module):
    def __init__(self,in_channels,out_channels,bn_act=True,**kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=not bn_act,**kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    def forward(self,x):
        if self.use_bn_act:
            return self.leakyrelu(self.batchnorm(self.conv1(x)))
        else:
            return self.conv1(x)
        
class TCNNBlockV3(nn.Module):
    def __init__(self,in_channels,out_channels,time_steps,bn_act=True,**kwargs) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=not bn_act,**kwargs)
        self.convf = nn.Conv2d(in_channels+out_channels, in_channels, bias=not bn_act,**kwargs)
        self.BN = snn.BatchNormTT2d(out_channels,time_steps=time_steps)
    def forward(self,x):
        x = x.transpose(0,1)
        timerange = x.shape[0]
        spk_rec = []
        for steps in range(timerange):
            x0 = x[steps]
            if steps == 0:
                x0 = self.conv1(x0)
            else:
                x0 = self.conv1(self.convf(torch.concat([x0,spk_rec[-1]],dim=1)))
            spk_rec.append(x0)
            
        x = torch.stack(spk_rec)
        spk_rec = []
        for idx,bn in enumerate(self.BN):
            spk_rec.append(bn(x[idx]))
        x = torch.stack(spk_rec)
        x = x.transpose(0,1)
        return x



class LCB(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride ,time_steps,padding,mem_output = False):
        super().__init__()
        self.time_steps = time_steps
        self.B1 = torch.rand(1)
        self.TH = torch.rand(1)
        self.LIF = snn.Leaky(beta=self.B1,threshold=self.TH,learn_beta=True,learn_threshold=True)#spike_grad=spike_grad1
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.convM = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.BN = snn.BatchNormTT2d(out_channels,time_steps=time_steps)
        self.mem_output = mem_output
    def forward(self,x):
        x = x.transpose(0,1)
        timerange = x.shape[0]
        spk_rec = []
        mem_rec = []
        mem = self.LIF.init_leaky()
        for steps in range(timerange):
            x0 = x[steps]
            x0,mem = self.LIF(x0,mem)
            mo = self.convM(mem)
            x0 = self.conv(x0)
            spk_rec.append(x0)
            mem_rec.append(mo)
        x = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
        spk_rec = []
        for idx,bn in enumerate(self.BN):
            spk_rec.append(bn(x[idx]))
        x = torch.stack(spk_rec)
        x = x.transpose(0,1)
        if self.mem_output:
            return x,mem_rec
        else:
            return x
class Tmaxpool(nn.Module):
    def __init__(self,kernel_size,stride,padding) -> None:
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding)
    def forward(self,x):
        x = x.transpose(0,1)
        timerange = x.shape[0]
        rec = []
        for step in range(timerange):
            rec.append(self.mp(x[step]))
        x = torch.stack(rec)
        x = x.transpose(0,1)
        return x

class Tupscale(nn.Module):
    def __init__(self,factor):
        super().__init__()
        self.up = nn.Upsample(scale_factor=factor)
    def forward(self,x):
        x = x.transpose(0,1)
        timerange = x.shape[0]
        rec = []
        for step in range(timerange):
            rec.append(self.up(x[step]))
        x = torch.stack(rec)
        x = x.transpose(0,1)
        return x

class MS(nn.Module):
    def __init__(self,in_channels,kernel_size,stride ,time_steps,padding):
        super().__init__()
        self.time_steps = time_steps
        self.LCB1 = LCB(in_channels=in_channels,out_channels=in_channels//2,kernel_size=3,stride=stride,time_steps=time_steps,padding=1)
        self.LCB2 = LCB(in_channels=in_channels//2,out_channels=in_channels,kernel_size=3,stride=stride,time_steps=time_steps,padding=1)
    def forward(self,x):
        xo = x
        x = self.LCB1(x)
        x = self.LCB2(x)
        x += xo
        return x


class SNNBlockV3M1(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,time_steps):
        super().__init__()
        assert in_channels >= out_channels
        self.L = nn.ModuleList(
            [
                LCB(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,time_steps=time_steps,padding=padding),
                LCB(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,time_steps=time_steps,padding=1),
            ]
        )
        self.R = nn.ModuleList(
            [
                Tmaxpool(kernel_size=kernel_size,stride=stride,padding=padding),
                LCB(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,time_steps=time_steps,padding=1)
            ]
        )
        self.Ms = MS(in_channels=out_channels,kernel_size=kernel_size,stride=1,time_steps=time_steps,padding=padding)
    def forward(self,x):
        xL = x
        xR = x
        for layer in self.L:
            xL = layer(xL)
        for layer in self.R:
            xR = layer(xR)
        x = xL+xR
        x = self.Ms(x)
        return x

class SNNBlockV3M2(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,time_steps):
        super().__init__()
        assert in_channels<out_channels
        R_C = out_channels - in_channels
        self.L = nn.ModuleList(
            [
                LCB(in_channels=in_channels,out_channels=out_channels,
                    kernel_size=kernel_size,stride=stride,
                    time_steps=time_steps,padding=padding),
                LCB(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,time_steps=time_steps,padding=padding),
            ]
        )
        self.mp = Tmaxpool(kernel_size=kernel_size,stride=stride,padding=padding)
        self.RLCB = LCB(in_channels=in_channels,out_channels=R_C,kernel_size=kernel_size,stride=1,time_steps=time_steps,padding=1)
        self.Ms = MS(in_channels=out_channels,kernel_size=kernel_size,stride=1,time_steps=time_steps,padding=padding)
    def forward(self,x):
        Xl = x
        Xr = x
        for layer in self.L:
            Xl = layer(Xl)
        Xr = self.mp(Xr)
        Xr0 = Xr
        Xr0 = self.RLCB(Xr0)#batch,time,c,w,h
        #print(Xr.shape)
        #print(Xr0.shape)
        Xr = torch.concat([Xr,Xr0],dim=2)#b,t,c,w,h
        x = Xl + Xr
        x = self.Ms(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self,channels, use_residual=True,num_repeats = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlockV3(in_channels=channels,out_channels=channels//2,kernel_size=1),
                    CNNBlockV3(channels//2,channels,kernel_size = 3,padding = 1)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats
    def forward(self,x):
        for layer in self.layers:
            if self.use_residual:
                x = x+layer(x)
            else:
                x = layer(x)
        return x
class TResidualBlock(nn.Module):
    def __init__(self,channels, use_residual=True,num_repeats = 1,time_steps=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    TCNNBlockV3(in_channels=channels,out_channels=channels//2,kernel_size=1,time_steps=time_steps),
                    TCNNBlockV3(channels//2,channels,kernel_size = 3,padding = 1,time_steps=time_steps)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats
    def forward(self,x):
        for layer in self.layers:
            if self.use_residual:
                x = x+layer(x)
            else:
                x = layer(x)
        return x
class SnnResidualBlock(nn.Module):
    def __init__(self,channels, time_steps,use_residual=True,num_repeats = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    LCB(in_channels=channels,out_channels=channels//2,kernel_size=1,stride=1,time_steps=time_steps,padding=0),
                    LCB(channels//2,channels,kernel_size = 3,padding = 1,stride=1,time_steps=time_steps)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats
    def forward(self,x):
        for layer in self.layers:
            if self.use_residual:
                x = x+layer(x)
            else:
                x = layer(x)
        return x

class SnnResidualBlockM(nn.Module):
    def __init__(self,channels, time_steps,use_residual=True,num_repeats = 1,men_true=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    LCB(in_channels=channels,out_channels=channels//2,kernel_size=1,stride=1,time_steps=time_steps,padding=0),
                    LCB(channels//2,channels,kernel_size = 3,padding = 1,stride=1,time_steps=time_steps,mem_output=True)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats
        self.men_true = men_true
    def forward(self,x):
        men_rec=[]
        for layer in self.layers:
            if self.use_residual:
                if self.men_true:
                    xt,men = layer(x)
                    x = x +xt
                    men_rec.append(men)
                else:
                    xt,_ = layer(x)
                    x = x +xt
                
            else:
                x,_ = layer(x)
        return x,men_rec.pop()

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlockV3(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlockV3(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
class SnnScalePrediction(nn.Module):
    def __init__(self,in_channels,numberC,time_steps) -> None:
        super().__init__()
        self.C = numberC
        self.LCB_out = LCB(in_channels=in_channels,out_channels = in_channels * 2,kernel_size=1,stride=1,time_steps=time_steps,padding=0,mem_output=True)
        self.outlay = CNNBlockV3(
                4 * in_channels, (numberC + 5) * 3, bn_act=False, kernel_size=1
            )
    def forward(self,x):
        spk, mem = self.LCB_out(x)
        spk = spk.transpose(0,1) # t,b,c,h,w ==> sum ==>bchw
        x = torch.concat([spk[-1],mem[-1]],dim=1)
        x = self.outlay(x)
        return (
            x.reshape(x.shape[0], 3, self.C + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
class adpMP(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.B1 = torch.rand(1)
        self.Th = torch.rand(1)
        self.LIF = snn.Leaky(beta=self.B1,threshold=self.Th,learn_beta=True,learn_threshold=True)#spike_grad=spike_grad1
        self.con1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
        self.Bn = nn.BatchNorm2d(in_channels)
    def forward(self,x):
        mem = self.LIF.init_leaky()
        spk_rec = []
        mem_rec = []
        x = x.transpose(0,1)
        #print(x.shape)
        for step in range(x.shape[0]):
            spk=self.Bn(self.con1(x[step]))
            spk,mem = self.LIF(spk,mem)

            spk_rec.append(spk)
            mem_rec.append(spk)
        return mem
        
class SnnScalePredictionM(nn.Module):
    def __init__(self,in_channels,numberC,time_steps) -> None:
        super().__init__()
        self.C = numberC
        self.LCB_out = LCB(in_channels=in_channels,out_channels = in_channels * 2,kernel_size=1,stride=1,time_steps=time_steps,padding=0,mem_output=True)
        self.enh = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
        self.outlayC = CNNBlockV3(
                4 * in_channels, (numberC + 5) * 3, bn_act=False, kernel_size=3,padding=1
            )
        self.outlayR = CNNBlockV3(
                3*in_channels, (5+numberC) * 3, bn_act=True, kernel_size=3,padding = 1
            )
        self.final = nn.Conv2d(in_channels=2*(5+numberC) * 3,out_channels=(5+numberC) * 3,kernel_size=1)
    def forward(self,x,m):
        spk, mem = self.LCB_out(x)
        spk = spk.transpose(0,1) # t,b,c,h,w ==> sum ==>bchw
        #spk = spk.sum(0)
        #print("m")
        #print(m.shape)
        #print("mem")
        #print(mem[-1].shape)
        #print("spk")
        #print(spk.sum(0).shape)
        x = self.outlayC(torch.concat([spk[-1],mem[-1]],dim=1))
        mem = torch.concat([self.enh(m), mem[-1]],dim=1)
        mem = self.outlayR(mem)
        
        x = torch.concat([x,mem],dim=1)
        x = self.final(x)
        return (
            x.reshape(x.shape[0], 3, self.C + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=3,config_set = config):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.config_set = config_set
        self.layers = self._create_conv_layers()
        

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in self.config_set:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlockV3(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlockV3(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers
        
class SnnYoloV3WM(nn.Module):
    def __init__(self,in_channels,numberC,time_steps,rn=8):
        super().__init__()
        self.numberC = numberC
        self.layerc = rn
        self.layer0 = TCNNBlockV3(in_channels=in_channels,out_channels=16,time_steps=time_steps,bn_act=False,kernel_size = 3,padding = 1)
        self.layer1 = SNNBlockV3M2(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1,time_steps=time_steps) # 2
        self.layer2 = SNNBlockV3M2(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1,time_steps=time_steps)# 4
        self.Res1 = TResidualBlock(channels=64,num_repeats=1,use_residual=True,time_steps=time_steps)
        self.layer3 = SNNBlockV3M2(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1,time_steps=time_steps)# 8
        self.Res2 = TResidualBlock(channels=128,num_repeats=round(rn/2),use_residual=True,time_steps=time_steps)
        self.mp1 = adpMP(128)
        self.layer4 = SNNBlockV3M2(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1,time_steps=time_steps)# 16
        self.Res3 = TResidualBlock(channels=256,num_repeats=rn,use_residual=True,time_steps=time_steps)
        self.mp2 = adpMP(256)
        self.layer5 = SNNBlockV3M2(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1,time_steps=time_steps)# 32
        #self.layer5_3 = SNNBlockV3M2(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1,time_steps=time_steps)# 32
        self.Res4 = TResidualBlock(channels=512,num_repeats=rn,use_residual=True,time_steps=time_steps)
        self.mp3 = adpMP(512)
        #self.layer5_4 = SNNBlockV3M1(in_channels=1024,out_channels=512,kernel_size=3,stride=1,padding=1,time_steps=time_steps)
        self.out1 = SnnScalePredictionM(in_channels=512,numberC=numberC,time_steps=time_steps)
        self.layer5_5 = SNNBlockV3M1(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0,time_steps=time_steps) #32
        self.ups1 = Tupscale(factor=2)# 16
        self.layer6 = SNNBlockV3M1(in_channels=256,out_channels=256,kernel_size=1,stride=1,padding=0,time_steps=time_steps)# 16
        # cat
        self.layer7 = SNNBlockV3M1(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0,time_steps=time_steps)# 16
        self.out2 = SnnScalePredictionM(in_channels=256,numberC=numberC,time_steps=time_steps) # 16 128
        self.layer7_5 = SNNBlockV3M1(in_channels=256,out_channels=128,kernel_size=1,stride=1,padding=0,time_steps=time_steps)
        self.ups2 = Tupscale(factor=2) # 8 128
        self.layer8 = SNNBlockV3M1(in_channels=256,out_channels=128,kernel_size=1,stride=1,padding=0,time_steps=time_steps)
        self.tool16_8 = nn.Upsample(scale_factor=4)
        self.fit1 = nn.Conv2d(512,128,kernel_size =3,padding=1)
        self.tool32_8 = nn.Upsample(scale_factor=2)
        self.fit2 = nn.Conv2d(256,128,kernel_size =3,padding=1)
        self.fit3 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size =3,padding=1),
            nn.BatchNorm2d(128))
        #cat
        #self.layer9 = SNNBlockV3M1(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,time_steps=time_steps)
        self.out3 = SnnScalePredictionM(in_channels=128,numberC=numberC,time_steps=time_steps)
    def forward(self,x):
        output = []
        res_rec = []
        men_rec = []

        #print("up1")
        x = self.layer0(x)
        #print("up1")
        x = self.layer1(x)# 32
        #print("up1")
        x = self.layer2(x)# 64
        #print("up1")
        x = self.Res1(x)
        x = self.layer3(x)# 128
        #print("up1")
        #print(x.shape)
        x = self.Res2(x)
        men = self.mp1(x)
        #print(men.shape)
        #print("up1")
        res_rec.append(x)#128 / 8
        men_rec.append(men)
        x = self.layer4(x)#256/16dasd
        x = self.Res3(x)
        men = self.mp2(x)
        #print(men.shape)
        #print("up1")
        res_rec.append(x) #256 / 16
        men_rec.append(men)
        x = self.layer5(x) # 512 /32
        x = self.Res4(x)
        #x = self.layer5_3(x)
        men = self.mp3(x)
        #print(men.shape)
        men_rec.append(men)
        #x = self.layer5_4(x)
        #print("up1")
        output.append(self.out1(x,men))
        x = self.layer5_5(x)# 256 /32
        x = self.ups1(x) # 256 / 16
        #print("up1")
        #print(x.shape)
        x = self.layer6(x)# 256 / 16
        x = torch.concat([x,res_rec[-1]],dim=2) # 512 /16
        x = self.layer7(x)# 256 /16
        output.append(self.out2(x,men_rec[-2]))
        x = self.layer7_5(x)# 128 /16
        x = self.ups2(x)# 128 /8
        x = torch.concat([x,res_rec[-2]],dim=2)# 256 /8
        x = self.layer8(x)# 128 /8
        #x = self.layer9(x) # 64 /8
        mem1 = self.tool16_8(men_rec[-1])
        
        mem1 = self.fit1(mem1)
        #print(mem1.shape)

        mem2 = self.tool32_8(men_rec[-2])
        
        mem2 = self.fit2(mem2)
        #print(mem2.shape)
        mem3 = self.fit3(mem2+mem1+men_rec[-3])

        output.append(self.out3(x,mem3))
        return output

if __name__ == "__main__":
        num_classes = 3
        IMAGE_SIZE = 416
        IMAGE_SIZEy = round(360 * 0.8888888)
        print(IMAGE_SIZEy)
        IMAGE_SIZEx = int(20*8)
        model = YOLOv3(num_classes=num_classes,config_set=config)
        x = torch.randn((2, 3, IMAGE_SIZEy, IMAGE_SIZEx))
        out = model(x)

        print(out[0].shape)
        print(out[1].shape)
        print(out[2].shape)
        assert model(x)[0].shape == (2, 3, IMAGE_SIZEy//32, IMAGE_SIZEx//32, num_classes + 5)
        assert model(x)[1].shape == (2, 3, IMAGE_SIZEy//16, IMAGE_SIZEx//16, num_classes + 5)
        assert model(x)[2].shape == (2, 3, IMAGE_SIZEy//8, IMAGE_SIZEx//8, num_classes + 5)
        print("Success!")
        
        def test_LCB():
            # Dummy input assuming the input shape is [batch_size, channels, height, width, time_steps]
            batch_size, channels, height, width, time_steps = 4, 3, 320, 32, 10
            dummy_input = torch.randn(batch_size, time_steps, channels,height, width )

            # Initialize the LCB module
            lcb_layer = LCB(in_channels=channels, out_channels=8, kernel_size=3,stride=1,time_steps=time_steps,padding=1)

            # Forward pass through the LCB layer
            output = lcb_layer(dummy_input)

            # Print output shape
            print("LCB Output shape:", output.shape)
        def test_MS():
            # Dummy input assuming the input shape is [batch_size, channels, height, width, time_steps]
            batch_size, channels, height, width, time_steps = 4, 3, 32, 32, 10
            dummy_input = torch.randn(batch_size, time_steps, channels,height, width )

            # Initialize the LCB module
            lcb_layer = MS(in_channels=channels, kernel_size=3,stride=1,time_steps=time_steps,padding=1)

            # Forward pass through the LCB layer
            output = lcb_layer(dummy_input)

            # Print output shape
            print("MS Output shape:", output.shape)
        def test_M1():
            # Dummy input assuming the input shape is [batch_size, channels, height, width, time_steps]
            batch_size, channels, height, width, time_steps = 4, 3, 32, 32, 10
            dummy_input = torch.randn(batch_size, time_steps, channels,height, width )

            # Initialize the LCB module
            lcb_layer = SNNBlockV3M1(in_channels=channels, out_channels=channels,kernel_size=3,stride=1,time_steps=time_steps,padding=1)

            # Forward pass through the LCB layer
            output = lcb_layer(dummy_input)

            # Print output shape
            print("M1 Output shape:", output.shape)
        def test_M2():
            # Dummy input assuming the input shape is [batch_size, channels, height, width, time_steps]
            batch_size, channels, height, width, time_steps = 4, 2, 32, 32, 10
            dummy_input = torch.randn(batch_size, time_steps, channels,height, width )

            # Initialize the LCB module
            lcb_layer = SNNBlockV3M2(in_channels=channels, out_channels=channels*8,kernel_size=3,stride=1,time_steps=time_steps,padding=1)

            # Forward pass through the LCB layer
            output = lcb_layer(dummy_input)

            # Print output shape
            print("M2 Output shape:", output.shape)
        def test_out():
            # Dummy input assuming the input shape is [batch_size, channels, height, width, time_steps]
            batch_size, channels, height, width, time_steps = 4, 2, 32, 32, 10
            dummy_input = torch.randn(batch_size, time_steps, channels,height, width )

            # Initialize the LCB module
            lcb_layer = SnnScalePrediction(in_channels=channels,numberC=5,time_steps=time_steps)

            # Forward pass through the LCB layer
            output = lcb_layer(dummy_input)

            # Print output shape
            print(" Output shape:", output.shape)
        def resB_out():
            # Dummy input assuming the input shape is [batch_size, channels, height, width, time_steps]
            batch_size, channels, height, width, time_steps = 4, 2, 32, 32, 10
            dummy_input = torch.randn(batch_size, time_steps, channels,height, width )

            # Initialize the LCB module
            lcb_layer = SnnResidualBlock(channels=channels,time_steps=time_steps,num_repeats=4)

            # Forward pass through the LCB layer
            output = lcb_layer(dummy_input)

            # Print output shape
            print(" Res Output shape:", output.shape)
        def Net_out():
            # Dummy input assuming the input shape is [batch_size, channels, height, width, time_steps]
            batch_size, channels, height, width, time_steps = 4, 2, 320, 128, 10
            dummy_input = torch.randn(batch_size, time_steps, channels,height, width )
            print("up0")

            # Initialize the LCB module
            lcb_layer = SnnYoloV3WM(in_channels=channels,numberC=5,time_steps=time_steps)

            print("up0")

            # Forward pass through the LCB layer
            output = lcb_layer(dummy_input)

            # Print output shape
            print(" Output 32 shape:", output[0].shape)
            print(" Output 16 shape:", output[1].shape)
            print(" Output 8 shape:", output[2].shape)
        test_LCB()
        test_MS()
        test_M1()
        test_M2()
        resB_out()
        test_out()
        Net_out()
    