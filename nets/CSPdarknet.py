import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MaxPool2d
from nets.attention import cbam_block, eca_block, se_block
attention_block = [se_block, cbam_block, eca_block]
# from nets.encoder import Attention
#################################################################################################3
def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features, dim=1)

        return features

class mppooling(nn.Module):
    def __init__(self,in_channels, out_channels,):
        super(mppooling, self).__init__()
        self.mppooling = nn.Sequential(
            # conv2d(in_channels, in_channels, 1),
            # SpatialPyramidPooling(), # channel*3
            # conv2d(3*in_channels,3*in_channels,3,stride=2),
            # nn.ConvTranspose2d(3*in_channels, 3*in_channels, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            # conv2d(3*in_channels,2*in_channels,1),
            # conv2d(2*in_channels,2*in_channels,3),
            # conv2d(2*in_channels,out_channels,1),

            conv2d(in_channels, in_channels, 1),
            conv2d(in_channels, in_channels, 3),
            MaxPool2d(kernel_size=3, stride=1, ceil_mode=False, padding=1),
            conv2d(in_channels, out_channels, 1),
            # nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        )
    def forward(self,x):
        x = self.mppooling(x)
        return x
class catcbam(nn.Module):
    def __init__(self,in_channels, out_channels,):
        super(catcbam, self).__init__()
        self.phi = 2
        if 1 <= self.phi and self.phi <= 3:
            self.x_att      = attention_block[self.phi - 1](in_channels)

        # self.conv2dd = conv2d(3*in_channels,in_channels,3)
        # self.conv2d0 = conv2d(2*in_channels,in_channels,1)
        # self.conv2d1 = conv2d(in_channels,out_channels,1)
        self.conv2d2 = conv2d(2*out_channels,2*out_channels,13)
        self.conv2d3 = conv2d(2*out_channels,out_channels,1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1,ceil_mode=False,padding=1)
        # self.conv2d4 = conv2d(2*out_channels,out_channels,3)
        # self.conv2d5 = conv2d(out_channels,out_channels,1)
    def forward(self,x,x_1):
        # x_1 = self.maxpool(x_1)
        # x = self.conv2dd(x)
        # a = torch.cat([x,x_1],axis=1)
        # a = self.conv2d0(a)
        a = self.x_att(x_1)
        # a = self.conv2d1(a)
        a = torch.cat([x, a], axis=1)
        a = self.conv2d2(a)
        a = self.avgpool(a)
        a = self.conv2d3(a)
        # k = torch.cat([x,a],axis=1)
        # k = self.conv2d4(k)
        # k = self.conv2d5(k)
        k = x + a
        return k
##################################################################################################

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        #----------------------------------------------------------------#
        #   利用一个步长为2x2的卷积块进行高和宽的压缩
        #----------------------------------------------------------------#
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:

            self.split_conv0 = BasicConv(out_channels, out_channels, 1)


            self.split_conv1 = BasicConv(out_channels, out_channels, 1)  
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )

            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:

            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)


            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )

            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)


        x = torch.cat([x1, x0], dim=1)

        x = self.concat_conv(x)

        return x


class CSPDarkNet(nn.Module):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            # 416,416,32 -> 208,208,64
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            # 208,208,64 -> 104,104,128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            # 104,104,128 -> 52,52,256
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            # 52,52,256 -> 26,26,512
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            # 26,26,512 -> 13,13,1024
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.num_features = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #############################################################################
        self.conv1_1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.stages_1 = nn.ModuleList([
            # 416,416,32 -> 208,208,64
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            # 208,208,64 -> 104,104,128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            # # 104,104,128 -> 52,52,256
            # Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            # # 52,52,256 -> 26,26,512
            # Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            # # 26,26,512 -> 13,13,1024
            # Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.mppooling0 = mppooling(64,64)
        self.mppooling1 = mppooling(128, 128)
        # self.mppooling2 = mppooling(256, 256)
        # self.mppooling3 = mppooling(512, 512)

        self.catcbam0 = catcbam(64,64)
        self.catcbam1 = catcbam(128,128)
        # self.catcbam2 = catcbam(256,256)
        # self.catcbam3 = catcbam(512,512)

        # self.transattention = Attention(dim = 64)
        ############################################################################


    def forward(self, x,x_1):
        x = self.conv1(x)
        x_1 = self.conv1_1(x_1)

        x = self.stages[0](x)
        x_1 = self.stages_1[0](x_1)

        # k = self.transattention(x,x_1)

        xm1_1 = self.mppooling0(x_1)
        x = self.catcbam0(x,xm1_1)

        x = self.stages[1](x)
        x_1 = self.stages_1[1](x_1)

        xm2_1 = self.mppooling1(x_1)
        x = self.catcbam1(x,xm2_1)

        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5
    
def darknet53(pretrained):
    model = CSPDarkNet([1, 2, 8, 8, 4])
    if pretrained:
        model.load_state_dict(torch.load("model_data/CSPdarknet53_backbone_weights.pth"))
    return model
