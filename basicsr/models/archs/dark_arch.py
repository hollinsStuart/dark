
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stx
import math


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):        
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img,mean_c], dim=1)
        x_1 = self.conv1(input)
        illu_map = self.conv2(self.depth_conv(x_1))
        return illu_map



##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    # def __init__(self, in_channels, height=3,reduction=8,bias=False):
    def __init__(self, in_channels, height=2,reduction=16,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V        

    
##########################################################################
### --------- Simplified Context Block (SCB) ----------
class SimplifiedContextBlock(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(SimplifiedContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)

    def forward(self, x):
        # Context modeling
        batch, channel, height, width = x.size()
        # Avoid redundant operations by merging view and unsqueeze
        context_mask = self.conv_mask(x).view(batch, 1, height * width)
        context_mask = F.softmax(context_mask, dim=2).view(batch, 1, height, width)

        # Using broadcasting instead of matrix multiplication for efficiency
        context = torch.sum(x * context_mask, dim=[2, 3], keepdim=True)
        return x + context  # Simplified channel addition

##########################################################################
### --------- Simplified Residual Context Block (SRCB) ----------
class SimplifiedRCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(SimplifiedRCB, self).__init__()
        self.body = nn.Conv2d(n_feat, n_feat, kernel_size, stride=1, padding=kernel_size//2, bias=bias, groups=groups)
        self.gcnet = SimplifiedContextBlock(n_feat, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.gcnet(res)
        res += x
        return res


##########################################################################
##---------- Resizing Modules ----------    
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


##########################################################################
##---------- Multi-Scale Resiudal Block (MRB) ----------
class MRB(nn.Module):
    def __init__(self, n_feat, height, width, chan_factor, bias,groups):
        super(MRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width

        self.dau_top = SimplifiedRCB(int(n_feat*chan_factor**0), bias=bias, groups=groups)
        self.dau_mid = SimplifiedRCB(int(n_feat*chan_factor**1), bias=bias, groups=groups)
        self.dau_bot = SimplifiedRCB(int(n_feat*chan_factor**2), bias=bias, groups=groups)

        self.down2 = DownSample(int((chan_factor**0)*n_feat),2,chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor**0)*n_feat),2,chan_factor), 
            DownSample(int((chan_factor**1)*n_feat),2,chan_factor)
        )

        self.up21_1 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up21_2 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up32_1 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)
        self.up32_2 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        # # only two inputs for SKFF
        self.skff_top = SKFF(int(n_feat*chan_factor**0), 2)
        self.skff_mid = SKFF(int(n_feat*chan_factor**1), 2)

    def forward(self, x):
        x_top = x.clone()
        x_mid = self.down2(x_top)
        x_bot = self.down4(x_top)

        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.skff_mid([x_mid, self.up32_1(x_bot)])
        x_top = self.skff_top([x_top, self.up21_1(x_mid)])

        out = self.conv_out(x_top)
        out = out + x

        return out
    

class MMRB(nn.Module):
    def __init__(self, n_feat, height, width, chan_factor, bias,groups):
        super(MMRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width

        self.dau_top = SimplifiedRCB(int(n_feat*chan_factor**0), bias=bias, groups=groups)
        #self.dau_mid = SimplifiedRCB(int(n_feat*chan_factor**1), bias=bias, groups=groups)

        self.down2 = DownSample(int((chan_factor**0)*n_feat),2,chan_factor)

        self.up21_1 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        self.skff_top = SKFF(int(n_feat*chan_factor**0), 2)

    def forward(self, x):
        x_top = x.clone()
        x_mid = self.down2(x_top)

        x_top = self.dau_top(x_top)

        x_top = self.skff_top([x_top, self.up21_1(x_mid)])

        out = self.conv_out(x_top)
        out = out + x

        return out

#---------- DarkNet  -----------------------
class DarkNet(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        n_feat=80,
        chan_factor=1.5,
        height=3,
        width=2,
        bias=False,
        task= None
    ):
        super(DarkNet, self).__init__()
        self.task = task
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)
        modules_body = []
        self.estimator = Illumination_Estimator(n_feat)
        modules_body.append(MMRB(n_feat, height, width, chan_factor, bias, groups=1))
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias))
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)

        

    def forward(self, inp_img):
        illu_map = self.estimator(inp_img)
        inp_img = inp_img * illu_map + inp_img

        shallow_feats = self.conv_in(inp_img)
        deep_feats = self.body(shallow_feats)
        out_img = self.conv_out(deep_feats)
        out_img += inp_img

        return out_img