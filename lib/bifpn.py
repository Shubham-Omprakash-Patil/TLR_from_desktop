import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
    
class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p1_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p2_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        
        self.p2_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p3_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        
        # TODO: Init weights
        self.w1 = nn.Parameter(torch.ones(2, 4))
        self.w1_relu = nn.LeakyReLU(negative_slope=0)
        self.w2 = nn.Parameter(torch.ones(3, 4))
        self.w2_relu = nn.LeakyReLU(negative_slope=0)
    
    def forward(self, p1_x, p2_x, p3_x, p4_x, p5_x):
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon
        
        p5_td = p5_x
        p4_td = self.p4_td(w1[0, 0] * p4_x + w1[1, 0] * F.interpolate(p5_td, scale_factor=2))        
        p3_td = self.p3_td(w1[0, 1] * p3_x + w1[1, 1] * F.interpolate(p4_td, scale_factor=2))
        p2_td = self.p2_td(w1[0, 2] * p2_x + w1[1, 2] * F.interpolate(p3_td, scale_factor=2))
        p1_td = self.p1_td(w1[0, 3] * p1_x + w1[1, 3] * F.interpolate(p2_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p1_out = p1_td
        p2_out = self.p2_out(w2[0, 0] * p2_x + w2[1, 0] * p2_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p1_out))
        p3_out = self.p3_out(w2[0, 1] * p3_x + w2[1, 1] * p3_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p2_out))
        p4_out = self.p4_out(w2[0, 2] * p4_x + w2[1, 2] * p4_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2[0, 3] * p5_x + w2[1, 3] * p5_td + w2[2, 3] * nn.Upsample(scale_factor=0.5)(p4_out))

        return p1_out, p2_out, p3_out, p4_out, p5_out
    
class BiFPN(nn.Module):
    def __init__(self, channels, feature_size=64, num_layers=2, epsilon=0.0001):
        super(BiFPN, self).__init__()
        self.p1 = nn.Conv2d(channels[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p2 = nn.Conv2d(channels[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p3 = nn.Conv2d(channels[2], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(channels[3], feature_size, kernel_size=1, stride=1, padding=0)        
        self.p5 = nn.Conv2d(channels[4], feature_size, kernel_size=1, stride=1, padding=0)
        
        self.bifpn_1 = BiFPNBlock(feature_size)
        self.bifpn_2 = BiFPNBlock(feature_size)
    
    def forward(self, c1, c2, c3, c4, c5):        
        # Calculate the input column of BiFPN
        p1_x = self.p1(c1)        
        p2_x = self.p2(c2)
        p3_x = self.p3(c3)
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)

        p1_out, p2_out, p3_out, p4_out, p5_out = self.bifpn_1(p1_x, p2_x, p3_x, p4_x, p5_x)
        _, p2_out, _, _, _ = self.bifpn_2(p1_out, p2_out, p3_out, p4_out, p5_out)

        return p2_out
