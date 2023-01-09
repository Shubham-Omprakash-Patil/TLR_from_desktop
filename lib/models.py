import torch
import torch.nn as nn
import math

from DCNv2.dcn_v2 import DCN
from lib.bifpn import BiFPN

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Pspblock(nn.Module):
    def __init__(self):
        super(Pspblock, self).__init__()


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, num_features):
        self.inplanes = num_features        
        self.deconv_with_bias = False

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, num_features, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_features, layers[0])
        self.layer2 = self._make_layer(block, num_features*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_features*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_features*8, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [num_features*4, num_features*2, num_features],
            [4, 4, 4],
        )        

        hm = nn.Conv2d(num_features, num_classes, 
                       kernel_size=1, stride=1, 
                       padding=0, bias=True)
        hm.bias.data.fill_(-2.19)
        self.hm = hm

        wh = nn.Conv2d(num_features, 2, 
                       kernel_size=1, stride=1, 
                       padding=0, bias=True)
        wh.bias.data.fill_(-2.19)
        self.wh = wh
        
        reg = nn.Conv2d(num_features, 2, 
                       kernel_size=1, stride=1, 
                       padding=0, bias=True)
        reg.bias.data.fill_(-2.19)
        self.reg = reg        

        cls = nn.Conv2d(num_features, 14, 
                       kernel_size=1, stride=1, 
                       padding=0, bias=True)
        cls.bias.data.fill_(-2.19)
        self.cls = cls

        nb = nn.Conv2d(num_features, 4, 
                       kernel_size=1, stride=1, 
                       padding=0, bias=True)
        nb.bias.data.fill_(-2.19)
        self.nb = nb

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1, 
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def deconv_layer(self, inplanes, planes, num_kernel):
        layers = []
        
        kernel, padding, output_padding = self._get_deconv_cfg(num_kernel, 0)            
        
        fc = DCN(inplanes, planes, 
                kernel_size=(3,3), stride=1,
                padding=1, dilation=1, deformable_groups=1)
        up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
        fill_up_weights(up)

        layers.append(fc)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))        

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)

        ret = {}        
        ret['hm'] = self.hm(x)
        ret['wh'] = self.wh(x)
        ret['reg'] = self.reg(x)
        ret['cls'] = self.cls(x)
        ret['nb'] = self.nb(x)
        
        return [ret]

class ResNetBiFpn(nn.Module):
    def __init__(self, block, layers, num_classes, num_features):
        self.inplanes = num_features        
        super(ResNetBiFpn, self).__init__()

        self.conv1 = nn.Conv2d(3, num_features, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.p2 = self._make_layer(block, num_features, layers[0])
        self.p3 = self._make_layer(block, num_features*2, layers[1], stride=2)
        self.p4 = self._make_layer(block, num_features*4, layers[2], stride=2)        
        self.p5 = self._make_layer(block, num_features*8, layers[3], stride=2)
        self.bifpn = BiFPN([num_features,num_features,num_features*2,num_features*4,num_features*8], num_features, 2)

        hm = nn.Conv2d(num_features, num_classes, 
                       kernel_size=1, stride=1, 
                       padding=0, bias=True)
        hm.bias.data.fill_(-2.19)
        self.hm = hm

        wh = nn.Conv2d(num_features, 2, 
                       kernel_size=1, stride=1, 
                       padding=0, bias=True)
        wh.bias.data.fill_(-2.19)
        self.wh = wh
        
        reg = nn.Conv2d(num_features, 2, 
                       kernel_size=1, stride=1, 
                       padding=0, bias=True)
        reg.bias.data.fill_(-2.19)
        self.reg = reg        

        cls = nn.Conv2d(num_features, 14, 
                       kernel_size=1, stride=1, 
                       padding=0, bias=True)
        cls.bias.data.fill_(-2.19)
        self.cls = cls

        nb = nn.Conv2d(num_features, 4, 
                       kernel_size=1, stride=1, 
                       padding=0, bias=True)
        nb.bias.data.fill_(-2.19)
        self.nb = nb

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        p1 = self.relu(x)
        p1_max_pool = self.maxpool(p1)

        p2 = self.p2(p1_max_pool)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        
        p2_out = self.bifpn(p1,p2,p3,p4,p5)
        
        ret = {}        
        ret['hm'] = self.hm(p2_out)
        ret['wh'] = self.wh(p2_out)
        ret['reg'] = self.reg(p2_out)
        ret['cls'] = self.cls(p2_out)
        ret['nb'] = self.nb(p2_out)
        
        return [ret]

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_resnet(num_layers, num_classes, num_features):
  block_class, layers = resnet_spec[num_layers]

  model = ResNet(block_class, layers, num_classes, num_features)
  return model

def get_resnet_bifpn(num_layers, num_classes, num_features):
    block_class, layers = resnet_spec[num_layers]

    model = ResNetBiFpn(block_class, layers, num_classes, num_features)
    return model