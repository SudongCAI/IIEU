from __future__ import absolute_import
import math

import torch
from torch.autograd import Variable 
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from functools import partial
from itertools import chain
from torch.utils.checkpoint import checkpoint

import torch.nn as nn
from torch.hub import load_state_dict_from_url

from einops import rearrange
from einops.layers.torch import Rearrange

import torch.linalg as linalg

from torch.autograd import Function
from torch.nn.init import calculate_gain

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import DropBlock2d, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, create_classifier

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg

import numpy as np



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }

default_cfg = {
    'mobilenetv2': _cfg(
        url='',
        interpolation='bicubic')
}





class IIEUB(nn.Module):
    def __init__(self, kernel_size, dilation, padding, stride, inplanes, planes, rate_reduct=8, spatial=8, g=1):
        super(IIEUB, self).__init__()
                
        self.kernel_size = kernel_size
        self.stride = stride
        self.g = g
        
        self.num_f, self.dim_f = planes, inplanes # [num of filters, filter dim, h, w]
        
        # nonlinear param
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding) if (kernel_size[0] > 1 or stride[0] > 1) else None
        
        self.sign = 'iieu basic example (IIEU-B)' # 
        
        # ch shift params
        self.stats = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(planes, eps=1e-06)
        nn.init.constant_(self.norm.weight, 0.01)
        nn.init.constant_(self.norm.bias, 0.)
        
        # r * sigma
        self.bound = nn.Parameter(0.05 * torch.ones(1, planes, 1, 1)) # 

    @torch.jit.script
    def compute(x, filter_len, x_pre_vec_len, shift, bound):
        mask = -bound + shift + filter_len * x * x_pre_vec_len # shift before mask
        return torch.where(mask<0, bound * x * torch.exp(mask.clamp(max=0)), bound * x + mask * x) # where(condition, x (if condition), y(otherwise))
    
    def forward(self, x, x_pre, Filt): 
        
        eps = 1e-08 
        
        # calculate the length of x_pre with regions
        coe_k = 1./self.kernel_size[0]
        
        # sim factors
        if self.kernel_size[0] > 1:
            x_pre_vec_len = torch.reciprocal(torch.sqrt(self.avgpool(linalg.vector_norm(x_pre, dim=1, keepdim=True).pow(2))) + eps) * coe_k
        elif self.stride[0] > 1:
            x_pre_vec_len = torch.reciprocal(linalg.vector_norm(self.avgpool(x_pre), dim=1, keepdim=True) + eps)
        else:
            x_pre_vec_len = torch.reciprocal(linalg.vector_norm(x_pre, dim=1, keepdim=True) + eps)
        
        filter_len = torch.reciprocal(linalg.vector_norm(Filt.weight.view(self.num_f, -1), dim=1) + eps).view(1,self.num_f,1,1)
        
        ## parametric adaptation
        shift = self.norm(self.stats(x).squeeze()).sigmoid().unsqueeze(-1).unsqueeze(-1) # b c 1 1
        
        return self.compute(x, filter_len, x_pre_vec_len, shift, self.bound)

########################





def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride, norm_layer):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        IIEUB(kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1), 
             stride=(2, 2), inplanes=inp, planes=oup)
    )


def conv_1x1_bn(inp, oup, norm_layer):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        IIEUB(kernel_size=(1, 1), dilation=(1, 1), padding=(0, 0), 
             stride=(1, 1), inplanes=inp, planes=oup)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        
        self.expand_ratio = expand_ratio
        if expand_ratio == 1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                norm_layer(hidden_dim)
                )
            self.act1 = IIEUB(kernel_size=self.conv1[0].kernel_size, dilation=self.conv1[0].dilation, 
                                      padding=self.conv1[0].padding, stride=self.conv1[0].stride, 
                                      inplanes=hidden_dim, planes=hidden_dim)
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup)
                )

            
        else:            
            self.conv1 = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                norm_layer(hidden_dim)
                )
            self.act1 = IIEUB(kernel_size=self.conv1[0].kernel_size, dilation=self.conv1[0].dilation, 
                                      padding=self.conv1[0].padding, stride=self.conv1[0].stride, 
                                      inplanes=inp, planes=hidden_dim)

            self.conv2 = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                norm_layer(hidden_dim)
                )
            self.act2 = IIEUB(kernel_size=self.conv2[0].kernel_size, dilation=self.conv2[0].dilation, 
                                      padding=self.conv2[0].padding, stride=self.conv2[0].stride, 
                                      inplanes=hidden_dim, planes=hidden_dim)

            self.conv3 = nn.Sequential(
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
                )



    def forward(self, x):
        if self.expand_ratio == 1:
            residual = x
            x = self.conv1(x) # [0] conv, [1] bn
            x = self.act1(x, residual, self.conv1[0])
            x = self.conv2(x) # [0] conv, [1] bn

        else:
            residual = x
            x = self.conv1(x) # [0] conv, [1] bn
            x = self.act1(x, residual, self.conv1[0])
            x_p2 = x
            x = self.conv2(x) # [0] conv, [1] bn
            x = self.act2(x, x_p2, self.conv2[0])
            x = self.conv3(x) # [0] conv, [1] bn
        
        if self.identity:
            return residual + x
        else:
            return x


class MobileNetV2(nn.Module):
    def __init__(
            self, block=InvertedResidual, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
            cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1,
            down_kernel_size=1, avg_down=False, act_layer=None, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None, 
            pretrained=False, width_mult=0.17): # new args
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        #layers = [conv_3x3_bn(3, input_channel, 2)]
        self.layer_open = conv_3x3_bn(3, input_channel, 2, norm_layer)
        layers = []
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, norm_layer))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel, norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x_p = x
        x = self.layer_open[0](x)
        x = self.layer_open[1](x)
        x = self.layer_open[2](x, x_p, self.layer_open[0]) 
        
        # feat
        x = self.features(x)
        
        # after feat
        x_p_conv = x
        x = self.conv[0](x)
        x = self.conv[1](x)
        x = self.conv[2](x, x_p_conv, self.conv[0])
        
        # after conv
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



"""
# simple definition (for debugging)
def iieub_mobilenetv2(**kwargs):
    return MobileNetV2(width_mult=0.17, **kwargs)
"""


# mobilenetv2 x0.17
@register_model
def iieub_mobilenetv2(pretrained=False, **kwargs):
    return build_model_with_cfg(
        MobileNetV2(width_mult=0.17), 'iieub_mobilenetv2', default_cfg=default_cfg['mobilenetv2'],
        pretrained=pretrained, **kwargs)

