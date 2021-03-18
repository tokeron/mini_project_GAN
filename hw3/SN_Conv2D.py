import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
from ..functions.max_sv import max_singular_value


class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SNConv2d, self).__init__(
            in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
            False, _pair(0), groups, bias)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def Weights(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, u = max_singular_value(w_mat, self.u)
        self.u.copy_(u)
        return self.weight / sigma

    def forward(self, x):
        return F.conv2d(x, self.Weights, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)