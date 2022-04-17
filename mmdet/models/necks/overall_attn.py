import torch.nn as nn
import torch
import torch.nn.functional as F

from mmcv.cnn import xavier_init
from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import build_conv_layer


@NECKS.register_module
class AttnNeck(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_sz = 3,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 ):
        super(AttnNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_sz - 1) // 2


        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_sz,
            stride=stride,
            padding=padding,
            bias=True)
        self.conv2 = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_sz,
            stride=stride,
            padding=padding,
            bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.gamma = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()  # Convert inputs to fp16
    def forward(self, x, ref_x):
        """Forward function."""
        out = []
        for x_feat, r_feat in zip(x, ref_x):
            print(x_feat.shape, r_feat.shape)

            x_f = self.relu(self.conv1(x_feat))
            r_f = self.relu(self.conv1(r_feat))

            (batchSize, feature_dim, H, W) = x_f.shape
            x_f = x_f.reshape(batchSize, feature_dim, -1) # c x wh
            r_f = r_f.permute(0, 2, 3, 1).reshape(batchSize, -1, feature_dim)  # wh x c
            corr = r_f.bmm(x_f).softmax(dim=1)  # wh x wh

            ref_c2 = self.relu(self.conv2(x_feat)) # c x w x h

            (batchSize, feature_dim, H, W) = ref_c2.shape
            ref_c2 = ref_c2.reshape(batchSize, feature_dim, -1) # c x wh
            A = ref_c2.bmm(corr).reshape(x_feat.shape)  # c x w x h
            output = A * self.gamma + x_feat
            out.append(output)

        return tuple(out)

