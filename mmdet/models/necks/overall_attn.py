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
                 inplanes,
                 outplanes,
                 gamma,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 ):
        super.__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            outplanes,
            3,
            stride=stride,
            padding="same",
            bias=False)
        self.conv2 = build_conv_layer(
            conv_cfg,
            inplanes,
            outplanes,
            3,
            stride=stride,
            padding="same",
            bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.gamma = torch.tensor(gamma, requires_grad=True)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()  # Convert inputs to fp16
    def forward(self, inputs, ref):
        """Forward function."""
        x0 = self.relu(self.conv1(inputs))
        x1 = self.relu(self.conv1(ref))

        (batchSize, feature_dim, H, W) = x0.shape
        x0 = x1.permute(0, 2, 3, 1).reshape(batchSize, -1, feature_dim)
        x1 = x1.reshape(batchSize, feature_dim, -1)
        corr = x0.bmm(x1).softmax(dim=1)

        x1 = self.relu(self.conv2(ref))

        (batchSize, feature_dim, H, W) = x1.shape
        x1 = x1.reshape(batchSize, feature_dim, -1)
        A = x1.bmm(corr)
        A = A.reshape(inputs.shape)
        output = A * self.gamma + ref
        return output
