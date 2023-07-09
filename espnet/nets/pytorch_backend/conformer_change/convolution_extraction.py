#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""

from torch import nn
import torch
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
class ConvolutionExtraction(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(self, channels, bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionExtraction, self).__init__()

        self.LNorm = LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x, l):
        """Compute convolution module.
        # conformer的convolution模块的定义与计算，参考论文结构，一样的
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        x = self.LNorm(x)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)
        l = l.transpose(1, 2)
        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, channel, dim)
        x = torch.cat((x, l), dim=1)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        return x.transpose(1, 2)
