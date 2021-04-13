from .registry import register

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from slowMRI.models import model_helper


class ConvBlock(nn.Module):
  """
  Convolution block consists of 2 blocks of (conv -> norm -> activation )
  """

  def __init__(self,
               in_channels: int,
               out_channels: int,
               mid_channels: int = None,
               kernel_size: int = 3,
               padding: int = 1,
               bias: bool = False,
               normalization: str = 'instancenorm',
               activation: str = 'leakyrelu'):
    super(ConvBlock, self).__init__()
    if not mid_channels:
      mid_channels = out_channels

    self.conv1 = nn.Conv2d(in_channels,
                           mid_channels,
                           kernel_size=kernel_size,
                           padding=padding,
                           bias=bias)
    self.norm1 = model_helper.normalization(normalization)(mid_channels)
    self.activation1 = model_helper.activation(activation)()

    self.conv2 = nn.Conv2d(mid_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=padding,
                           bias=bias)
    self.norm2 = model_helper.normalization(normalization)(out_channels)
    self.activation2 = model_helper.activation(activation)()

  def forward(self, x):
    outputs = self.conv1(x)
    outputs = self.norm1(outputs)
    outputs = self.activation1(outputs)
    outputs = self.conv2(outputs)
    outputs = self.norm2(outputs)
    outputs = self.activation2(outputs)
    return outputs


class DownScale(nn.Module):
  """ down scale block with max pool followed by convolution block """

  def __init__(self,
               in_channels: int,
               out_channels: int,
               normalization: str = 'instancenorm',
               activation: str = 'lrelu'):
    super(DownScale, self).__init__()
    self.max_pool = nn.MaxPool2d(kernel_size=2)
    self.conv_block = ConvBlock(in_channels=in_channels,
                                out_channels=out_channels,
                                normalization=normalization,
                                activation=activation)

  def forward(self, x):
    outputs = self.max_pool(x)
    outputs = self.conv_block(outputs)
    return outputs


class UpScale(nn.Module):
  """ up scale block with transpose convolution followed by convolution block """

  def __init__(self,
               in_channels: int,
               out_channels: int,
               normalization: str = 'instancenorm',
               activation: str = 'lrelu'):
    super(UpScale, self).__init__()
    self.transpose_conv = nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=in_channels // 2,
                                             kernel_size=2,
                                             stride=2)
    self.conv_block = ConvBlock(in_channels=in_channels,
                                out_channels=out_channels,
                                normalization=normalization,
                                activation=activation)
    self.padding = None

  def forward(self, x, skip):
    outputs = self.transpose_conv(x)
    if self.padding is None:
      h_diff = skip.shape[2] - outputs.shape[2]
      w_diff = skip.shape[3] - outputs.shape[3]
      self.padding = [
          w_diff // 2,
          w_diff - (w_diff // 2),
          h_diff // 2,
          h_diff - (h_diff // 2),
      ]
    outputs = F.pad(outputs, pad=self.padding)
    outputs = torch.cat([outputs, skip], dim=1)
    outputs = self.conv_block(outputs)
    return outputs


@register('unet')
class UNet(nn.Module):
  """
  UNet model
  reference: https://github.com/milesial/Pytorch-UNet/tree/master/unet
  """

  def __init__(self, args, max_blocks: int = 5):
    """ initialize UNet model
    Args:
      args
      max_blocks: the maximum number of down scale blocks
      return_logits: return logits or activated outputs
    """
    super(UNet, self).__init__()
    in_channels = args.input_shape[0]
    out_channels = args.input_shape[0]
    num_filters = args.num_filters
    normalization = args.normalization
    activation = args.activation

    # calculate the number of down scale blocks s.t. the smallest block output
    # is at least 2x2 in height and width
    num_blocks = int(math.log(args.input_shape[1], 2))
    self.filters = [
        num_filters * (2**i) for i in range(min(num_blocks, max_blocks))
    ]

    self.input_block = ConvBlock(in_channels=in_channels,
                                 out_channels=self.filters[0],
                                 normalization=normalization,
                                 activation=activation)

    self.down_blocks = nn.ModuleList([
        DownScale(in_channels=self.filters[i],
                  out_channels=self.filters[i + 1],
                  normalization=normalization,
                  activation=activation) for i in range(len(self.filters) - 1)
    ])

    self.up_blocks = nn.ModuleList([
        UpScale(in_channels=self.filters[i],
                out_channels=self.filters[i - 1],
                normalization=normalization,
                activation=activation)
        for i in range(len(self.filters) - 1, 0, -1)
    ])

    self.output_conv = nn.Conv2d(in_channels=self.filters[0],
                                 out_channels=out_channels,
                                 kernel_size=1)
    self.sigmoid = None
    if not args.output_logits:
      self.sigmoid = model_helper.activation('sigmoid')()

  def forward(self, x):
    outputs = self.input_block(x)

    skips = [outputs]
    for i in range(len(self.down_blocks)):
      outputs = self.down_blocks[i](outputs)
      skips.append(outputs)

    skips = skips[-2::-1]

    for i in range(len(self.up_blocks)):
      outputs = self.up_blocks[i](outputs, skips[i])

    outputs = self.output_conv(outputs)
    if self.sigmoid is not None:
      outputs = self.sigmoid(outputs)
    return outputs
