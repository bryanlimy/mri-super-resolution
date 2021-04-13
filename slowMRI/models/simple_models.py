from .registry import register

import torch
from torch import nn


@register('identity')
class Identity(nn.Module):
  """
  Dummy model
  """

  def __init__(self,
               args,
               in_channels: int = 1,
               out_channels: int = 1,
               filters: int = 128,
               n_conv_blocks: int = 3,
               residual_connections: bool = False):
    super().__init__()
    self.dummy_weights = torch.nn.Parameter(torch.randn(1), requires_grad=True)

  def forward(self, patch: torch.Tensor) -> torch.Tensor:
    # this is necessary to make this model work with our training loop
    return patch - self.dummy_weights + self.dummy_weights.detach()


@register('simple_cnn')
class SimpleCNN(nn.Module):
  """
  Very basic CNN that passes the input through a number of SimpleConvBlocks.
  """

  def __init__(self,
               args,
               filters: int = 128,
               n_conv_blocks: int = 3,
               residual_connections: bool = False):
    super().__init__()

    self.in_channels = args.input_shape[0]
    self.out_channels = args.input_shape[0]
    self.filters = filters
    self.n_conv_blocks = n_conv_blocks
    self.residual_connections = residual_connections

    self.input_conv = nn.Sequential(
        nn.Conv2d(in_channels=self.in_channels,
                  out_channels=512,
                  kernel_size=3,
                  padding=1,
                  bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))

    conv_blocks = []
    for i in range(n_conv_blocks):
      conv_blocks.append(
          SimpleConvBlock(in_channels=512 if i == 0 else filters,
                          out_channels=filters,
                          filters=filters,
                          residual_connection=residual_connections))
    self.conv_blocks = nn.Sequential(*conv_blocks)

    self.output_conv = nn.Sequential(
        nn.Conv2d(in_channels=filters,
                  out_channels=self.out_channels,
                  kernel_size=1,
                  padding=0,
                  bias=True), nn.Sigmoid())

  def forward(self, patch: torch.Tensor) -> torch.Tensor:
    x = self.input_conv(patch)
    x = self.conv_blocks(x)
    x = self.output_conv(x)
    return x


@register('simple_cnn_residual')
class SimpleCNNResidual(SimpleCNN):
  """
  Like SimpleCNN but uses residual connections by default.
  """

  def __init__(self,
               args,
               filters: int = 128,
               n_conv_blocks: int = 3,
               residual_connections: bool = True):
    super().__init__(args=args,
                     filters=filters,
                     n_conv_blocks=n_conv_blocks,
                     residual_connections=residual_connections)


@register('simple_cnn_skip')
class SimpleCNNSkip(SimpleCNN):
  """
  Like Simple CNN but has a single skip connection from input to output so that
  this model estimates the difference (target-input) instead of target.
  """

  def __init__(self,
               args,
               filters: int = 128,
               n_conv_blocks: int = 3,
               residual_connections: bool = False):
    super().__init__(args=args,
                     filters=filters,
                     n_conv_blocks=n_conv_blocks,
                     residual_connections=residual_connections)
    self.output_conv = nn.Sequential(
        nn.Conv2d(in_channels=filters,
                  out_channels=self.out_channels,
                  kernel_size=1,
                  padding=0,
                  bias=True),
        # skip model uses tanh so it can add and subtract
        nn.Tanh())

  def forward(self, patch: torch.Tensor) -> torch.Tensor:
    x = self.input_conv(patch)
    x = self.conv_blocks(x)
    x = self.output_conv(x)
    return x + patch


@register('simple_cnn_skip_res')
class SimpleCNNSkipRes(SimpleCNNSkip):
  """
  Like SimpleCNNSkip but uses residual connections by default.
  """

  def __init__(self,
               args,
               filters: int = 128,
               n_conv_blocks: int = 3,
               residual_connections: bool = True):
    super().__init__(args=args,
                     filters=filters,
                     n_conv_blocks=n_conv_blocks,
                     residual_connections=residual_connections)


class SimpleConvBlock(nn.Module):
  """
  Very basic convolutional block of 2 Conv2d layers each followed by LeakyRelu.
  """

  def __init__(self,
               in_channels: int = 1,
               out_channels: int = 1,
               filters: int = 32,
               residual_connection: bool = False):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filters = filters
    self.residual_connection = residual_connection

    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=filters,
                  kernel_size=3,
                  padding=1,
                  bias=True),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=filters,
                  out_channels=out_channels,
                  kernel_size=3,
                  padding=1,
                  bias=True),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )

    if self.residual_connection:
      self.projection = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  padding=0,
                                  bias=True)

  def forward(self, patch: torch.Tensor) -> torch.Tensor:
    if self.residual_connection:
      return self.projection(patch) + self.layers(patch)
    return self.layers(patch)
