import torch.nn as nn

from slowMRI.critic.critic import register
from slowMRI.models.model_helper import get_conv_shape


@register('dcgan')
class DCGAN(nn.Module):
  """
  DCGAN discriminator
  reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
  """

  def __init__(self, args):
    super(DCGAN, self).__init__()
    num_filters = args.num_filters
    num_blocks = 3
    kernel_size = 4
    stride = 2
    padding = 1
    dropout = 0.5

    c, h, w = args.input_shape
    in_channels, out_channels = c, num_filters

    h, w = get_conv_shape(h, w, kernel_size, stride, padding)
    self.input_conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False)
    self.input_activation = nn.LeakyReLU()
    self.input_dropout = nn.Dropout2d(dropout)

    conv_blocks = []
    for i in range(num_blocks):
      new_h, new_w = get_conv_shape(h, w, kernel_size, stride, padding)
      if h <= kernel_size or w <= kernel_size:
        break
      in_channels, out_channels = out_channels, out_channels * 2
      conv_blocks.append(
          nn.Sequential(
              nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False),
              nn.LeakyReLU(),
              nn.Dropout2d(dropout),
          ))
      h, w = new_h, new_w
    self.conv_blocks = nn.ModuleList(conv_blocks)

    h, w = get_conv_shape(h, w, kernel_size, 1, 0)
    self.output_conv = nn.Conv2d(in_channels=out_channels,
                                 out_channels=1,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=0,
                                 bias=False)
    self.flatten = nn.Flatten()
    self.output_dense = nn.Linear(in_features=h * w, out_features=1)

  def forward(self, x):
    outputs = self.input_conv(x)
    outputs = self.input_activation(outputs)
    outputs = self.input_dropout(outputs)
    for i in range(len(self.conv_blocks)):
      outputs = self.conv_blocks[i](outputs)
    outputs = self.output_conv(outputs)
    outputs = self.flatten(outputs)
    outputs = self.output_dense(outputs)
    return outputs
