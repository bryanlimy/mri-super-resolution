from .registry import register

import torch
import numpy as np
from torch import nn

from slowMRI.models import model_helper


@register('mlp')
class MLP(nn.Module):

  def __init__(self, args):
    super(MLP, self).__init__()

    num_features = int(np.prod(args.input_shape))

    self.sequential = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_features, 512),
        model_helper.activation(args.activation),
        nn.Dropout2d(0.5),
        nn.Linear(512, 256),
        model_helper.activation(args.activation),
        nn.Dropout2d(0.25),
        nn.Linear(256, 512),
        model_helper.activation(args.activation),
        nn.Dropout2d(0.25),
        nn.Linear(512, num_features),
    )
    self.sigmoid = None
    if not args.output_logits:
      self.sigmoid = model_helper.activation('sigmoid')()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    shape = inputs.shape
    outputs = self.sequential(inputs)
    outputs = outputs.view(shape)
    if self.sigmoid is not None:
      outputs = self.sigmoid(outputs)
    return outputs
