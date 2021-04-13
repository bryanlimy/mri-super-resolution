import numpy as np
import torch


class WeightedMSE():
  """
  Mean squared error with lower weights for the pixels closer to the edge.
  Currently supports only square inputs of at least size (30,30)
  and progressively discounts the outer 12 pixels on each edge.
  
  The idea is that for the pixels closer to the edge,
  there is less context for the super resolution model to work with
  and thus those reconstruction errors are weighted less.
  """

  def __init__(self, square_dim: int = 64):
    self.square_dim = square_dim
    self.weights = self.__init_weights(square_dim=square_dim)

  def __init_weights(self, square_dim: int = 64) -> torch.Tensor:
    if square_dim < 30:
      # smaller inputs are currently not supported
      weights = np.ones((square_dim, square_dim))
    else:
      # each padding adds four pixels to each edge, so both dims are increased by 3x8=24 overall
      square_dim -= 24
      weights = np.ones((square_dim, square_dim))
      weights = np.pad(weights, 4, constant_values=0.75)
      weights = np.pad(weights, 4, constant_values=0.5)
      weights = np.pad(weights, 4, constant_values=0.25)
    return torch.tensor(weights)

  def __call__(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    se = ((X - y)**2) * self.weights
    return torch.mean(se)
