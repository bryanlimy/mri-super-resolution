import os
import csv
import json
import copy
import torch
import warnings
import subprocess
import numpy as np
import typing as t
from math import ceil
from pathlib import Path
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt

from slowMRI.metrics import metrics


def get_loss_function(name: str):
  """
  Args:
    name: name of loss function
  Returns:
    loss function
  """
  name = name.lower()
  if name in ['mse', 'meansquarederror', 'l2']:
    return F.mse_loss
  elif name in ['mae', 'meanabsoluteerror', 'l1']:
    return F.l1_loss
  elif name in ['bce', 'binarycrossentropy']:
    return F.binary_cross_entropy_with_logits
  elif name in ['ssim']:
    return lambda x, y: 1 - metrics.ssim(x, y)
  raise ValueError(f'Unknown loss function name {name}')


def get_current_git_hash():
  """ return the current Git hash """
  try:
    return subprocess.check_output(['git', 'describe',
                                    '--always']).strip().decode()
  except Exception:
    warnings.warn('Unable to get git hash.')


def save_json(filename: Path, data: t.Dict):
  """ Save dictionary data to filename as a json file """
  assert type(data) == dict
  for key, value in data.items():
    if isinstance(value, np.ndarray):
      data[key] = value.tolist()
    elif isinstance(value, np.float32):
      data[key] = float(value)
    elif isinstance(value, Path) or isinstance(value, torch.device):
      data[key] = str(value)
  with open(filename, 'w') as file:
    json.dump(data, file)


def load_json(filename: Path) -> t.Dict:
  """ Load json file as a dictionary"""
  with open(filename, 'r') as file:
    data = json.load(file)
  return data


def update_json(filename: Path, data: t.Dict):
  """ Update json file with items in data """
  content = {}
  if os.path.exists(filename):
    content = load_json(filename)
  for key, value in data.items():
    content[key] = value
  save_json(filename, content)


def save_args(args):
  """ Save input arguments as a json file in args.output_dir"""
  args.git_hash = get_current_git_hash()
  save_json(args.output_dir / 'args.json', copy.deepcopy(args.__dict__))


def load_args(args, filename=None):
  """ Load input arguments from filename """
  if filename is None:
    filename = args.output_dir / 'args.json'
  content = load_json(filename)
  for key, value in content.items():
    if (not hasattr(args, key)) or (getattr(args, key) == None):
      setattr(args, key, value)


def save_csv(filename, data: t.Dict[str, t.Union[torch.Tensor, np.ndarray]]):
  content = {
      key: data[key].item() if torch.is_tensor(data[key]) else float(data[key])
      for key in sorted(data.keys())
  }
  with open(filename, 'w') as file:
    writer = csv.DictWriter(file, content.keys())
    writer.writeheader()
    writer.writerow(content)


def save_model(args, model):
  """ save model state_dict to args.checkpoint_dir """
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  torch.save(model.state_dict(), args.checkpoint_dir / f'model_{timestamp}.pt')


def load_weights(args, model):
  """ Loads model weights from checkpoints folder. """
  checkpoint_dir = args.output_dir / "checkpoints"
  assert checkpoint_dir.exists(), \
    f'Checkpoint directory {checkpoint_dir} not found.'
  checkpoints = sorted(list(checkpoint_dir.glob('*.pt')))
  if len(checkpoints) > 1:
    print(f'{len(checkpoints)} checkpoints found in {checkpoint_dir}. '
          f'Loading from latest checkpoint...')
  model.load_state_dict(torch.load(checkpoints[-1], map_location=args.device))
  if args.verbose:
    print(f'\nLoaded model weights from {checkpoints[-1]}\n')
  return model


def to_numpy(x: t.Union[np.ndarray, torch.tensor]) -> np.ndarray:
  """ Convert torch.tensor to CPU numpy array"""
  return (x.cpu().numpy()).astype(np.float32) if torch.is_tensor(x) else x


def update_dict(source: dict, target: dict, replace: bool = False):
  """ replace or append items in target to source """
  for key, value in target.items():
    if replace:
      source[key] = value
    else:
      if key not in source:
        source[key] = []
      source[key].append(value)


def save_array_as_pdf(filename: t.Union[Path, str],
                      array: t.Union[np.ndarray, torch.Tensor],
                      dpi: int = 120):
  """
  Save 2D array as PDF image

  Args:
    filename: name of the output file
    img: 2D image
    dpi: dpi of the output image
  """
  assert len(array.shape) == 2
  if torch.is_tensor(array):
    array = array.numpy()
  figure, ax = plt.subplots(1, figsize=(8, 8), squeeze=False, dpi=dpi)
  axes = ax.ravel()
  axes[0].imshow(array, cmap='gray', interpolation='none')
  axes[0].axis('off')
  plt.tight_layout()
  figure.savefig(str(filename),
                 dpi=dpi,
                 format="pdf",
                 bbox_inches='tight',
                 pad_inches=0,
                 transparent=True)
  plt.close(figure)


def get_padding(shape1, shape2):
  """ Return the padding needed to convert shape2 to shape1 """
  assert len(shape1) == len(shape2)
  h_diff, w_diff = shape1[1] - shape2[1], shape1[2] - shape2[2]
  padding_left = w_diff // 2
  padding_right = w_diff - padding_left
  padding_top = h_diff // 2
  padding_bottom = h_diff - padding_top
  return (padding_left, padding_right, padding_top, padding_bottom)


def convert_square_shape(shape):
  """ Return new shape that are multiple 2 and square
  Args:
    shape: the existing shape
  Returns:
    new_shape: the new shape with square H,W and are multiple of 2
    padding: padding needed to convert shape to new shape
  """
  square_dim = 2 * ceil(max(shape[1:]) / 2)
  new_shape = (shape[0], square_dim, square_dim)
  return new_shape, get_padding(new_shape, shape)
