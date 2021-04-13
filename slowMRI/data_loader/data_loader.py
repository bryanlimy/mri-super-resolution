import os
import sys
import torch
import torchvision
import numpy as np
import typing as t
from torch import nn
from pathlib import Path
from torch.fft import fftn, ifftn
from typing import Union, List, Tuple
from skimage.transform import rescale, resize
from torch.utils.data import Dataset, DataLoader

from slowMRI.utils import utils
from slowMRI.utils import h5_io
from slowMRI.data_loader import data_handling


def load_dataset_info(args):
  """ Load dataset information from args.input / info.json to args """
  info = utils.load_json(args.input_dir / 'info.json')
  args.scan_types = info['scan_types']
  args.ds_min = info['ds_min']
  args.ds_max = info['ds_max']
  args.ds_name = info['ds_name']
  args.scan_shape = tuple(info['scan_shape'])


def load_h5_from_dir(args) -> DataLoader:
  """
  Sister function to get_loaders but obtains all scans input dir.

  Returns:
    data_loader: Dataset data loader
  """
  load_dataset_info(args)
  args.patch_dim = (args.patch_dim, args.patch_dim)

  # get hr_samples.h5 and lr_samples if available)
  hr_filename = args.input_dir / 'hr_samples.h5'
  assert hr_filename.exists(), f"File {hr_filename} not found."

  lr_filename = args.input_dir / 'lr_samples.h5'
  if not lr_filename.exists():
    lr_filename = None

  scan_paths = data_handling.get_all_scan_paths(args, hr_filename)
  dataset = MRIDataset(args,
                       hr_filename=hr_filename,
                       lr_filename=lr_filename,
                       dataset_paths=scan_paths,
                       create_patches=False)
  data_kwargs = {
      'batch_size': args.batch_size,
      'num_workers': 2,
      'shuffle': False
  }

  if args.cuda:
    cuda_kwargs = {'prefetch_factor': 2, 'pin_memory': True}
    data_kwargs.update(cuda_kwargs)

  data_loader = DataLoader(dataset, **data_kwargs)

  return data_loader


def get_loaders(args):
  args.input_dir = Path(args.input_dir)
  assert args.input_dir.exists()

  load_dataset_info(args)

  args.patch_dim = (args.patch_dim, args.patch_dim)
  num_channels = len(args.scan_types) if args.merge_scan_type else 1
  padding = None
  args.input_shape = (num_channels,) + args.patch_dim
  if args.scan_input:
    args.input_shape, padding = utils.convert_square_shape((num_channels,) +
                                                           args.scan_shape[1:])

  # get hr_samples.h5 and lr_samples if available)
  hr_filename = args.input_dir / 'hr_samples.h5'
  assert hr_filename.exists(), f"File {hr_filename} not found."

  lr_filename = args.input_dir / 'lr_samples.h5'
  if not lr_filename.exists():
    lr_filename = None

  scan_paths = data_handling.get_all_scan_paths(args, hr_filename)

  train_paths, val_paths, test_paths = data_handling.split_data(
      args, scan_paths, random_state=args.seed)

  # TODO make this configurable
  train_transforms = torchvision.transforms.Compose([
      torchvision.transforms.RandomHorizontalFlip(0.5),
      torchvision.transforms.RandomVerticalFlip(0.05)
  ])

  train_dataset = MRIDataset(args,
                             hr_filename=hr_filename,
                             lr_filename=lr_filename,
                             dataset_paths=train_paths,
                             transforms=train_transforms,
                             padding=padding,
                             create_patches=not args.scan_input,
                             shuffle_slice=True)
  val_dataset = MRIDataset(args,
                           hr_filename=hr_filename,
                           lr_filename=lr_filename,
                           dataset_paths=val_paths,
                           padding=padding,
                           create_patches=not args.scan_input)
  test_dataset = MRIDataset(args,
                            hr_filename=hr_filename,
                            lr_filename=lr_filename,
                            dataset_paths=test_paths,
                            padding=padding,
                            create_patches=False)

  # initialize data loaders
  train_kwargs = {'batch_size': 1, 'num_workers': 2, 'shuffle': True}
  test_kwargs = {'batch_size': 1, 'num_workers': 2, 'shuffle': False}
  if args.cuda:
    cuda_kwargs = {'prefetch_factor': 2, 'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

  train_loader = DataLoader(train_dataset, **train_kwargs)
  val_loader = DataLoader(val_dataset, **test_kwargs)
  test_loader = DataLoader(test_dataset, **test_kwargs)

  return train_loader, val_loader, test_loader


def scan_to_patches(
    scan1: Union[np.ndarray, torch.Tensor],
    scan2: Union[np.ndarray, torch.Tensor] = None,
    patch_dim: Tuple[int, int] = (64, 64),
    n_patches: int = 100
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
  """
  Creates random 2d patches from scan

  Input scan has format (channel, slice, height, width) and the output patches
  has format (n_patches, channel, height, width)

  Args:
    scan1: scan in 4D
    scan2: scan in 4D, None if not needed
    patch_dim: dimensions of the patches
    n_patches: number of patches to be returned
  Returns:
    patches1: tensor of patches
    patches2: tensor of patches if scan2 is provided
  """
  if scan2 is not None:
    assert scan1.shape == scan2.shape
  shape = scan1.shape
  patches1 = torch.zeros((n_patches, shape[0]) + patch_dim, dtype=torch.float)
  patches2 = torch.zeros((n_patches, shape[0]) + patch_dim, dtype=torch.float)

  for i in range(n_patches):
    # select slice - we ignore the first and last 20
    s = np.random.randint(low=20, high=shape[1] - 20)
    # select h, w ranges - we ignore the first and last 5
    h = np.random.randint(low=5, high=shape[2] - patch_dim[0] - 5)
    w = np.random.randint(low=5, high=shape[3] - patch_dim[1] - 5)

    patches1[i, ...] = scan1[:, s, h:h + patch_dim[0], w:w + patch_dim[0]]
    if scan2 is not None:
      patches2[i, ...] = scan2[:, s, h:h + patch_dim[0], w:w + patch_dim[0]]

  if scan2 is not None:
    return patches1, patches2
  return patches1


def torch_blur(hr_samples: torch.Tensor,
               kernel_size: int = 3,
               sigma: float = 2) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Placeholder downsampling function that applies a simple Gaussian blur.
  Args:
    hr_samples: tensor (n_patches, 1, patch_dim, patch_dim)
    kernel_size: size of the gaussian kernel
    sigma: standard deviation of the gaussian kernel
  Returns:
    lr_samples: batch of blurry images
    hr_samples: batch of highres images
  """
  lr_samples = torchvision.transforms.functional.gaussian_blur(
      hr_samples, kernel_size=kernel_size, sigma=sigma)
  return lr_samples, hr_samples


def kspace_downsample_patch(hr_patch: torch.Tensor,
                            center_crop: int = 25,
                            end_crop: int = 6) -> torch.Tensor:
  """
  Down-sample high resolution patch by k-space truncation.

  Note: end_crop worsens the picture quality much more than center_crop.

  Args:
    hr_patch: original high resolution patch
    center_crop: square dimension of center to be removed
    end_crop: final rows/cols to be removed
  Returns:
    lr_patch in shape (channel, patch_dim, patch_dim)
  """
  lr_patch = fftn(hr_patch)

  # remove last n cols and rows
  if end_crop > 0:
    lr_patch[:, -end_crop:, :] = 0
    lr_patch[:, :, -end_crop:] = 0
  # remove square center
  if center_crop > 0:
    i = max(center_crop // 2, 1)
    x_center = lr_patch.shape[1] // 2
    y_center = lr_patch.shape[2] // 2
    lr_patch[:, x_center - i:x_center + i, y_center - i:y_center + i] = 0

  return torch.abs(ifftn(lr_patch))


def kspace_downsample_patches(hr_patches: torch.Tensor,
                              center_crop: int = 25,
                              end_crop: int = 6) -> torch.Tensor:
  """
  Down-sample high resolution patches by k-space truncation.

  Args:
    hr_patches: original high resolution patches
    center_crop: square dimension of center to be removed
    end_crop: final rows/cols to be removed
  Returns:
    lr_scan in shape (n_patches, channel, patch_dim, patch_dim)
  """
  return torch.stack([
      kspace_downsample_patch(hr_patch=patch,
                              center_crop=center_crop,
                              end_crop=end_crop) for patch in hr_patches
  ])


def skimg_down_then_upsample(
    hr_samples: List[np.ndarray],
    downscale_factor: float = 0.6) -> Tuple[List, List]:
  """
  This is a placeholder downsample function. Take a list of 2d arrays and
  downsamples it to 60% of the size and then upsamples it to the original dimensions.
  Args:
    hr_samples: List of high res 2d arrays
    downscale_factor: Factor by which to downscale mri scan
  Returns:
    lr_samples: List of low res 2d arrays
    hr_samples: List of high res 2d arrays
  """
  lr_samples = []
  for sample in hr_samples:
    shape = sample.shape
    sample = rescale(sample, downscale_factor, preserve_range=True)
    sample = resize(
        sample,
        output_shape=shape,
        preserve_range=True,
        # order=3 is bicubic interpolation
        # (see https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp)
        order=3)
    # lr_samples.append(torch.as_tensor(sample, dtype=torch.int16))
    lr_samples.append(torch.as_tensor(sample, dtype=torch.float))

  return lr_samples, hr_samples


def list2tensor(inputs: List[np.ndarray]) -> torch.Tensor:
  """ Convert a list of numpy array to tensor

  This temporary helper function is needed due to a bug in PyTorch where
  converting a list of numpy array is a lot slower than np.stack then
  convert to tensor.

  See issues:
  - https://github.com/pytorch/pytorch/issues/39842
  - https://github.com/pytorch/pytorch/issues/13918
  Potential PR to fix this issue
  - https://github.com/pytorch/pytorch/pull/51731
  """
  return torch.as_tensor(np.stack(inputs), dtype=torch.float)


def normalize(scan: torch.Tensor) -> torch.Tensor:
  """ Normalization to [0, 1] """
  s_min, s_max = scan.min(), scan.max()
  return (scan - s_min) / (s_max - s_min)


class MRIDataset(Dataset):
  """
  Torch Dataset class. Loads 3d-scans from a specified hdf5 file, transforms
  them into many patches, applies transforms, and returns
  (low resolution, high resolution) tuples of image tensors.

  Each image tensor has format
    (n_patches, channel, patch_dim, patch_dim),
  so a batch generated from this dataset will be
    (batch_size, n_patches, channel, patch_dim, patch_dim)
  and should be reshaped to
    (batch_size * n_patches, channel, patch_dim, patch_dim)
  in the training loop.
  """

  def __init__(self,
               args,
               hr_filename: Union[str, Path],
               lr_filename: Union[str, Path],
               dataset_paths: List[str],
               transforms=None,
               padding: t.Tuple[int, int, int] = None,
               create_patches: bool = True,
               shuffle_slice: bool = False):
    """
    Args:
      hr_filename: filename of the h5 file with the high resolution scans
      lr_filename: filename of the h5 file with the low resolution scans
                  None if low resolution scans are not available
      dataset_paths: dataset paths in the h5 file to extract
      transforms: torch functions to transform data
      create_patches: whether to create patches (train+validation) or not (test)
    """
    self.hr_filename = hr_filename
    self.lr_filename = lr_filename
    self.dataset_paths = dataset_paths
    self.transforms = transforms
    self.create_patches = create_patches
    self.shuffle_slice = shuffle_slice

    self.seed = args.seed
    self.n_patches = args.n_patches
    self.patch_dim = args.patch_dim
    self.merge_scan_type = args.merge_scan_type

    self.zero_padding = None
    if padding is not None:
      self.zero_padding = nn.ZeroPad2d(padding)

    # loses some information and introduces interpolation artifacts
    self.downsampling_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(args.patch_dim[0] - 6,
                                            args.patch_dim[1] - 6)),
        torchvision.transforms.Resize(size=args.patch_dim)
    ])

  def __getitem__(self, idx):
    """
    Return lr_patches and hr_patches in (n_patches, channel, patch_dim, patch_dim)
    """
    if torch.is_tensor(idx):
      idx = idx.tolist()

    hr_scan = h5_io.read_ds(self.hr_filename, name=self.dataset_paths[idx])
    # create tensor with format (channel, slice, height, width)
    hr_scan = list2tensor(hr_scan if self.merge_scan_type else [hr_scan])

    # read low resolution images if provided, down-sample high resolution
    # images otherwise
    if self.lr_filename is None:
      if self.create_patches:
        hr_patches = scan_to_patches(scan1=hr_scan, n_patches=self.n_patches)
      else:
        # convert scans to (slice, channel, height, width)
        hr_patches = hr_scan.permute(1, 0, 2, 3)
      # currently, transforms are only supported for downsampling case as random
      # transforms otherwise cause issues
      # TODO add transforms for non-downsampling case
      if self.transforms:
        hr_patches = self.transforms(hr_patches)
      # add desired function here - this could be a config setting in the future
      lr_patches = kspace_downsample_patches(hr_patches)
      if self.create_patches:
        lr_patches = self.downsampling_transforms(lr_patches)
    else:
      lr_scan = h5_io.read_ds(self.lr_filename, name=self.dataset_paths[idx])
      lr_scan = list2tensor(lr_scan if self.merge_scan_type else [lr_scan])
      if self.create_patches:
        hr_patches, lr_patches = scan_to_patches(scan1=hr_scan,
                                                 scan2=lr_scan,
                                                 patch_dim=self.patch_dim,
                                                 n_patches=self.n_patches)
      else:
        # convert scans to (slice, channel, height, width)
        hr_patches = hr_scan.permute(1, 0, 2, 3)
        lr_patches = lr_scan.permute(1, 0, 2, 3)
        if self.shuffle_slice:
          indexes = np.arange(hr_patches.shape[0])
          np.random.shuffle(indexes)
          hr_patches = hr_patches[indexes]
          lr_patches = lr_patches[indexes]

    # TODO: add this to self.transforms
    lr_patches = normalize(lr_patches)
    hr_patches = normalize(hr_patches)

    if self.zero_padding is not None:
      lr_patches = self.zero_padding(lr_patches)
      hr_patches = self.zero_padding(hr_patches)

    return lr_patches, hr_patches

  def __len__(self):
    return len(self.dataset_paths)
