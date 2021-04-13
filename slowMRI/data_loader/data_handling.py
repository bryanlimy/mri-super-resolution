import os
import h5py
import torch
import numpy as np
import typing as t
from math import ceil
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from torch.nn import Module
import torch.nn.functional as F

from config import DATA_FOLDER
from slowMRI.utils import utils
from slowMRI.metrics import metrics
from slowMRI.critic.critic import Critic
from slowMRI.data_loader import data_loader
from slowMRI.utils.tensorboard import Summary


def get_dataset_name(path: Path):
  """ Return the name of the dataset (HCP, MDS, fastMRI) given path """
  path = str(path).lower()
  if 'hcp' in path:
    return 'hcp'
  elif 'mds' in path:
    return 'mds'
  elif 'fastmri' in path:
    return 'fastmri'

  raise NotImplementedError('Only HCP, MDS and fastMRI datasets are supported')


def extract_path_structure(data_path: Path = DATA_FOLDER,
                           file_format: str = 'nii.gz') -> dict:
  """
    Creates path structure for the datasets in a given path.

    Args:
      data_path: path to the data folder
      file_format: format of the file to be searched for in `data_path`

    Returns:
      data_paths_dict: hierarchical dict of data file {dataset_type : {mri_type : [paths] } }
        eg {'hcp':
              {'MPR1':
                [PosixPath('115320/unprocessed/3T/T1w_MPR1/115320_3T_T1w_MPR1.nii.gz'),
                PosixPath('195041/unprocessed/3T/T1w_MPR1/195041_3T_T1w_MPR1.nii.gz'),
                PosixPath('103818/unprocessed/3T/T1w_MPR1/103818_3T_T1w_MPR1.nii.gz'),
                PosixPath('660951/unprocessed/3T/T1w_MPR1/660951_3T_T1w_MPR1.nii.gz') ...]
              ...,
             ..., }
           }
    """
  paths_list = list(Path(data_path).rglob(f'*.{file_format}'))
  data_paths_dict = {}

  for full_path in paths_list:
    # convert path to str to allow for preprocessing
    path = str(full_path)
    # os path adds a '/' or '\' depending on os
    if os.path.join('hcp', '') in path.lower():
      # remove irrelevant parts of paths
      # in this case /Users/leo/Documents/code/slowMRI/data/hcp/small/'
      _, hcp_path = path.split(os.path.join('small', ''))
      # extract information from path
      # eg /627549/unprocessed/3T/T1w_MPR2/627549_3T_T1w_MPR2.nii.gz
      #   / patient_code / _ / resolution / MRI_type / filename
      patient_code, _, resolution, mri_type, file_name = hcp_path.split(os.sep)
      mri_type = mri_type.split('_')[-1]
      # if the file contains the MRI:
      if mri_type in file_name:
        # create HCP dataset type:
        if 'hcp' not in data_paths_dict:
          data_paths_dict['hcp'] = {}
        # append MRI Path
        if mri_type in data_paths_dict['hcp']:
          data_paths_dict['hcp'][mri_type].append(full_path)
        else:
          # create MRI Type and add path:
          data_paths_dict['hcp'][mri_type] = [full_path]
    else:
      raise ValueError(f'Unknown dataset in {full_path}')

  return data_paths_dict


def get_all_scan_paths(args, filename: Path):
  """
  Return all the scan paths from H5 for each dataset type.

  Note: This function assumes all of the patients scans will be used in the
  same set. Ie. if MPR1 is used in train_epoch set, so will MPR2 and SPC1.

  Args:
    args
    filename: path to hdf5
  Returns:
    list of list of scan paths from the H5 file
    e.g.
    [ [scan1_FLAIR, scan1_T1, scan1_T2] ... ] for MDS
    [ [scan1_MPR1, scan1_MPR2, scan1_SCP1] ... ] for HCP
  """
  scan_paths = []
  with h5py.File(filename, 'r') as dataset:
    for scan_id in dataset:
      scan_paths.append(
          [f'{scan_id}/{scan_type}' for scan_type in args.scan_types])
  return scan_paths


def split_data(
    args,
    data_paths: t.List[Path],
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    random_state: int = 42,
) -> (t.List[str], t.List[str], t.List[str]):
  """
  Split the data into train, val and test and save the splits to
  file.

  Args:
    args
    data_paths: list of list of scan paths in H5 file
                e.g. [ [scan1_FLAIR, scan1_T1, scan1_T2] ... ]
    val_ratio: validation set ratio
    test_ratio: test set ratio
    random_state: random state to be passed
  Returns:
    train_paths: list of scan paths for training
    val_paths: list of scan paths for validation
    test_paths: list of scan paths for testing
  """
  test_size = int(len(data_paths) * test_ratio)
  val_size = int(len(data_paths) * val_ratio)
  train_size = len(data_paths) - val_size - test_size

  data_paths = np.asarray(data_paths)

  # shuffle indexes
  rng = np.random.default_rng(random_state)
  indexes = np.arange(len(data_paths))
  rng.shuffle(indexes)

  # split data into train validation and test set
  train_paths = data_paths[indexes[:train_size]]
  val_paths = data_paths[indexes[train_size:train_size + val_size]]
  test_paths = data_paths[indexes[train_size + val_size:]]

  if not args.merge_scan_type:
    # treat each scan type separately
    train_paths = train_paths.flatten()
    val_paths = val_paths.flatten()
    test_paths = test_paths.flatten()

  return train_paths.tolist(), val_paths.tolist(), test_paths.tolist()


def load_splits_from_file(
    splits_paths: t.List[Path],) -> (t.List[Path], t.List[Path], t.List[Path]):
  """
  Load splits of train, val and test from path.

  Args:
    splits_paths: List of Paths to (train, test, val) splits

  Returns:
    paths_train: List of training paths
    paths_test: List of test paths
    paths_val: List of validation paths
  """
  # load splits from txt:
  paths_train = np.genfromtxt(splits_paths[0], dtype='str')
  paths_test = np.genfromtxt(splits_paths[1], dtype='str')
  paths_val = np.genfromtxt(splits_paths[2], dtype='str')

  return paths_train, paths_test, paths_val


class SliceUpsampler:
  """
  Given a slice of MRI, it creates patches which are then upsampled and
  stitched together to form a final image of the MRI slice.
  """

  def __init__(self,
               args,
               model,
               stride: int,
               loss_function=None,
               critic: t.Type[Critic] = None,
               pad_value: int = 0):
    """
    Up-samples an MRI slice given a model

    Scan should be in format (NSCHW)

    Args:
      args
      model: PyTorch model used for upsampling. For testing purposes a
        function like scikit-image upsampler could be used.
      stride: Step size to move the patch
      loss_function: function for loss calculations
      critic: critic model
      pad_value: fill value for padding, default = 0
    """
    if stride > args.input_shape[1]:
      print(f'\nStride of {stride} is larger than {args.input_shape[1]}, which '
            f'would leave gaps in the final image. Setting stride size to 1.\n')
      stride = 1

    self.model = model
    self.loss_function = loss_function
    self.critic = critic
    self.stride = stride
    self.pad_value = pad_value

    self.device = args.device
    self.scan_input = args.scan_input
    self.batch_size = args.batch_size
    self.input_shape = args.input_shape
    self.slice_shape = (args.input_shape[0], args.scan_shape[1],
                        args.scan_shape[2])
    self.output_logits = args.output_logits
    self.merge_scan_type = args.merge_scan_type

    # calculate up-sampling factor
    dummy_input = torch.zeros(size=(1,) + args.input_shape,
                              dtype=torch.float,
                              device=self.device)
    resized_slice = model(dummy_input)
    self.stretch_factor = resized_slice.shape[-1] / dummy_input.shape[-1]

    self.check_inputs()

    # calculate padding needed to patch slice evenly
    h_diff, w_diff, h_pad, w_pad = self.calculate_pad_for_patches()
    self.h_pad = h_pad
    self.w_pad = w_pad
    self.h_diff = h_diff
    self.w_diff = w_diff
    self.slice_padded_shape = (
        self.slice_shape[0],
        self.slice_shape[1] + self.h_pad,
        self.slice_shape[2] + self.w_pad,
    )

    # increase the size to match upsampling
    self.final_slice_shape = (
        self.slice_padded_shape[0],
        int(self.slice_padded_shape[1] * self.stretch_factor),
        int(self.slice_padded_shape[2] * self.stretch_factor),
    )
    self.stretched_stride = int(self.stride * self.stretch_factor)

  def check_inputs(self):
    """
    Checks whether user inputs comply to standards.

    Raises:
      AssertionError: if types are incompatible with function.
    """
    assert self.input_shape[1] == self.input_shape[2], \
      f'Input dimension should be equal, got {self.input_shape[2:]}.'
    assert len(self.slice_shape) == 3, \
      f'Expected MRI shapes to be 2D but got {self.slice_shape}.'
    assert self.stretch_factor >= 1, \
      f'Stretch factor is below 1: {self.stretch_factor}'

  def calculate_pad_for_patches(self) -> t.Tuple[int, int, int, int]:
    """
    Calculates how much padding the MRI slice requires based on input values

    Returns:
      h_diff: height difference between slice and patch
      w_diff: width difference between slice and patch
      h_pad: height padding needed
      w_pad: width padding needed
    """
    # calculate the difference between slice height and width against patch dim
    h_diff = self.slice_shape[1] - self.input_shape[1]
    w_diff = self.slice_shape[2] - self.input_shape[2]
    h_pad, w_pad = 0, 0

    # check width and height are divisible by stride
    if h_diff % self.stride != 0:
      h_strides = h_diff / self.stride
      # multiply decimals by stride to get whole number of extra padding needed
      h_pad = int(ceil((h_strides - int(h_strides))) * self.stride)
    if w_diff % self.stride != 0:
      w_strides = w_diff / self.stride
      # multiply decimals by stride to get whole number of extra padding needed
      w_pad = int(ceil((w_strides - int(w_strides))) * self.stride)

    return h_diff, w_diff, h_pad, w_pad

  def create_patches(self, slice: torch.Tensor) -> torch.Tensor:
    """
    Creates patches from an image based on a stride that moves a default
    patch. These are saved into a 4D list indicating:
     (position_y, position_x, channel, patch_dim, patch_dim)

    The array is padded with a constant value of 0 (default) to make sure
    that patches can be evenly spaced.

    Returns:
      patches_list: 4D tensor with list of patches.
    """
    padded_slice = slice
    h_diff, w_diff = self.h_diff, self.w_diff
    if self.h_pad > 0 or self.w_pad > 0:
      # pad slice bottom and right
      padded_slice = F.pad(slice, (0, self.w_pad, 0, self.h_pad),
                           value=self.pad_value)
      padded_shape = padded_slice.shape
      h_diff = padded_shape[1] - self.input_shape[1]
      w_diff = padded_shape[2] - self.input_shape[2]

    # number patches in height and width dimension
    n_h_patches = int(1 + (h_diff / self.stride))
    n_w_patches = int(1 + (w_diff / self.stride))

    patched_slice = torch.zeros(
        (
            n_h_patches,
            n_w_patches,
        ) + self.input_shape,
        dtype=torch.float,
        device=self.device,
    )
    for h in range(n_h_patches):
      h_start = h * self.stride
      for w in range(n_w_patches):
        w_start = w * self.stride
        patch = padded_slice[:, h_start:h_start + self.input_shape[1],
                             w_start:w_start + self.input_shape[2]]
        patched_slice[h, w] = patch
    return patched_slice

  def upsample_patches(self, lr_patches: torch.Tensor) -> torch.Tensor:
    """
    Uses Pytorch model to upsample patch.

    Args:
      lr_patches: 5D array with patches
    Returns:
      upsampled_image: 2D numpy array with upsampled patch.
    """
    shape = lr_patches.shape
    # create an empty array with the size of the upsampled image
    output_patches = torch.zeros(
        (
            shape[0],
            shape[1],
            shape[2],
            int(shape[3] * self.stretch_factor),
            int(shape[4] * self.stretch_factor),
        ),
        dtype=torch.float,
        device=self.device,
    )

    with torch.no_grad():
      # convert from (n_h_patches, n_w_patches, channel, patch_dim, patch_dim)
      # to (n_h_patches * n_w_patches, channel, patch_dim, patch_dim)
      inputs = torch.flatten(lr_patches, start_dim=0, end_dim=1)
      outputs = torch.zeros_like(inputs,
                                 dtype=inputs.dtype,
                                 device=inputs.device)
      for i in range(0, inputs.shape[0], self.batch_size):
        logits = self.model(inputs[i:i + self.batch_size])
        outputs[i:i + self.batch_size] = F.sigmoid(logits) \
          if self.output_logits else logits

    # stitch patches together
    count = 0
    for h in range(shape[0]):
      for w in range(shape[1]):
        output_patches[h, w] = outputs[count]
        count += 1
    return output_patches

  def stitch_patches(self, patches: torch.Tensor) -> torch.Tensor:
    """
    Stitches patches together.

    Args:
      patches:(n_h_patches, n_w_patches, channel, patch_dim, patch_dim)
    Returns:
      stitched_output: 2D array with stitched matrix, without padding.
    """
    stitched_output = torch.zeros(self.final_slice_shape,
                                  dtype=torch.float,
                                  device=self.device)

    for h in range(patches.shape[0]):
      h_start = h * self.stretched_stride
      for w in range(patches.shape[1] - 1):
        patch = patches[h, w]
        next_patch = patches[h, w + 1]
        # calculate differences in left and right directions
        left_diff = patch.shape[2] - (patch.shape[2] - self.stretched_stride)
        right_diff = next_patch.shape[2] - self.stretched_stride
        # obtain intersection of matrices and average them
        w_intersect = (patch[:, :, left_diff:] +
                       next_patch[:, :, :right_diff]) / 2
        # replace intersect into the left matrix
        patch[:, :, left_diff::] = w_intersect
        # stack whatever is left of the right matrix
        stacked_patches = torch.cat([patch, next_patch[:, :, right_diff::]],
                                    dim=-1)
        # extract shape of the left + right matrix
        added_shape = stacked_patches.shape
        # slice and replace original matrix
        w_start = w * self.stretched_stride
        stitched_output[:, h_start:h_start + added_shape[1],
                        w_start:w_start + added_shape[2],] = stacked_patches

    if stitched_output.shape != self.slice_shape:
      stitched_output = stitched_output[:, :self.slice_shape[1], :self.
                                        slice_shape[2]]
    return stitched_output

  def upsample_slice(
      self,
      lr_slice: torch.Tensor,
      return_critic: bool = False) -> (torch.Tensor, torch.Tensor):
    """
    Upsamples a lr slice (CHW) into higher resolution

    Args:
      lr_slice: lr slice of MRI to upsample
      return_critic: return critic inputs
    Returns:
      output_slice: upsampled slice using model
      critic_input: input for critic model if return_critic is True
    """
    if self.scan_input:
      output_slice = self.model(torch.unsqueeze(lr_slice, dim=0))[0]
      if self.output_logits:
        output_slice = F.sigmoid(output_slice)
      critic_input = torch.unsqueeze(output_slice, dim=0)
    else:
      lr_patches = self.create_patches(lr_slice)
      output_patches = self.upsample_patches(lr_patches)
      output_slice = self.stitch_patches(output_patches)
      critic_input = torch.flatten(output_patches, end_dim=1)
    if return_critic:
      return output_slice, critic_input
    else:
      return output_slice

  def upsample_batch(self, lr_scans: torch.Tensor, hr_scans: torch.Tensor,
                     slice_idx: int) -> t.Tuple[dict, dict]:
    """
    Up-sample and stitch lr_scans with slice slice_idx

    Args:
      lr_scans: low resolution scans in shape (NCHW)
      hr_scans: high resolution scans in shape (NCHW)
      slice_idx: slice index to up-sample and stitch
    Returns:
      samples: input, target and output samples of slice_idx in shape (NCHW)
        dictionary in {'inputs': tensor, 'targets': tensor, 'outputs': tensor}
      result: dictionary of metrics between generated and target samples
    """
    critic_scores = []
    lr_batch, hr_batch, output_batch = [], [], []
    for lr_scan, hr_scan in zip(lr_scans, hr_scans):
      lr_slice = lr_scan[slice_idx, :, :, :]
      hr_slice = hr_scan[slice_idx, :, :, :]

      output_slice, critic_input = self.upsample_slice(lr_slice,
                                                       return_critic=True)

      lr_batch.append(lr_slice)
      hr_batch.append(hr_slice)
      output_batch.append(output_slice)

      if self.critic is not None:
        critic_scores.append(self.critic.predict(critic_input))

    lr_batch = torch.stack(lr_batch)
    hr_batch = torch.stack(hr_batch)
    output_batch = torch.stack(output_batch)

    samples = {'inputs': lr_batch, 'targets': hr_batch, 'outputs': output_batch}

    result = {
        'MAE': metrics.mae(output_batch, hr_batch),
        'NMSE': metrics.nmse(output_batch, hr_batch),
        'PSNR': metrics.psnr(output_batch, hr_batch),
        'SSIM': metrics.ssim(output_batch, hr_batch),
    }

    if self.loss_function is not None:
      result.update({'Loss': self.loss_function(output_batch, hr_batch)})

    if critic_scores:
      result.update(
          {'critic/generation_score': torch.stack(critic_scores).mean()})

    return samples, result

  def upsample(self, lr_scan: torch.Tensor, verbose: int = 0) -> torch.Tensor:
    """
    Up-sample and stitch lr_scans with slice slice_idx

    Note: this function should be used for inference only

    Args:
      lr_scan: low resolution scans in shape (SCHW)
      verbose: print tqdm progress bar
    Returns:
      output_scan: the up-sampled scan in (SCHW)
    """
    shape = lr_scan.shape
    assert len(shape) == 4, 'scan must be in shape (SCHW)'

    # pad scan to have square dimension if needed
    shape, padding = lr_scan.shape, None
    if self.scan_input and self.input_shape != shape:
      padding = utils.get_padding(self.input_shape, shape[1:])
      lr_scan = F.pad(lr_scan, pad=padding)

    output_scan = torch.zeros_like(lr_scan,
                                   dtype=lr_scan.dtype,
                                   device=lr_scan.device)
    for slice in tqdm(range(lr_scan.shape[0]),
                      desc='Slice',
                      leave=False,
                      disable=verbose == 0):
      lr_slice = lr_scan[slice, ...]
      if self.merge_scan_type:
        output_scan[slice] = self.upsample_slice(lr_slice)
      else:
        for channel in range(lr_slice.shape[0]):
          output_scan[slice, channel] = self.upsample_slice(
              torch.unsqueeze(lr_slice[channel], dim=0))[0]

    if padding is not None:
      output_scan = output_scan[:, :, padding[2]:-padding[3],
                                padding[0]:-padding[1]]

    assert output_scan.shape == shape

    return output_scan


def scale_data(data,
               min_max: Tuple[float, float] = None,
               return_min_max: bool = False):
  """
  Scales data.
  Optionally takes min_max tuple for scaling or return this tuple to scale
  another dataset equally.
  Args:
    data:
    min_max: tuple containing minimum and maximum to be used for scaling
    return_min_max: whether to return min_max tuple calculated on data

  Returns:
    data: scaled data
    min_max: tuple containing minimum and maximum value of scan
  """
  data = data.astype(np.float32)

  if min_max is None:
    min_max = (data.min(), data.max())

  data = (data - min_max[0]) / (min_max[1] - min_max[0])

  # we keep this optional for compatibility, since we don't always need it
  if return_min_max:
    return data, min_max

  return data
