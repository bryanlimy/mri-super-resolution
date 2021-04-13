import re
import os
import torch
import argparse
import scipy.io
import typing as t
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.io import savemat

from slowMRI.utils import utils
from slowMRI.models.registry import get_model
from slowMRI.data_loader.data_loader import normalize
from slowMRI.data_loader.data_handling import SliceUpsampler


def load_args(args):
  """
  Loads settings from output_dir/args.json, upsampler stride and input dir.
  """
  utils.load_args(args, filename=args.output_dir / 'args.json')
  args.input_shape = tuple(args.input_shape)
  # allow for backward compatibility
  if not hasattr(args, "upsampler_stride"):
    stride_size = min(args.input_shape[1], args.upsampler_stride)
    setattr(args, "upsampler_stride", stride_size)
  args.output_dir = Path(args.output_dir)
  # create directory for upsample outputs
  if args.upsample_dir is None:
    args.upsample_dir = args.input_dir / 'upsampled'
  else:
    args.upsample_dir = Path(args.upsample_dir)
  if args.upsample_dir.exists():
    raise FileExistsError(
        f'Upsample director {args.upsample_dir} already exits.')
  args.upsample_dir.mkdir(parents=True)
  # create directory for plots
  args.plot_dir = args.upsample_dir / 'plots'
  args.plot_dir.mkdir(parents=True, exist_ok=True)


def load_mat(args, filename: Path) -> torch.Tensor:
  """
  Load Mat MRI file to a tensor.

  Args:
    filename: .mat file with MRI
  Returns:
    scan: MRI scan with channel order [FLAIR, T1, T2]
  """
  data = scipy.io.loadmat(str(filename))
  flair = data['FLAIRarray'].astype(np.float32)
  t1 = data['T1array'].astype(np.float32)
  t2 = data['T2array'].astype(np.float32)
  if np.isnan(flair).any():
    print(f'\t{os.path.basename(filename)}/FLAIR has NaN values')
    flair = np.nan_to_num(flair)
  if np.isnan(t1).any():
    print(f'\t{os.path.basename(filename)}/T1 has NaN values')
    t1 = np.nan_to_num(t1)
  if np.isnan(t2).any():
    print(f'\t{os.path.basename(filename)}/T2 has NaN values')
    t2 = np.nan_to_num(t2)
  scan = np.stack([flair, t1, t2])
  args.original_shape = scan.shape
  scan = np.transpose(scan, axes=[3, 0, 1, 2])
  scan = normalize(torch.from_numpy(scan))
  return scan


def load_scans(args) -> (t.List[torch.Tensor], t.List[str]):
  """
  Load all scan files from input_dir

  Args:
    input_dir: input directory with .mat files

  Returns:
    scans: list of MRI scans
    scan_names: list of patients IDs in same order as scans
  """
  print(f'Loading scans from {args.input_dir}...')
  filenames = args.input_dir.glob('*.mat')
  scans, scan_names, original_shape = [], [], None
  for filename in filenames:
    # extract patient ID
    matches = re.match(r"(.+).mat$", filename.name)
    scan_name = matches.groups()[0]
    scan_names.append(scan_name)
    # load mat to tensor
    scan = load_mat(args, filename)
    scans.append(scan)
  print(f"Found {len(scans)} .mat files")
  return scans, scan_names


def save_mat(args, filename: t.Union[Path, str], scan: torch.Tensor):
  """
  Convert a patient scan into a .mat file.

  Args:
    filename: .mat filename to be stored
    scan: Tensor scan with format NCHW
  """
  scan = scan.permute((1, 2, 3, 0))
  assert scan.shape == args.original_shape
  flair = utils.to_numpy(scan[0, ...])
  t1 = utils.to_numpy(scan[1, ...])
  t2 = utils.to_numpy(scan[2, ...])
  savemat(filename, {"FLAIRarray": flair, "T1array": t1, "T2array": t2})


def plot_slice(args, scan_name: str, samples: dict):
  """
  Plot upsampled images as well as input and target image to file.

  Args:
    args
    scan_name: name of the scan
    samples: dictionary of inputs, outputs and target images
  """
  samples = {k: utils.to_numpy(v) for k, v in samples.items()}
  shape = samples['inputs'].shape
  for channel in range(shape[0]):
    utils.save_array_as_pdf(
        args.plot_dir / f"{scan_name}_{args.scan_types[channel]}_original.pdf",
        samples['inputs'][channel], args.dpi)
    utils.save_array_as_pdf(
        args.plot_dir / f"{scan_name}_{args.scan_types[channel]}_upsampled.pdf",
        samples['outputs'][channel], args.dpi)


def main(args):
  args.input_dir = Path(args.input_dir)
  assert args.input_dir.exists(), f"--input_dir {args.input_dir} not found."
  args.output_dir = Path(args.output_dir)
  assert args.output_dir.exists(), f"--output_dir {args.output_dir} not found."

  args.cuda = not args.no_cuda and torch.cuda.is_available()
  args.device = torch.device("cuda" if args.cuda else "cpu")

  load_args(args)

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  scans, scan_names = load_scans(args)

  model = get_model(args)
  model = utils.load_weights(args, model=model)

  model.eval()
  slice_upsampler = SliceUpsampler(args, model, stride=args.upsampler_stride)
  with torch.no_grad():
    for scan_name, scan in tqdm(zip(scan_names, scans),
                                desc='Scan',
                                total=len(scans),
                                disable=args.verbose == 0):
      scan = scan.to(args.device)
      output_scan = slice_upsampler.upsample(scan, verbose=args.verbose)
      save_mat(filename=args.upsample_dir / f'{scan_name}.mat', scan=scan)
      # plot middle slice
      mid_slice = len(output_scan) // 2
      plot_slice(args,
                 scan_name=scan_name,
                 samples={
                     'inputs': scan[mid_slice, ...],
                     'outputs': output_scan[mid_slice, ...]
                 })

  print(f'Upsampled {len(scans)} scans and stored them in {args.upsample_dir}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Predict scans')
  parser.add_argument('--input_dir',
                      type=str,
                      required=True,
                      help='Path to directory with scans to upsample.')
  parser.add_argument('--output_dir',
                      type=str,
                      required=True,
                      help='Path to directory with model checkpoint saved.')
  parser.add_argument('--upsample_dir',
                      type=str,
                      default=None,
                      help='Path to directory to store the upsampled scans. '
                      'Store upsample scans in input_dir/upsampled by default.')
  parser.add_argument('--upsampler_stride',
                      type=int,
                      default=None,
                      help='Stride size to use in upsampler. '
                      'By default, use the setting from the model checkpoint.')
  parser.add_argument('--batch_size',
                      type=int,
                      default=None,
                      help='Number of samples the network process at once. '
                      'By default, use the setting from the model checkpoint.')
  parser.add_argument('--no_cuda',
                      action='store_true',
                      help='Disable CUDA compute.')
  parser.add_argument('--dpi',
                      default=120,
                      type=int,
                      help='DPI of matplotlib plots. (Default: 120)')
  parser.add_argument('--verbose', default=1, choices=[0, 1, 2])
  params = parser.parse_args()
  main(params)
