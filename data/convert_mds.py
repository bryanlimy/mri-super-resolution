import os
import re
import scipy.io
import argparse
import numpy as np
import typing as t
from glob import glob
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree

from slowMRI.utils import utils
from slowMRI.utils.h5_io import write_ds


def load_mat(
    filename: t.Union[str, Path]) -> (np.ndarray, np.ndarray, np.ndarray):
  """ Load and transpose MatLab all scan types to have top-view format of
  (slice, height, width) """
  data = scipy.io.loadmat(filename)
  flair = np.transpose(data['FLAIRarray'].astype(np.float32), axes=(2, 0, 1))
  t1 = np.transpose(data['T1array'].astype(np.float32), axes=(2, 0, 1))
  t2 = np.transpose(data['T2array'].astype(np.float32), axes=(2, 0, 1))
  if np.isnan(flair).any():
    print(f'\t{os.path.basename(filename)}/FLAIR has NaN values')
    flair = np.nan_to_num(flair)
  if np.isnan(t1).any():
    print(f'\t{os.path.basename(filename)}/T1 has NaN values')
    t1 = np.nan_to_num(t1)
  if np.isnan(t2).any():
    print(f'\t{os.path.basename(filename)}/T2 has NaN values')
    t2 = np.nan_to_num(t2)
  return flair, t1, t2


def main(args):
  """
  Read and extract all .mat MATLAB file in args.input_dir.
  Save the extracted low and high resolution scans to lr_samples.h5 and
  hr_samples.h5 under args.output_dir respectively.

  Note: MDS has a top-view format of (height, width, slice) and we store scans
  in format for (slice, height, width)

  The args.input_dir should have the file structure of
  - args.input_dir
    - SR_002_NHSRI_V0.mat
    - SR_002_NHSRI_V1.mat
    - ...
  where V0 indicate low resolution scan and V1 indicate high resolution scan.

  The .h5 file has the format of
  - scan name
    - FLAIR
    - T1
    - T2
  """
  args.input_dir, args.output_dir = Path(args.input_dir), Path(args.output_dir)
  if not args.input_dir.exists():
    raise FileNotFoundError(f'{args.input_dir} not found.')
  if args.output_dir.exists() and args.overwrite:
    rmtree(args.output_dir)
  args.output_dir.mkdir(parents=True)

  lr_filename = args.output_dir / 'lr_samples.h5'
  hr_filename = args.output_dir / 'hr_samples.h5'

  # record the minimum and maximum values of the dataset
  ds_min, ds_max = np.inf, -np.inf
  scan_shape = None

  for filename in tqdm(glob(os.path.join(args.input_dir, '*.mat')),
                       disable=not args.verbose):
    # extract the name of the scan and the resolution of the scan
    # group 0 is the scan name, group 1 is the scan resolution
    matches = re.match(r"(.+)_V(\d).mat$", os.path.basename(filename))
    scan_name = matches.groups()[0]
    # load MATLAB file
    flair, t1, t2 = load_mat(filename)

    s_min = min(np.min(flair), np.min(t1), np.min(t2))
    s_max = max(np.max(flair), np.max(t1), np.max(t2))
    if s_min < ds_min:
      ds_min = s_min
    if s_max > ds_max:
      ds_max = s_max

    write_ds(lr_filename if matches.groups()[1] == '0' else hr_filename,
             content={
                 f'{scan_name}/FLAIR': flair,
                 f'{scan_name}/T1': t1,
                 f'{scan_name}/T2': t2
             })
    if scan_shape is None:
      scan_shape = flair.shape

  # save dataset information to args.output_dir/info.json
  utils.save_json(args.output_dir / 'info.json',
                  data={
                      'ds_name': 'mds',
                      'ds_min': ds_min,
                      'ds_max': ds_max,
                      'scan_types': ['FLAIR', 'T1', 'T2'],
                      'scan_shape': scan_shape
                  })

  if args.verbose:
    print(f'processed and saved scans to {lr_filename} and {hr_filename}.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', type=str, default='raw_data/MDS')
  parser.add_argument('--output_dir', type=str, default='dataset/MDS')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--verbose', type=int, default=1)

  main(parser.parse_args())
