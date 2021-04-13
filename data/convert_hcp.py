import re
import nibabel
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree

from slowMRI.utils import utils
from slowMRI.utils.h5_io import write_ds


def read_nii_as_npy(filepath: Path) -> np.ndarray:
  """
  Converts a .nii or .nii.gz file and returns it as numpy array with top-view
  format of (slice, height, width).
  """
  file = nibabel.load(filepath)
  scan = np.array(file.dataobj, dtype=np.float32)
  return np.transpose(scan, axes=[2, 0, 1])


def save_scans(args, scan_name: str, scans: dict):
  """ Save scans with scan_name to args.hr_filename """
  print(f'writing scan {scan_name}...')
  content = {}
  for scan_type, path in scans.items():
    scan = read_nii_as_npy(filepath=path)
    s_min, s_max = np.min(scan), np.max(scan)
    if s_min < args.ds_min:
      args.ds_min = s_min
    if s_max > args.ds_max:
      args.ds_max = s_max
    if args.scan_shape is None:
      args.scan_shape = scan.shape
    content[f'{scan_name}/{scan_type}'] = scan
  write_ds(args.hr_filename, content=content)


def main(args):
  """
  Converts the HCP data into a single hdf5 file with scan types 
  'MPR1', 'MPR2' and 'SPC1' or 'MPR1' if args.MRI1_only is specified.
  
  HCP has a top-view format of (height, width, slice) and we store scans in 
  format for (slice, height, width)
  
  Note: we filter patients that don't have all scan_types
  
  The output H5 has the format
  {
    scanID1/MPR1: scan,
    scanID1/MPR2: scan,
    scanID1/SPC1: scan,
    scanID2/MPR1: scan,
    ...
  }
  """
  args.input_dir, args.output_dir = Path(args.input_dir), Path(args.output_dir)
  if not args.input_dir.exists():
    raise FileNotFoundError(f'No such directory: {args.input_dir}')
  if args.output_dir.exists() and args.overwrite:
    rmtree(args.output_dir)
  args.output_dir.mkdir(parents=True)

  if args.MPR1_only:
    scan_types = ['MPR1']
  else:
    scan_types = ['MPR1', 'MPR2', 'SPC1']

  args.hr_filename = args.output_dir / 'hr_samples.h5'

  # record the minimum and maximum values of the dataset
  args.ds_min, args.ds_max = np.inf, -np.inf
  args.scan_shape = None

  scan_pairs = {}
  for scan_type in scan_types:
    for path in args.input_dir.rglob(f'*{scan_type}.nii.gz'):
      matches = re.match(r"^(\d*)_", path.name)
      scan_name = matches.groups()[0]
      if scan_name not in scan_pairs:
        scan_pairs[scan_name] = {}
      scan_pairs[scan_name][scan_type] = path
      # save scans when all corresponding scan types have been found
      if len(scan_pairs[scan_name]) == len(scan_types):
        save_scans(args, scan_name, scan_pairs[scan_name])
        del scan_pairs[scan_name]

  # save dataset information to args.output_dir/info.json
  utils.save_json(args.output_dir / 'info.json',
                  data={
                      'ds_name': 'hcp',
                      'ds_min': args.ds_min,
                      'ds_max': args.ds_max,
                      'scan_types': scan_types,
                      'scan_shape': args.scan_shape
                  })

  if args.verbose:
    print(f'processed and saved scans to {args.hr_filename}.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', type=str, default='raw_data/HCP')
  parser.add_argument('--output_dir', type=str, default='dataset/HCP')
  parser.add_argument('--MPR1_only', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--verbose', type=int, default=1)
  main(parser.parse_args())
