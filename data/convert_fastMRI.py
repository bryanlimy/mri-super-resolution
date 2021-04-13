import os
import h5py
import argparse
import numpy as np
from glob import glob
from multiprocessing import Pool

from slowMRI.utils.h5_io import write_ds


def rss(x: np.array, axis: int = 0) -> np.array:
  """
  Compute the Root Sum of Squares (RSS).
  RSS is computed assuming that axis is the coil dimension.
  Args:
    data: np.ndarray, the input array
    axis: int, the axis along which to apply the RSS transform
  Returns:
    RSS value
  """
  return np.sqrt(np.sum(np.square(x), axis))


def multicoil_to_monocoil(hparams, path: str):
  """
  Convert multi-coil image in path to mono-coil image. Store the new image 
  and its metadata to args.output_dir with the same filename.
  Args:
    args
    path: str, path to the .h5 file
  """
  try:
    if hparams.verbose:
      print(f'processing {path}...')
    data = {}
    with h5py.File(path, mode='r') as file:
      for key in file.keys():
        data[key] = file[key][()]
        if key == 'kspace':
          data[key] = rss(data[key])
    write_ds(os.path.join(hparams.output_dir, os.path.basename(path)), data)
  except OSError as error:
    print(f'{path} Error: {error}')


def main(args):
  if not os.path.exists(args.input_dir):
    raise FileNotFoundError(f'No such directory: {args.input_dir}')
  if args.output_dir is None:
    # if output_dir is not specified, store updated images to input_dir/convert
    args.output_dir = os.path.join(args.input_dir, 'convert')
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  paths = glob(os.path.join(args.input_dir, '*.h5'))

  # create args.num_processors pool to convert paths
  pool = Pool(args.num_processors)
  pool.starmap(multicoil_to_monocoil, [(args, path) for path in paths])
  pool.close()

  print(f'converted {len(paths)} images to {args.output_dir}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir',
                      type=str,
                      default='data/fastMRI/multicoil_test')
  parser.add_argument('--output_dir', type=str, default=None)
  parser.add_argument('--verbose', type=int, default=1)
  parser.add_argument('--num_processors', default=6, type=int)
  main(parser.parse_args())
