import h5py
import numpy as np
import typing as t
from pathlib import Path


def write_ds(filename: t.Union[str, Path], content: dict):
  """
  Writes a dictionary to a dataset in a hdf5 file. Supports groups.

  Args:
    filename: name of the hdf5 file (including extension if applicable)
    content: dictionary with name-data pairs
  """
  with h5py.File(filename, mode='a') as file:
    for k, v in content.items():
      if k not in file:
        shape, dtype = None, None
        if type(v) == np.ndarray:
          shape, dtype = v.shape, v.dtype
        file.create_dataset(k, shape=shape, dtype=dtype, data=v)
      else:
        # we can add appending/overwriting to a dataset (i.e. a single 3d scan)
        # later if needed, but for now we ensure this doesn't happen accidentally
        raise ValueError(f'key {k} already in {filename}.')


def read_ds(filename: t.Union[str, Path], name: t.Union[str, t.List[str]]):
  """
  Reads content from a dataset in a hdf5 file.

  Args:
    filename: path to the H5 file
    name: name or list of names to read
  Returns:
    data: content or list of content with name from dataset
  """
  with h5py.File(filename, mode='r') as file:
    data = file[name][()] if type(name) is str else [file[n][()] for n in name]
  return data
