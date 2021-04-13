import io
import torch
import platform
import warnings
import numpy as np
import typing as t
from pathlib import Path
from typing import Union
from torch.utils.tensorboard import SummaryWriter

import matplotlib
if platform.system() == 'Darwin':
  matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('seaborn-deep')
warnings.simplefilter("ignore", UserWarning)

from slowMRI.utils import utils


class Summary():
  """
  TensorBoard Summary class to log model training performance and record 
  generated samples
  
  By default, training summary are stored under hparams.output_dir and 
  validation summary are stored at hparams.output_dir/validation
  """

  def __init__(self, args):
    self.train_writer = SummaryWriter(args.output_dir)
    self.val_writer = SummaryWriter(args.output_dir / 'validation')
    self.test_writer = SummaryWriter(args.output_dir / 'test')
    self.writers = [self.train_writer, self.val_writer, self.test_writer]

    self.dpi = args.dpi
    # save plots and figures to disk
    self.save_plots = args.save_plots
    if self.save_plots:
      self.plot_dir = args.output_dir / 'plots'
      self.plot_dir.mkdir(parents=True, exist_ok=True)
      self.format = 'pdf'

    self.scan_types = args.scan_types
    self.random_patches = None

  def get_writer(self, mode: int):
    """ get writer for the specified mode
    Args:
      mode: int, 0 - train, 1 - validation, 2 - test
    Returns:
      writer
    """
    assert mode in [0, 1, 2], f'No writer with mode {mode}'
    return self.writers[mode]

  def patch_indexes(self, num_samples: int, random: bool):
    if random and self.random_patches is None:
      # select 7 random patches to plot
      self.random_patches = np.random.choice(num_samples, size=7, replace=False)
    return self.random_patches if random else range(num_samples)

  def save_figure(self, filename: Union[str, Path], figure: t.Type[plt.Figure]):
    """ Save matplotlib figure to self.plot_dir/filename """
    fname = self.plot_dir / f'{filename}.{self.format}'
    if not fname.parent.exists():
      fname.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(fname), dpi=self.dpi, format=self.format)

  def flush(self):
    """ Flush all writers """
    self.train_writer.flush()
    self.val_writer.flush()

  def close(self):
    """ Close all writers """
    self.flush()
    self.train_writer.close()
    self.val_writer.close()

  def scalar(self, tag, value, step: int = 0, mode: int = 0):
    """ Write scalar value to summary
    Args:
      tag: data identifier
      value: scalar value
      step: global step value to record
      mode: mode of the data. 0 - train, 1 - validation, 2 - test
    """
    writer = self.get_writer(mode)
    writer.add_scalar(tag, value, global_step=step)
    writer.flush()

  def histogram(self, tag, value, step=0, mode: int = 0):
    """ Write histogram to summary
    Args:
      tag: data identifier
      value: values to build histogram
      step: global step value to record
      mode: mode of the data. 0 - train, 1 - validation, 2 - test
    """
    writer = self.get_writer(mode)
    writer.add_histogram(tag, values=value, global_step=step)

  def image(self, tag, value, step=0, dataformats: str = 'CHW', mode: int = 0):
    """ Write image to summary
    
    Note: TensorBoard only accept images with value within [0, 1)
    
    Args:
      tag: data identifier
      value: image array dataformats
      dataformats: image data format, e.g. (CHW) or (NCHW)
      step: global step value to record
      mode: mode of the data. 0 - train, 1 - validation, 2 - test
    """
    assert len(value.shape) in [3, 4] and len(value.shape) == len(dataformats)
    writer = self.get_writer(mode)
    if len(dataformats) == 3:
      writer.add_image(tag, value, global_step=step, dataformats=dataformats)
    else:
      writer.add_images(tag, value, global_step=step, dataformats=dataformats)

  def figure(self, tag, figure, step=0, close=True, mode: int = 0):
    """ Write matplotlib figure to summary
    Args:
      tag: data identifier
      figure: matplotlib figure or a list of figures
      step: global step value to record
      close: flag to close figure
      mode: mode of the data. 0 - train, 1 - validation, 2 - test
    """
    plt.tight_layout()
    if self.save_plots:
      self.save_figure(f'{tag}/step{step:03d}', figure=figure)
    figure.canvas.draw()
    w, h = figure.canvas.get_width_height()
    # matplotlib canvas store image in (HWC) format
    image = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = np.reshape(image, newshape=(h, w, 3))
    self.image(tag, image, step=step, dataformats='HWC', mode=mode)
    if close:
      plt.close()

  def plot_patches(self, tag, values, step=0, mode: int = 0):
    """ Plot randomly selected patches to TensorBoard """
    for i in self.patch_indexes(values.shape[0], random=True):
      self.image(f'{tag}/patch_#{i:03d}',
                 values[i, ...],
                 step=step,
                 dataformats='CHW',
                 mode=mode)

  def plot_side_by_side(self,
                        tag: str,
                        samples: dict,
                        random_patches: bool,
                        step: int = 0,
                        mode: int = 0):
    """ Plot side by side comparison in a grid for each channel
    Args:
      tag: name of the plot in TensorBoard
      samples: inputs, targets and outputs in shape (NCHW)
        dictionary in {'inputs': tensor, 'targets': tensor, 'outputs': tensor}
      random_patches: plot random patches or all slices
      step: the current step or epoch
      mode: mode of the data. 0 - train, 1 - validation, 2 - test
    """
    samples = {k: utils.to_numpy(v) for k, v in samples.items()}
    shape = samples['inputs'].shape
    figsize = (8, 3) if shape[1] == 1 else (7.5, 2.5 * shape[1])

    # plot random patches if random_patches, otherwise plot all slices
    for patch in self.patch_indexes(shape[0], random=random_patches):
      figure, axes = plt.subplots(nrows=shape[1],
                                  ncols=3,
                                  figsize=figsize,
                                  squeeze=False,
                                  dpi=self.dpi)
      for channel in range(shape[1]):
        # extract images for current channel:
        input_image = samples['inputs'][patch, channel]
        output_image = samples['outputs'][patch, channel]
        target_image = samples['targets'][patch, channel]
        # if save plots in test mode:
        if self.save_plots and mode == 2:
          labels = ["input", "generated", "target"]
          images = [input_image, output_image, target_image]
          for label, img in zip(labels, images):
            clean_tag = tag.split("/")[1]
            outfile_name = self.plot_dir / f"upsampled_{clean_tag}_patch_" \
                                           f"{patch:03d}_" \
                                           f"{self.scan_types[channel]}_" \
                                           f"{label}.pdf"
            utils.save_array_as_pdf(outfile_name, img, self.dpi)

        kwargs = {'cmap': 'gray', 'interpolation': 'none'}
        axes[channel, 0].imshow(input_image, **kwargs)
        axes[channel, 1].imshow(output_image, **kwargs)
        axes[channel, 2].imshow(target_image, **kwargs)
        # show scan type if scan is multi-channel
        if shape[1] > 1:
          axes[channel, 0].set_ylabel(self.scan_types[channel])
        # set title for top row only
        if channel == 0:
          axes[channel, 0].set_title('input')
          axes[channel, 1].set_title('generated')
          axes[channel, 2].set_title('target')

      plt.setp(axes, xticks=[], yticks=[])
      figure.subplots_adjust(wspace=0.05, hspace=0.05)
      self.figure(f'{tag}/patch_#{patch:03d}', figure, step=step, mode=mode)

  def plot_difference_maps(self,
                           tag,
                           samples,
                           random_patches: bool,
                           step: int = 0,
                           mode: int = 0):
    """ Plot difference maps in a grid for each channel
    Args:
      tag: name of the plot in TensorBoard
      samples: inputs, targets and outputs in shape (NCHW)
        dictionary in {'inputs': tensor, 'targets': tensor, 'outputs': tensor}
      random_patches: plot random patches or all slices
      step: the current step or epoch
      mode: mode of the data. 0 - train, 1 - validation, 2 - test
    """
    samples = {k: utils.to_numpy(v) for k, v in samples.items()}
    shape = samples['inputs'].shape
    figsize = (7.5, 2.5) if shape[1] == 1 else (8, 2.5 * shape[1])

    # calculate difference map per batch
    targets_inputs = samples['targets'] - samples['inputs']
    targets_outputs = samples['targets'] - samples['outputs']
    inputs_outputs = samples['inputs'] - samples['outputs']

    # get the minimum and maximum for each channel difference maps
    vmin = np.min([targets_inputs, targets_outputs, inputs_outputs],
                  axis=(0, -2, -1))
    vmax = np.max([targets_inputs, targets_outputs, inputs_outputs],
                  axis=(0, -2, -1))

    # plot random patches if random_patches, otherwise plot all slices
    for patch in self.patch_indexes(shape[0], random=random_patches):
      figure, axes = plt.subplots(nrows=shape[1],
                                  ncols=4,
                                  gridspec_kw={'width_ratios': [1, 1, 1, 0.05]},
                                  figsize=figsize,
                                  squeeze=False,
                                  dpi=self.dpi)
      for channel in range(shape[1]):
        absolute_max = max(np.abs(vmin[patch, channel]),
                           np.abs(vmax[patch, channel])) + 0.05
        kwargs = {
            'cmap': 'bwr',
            'interpolation': 'none',
            'vmin': -absolute_max,
            'vmax': absolute_max
        }
        axes[channel, 0].imshow(targets_inputs[patch, channel], **kwargs)
        axes[channel, 1].imshow(targets_outputs[patch, channel], **kwargs)
        im = axes[channel, 2].imshow(inputs_outputs[patch, channel], **kwargs)
        if shape[1] > 1:
          axes[channel, 0].set_ylabel(self.scan_types[channel])
        if channel == 0:
          axes[channel, 0].set_title('target - input')
          axes[channel, 1].set_title('target - generated')
          axes[channel, 2].set_title('input - generated')
        figure.colorbar(im, cax=axes[channel, 3])

      plt.setp(axes, xticks=[], yticks=[])
      figure.subplots_adjust(wspace=0.01, hspace=0.01)
      self.figure(f'{tag}/patch_#{patch:03d}', figure, step=step, mode=mode)
