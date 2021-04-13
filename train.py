import torch
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler

from slowMRI.utils import utils
from slowMRI.metrics import metrics
from slowMRI.critic.critic import Critic
from slowMRI.models.registry import get_model
from slowMRI.utils.tensorboard import Summary
from slowMRI.data_loader.data_loader import get_loaders
from slowMRI.data_loader.data_handling import SliceUpsampler


def step(args,
         model,
         inputs,
         targets=None,
         optimizer=None,
         loss_function=None,
         scaler=None,
         critic=None,
         training: bool = False):
  """ batch inputs and targets into args.batch_size """
  outputs = torch.zeros_like(targets,
                             dtype=targets.dtype,
                             device=targets.device,
                             requires_grad=False)

  if training:
    model.train()
  else:
    model.eval()

  results = {}
  for i in range(0, inputs.shape[0], args.batch_size):
    x = inputs[i:i + args.batch_size]
    y = None if targets is None else targets[i:i + args.batch_size]

    loss = torch.tensor(0.0, requires_grad=True)
    with autocast(enabled=args.mixed_precision):
      logits = model(x)
      y_pred = F.sigmoid(logits) if args.output_logits else logits
      if loss_function is not None:
        loss = loss_function(logits, y)
        utils.update_dict(results, {'Loss': loss.detach()})

    outputs[i:i + args.batch_size] = y_pred.detach()

    critic_score = torch.tensor(0.0, requires_grad=True)
    if critic is not None:
      if y is not None:
        if training:
          critic_results = critic.train(y.detach(), y_pred.detach())
        else:
          critic_results = critic.validate(y.detach(), y_pred.detach())
        utils.update_dict(results, critic_results)
      if args.critic_loss > 0:
        critic_score = critic.predict(y_pred)

    total_loss = loss + args.critic_loss * (1 - critic_score)
    utils.update_dict(results, {'Loss/total_loss': total_loss})

    if optimizer is not None:
      optimizer.zero_grad()
      scaler.scale(total_loss).backward()
      scaler.step(optimizer)
      scaler.update()

  if loss_function is None:
    return outputs
  else:
    results = {k: torch.stack(v).mean() for k, v in results.items()}
    return outputs, results


def train(args,
          model,
          data,
          optimizer,
          loss_function,
          scaler,
          summary,
          epoch: int = 0,
          critic=None) -> dict:
  results = {}
  for inputs, targets in tqdm(data, desc='Train'):
    inputs = torch.flatten(inputs, end_dim=1)
    targets = torch.flatten(targets, end_dim=1)
    inputs, targets = inputs.to(args.device), targets.to(args.device)
    outputs, step_results = step(args,
                                 model,
                                 inputs,
                                 targets=targets,
                                 optimizer=optimizer,
                                 loss_function=loss_function,
                                 scaler=scaler,
                                 critic=critic,
                                 training=True)
    utils.update_dict(results, step_results)
    utils.update_dict(results, {'NMSE': metrics.nmse(outputs, targets)})
    args.global_step += 1
    if args.dry_run:
      break

  for key, value in results.items():
    results[key] = torch.stack(value).mean()
    summary.scalar(key, results[key], step=epoch, mode=0)

  return results


def validate(args,
             model,
             data,
             loss_function,
             summary,
             epoch: int = 0,
             critic=None) -> dict:
  samples, results = None, {}
  with torch.no_grad():
    for inputs, targets in tqdm(data, desc='Validation'):
      inputs = torch.flatten(inputs, end_dim=1)
      targets = torch.flatten(targets, end_dim=1)
      inputs, targets = inputs.to(args.device), targets.to(args.device)
      outputs, step_results = step(args,
                                   model,
                                   inputs,
                                   targets=targets,
                                   loss_function=loss_function,
                                   critic=critic,
                                   training=False)
      utils.update_dict(results, step_results)
      utils.update_dict(
          results, {
              'MAE': metrics.mae(outputs, targets),
              'NMSE': metrics.nmse(outputs, targets),
              'PSNR': metrics.psnr(outputs, targets),
              'SSIM': metrics.ssim(outputs, targets)
          })
      # store samples to plot
      if samples is None:
        samples = {'inputs': inputs, 'targets': targets, 'outputs': outputs}

  for key, value in results.items():
    results[key] = torch.stack(value).mean()
    summary.scalar(key, results[key], step=epoch, mode=1)

  if (epoch % 10 == 0) or (epoch + 1 == args.epochs):
    summary.plot_side_by_side('side_by_side',
                              samples,
                              random_patches=True,
                              step=epoch,
                              mode=1)
    summary.plot_difference_maps('diff_maps',
                                 samples,
                                 random_patches=True,
                                 step=epoch,
                                 mode=1)

  return results


def test(args,
         model,
         data,
         loss_function,
         summary,
         epoch: int = 0,
         critic=None) -> dict:
  model.eval()

  slice_upsampler = SliceUpsampler(args,
                                   model,
                                   stride=args.upsampler_stride,
                                   loss_function=loss_function,
                                   critic=critic)

  results = {}
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(tqdm(data, desc='Test')):
      # shape (batch, slices, channels, height, width)
      inputs, targets = inputs.to(args.device), targets.to(args.device)
      # select middle slice to up-sample
      samples, result = slice_upsampler.upsample_batch(
          lr_scans=inputs, hr_scans=targets, slice_idx=inputs.shape[1] // 2)
      utils.update_dict(results, result)

      summary.plot_side_by_side(f'stitched/batch_#{batch_idx+1:02d}',
                                samples,
                                random_patches=False,
                                step=epoch,
                                mode=2)

  for key, value in results.items():
    results[key] = torch.stack(value).mean()
    summary.scalar(key, results[key], step=epoch, mode=2)

  print(f'Test\t\tLoss: {results["Loss"]:.04f}\t'
        f'MAE: {results["MAE"]:.4f}\t'
        f'PSNR: {results["PSNR"]:.02f}\t'
        f'SSIM: {results["SSIM"]:.04f}\n')
  utils.save_csv(filename=args.output_dir / 'test_results.csv', data=results)
  return results


def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  # delete args.output_dir if the flag is set and the directory exists
  if args.clear_output_dir and args.output_dir.exists():
    rmtree(args.output_dir)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  args.checkpoint_dir = args.output_dir / 'checkpoints'
  args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

  args.cuda = not args.no_cuda and torch.cuda.is_available()
  args.device = torch.device("cuda" if args.cuda else "cpu")

  train_loader, val_loader, test_loader = get_loaders(args)

  summary = Summary(args)

  scaler = GradScaler(enabled=args.mixed_precision)
  args.output_logits = (args.loss in ['bce', 'binarycrossentropy'] and
                        args.model != 'identity')

  model = get_model(args, summary)
  if args.weights_dir is not None:
    model = utils.load_weights(args, model)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  scheduler = StepLR(optimizer,
                     step_size=5,
                     gamma=args.gamma,
                     verbose=args.verbose == 2)
  loss_function = utils.get_loss_function(name=args.loss)

  critic = None if args.critic is None else Critic(args, summary=summary)

  utils.save_args(args)

  args.global_step = 0
  for epoch in range(args.epochs):
    print(f'Epoch {epoch + 1:03d}/{args.epochs:03d}')
    start = time()
    train_results = train(args,
                          model=model,
                          data=train_loader,
                          optimizer=optimizer,
                          loss_function=loss_function,
                          scaler=scaler,
                          summary=summary,
                          epoch=epoch,
                          critic=critic)
    val_results = validate(args,
                           model=model,
                           data=val_loader,
                           loss_function=loss_function,
                           summary=summary,
                           epoch=epoch,
                           critic=critic)
    end = time()

    scheduler.step()

    summary.scalar('elapse', end - start, step=epoch, mode=0)
    summary.scalar('lr', scheduler.get_last_lr()[0], step=epoch, mode=0)
    summary.scalar('gradient_scale', scaler.get_scale(), step=epoch, mode=0)

    print(f'Train\t\tLoss: {train_results["Loss"]:.04f}\n'
          f'Validation\tLoss: {val_results["Loss"]:.04f}\t'
          f'MAE: {val_results["MAE"]:.04f}\t'
          f'PSNR: {val_results["PSNR"]:.02f}\t'
          f'SSIM: {val_results["SSIM"]:.04f}\n')

  utils.save_model(args, model)

  test(args,
       model=model,
       data=test_loader,
       loss_function=loss_function,
       summary=summary,
       epoch=args.epochs,
       critic=critic)

  summary.close()


if __name__ == '__main__':
  # Training settings
  parser = argparse.ArgumentParser(description='Bryan GAN')
  parser.add_argument('--input_dir', type=str, help='path to dataset')
  parser.add_argument('--output_dir',
                      type=str,
                      default='runs',
                      help='directory to write TensorBoard summary.')
  parser.add_argument('--model',
                      type=str,
                      default='simple_cnn',
                      help='model to use')
  parser.add_argument('--critic',
                      type=str,
                      default=None,
                      help='adversarial loss to use.')
  parser.add_argument('--critic_steps',
                      type=int,
                      default=1,
                      help='number of update steps for critic per global step')
  parser.add_argument('--critic_loss',
                      type=float,
                      default=0.0,
                      help='critic loss coefficient to the training objective')
  parser.add_argument('--weights_dir',
                      type=str,
                      default=None,
                      help='path to directory to load model weights from')
  parser.add_argument('--num_filters',
                      type=int,
                      default=64,
                      help='number of filters or hidden units')
  parser.add_argument('--normalization', type=str, default='instancenorm')
  parser.add_argument('--activation', type=str, default='leakyrelu')
  parser.add_argument('--batch_size',
                      type=int,
                      default=32,
                      metavar='N',
                      help='input batch size for training (default: 4)')
  parser.add_argument('--epochs',
                      type=int,
                      default=100,
                      metavar='N',
                      help='number of epochs to train_epoch (default: 100)')
  parser.add_argument('--loss',
                      type=str,
                      default='bce',
                      help='loss function to use')
  parser.add_argument('--lr',
                      type=float,
                      default=0.001,
                      metavar='LR',
                      help='learning rate (default: 0.001)')
  parser.add_argument('--gamma',
                      type=float,
                      default=0.7,
                      metavar='M',
                      help='Learning rate step gamma (default: 0.7)')
  parser.add_argument('--no_cuda',
                      action='store_true',
                      default=False,
                      help='disables CUDA training')
  parser.add_argument('--dry_run',
                      action='store_true',
                      default=False,
                      help='quickly check a single pass')
  parser.add_argument('--seed',
                      type=int,
                      default=42,
                      metavar='S',
                      help='random seed (default: 42)')
  parser.add_argument('--patch_dim',
                      type=int,
                      default=64,
                      help='patch dimension (default: 64)')
  parser.add_argument('--upsampler_stride',
                      type=int,
                      default=1,
                      help='Upsampler stride size (default: 1)')
  parser.add_argument('--n_patches',
                      type=int,
                      default=500,
                      help='number of patches to generate per sample')
  parser.add_argument('--merge_scan_type',
                      action='store_true',
                      help='treat FLAIR, T1 and T2 as an image with 3 channels')
  parser.add_argument('--scan_input',
                      action='store_true',
                      help='feed entire scan to model instead of patches')
  parser.add_argument('--save_plots',
                      action='store_true',
                      help='save TensorBoard figures and images to disk.')
  parser.add_argument('--save_upsampled_image',
                      action='store_true',
                      help='saves upsampled images to disk. This is only '
                      'activated if save_plots is True.')
  parser.add_argument('--dpi',
                      type=int,
                      default=120,
                      help='DPI of matplotlib figures')
  parser.add_argument('--clear_output_dir',
                      action='store_true',
                      help='overwrite existing output directory')
  parser.add_argument('--mixed_precision',
                      action='store_true',
                      help='use mixed precision compute')
  parser.add_argument('--verbose', choices=[0, 1, 2], default=1, type=int)
  params = parser.parse_args()
  # create output directory
  params.output_dir = Path(params.output_dir)
  params.output_dir.mkdir(parents=True, exist_ok=True)

  main(params)
