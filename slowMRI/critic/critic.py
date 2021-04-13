import torch
import typing as t
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from slowMRI.utils import utils
from slowMRI.models import torchsummary

_CRITICS = dict()


def register(name):
  """ Note: update __init__.py for additional models """

  def add_to_dict(fn):
    global _CRITICS
    _CRITICS[name] = fn
    return fn

  return add_to_dict


class Critic:
  """ Adversarial loss for targets and up-sampled images """

  def __init__(self,
               args,
               lr: float = 0.0002,
               betas: tuple = (0.5, 0.999),
               summary=None):
    """
    Args:
      args
      lr: learning rate for the optimizer
      betas: betas for optimizer
    """
    assert args.critic in _CRITICS, f'Critic {args.critic} not found.'

    self.lr = lr
    self.betas = betas
    self.critic_steps = args.critic_steps
    self.mixed_precision = args.mixed_precision

    self.critic = _CRITICS[args.critic](args)
    self.critic.to(args.device)

    self.output_shape = self.critic(
        torch.rand(2, *args.input_shape, device=args.device)).shape[1:]

    self.optimizer = optim.Adam(self.critic.parameters(),
                                lr=self.lr,
                                betas=self.betas)
    self.scaler = GradScaler(enabled=self.mixed_precision)

    self.loss_function = F.binary_cross_entropy_with_logits

    if summary is not None:
      summary_readout, trainable_parameters = torchsummary.summary(
          self.critic, input_size=args.input_shape, device=args.device)
      with open(args.output_dir / 'critic.txt', 'w') as file:
        file.write(summary_readout)
      summary.scalar('critic/trainable_parameters', trainable_parameters)
      if args.verbose == 2:
        print(summary_readout)

  def critic_loss(self, discriminate_real, discriminate_fake):
    real_loss = self.loss_function(discriminate_real,
                                   torch.ones_like(discriminate_real))
    fake_loss = self.loss_function(discriminate_fake,
                                   torch.zeros_like(discriminate_fake))
    return (real_loss + fake_loss) / 2

  def train(self, real: torch.Tensor, fake: torch.Tensor) -> dict:
    """ Train critic model on real and fake samples for self.critic_steps
    Args:
      real: real samples
      fake: fake samples
    Returns:
      results: dictionary containing the average loss and the average outputs
              of the critic on real and fake samples
    """
    assert real.shape == fake.shape
    self.critic.train()

    results = {}
    for _ in range(self.critic_steps):
      self.optimizer.zero_grad()
      with autocast(enabled=self.mixed_precision):
        discriminate_real = self.critic(real)
        discriminate_fake = self.critic(fake)

        loss = self.critic_loss(discriminate_real, discriminate_fake)

        discriminate_real = F.sigmoid(discriminate_real)
        discriminate_fake = F.sigmoid(discriminate_fake)

      self.scaler.scale(loss).backward()
      self.scaler.step(self.optimizer)
      self.scaler.update()

      utils.update_dict(
          source=results,
          target={
              'critic/loss': loss,
              'critic/discriminate_real': discriminate_real.mean(),
              'critic/discriminate_fake': discriminate_fake.mean()
          })

    return {k: torch.stack(v).mean() for k, v in results.items()}

  def validate(self, real: torch.Tensor, fake: torch.Tensor) -> dict:
    """ Validate critic model on real and fake samples
    Args:
      real: real samples
      fake: fake samples
    Returns:
      results: dictionary containing the average loss and the average outputs
              of the critic on real and fake samples
    """
    assert real.shape == fake.shape
    self.critic.eval()

    with autocast(enabled=self.mixed_precision), torch.no_grad():
      discriminate_real = self.critic(real)
      discriminate_fake = self.critic(fake)

      loss = self.critic_loss(discriminate_real, discriminate_fake)

      discriminate_real = F.sigmoid(discriminate_real)
      discriminate_fake = F.sigmoid(discriminate_fake)

    return {
        'critic/loss': loss,
        'critic/discriminate_real': discriminate_real.mean(),
        'critic/discriminate_fake': discriminate_fake.mean()
    }

  def predict(self, x: torch.Tensor, return_mean: bool = True) -> torch.Tensor:
    """ Discriminate sample
    Args:
      x: sample to discriminate
      return_mean: return average of the output
    Returns:
      discriminate: the output of the critic
    """
    with autocast(enabled=self.mixed_precision):
      discriminate = self.critic(x)
      discriminate = F.sigmoid(discriminate)
    return discriminate.mean() if return_mean else discriminate
