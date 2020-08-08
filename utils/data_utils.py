import os
from math import log

import torch
from torch import distributions
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms

from args import args


class DataInfo():
    def __init__(self, name, channel, size):
        """Instantiates a DataInfo.

        Args:
            name: name of dataset.
            channel: number of image channels.
            size: height and width of an image.
        """
        self.name = name
        self.channel = channel
        self.size = size


def load_dataset():
    """Load dataset.

    Returns:
        a torch dataset and its associated information.
    """
    if args.data == 'celeba32':
        data_info = DataInfo(args.data, 3, 32)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(os.path.join(args.data_path,
                                                      'celeba32'),
                                         transform=transform)
        [train_split, val_split] = data.random_split(train_set,
                                                     [180000, 22599])

    elif args.data == 'celeba64':
        data_info = DataInfo(args.data, 3, 64)
        transform = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        # train_set = datasets.CelebA(os.path.join(args.data_path, 'celeba'),
        #                             download=True,
        #                             transform=transform)
        train_set = datasets.ImageFolder(os.path.join(args.data_path,
                                                      'celeba'),
                                         transform=transform)
        [train_split, val_split] = data.random_split(train_set,
                                                     [180000, 22599])

    elif args.data == 'mnist32':
        data_info = DataInfo(args.data, 1, 32)
        transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
        ])
        train_set = datasets.MNIST(os.path.join(args.data_path, 'mnist'),
                                   train=True,
                                   download=True,
                                   transform=transform)
        [train_split, val_split] = data.random_split(train_set, [50000, 10000])

    else:
        raise ValueError('Unknown data: {}'.format(args.data))

    assert data_info.channel == args.nchannels
    assert data_info.size == args.L

    return train_split, val_split, data_info


def logit_transform(x, constraint=0.9, reverse=False):
    """Transforms data from [0, 1] into unbounded space.

    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).

    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    """
    if reverse:
        x = 1 / (torch.exp(-x) + 1)    # [0.05, 0.95]
        # x *= 2    # [0.1, 1.9]
        # x -= 1    # [-0.9, 0.9]
        # x /= constraint    # [-1, 1]
        # x += 1    # [0, 2]
        # x /= 2    # [0, 1]
        return x, 0
    else:
        B, C, H, W = x.shape

        # dequantization
        noise = distributions.Uniform(0, 1).sample((B, C, H, W)).to(x.device)
        x = (x * 255 + noise) / 256

        # restrict data
        x *= 2    # [0, 2]
        x -= 1    # [-1, 1]
        x *= constraint    # [-0.9, 0.9]
        x += 1    # [0.1, 1.9]
        x /= 2    # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1 - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(log(constraint) - log(1 - constraint))
        log_diag_J = (F.softplus(logit_x) + F.softplus(-logit_x) -
                      F.softplus(-pre_logit_scale))

        return logit_x, log_diag_J.sum(dim=(1, 2, 3))
