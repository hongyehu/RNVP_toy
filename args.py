import argparse
import os
from math import log2

import torch

parser = argparse.ArgumentParser()

group = parser.add_argument_group('dataset parameters')
group.add_argument('--data',
                   type=str,
                   default='pinwheel',
                   choices=['pinwheel'],
                   help='dataset name')

group.add_argument('--nvars', type=int, default=2, help='edge length of images')

group = parser.add_argument_group('network parameters')
group.add_argument('--prior',
                   type=str,
                   default='laplace',
                   choices=['gaussian', 'cauchy', 'laplace'],
                   help='prior of latent variables ')


group.add_argument('--nlayers',
                   type=int,
                   default=40,
                   help='number of NVP layers')
group.add_argument('--nresblocks',
                   type=int,
                   default=4,
                   help='number of residual blocks')
group.add_argument('--nmlp',
                   type=int,
                   default=4,
                   help='number of MLP hidden layers in an residual block')
group.add_argument('--nhidden',
                   type=int,
                   default=128,
                   help='width of MLP hidden layers')
group.add_argument('--dtype',
                   type=str,
                   default='float32',
                   choices=['float32', 'float64'],
                   help='dtype')

group = parser.add_argument_group('optimizer parameters')
group.add_argument('--batch_size', type=int, default=256, help='batch size')
group.add_argument('--lr', type=float, default=1e-3, help='learning rate')
group.add_argument('--weight_decay',
                   type=float,
                   default=5e-5,
                   help='weight decay')
group.add_argument('--epoch', type=int, default=500, help='number of epoches')
group.add_argument('--clip_grad',
                   type=float,
                   default=1,
                   help='global norm to clip gradients, 0 for disabled')

group = parser.add_argument_group('system parameters')
group.add_argument('--no_stdout',
                   action='store_true',
                   help='do not print log to stdout, for better performance')
group.add_argument('--print_step',
                   type=int,
                   default=1,
                   help='number of batches to print log, 0 for disabled')
group.add_argument(
    '--save_epoch',
    type=int,
    default=1,
    help='number of epochs to save network weights, 0 for disabled')
group.add_argument(
    '--keep_epoch',
    type=int,
    default=50,
    help='number of epochs to keep saved network weights, 0 for disabled')
group.add_argument('--plot_epoch',
                   type=int,
                   default=1,
                   help='number of epochs to plot samples, 0 for disabled')
group.add_argument('--cuda',
                   type=str,
                   default='',
                   help='IDs of GPUs to use, empty for disabled')
group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs')
group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='./saved_model',
    help='directory prefix for output, empty for disabled')

args = parser.parse_args()

if args.dtype == 'float32':
    torch.set_default_tensor_type(torch.FloatTensor)
elif args.dtype == 'float64':
    torch.set_default_tensor_type(torch.DoubleTensor)
else:
    raise ValueError('Unknown dtype: {}'.format(args.dtype))

if args.cuda:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


def get_net_name():
    net_name = ''
    if args.prior != 'gaussian':
        net_name += '{prior}_'
    net_name += 'nl{nlayers}_nr{nresblocks}_nm{nmlp}_nh{nhidden}'
    net_name = net_name.format(**vars(args))
    return net_name


args.net_name = get_net_name()

if args.out_dir:
    args.out_filename = os.path.join(
        args.out_dir,
        args.data,
        args.net_name,
        'out{out_infix}'.format(**vars(args)),
    )
    args.plot_filename = os.path.join(
        args.out_dir,
        args.data,
        args.net_name,
        'epoch_sample',
    )
else:
    args.out_filename = None
    args.plot_filename = None



