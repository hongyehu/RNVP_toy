import re
from collections import namedtuple
from math import isinf, isnan

import numpy as np

RunRow = namedtuple('RunRow',
                    ['epoch', 'batch', 'bpp', 'loss', 'loss_std', 'time'])

Run = namedtuple('Run',
                 ['steps', 'losses', 'loss_stds', 'nparams', 'time_per_step'])


def parse_run_row(s):
    match = re.compile(
        r'epoch (?P<epoch>.*?) batch (?P<batch>.*?) (bpp (?P<bpp>.*?) )?loss (?P<loss>.*?) \+- (?P<loss_std>.*?) time (?P<time>.*)'
    ).match(s)
    if not match:
        return None

    loss = float(match.group('loss'))
    if isinf(loss) or isnan(loss):
        return None

    return RunRow(
        epoch=int(match.group('epoch')),
        batch=int(match.group('batch')),
        bpp=float(match.group('bpp')),
        loss=loss,
        loss_std=float(match.group('loss_std')),
        time=float(match.group('time')),
    )


def read_log(filename):
    step = -1
    steps = []
    losses = []
    loss_stds = []
    nparams = 0
    last_time = None
    time_sum = 0
    step_sum = 0
    with open(filename, 'r') as f:
        for line in f:
            match = re.compile(r'.*parameters: (.*)').match(line)
            if match:
                nparams = int(match.group(1))
                continue

            run_row = parse_run_row(line)
            if run_row:
                step += 1
                steps.append(step)
                losses.append(run_row.loss)
                loss_stds.append(run_row.loss_std)

                time = run_row.time
                if last_time is None:
                    last_time = time
                elif time < last_time or time > last_time + 60:
                    last_time = None
                else:
                    time_sum += time - last_time
                    step_sum += 1
                    last_time = time

    steps = np.array(steps, dtype=int)
    losses = np.array(losses)
    loss_stds = np.array(loss_stds)
    time_per_step = time_sum / step_sum

    return Run(steps, losses, loss_stds, nparams, time_per_step)


def ema(steps, losses, alpha):
    out = np.empty_like(losses)
    out[0] = losses[0]
    for i in range(1, losses.size):
        a = alpha**(steps[i] - steps[i - 1])
        out[i] = a * out[i - 1] + (1 - a) * losses[i]
    return out


def get_loss(filename, alpha=0):
    run = read_log(filename)
    steps = run.steps
    losses = run.losses
    if alpha > 0:
        losses = ema(steps, losses, alpha)
    loss = losses.min()
    return run, loss
