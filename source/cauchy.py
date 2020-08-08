from math import log, pi

import torch

from .source import Source


class Cauchy(Source):
    def __init__(self, nvars, scale=1):
        super().__init__(nvars)
        self.register_buffer(
            'scale', torch.tensor(scale, dtype=torch.get_default_dtype()))

    def sample(self, batch_size):
        shape = [batch_size] + self.nvars
        out = self.scale.new_empty(shape).cauchy_()
        out = out * self.scale
        return out

    def log_prob(self, x):
        out = (-(1 + (x / self.scale)**2).log() - torch.log(self.scale) -
               log(pi))
        out = out.view(out.shape[0], -1).sum(dim=1)
        return out
