import torch
from torch import nn

from .flow import Flow


def generate_masks(nvars):
    left = torch.zeros([nvars])
    left[:int(nvars/2)] = 1
    right = torch.ones_like(left) - left
    return left, right


class RNVP(Flow):
    def __init__(self,
                 nvars,
                 s_layers,
                 t_layers,
                 prior=None):
        super().__init__(prior)
        assert len(s_layers) == len(t_layers)
        self.s_layers = nn.ModuleList(s_layers)
        self.t_layers = nn.ModuleList(t_layers)

        mask_left, mask_right = generate_masks(nvars)
        self.register_buffer('mask_left', mask_left)
        self.register_buffer('mask_right', mask_right)

    def _forward(self, x, s_layer, t_layer, mask, inv_mask):
        # x.shape = (B, nvars)
        mask_x = mask * x
        s = inv_mask * s_layer(mask_x)
        t = inv_mask * t_layer(mask_x)
        z = mask_x + inv_mask * (x * torch.exp(s) + t)
        ldj = s.sum(dim=(1))
        return z, ldj

    def _inverse(self, z, s_layer, t_layer, mask, inv_mask):
        # z.shape = (B, C, W, H)
        mask_z = mask * z
        s = inv_mask * s_layer(mask_z)
        t = inv_mask * t_layer(mask_z)
        x = mask_z + inv_mask * (z - t) * torch.exp(-s)
        inv_ldj = -s.sum(dim=(1))
        return x, inv_ldj

    # Each layer does _forward() twice, so all parameters in the layer are used
    def forward(self, x):
        ldj = x.new_zeros(x.shape[0])
        for i in range(len(self.s_layers)):
            x, ldj_ = self._forward(x, self.s_layers[i], self.t_layers[i],
                                    self.mask_left, self.mask_right)
            ldj = ldj + ldj_
            x, ldj_ = self._forward(x, self.s_layers[i], self.t_layers[i],
                                    self.mask_right, self.mask_left)
            ldj = ldj + ldj_
        return x, ldj

    def inverse(self, z):
        inv_ldj = z.new_zeros(z.shape[0])
        for i in reversed(range(len(self.s_layers))):
            z, inv_ldj_ = self._inverse(z, self.s_layers[i], self.t_layers[i],
                                        self.mask_right, self.mask_left)
            inv_ldj = inv_ldj + inv_ldj_
            z, inv_ldj_ = self._inverse(z, self.s_layers[i], self.t_layers[i],
                                        self.mask_left, self.mask_right)
            inv_ldj = inv_ldj + inv_ldj_
        return z, inv_ldj
