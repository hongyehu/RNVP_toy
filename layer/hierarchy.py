from torch import nn

import utils

from .flow import Flow


class HierarchyBijector(Flow):
    def __init__(self, indexI, indexJ, layers, prior=None):
        super().__init__(prior)
        assert len(layers) == len(indexI)
        assert len(layers) == len(indexJ)
        self.depth = len(layers)
        self.layers = nn.ModuleList(layers)
        self.indexI = indexI
        self.indexJ = indexJ

    def forward(self, x):
        # dim(x) = (B, C, H, W)
        batch_size = x.shape[0]
        ldj = x.new_zeros(batch_size)
        for i in range(self.depth):
            x, x_ = utils.dispatch(self.indexI[i], self.indexJ[i], x)
            # dim(x_) = (B, C, num_of_block, K * K)
            x_ = utils.stackRGblock(x_)
            # dim(x_) = (batch * num_of_block, color, kernelSize, kernelSize)

            x_, log_prob = self.layers[i].forward(x_)
            ldj = ldj + log_prob.view(batch_size, -1).sum(dim=1)

            x_ = utils.unstackRGblock(x_, batch_size)
            x = utils.collect(self.indexI[i], self.indexJ[i], x, x_)

        return x, ldj

    def inverse(self, z):
        batch_size = z.shape[0]
        inv_ldj = z.new_zeros(batch_size)
        for i in reversed(range(self.depth)):
            z, z_ = utils.dispatch(self.indexI[i], self.indexJ[i], z)
            z_ = utils.stackRGblock(z_)

            z_, log_prob = self.layers[i].inverse(z_)
            inv_ldj = inv_ldj + log_prob.view(batch_size, -1).sum(dim=1)

            z_ = utils.unstackRGblock(z_, batch_size)
            z = utils.collect(self.indexI[i], self.indexJ[i], z, z_)

        return z, inv_ldj
