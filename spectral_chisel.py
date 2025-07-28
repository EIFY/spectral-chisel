import itertools
import math

import torch


def power_iter(mat, v, steps):
    for _ in range(steps):
        u = mat @ v
        v = mat.mT @ u
        v /= torch.linalg.vector_norm(v)
    return v


class SpectralChisel:

    def __init__(self, mat, steps=5):
        n = mat.size(1)
        v = torch.normal(0, n**-0.5, (n,)).to(mat)
        self.v = power_iter(mat, v, steps)

    def hammer(self, mat, s_max, steps=math.inf, iter_per_step=1):
        it = range(steps) if steps < math.inf else itertools.repeat(None)
        for _ in it:
            self.v = power_iter(mat, self.v, iter_per_step - 1)
            u = mat @ self.v
            self.v = mat.mT @ u
            v_norm, u_norm = torch.linalg.vector_norm(self.v), torch.linalg.vector_norm(u)
            self.v /= v_norm
            u /= u_norm
            spectral_norm = v_norm / u_norm
            mat -= (spectral_norm - s_max) * torch.outer(u, self.v)
        return mat

    def decay(self, mat, wd, steps=math.inf, iter_per_step=1):
        it = range(steps) if steps < math.inf else itertools.repeat(None)
        for _ in it:
            self.v = power_iter(mat, self.v, iter_per_step - 1)
            u = mat @ self.v
            self.v = mat.mT @ u
            v_norm, u_norm = torch.linalg.vector_norm(self.v), torch.linalg.vector_norm(u)
            self.v /= v_norm
            u /= u_norm
            spectral_norm = v_norm / u_norm
            mat -= spectral_norm * wd * torch.outer(u, self.v)
        return mat
