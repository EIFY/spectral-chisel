import itertools
import math

import torch


class SpectralChisel:

    def _iterate(self, mat, steps):
        for _ in range(steps):
            u = mat @ self.v
            self.v = mat.mT @ u
            self.v /= torch.linalg.vector_norm(self.v)

    def __init__(self, mat, steps=5):
        self.v = torch.normal(0, 1, (mat.size(1),)).to(mat)
        self._iterate(mat, steps)

    def shave(self, mat, s_max, steps=math.inf, iter_per_step=1):
        it = range(steps) if steps < math.inf else itertools.repeat(None)
        for _ in it:
            self._iterate(mat, iter_per_step - 1)
            u = mat @ self.v
            self.v = mat.mT @ u
            v_norm, u_norm = torch.linalg.vector_norm(self.v), torch.linalg.vector_norm(u)
            self.v /= v_norm
            spectral_norm = v_norm / u_norm
            if spectral_norm > s_max:
                u /= u_norm
                mat -= (spectral_norm - s_max) * torch.outer(u, self.v)
            else:
                return

    def weight_decay(self, mat, wd, s_max=0., steps=math.inf, iter_per_step=1):
        it = range(steps) if steps < math.inf else itertools.repeat(None)
        for _ in it:
            self._iterate(mat, iter_per_step - 1)
            u = mat @ self.v
            self.v = mat.mT @ u
            v_norm, u_norm = torch.linalg.vector_norm(self.v), torch.linalg.vector_norm(u)
            self.v /= v_norm
            spectral_norm = v_norm / u_norm
            if spectral_norm > s_max:
                u /= u_norm
                mat -= spectral_norm * wd * torch.outer(u, self.v)
            else:
                return
