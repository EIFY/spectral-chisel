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


eps = 1e-8

# Polar Express (https://arxiv.org/abs/2505.16932) w/ eps but no bfloat16 casting for fair comparison
coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]

# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]

@torch.compile
def PolarExpress(X: torch.Tensor, steps: int) -> torch.Tensor:
    assert X.ndim >= 2
    if transpose := X.size(-2) > X.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim = True) * 1.01 + eps)
    hs = coeffs_list[:steps] + list(itertools.repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX ˆ3 + cX ˆ5
    if transpose: X = X.mT
    return X


# https://leloykun.github.io/ponder/spectral-clipping/#33-unbounded-below-spectral-hardcapping

@torch.compile
def msign_hard_cap(W, beta, steps=8):
    if transpose := W.shape[0] > W.shape[1]:
        W = W.T
    OW = PolarExpress(W, steps=steps)
    aW = beta * OW - W
    result = (1/2) * (beta*OW + W - aW @ PolarExpress(aW, steps=steps).T @ OW)
    if transpose:
        result = result.T
    return result


@torch.compile
def direct_hard_cap(M, beta):
    return beta * _direct_hard_cap(M / beta)


# Implementations below are ported from https://github.com/Arongil/lipschitz-transformers

@torch.compile
def _direct_hard_cap(M):
    """Apply min(1, x) approximately to the singular values of a single matrix. Credit: Franz Cesista."""
    coeffs = [
        (0.805032, 0.206361, -0.019763),
        (0.649867, 0.162935, -0.011150),
        (1.810259, -0.200265, 0.008251),
        (1.004384, -0.183490, 0.014413),
    ]
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    for a, b, c in coeffs:
        A = M.T @ M
        I = torch.eye(A.shape[0], dtype=M.dtype, device=M.device)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M


@torch.compile
def _soft_cap(M, alpha):
    """Apply min(1, x) approximately to the singular values of a single matrix."""
    coeffs = [
        (1, -alpha),
        (1, alpha),
    ]
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    for a, b in coeffs:
        A = M.T @ M
        I = torch.eye(A.shape[0], dtype=M.dtype, device=M.device)
        M = M @ (a * I + b * A)
    if transpose:
        M = M.T
    return M
