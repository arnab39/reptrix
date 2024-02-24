import numpy as np
import torch

from reptrix import utils


def test_get_eigenspectrum() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    activations_arr = torch.randn(1000, 1024)

    eigen = utils.get_eigenspectrum(activations_arr)
    assert np.allclose(np.sum(eigen), 1.0)


def test_sqrt_inv() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    A = torch.randn(100, 64)
    A_cov = A.T @ A

    A_sqrt_inv = utils.mat_sqrt_inv(A_cov)
    res = A_sqrt_inv @ A_cov @ A_sqrt_inv
    assert torch.allclose(res, torch.eye(A_cov.shape[0]), atol=1e-5)
