import numpy as np
import torch

from reptrix import utils


def test_get_eigenspectrum() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    activations_arr = torch.randn(1000, 1024)

    eigen = utils.get_eigenspectrum(activations_arr)
    assert np.allclose(np.sum(eigen), 1.0)
