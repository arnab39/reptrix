import numpy as np
import torch

from reptrix import alpha


def test_get_alpha() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    activations_arr = torch.randn(1000, 1024)

    metric_alpha = alpha.get_alpha(activations_arr)
    # print(metric_alpha[0], metric_alpha[2], metric_alpha[3])
    assert np.allclose(metric_alpha[0], 0.148936)
