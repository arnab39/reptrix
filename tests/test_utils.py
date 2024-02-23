import numpy as np
import torch

from reptrix import alpha, rankme


def test_alpha_rankme() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    activations_arr = torch.randn(1000, 1024)

    metric_alpha = alpha.get_alpha(activations_arr)
    metric_rankme = rankme.get_rankme(activations_arr)

    assert np.allclose(metric_alpha[0], 0.12024)
    assert np.allclose(metric_rankme, 613.4726)
