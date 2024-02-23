import numpy as np
import torch

from reptrix import rankme


def test_get_rankme() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    activations_arr = torch.randn(1000, 1024)

    metric_rankme = rankme.get_rankme(activations_arr)
    assert np.allclose(metric_rankme, 613.4726)
