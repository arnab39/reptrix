import numpy as np
import torch

from reptrix import lidar


def test_get_lidar() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    activations_arr = torch.randn(1000, 10, 1024)

    metric_lidar = lidar.get_lidar(activations_arr, 1000, 10)
    assert np.allclose(metric_lidar, 348.08795)
