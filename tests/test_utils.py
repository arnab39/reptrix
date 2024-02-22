import numpy as np

from reptrix.utils import get_eigenspectrum


def test_get_eigenspectrum() -> None:
    np.random.seed(0)
    arr = np.random.rand(3, 2)

    result = get_eigenspectrum(arr)

    assert result[0] == 0.6996675118986713
