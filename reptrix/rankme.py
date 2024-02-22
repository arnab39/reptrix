import numpy as np
import torch
import reptrix.utils as utils

def get_rank(eigen: np.ndarray) -> float:
    """Get effective rank of the representation covariance matrix

    Args:
        eigen (np.ndarray): Eigenspectrum of the representation covariance matrix

    Returns:
        float: Effective rank
    """
    l1 = np.sum(np.abs(eigen))
    eps = 1e-7
    eigen_norm = eigen / l1 + eps
    entropy = -np.sum(eigen_norm*np.log(eigen_norm))
    return np.exp(entropy)

def get_rankme(activations: torch.Tensor,
              max_eigenvals: int = 2048) -> float:
    """Get RankMe metric
    (https://proceedings.mlr.press/v202/garrido23a)

    Args:
        activations (np.ndarray): Activation tensor of shape (bsz,d1,d2...dn)
        max_eigenvals (int, optional): Maximum #eigenvalues to compute. 
                                    Defaults to 2048.

    Returns:
        float: RankMe metric
    """
    try:
        activations_arr = activations.detach()
    except:
        activations_arr = activations
    activations_arr = activations_arr.cpu().numpy()
    eigen = utils.get_eigenspectrum(activations_np=activations_arr,
                              max_eigenvals=max_eigenvals)
    rank = get_rank(eigen)
    return rank
