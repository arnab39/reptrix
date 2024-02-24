import numpy as np
import torch

import reptrix.utils as utils


def get_rank(eigen: np.ndarray) -> float:
    """Get effective rank of the LDA covariance matrix

    Args:
        eigen (np.ndarray): Eigenspectrum of the LDA covariance matrix

    Returns:
        float: Effective rank
    """
    l1 = np.sum(np.abs(eigen))
    eps = 1e-7
    eigen_norm = eigen / l1 + eps
    entropy = -np.sum(eigen_norm * np.log(eigen_norm))
    return np.exp(entropy)


def get_lidar(
    activations: torch.Tensor,
    num_samples: int,
    num_augs: int,
    del_sigma_augs: float = 1e-6,
    max_eigenvals: int = 2048,
) -> float:
    """Get RankMe metric
    (https://openreview.net/forum?id=f3g5XpL9Kb)

    Args:
        activations (torch.Tensor): Activation tensor of shape either
                                    (num_samples, num_augs, d1,d2...dn) or
                                    (num_samples * num_augs, d1,d2...dn)
        num_samples (int): Number of unique inputs/samples
        num_augs (int): Number of augmentations for each input used to estimate
                        object manifold
        del_sigma_augs (float, optional): A small positive constant that is added
                                        to make sure matrices are invertible.
                                        Defaults to 1e-6.
        max_eigenvals (int, optional): Maximum #eigenvalues to compute.
                                    Defaults to 2048.

    Returns:
        float: LiDAR metric
    """
    if activations.requires_grad:
        activations_arr = activations.detach().cpu()
    else:
        activations_arr = activations.cpu()
    if activations_arr.shape[0] == num_samples * num_augs:
        activations_arr = activations_arr.reshape(num_samples, num_augs, -1)
    else:
        d0 = activations_arr.shape[0]
        d1 = activations_arr.shape[1]
        assert (
            d0 == num_samples and d1 == num_augs
        ), "Tensor activations should have shape (num_samples, num_augs,...)"
        activations_arr = activations_arr.reshape(d0, d1, -1)

    object_activations = activations_arr.mean(dim=1, keepdim=True)  # mean over augs
    mean_activations = object_activations.mean(dim=0, keepdim=True)
    # compute the inter-object manifold covariance
    sigma_obj = (object_activations - mean_activations).squeeze().T @ (
        object_activations - mean_activations
    ).squeeze()

    # compute the intra-object manifold covariance and take mean across objects
    sigma_augs = torch.bmm(
        (activations_arr - object_activations).permute((0, 2, 1)),
        (activations_arr - object_activations),
    ).mean(dim=0)
    # add Identity to ensure invertibility
    sigma_augs += del_sigma_augs * torch.eye(sigma_augs.shape[0])
    sigma_augs_inv_sqrt = utils.mat_sqrt_inv(sigma_augs)

    # compute LIDAR matrix
    sigma_lidar = sigma_augs_inv_sqrt @ sigma_obj @ sigma_augs_inv_sqrt
    sigma_lidar_np = sigma_lidar.numpy()

    eigen = utils.get_eigenspectrum(
        activations_np=sigma_lidar_np, max_eigenvals=max_eigenvals
    )
    lidar = get_rank(eigen)
    return lidar
