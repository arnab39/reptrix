import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA  # type: ignore


def get_eigenspectrum(
    activations_np: np.ndarray, max_eigenvals: int = 2048
) -> np.ndarray:
    """Get eigenspectrum of activation covariance matrix.

    Args:
        activations_np (np.ndarray): Numpy arr of activations,
                                    shape (bsz,d1,d2...dn)
        max_eigenvals (int, optional): Maximum #eigenvalues to compute.
                                        Defaults to 2048.

    Returns:
        np.ndarray: Returns the eigenspectrum of the activation covariance matrix
    """
    feats = activations_np.reshape(activations_np.shape[0], -1)
    feats_center = feats - feats.mean(axis=0)
    pca = PCA(
        n_components=min(max_eigenvals, feats_center.shape[0], feats_center.shape[1]),
        svd_solver='full',
    )
    pca.fit(feats_center)
    eigenspectrum = pca.explained_variance_ratio_
    return eigenspectrum


def plot_eigenspectrum(eigenspectrum: np.ndarray) -> None:
    """Plot eigenspectrum in log-log scale

    Args:
        eigenspectrum (np.ndarray): Eigenspectrum of activation covariance matrix
    """
    xrange = np.arange(1, 1 + len(eigenspectrum))
    plt.loglog(xrange, eigenspectrum, c='blue', lw=2.0, label='Eigenspectrum')
    xlim_max = 1024 if len(eigenspectrum) > 512 else len(eigenspectrum) - 20
    plt.xlim(right=xlim_max)
    plt.ylim(bottom=1e-4)
    plt.legend()
    plt.grid(True, color='gray', lw=1.0, alpha=0.3)
    plt.xlabel('i')
    plt.ylabel(r'$\lambda_i$')


def mat_sqrt_inv(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the square root of inverse of a square (positive definite) matrix.

    Args:
        matrix (torch.Tensor): Matrix to be be sqrt inverted

    Returns:
        torch.Tensor: Square root of inverse of matrix
    """
    # Ensure the input is a square matrix
    assert matrix.size(0) == matrix.size(1), "Input must be a square matrix."

    # Compute the inverse of the matrix
    matrix_inv = torch.inverse(matrix)

    # Compute the eigenvalues and eigenvectors of the inverse matrix
    eigenvals, eigenvecs = torch.linalg.eig(matrix_inv)
    sqrt_eigenvals = torch.sqrt(eigenvals.real).type_as(matrix)
    sqrt_diag = torch.diag(sqrt_eigenvals)
    eigenvecs_real = eigenvecs.real.type_as(matrix)

    # Reconstruct the matrix
    result = torch.mm(
        torch.mm(eigenvecs_real, sqrt_diag), torch.linalg.inv(eigenvecs_real)
    )

    return result
