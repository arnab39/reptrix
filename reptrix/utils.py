import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


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
    plt.xlim(right=1024)
    plt.ylim(bottom=1e-6)
    plt.legend()
    plt.grid('on', color='gray', lw=1.0, alpha=0.3)
    plt.xlabel('i')
    plt.ylabel(r'$\lambda_i$')
