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
