import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score  # type: ignore

import reptrix.utils as utils


def get_powerlaw(eigen: np.ndarray, trange: np.ndarray) -> tuple:
    """Fit powerlaw and return decay, powerlaw fit and the goodness of fit

    Args:
        eigen (np.ndarray): Eigenspectrum of activation covariance matrix
        trange (np.ndarray): Range to fit the powerlaw.
                            Tip: Ignore the first couple of eigenvalues
                                because we want to fit the tail of the spectrum

    Returns:
        tuple: Result of size 4
            alpha: powerlaw decay coefficient
            ypred: Powerlaw fit
            fit_R2: goodness of powerlaw fit
            fit_R2_100: goodness of powerlaw fit (computed till the 100 eigvals)
    """
    # Inspired by Stringer+Pachitariu 2018b github repo!
    # (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
    logss = np.log(np.abs(eigen))
    y = logss[trange][:, np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, eigen.size).astype(int) + 1
    x = np.concatenate(
        (-np.log(allrange)[:, np.newaxis], np.ones((eigen.size, 1))), axis=1
    )
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    max_range = (
        500 if len(eigen) >= 512 else len(eigen) - 10
    )  # subtracting 10 here arbitrarily because we want to avoid the last tail!
    fit_R2 = r2_score(
        y_true=logss[trange[0] : max_range],
        y_pred=np.log(np.abs(ypred))[trange[0] : max_range],
    )
    try:
        fit_R2_100 = r2_score(
            y_true=logss[trange[0] : 100], y_pred=np.log(np.abs(ypred))[trange[0] : 100]
        )
    except:  # noqa: E722
        fit_R2_100 = None
    return (alpha, ypred, fit_R2, fit_R2_100)


def get_alpha(
    activations: torch.Tensor,
    max_eigenvals: int = 2048,
    fit_range: np.ndarray = None,
) -> tuple:
    """Get alpha and powerlaw fit
    (https://proceedings.neurips.cc/paper_files/paper/2022/hash/70596d70542c51c8d9b4e423f4bf2736-Abstract-Conference.html)

    Args:
        activations (np.ndarray): Activation tensor of shape (bsz,d1,d2...dn)
        max_eigenvals (int, optional): Maximum #eigenvalues to compute.
                                    Defaults to 2048.
        fit_range (np.ndarray, optional): Range to fit the powerlaw.
                                        Defaults to np.arange(10,100).

    Returns:
        tuple: Result of size 4
            alpha: powerlaw decay coefficient
            ypred: Powerlaw fit
            fit_R2: goodness of powerlaw fit
            fit_R2_100: goodness of powerlaw fit (computed till the 100 eigvals)
    """
    if fit_range is None:
        fit_range = np.arange(10, 100)
    if activations.requires_grad:
        activations_arr = activations.detach().cpu()
    else:
        activations_arr = activations.cpu()
    activations_arr_np = activations_arr.numpy()
    eigen = utils.get_eigenspectrum(
        activations_np=activations_arr_np, max_eigenvals=max_eigenvals
    )
    alpha_res = get_powerlaw(eigen=eigen, trange=fit_range)
    return alpha_res


def plot_powerlaw(eigenspectrum: np.ndarray, alpha_res: tuple) -> None:
    """Plot eigenspectrum and powerlaw fit

    Args:
        eigenspectrum (np.ndarray): Eigenspectrum of activation covariance matrix
        alpha_res (tuple): Tuple containing alpha, powerlaw fit,
                        goodness of powerlaw fit,
                        goodness of powerlaw fit for first 100 eigenvalues
    """
    utils.plot_eigenspectrum(eigenspectrum=eigenspectrum)
    alpha, ypred, r2, r2_100 = alpha_res
    xrange = np.arange(1, 1 + len(ypred))
    plt.loglog(xrange, ypred, c='red', ls='--', lw=2.0, label='Powerlaw fit')
    plt.title(
        rf'Eigenspectrum and powerlaw fit, $\alpha$ = {alpha:.3f} '
        + rf'(r2_100 = {r2_100:.3f})'
    )
