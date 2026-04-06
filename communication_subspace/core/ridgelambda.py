import numpy as np
from scipy import linalg


def get_ridge_lambda(d_max_shrink, X, scale=True):
    """
    Computes an appropriate range for Ridge regularization parameters (lambdas)
    based on the eigenvalues of the data's covariance matrix.

    Parameters:
    -----------
    d_max_shrink : array-like
        Array of shrinkage factors (e.g., np.arange(0.5, 1.01, 0.01)).
        Values should be in (0, 1].
        1.0 implies no shrinkage (lambda=0).
    X : np.ndarray
        Source data matrix (n_samples, n_features).
    scale : bool, default=True
        If True, z-score (standardize) X before computing eigenvalues.

    Returns:
    --------
    lambdas : np.ndarray
        The calculated regularization parameters.
    dof : np.ndarray
        The effective degrees of freedom for each lambda.
    """
    X = np.asarray(X)
    d_max_shrink = np.asarray(d_max_shrink)

    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0, ddof=1)

    epsilon = np.finfo(X.dtype).eps
    valid_cols_mask = np.abs(x_std) >= np.sqrt(epsilon)

    X_filtered = X[:, valid_cols_mask]

    if scale:
        Z = (X_filtered - x_mean[valid_cols_mask]) / x_std[valid_cols_mask]
    else:
        Z = X_filtered - x_mean[valid_cols_mask]

    cov_matrix = Z.T @ Z

    d = linalg.eigvalsh(cov_matrix)
    d_max = np.max(d)

    with np.errstate(divide="ignore"):
        lambdas = d_max * (1 - d_max_shrink) / d_max_shrink

    lambdas = np.clip(lambdas, a_min=1e-6, a_max=None)

    d_col = d[:, np.newaxis]
    lam_row = lambdas[np.newaxis, :]
    dof = np.sum(d_col / (d_col + lam_row), axis=0)

    return lambdas, dof
