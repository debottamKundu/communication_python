import numpy as np
from scipy import linalg
from sklearn.linear_model import RidgeCV, Ridge
from communication_subspace.core.ridgelambda import get_ridge_lambda


def reduced_rank_regression(X, Y, dims, ridge_init=True):
    ### Y: target data matrix (n_data_points x neurons_a)
    ### X : source data matrix (n_data_points x neurons_b)
    """
    Performs Reduced Rank Regression or PCA-based regression.

    Parameters:
    -----------
    X : np.ndarray
        Source matrix (n_samples, n_features).
    Y : np.ndarray
        Target matrix (n_samples, n_targets).
    dims : list or int
        Dimensions (ranks) to project onto.
    ridge_init : bool, default=True
        If True, use Ridge Regression for initialization.

    Returns:
    --------
    B_final : np.ndarray
        The regression coefficients including intercept term at index 0.
    """

    X = np.asarray(X)
    Y = np.asarray(Y)

    if np.isscalar(dims):
        dims = [dims]

    n, p = X.shape
    _, K = Y.shape

    x_std = np.std(X, axis=0, ddof=1)
    epsilon = np.finfo(X.dtype).eps

    valid_cols_mask = np.abs(x_std) >= np.sqrt(epsilon)

    if not np.all(valid_cols_mask):
        X_filtered = X[:, valid_cols_mask]
        x_mean_filtered = np.mean(X_filtered, axis=0)
    else:
        X_filtered = X
        x_mean_filtered = np.mean(X, axis=0)

    Z = X_filtered - x_mean_filtered

    if ridge_init:

        shrinkage_factors = np.linspace(0.5, 1.0, 51)
        lambdas, dofs = get_ridge_lambda(shrinkage_factors, X, scale=True)
        clf = RidgeCV(alphas=lambdas, fit_intercept=True)
        clf.fit(X_filtered, Y)
        Bfull = clf.coef_.T

    else:
        Bfull, _, _, _ = linalg.lstsq(Z, Y)  # type: ignore #

    Yhat = Z @ Bfull
    Yhat_centered = Yhat - np.mean(Yhat, axis=0)

    _, _, Vt = linalg.svd(Yhat_centered, full_matrices=False)
    V = Vt.T

    B_list = []

    y_mean = np.mean(Y, axis=0)
    for d in dims:
        if d == 0:
            B_d = np.zeros((X_filtered.shape[1], K))
        else:
            V_d = V[:, :d]
            B_d = Bfull @ V_d @ V_d.T

        intercept = y_mean - (x_mean_filtered @ B_d)

        B_with_intercept = np.vstack([intercept.reshape(1, -1), B_d])

        B_list.append(B_with_intercept)

    B_final_reduced = B_list

    if not np.all(valid_cols_mask):
        original_p = X.shape[1]

        B_restored_list = []

        for B_reduced in B_final_reduced:

            B_full = np.zeros((original_p + 1, K))
            B_full[0, :] = B_reduced[0, :]
            B_full[1:][valid_cols_mask] = B_reduced[1:]

            B_restored_list.append(B_full)

        return B_restored_list

    else:
        return B_final_reduced
