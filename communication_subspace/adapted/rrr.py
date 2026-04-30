import numpy as np
from scipy.linalg import sqrtm, inv


def svd_RRR(X, Y, rnk, lambda_=0):
    """
    Perform Ridge Regularized Reduced Rank Regression (RRR) using SVD.
    Parameters:
        X : np.ndarray
            Input data matrix (n_samples, n_input_neurons).
        Y : np.ndarray
            Output data matrix (n_samples, n_output_neurons).
        rnk : int
            Dimensionaility of communication
        lambda_ : float
            Regularization parameter (default is 0 for no regularization).
    Returns:
        w0 : np.ndarray
            Estimate of the communication strength (n_input_neurons, n_output_neurons).
        urrr : np.ndarray
            Input axes (n_input_neurons, rnk).
        vrrr : np.ndarray
            Output axes, orthonormal (n_output_neurons, rnk).
    """
    # Check if X and Y are 2D arrays
    # Ridge regularization
    XX = X.T @ X + lambda_ * np.eye(X.shape[1])

    # Least squares estimate with ridge
    if np.linalg.cond(XX) < 1e10:
        wridge = np.linalg.solve(XX, X.T @ Y)
    else:
        wridge = np.linalg.pinv(XX) @ (X.T @ Y)

    # SVD of relevant matrix
    _, _, vrrr = np.linalg.svd(Y.T @ X @ wridge)

    # Get the top 'rnk' components
    vrrr = vrrr[:rnk, :].T  # shape: (features, rnk)
    urrr = wridge @ vrrr  # shape: (features, rnk)

    # Construct full RRR estimate
    w0 = urrr @ vrrr.T
    vrrr = vrrr.T  # for compatibility with original code's return

    return w0, urrr, vrrr


def svd_RRR_noniso(X, Y, rnk, C=None):
    """
    Perform Reduced Rank Regression (RRR) using SVD with non-isotropic noise.
    Parameters:
        X : np.ndarray
            Input data matrix (n_samples, n_input_neurons).
        Y : np.ndarray
            Output data matrix (n_samples, n_output_neurons).
        rnk : int
            Dimensionaility of communication
        C : np.ndarray, optional
            Covariance matrix of the noise (default is None, estimate from data).
    Returns:
        w0 : np.ndarray
            Estimate of the communication strength (n_input_neurons, n_output_neurons).
        urrr : np.ndarray
            Input axes (n_input_neurons, rnk).
        vrrr : np.ndarray
            Output axes (n_output_neurons, rnk).
    """
    # Least squares estimate
    wls = np.linalg.solve(X.T @ X, X.T @ Y)

    # Compute covariance of residuals if C is not provided
    if C is None:
        res_wls = Y - X @ wls
        C = (res_wls.T @ res_wls) / (X.shape[0] - 1)

    # Compute inverse sqrt and sqrt of C
    C_sqrt = sqrtm(C)
    C_inv_sqrt = inv(C_sqrt)

    # SVD of whitened cross-covariance
    _, _, vrrr = np.linalg.svd(C_inv_sqrt @ Y.T @ X @ wls @ C_inv_sqrt)
    vrrr = vrrr[:rnk, :].T  # shape: (features, rnk)

    # Adjust for non-isotropic noise
    vrrr = C_sqrt @ vrrr
    urrr = np.linalg.solve(X.T @ X, X.T @ Y @ inv(C) @ vrrr)

    # Reconstruct estimate
    w0 = urrr @ vrrr.T

    return w0, urrr, vrrr


def svd_RRR_noniso_ridge(X, Y, rank, lambda_, alpha_C=1e-5):
    """extends nonisotropic reduced rank regression with ridge

    Args:
        X (np.array): trials x neurons
        Y (np.array): trails x neurons
        rank (int): reduced rank
        lambda_ (float): ridge penalty
    """

    XX = X.T @ X + lambda_ * np.eye(X.shape[1])

    if np.linalg.cond(XX) < 1e10:
        wridge = np.linalg.solve(XX, X.T @ Y)
    else:
        wridge = np.linalg.pinv(XX) @ (X.T @ Y)

    res_wridge = Y - X @ wridge
    C = (res_wridge.T @ res_wridge) / (X.shape[0] - 1)
    C += alpha_C * np.eye(C.shape[0])   

    C_sqrt = sqrtm(C)
    C_sqrt = np.real(C_sqrt)
    C_inv_sqrt = inv(C_sqrt)

    _, _, vrrr = np.linalg.svd(C_inv_sqrt @ Y.T @ X @ wridge @ C_inv_sqrt)
    vrrr = vrrr[:rank, :].T  # shape: (features, rnk)

    vrrr = C_sqrt @ vrrr

    # 6. Extract U using the Ridge weights
    # Mathematically: urrr = (X^T X + lambda*I)^-1 X^T Y C^-1 vrrr
    # Since wridge = (X^T X + lambda*I)^-1 X^T Y, we can substitute directly:
    urrr = wridge @ inv(C) @ vrrr

    w0 = urrr @ vrrr.T

    return w0, urrr, vrrr
