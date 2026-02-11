import numpy as np
from scipy import linalg

from communication_subspace.faCrossVal import cross_val_fa
from communication_subspace.factorAnalysis import factor_analysis


def factor_regress(Y, X, q_dims, q_opt=None, var_threshold=0.95):
    """
    Fits a Factor Regression model.

    1. Models X using Factor Analysis (dimensionality q_opt).
    2. Projects X into latent factors Z.
    3. Regresses Y on Z using varying numbers of factors (specified in q_dims).

    Parameters:
    -----------
    Y : np.ndarray
        Target matrix (n_samples, n_targets).
    X : np.ndarray
        Source matrix (n_samples, n_features).
    q_dims : list or int
        Latent dimensionalities to use for the REGRESSION step.
    q_opt : int, optional
        Optimal factor analysis dimensionality for X.
        If None, it is determined automatically via Cross-Validation.
    var_threshold : float, default=0.95
        Threshold for selecting q_opt if running CV.

    Returns:
    --------
    B_list : list of np.ndarray
        List of mapping matrices (p+1, K).
        One for each dimension in q_dims.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    q_dims = np.sort(np.atleast_1d(q_dims).astype(int))

    n, p = X.shape
    _, K = Y.shape

    if q_opt is None:
        print("Determining optimal FA dimension (q_opt) via CV...")

        q_candidates = np.arange(p)
        cv_loss, _ = cross_val_fa(X, q_candidates, cv_num_folds=10)

        if np.all(np.isnan(cv_loss)):
            q_opt = 0
        else:
            explained_var = 1.0 - cv_loss
            valid_indices = np.where(explained_var > var_threshold)[0]
            if len(valid_indices) > 0:
                q_opt = int(q_candidates[valid_indices[0]])
            else:

                valid_mask = ~np.isnan(cv_loss)
                q_opt = int(q_candidates[valid_mask][-1])
        print(f"q_opt selected: {q_opt}")

    if q_opt == 0:

        y_mean = np.mean(Y, axis=0)
        B_list = []
        for _ in q_dims:
            B_zero = np.zeros((p + 1, K))
            B_zero[0, :] = y_mean
            B_list.append(B_zero)
        return B_list

    valid_q_dims = q_dims[q_dims <= q_opt]
    if len(valid_q_dims) == 0:

        valid_q_dims = np.array([q_opt])

    x_mean = np.mean(X, axis=0)

    Sigma = np.cov(X, rowvar=False, bias=True)

    s_diag = np.diag(Sigma)
    epsilon = np.finfo(float).eps
    valid_cols_mask = np.abs(s_diag) >= np.sqrt(epsilon)

    if not np.all(valid_cols_mask):
        Sigma_filtered = Sigma[valid_cols_mask][:, valid_cols_mask]
        X_filtered = X[:, valid_cols_mask]
        x_mean_filtered = np.mean(X_filtered, axis=0)
        p_filtered = X_filtered.shape[1]
    else:
        Sigma_filtered = Sigma
        X_filtered = X
        x_mean_filtered = x_mean
        p_filtered = p

    L, psi, _ = factor_analysis(Sigma_filtered, q_opt)
    Psi_mat = np.diag(psi)
    C = (L @ L.T) + Psi_mat

    U, s, Vt = linalg.svd(L, full_matrices=False)
    V = Vt.T
    S_mat = np.diag(s)

    RHS = L @ V @ S_mat.T

    try:
        Q = linalg.solve(C, RHS, assume_a="pos")
    except linalg.LinAlgError:
        Q = linalg.solve(C, RHS)

    B_list_reduced = []

    X_centered = X_filtered - x_mean_filtered

    for q_curr in valid_q_dims:
        if q_curr == 0:
            B_slopes = np.zeros((p_filtered, K))
        else:

            Q_sub = Q[:, :q_curr]
            EZ = X_centered @ Q_sub

            beta_latent, _, _, _ = linalg.lstsq(EZ, Y)  # type: ignore

            B_slopes = Q_sub @ beta_latent

        B_list_reduced.append(B_slopes)

    B_list_final = []
    y_mean = np.mean(Y, axis=0)

    for B_red in B_list_reduced:

        B_full = np.zeros((p + 1, K))

        if not np.all(valid_cols_mask):
            B_full[1:][valid_cols_mask] = B_red
        else:
            B_full[1:] = B_red

        slopes_full = B_full[1:]
        intercept = y_mean - (x_mean @ slopes_full)

        B_full[0, :] = intercept
        B_list_final.append(B_full)

    return B_list_final
