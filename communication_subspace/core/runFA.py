import numpy as np
from scipy import linalg

from communication_subspace.core.faCrossVal import cross_val_fa
from communication_subspace.core.factorAnalysis import factor_analysis




def extract_fa_latents(X, q=None, var_threshold=0.95):
    """
    Fits a Factor Analysis model and extracts orthonormalized latent variables.

    If 'q' is a list or None, it performs Cross-Validation to select the
    optimal latent dimensionality using a shared variance threshold
    relative to the max-likelihood model.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix (n_samples, n_features).
    q : int, list, or None, default=None
        - If int: Uses this dimensionality directly.
        - If list/None: Tests dimensions via CV.
    var_threshold : float, default=0.95
        Threshold for explained shared variance when selecting q_opt.
        Set to 1.0 (or 1-eps) to simply select the Max-Likelihood dimension.

    Returns:
    --------
    Z : np.ndarray
        Latent variables (n_samples, q_opt).
    U : np.ndarray
        Dominant dimensions (p, q_opt).
    Q : np.ndarray
        Decoding matrix (p, q_opt).
    q_opt : int
        The optimal dimensionality used.
    """
    X = np.asarray(X)
    n, p = X.shape

    if q is None:
        q_candidates = np.arange(p)
    elif np.isscalar(q):
        q_candidates = np.array([q])
    else:
        q_candidates = np.asarray(q)

    if len(q_candidates) > 1:
        print(f"Cross-validating dimensions: {q_candidates}")
        cv_loss, _ = cross_val_fa(X, q_candidates, cv_num_folds=10)

        if np.all(np.isnan(cv_loss)):
            q_opt = 0
        else:

            explained_var = 1.0 - cv_loss
            valid_indices = np.where(explained_var > var_threshold)[0]

            if len(valid_indices) > 0:
                best_idx = valid_indices[0]
                q_opt = int(q_candidates[best_idx])
            else:
                valid_mask = ~np.isnan(cv_loss)
                q_opt = int(q_candidates[valid_mask][-1])

        print(f"Optimal dimension selected: {q_opt}")
    else:
        q_opt = int(q_candidates[0])

    Sigma = np.cov(X, rowvar=False, bias=True)

    if q_opt == 0:

        U = np.zeros((p, 0))
        Q = np.zeros((p, 0))
        Z = np.zeros((n, 0))
        return Z, U, Q, q_opt

    L, psi, _ = factor_analysis(Sigma, q_opt)

    Psi_mat = np.diag(psi)
    C = (L @ L.T) + Psi_mat

    U, s, Vt = linalg.svd(L, full_matrices=False)
    S_mat = np.diag(s)
    V = Vt.T

    RHS = L @ V @ S_mat.T

    try:
        Q = linalg.solve(C, RHS, assume_a="pos")
    except linalg.LinAlgError:
        Q = linalg.solve(C, RHS)

    m = np.mean(X, axis=0)
    Z = (X - m) @ Q

    return Z, U, Q, q_opt
