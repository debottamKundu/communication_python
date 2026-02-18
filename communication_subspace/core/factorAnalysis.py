import numpy as np
from scipy import linalg


def factor_analysis(S, q, method="FA", tol=1e-8, max_iter=int(1e8)):
    """
    Applies Factor Analysis (FA) or Probabilistic PCA (PPCA) to the sample
    covariance matrix S, using Expectation-Maximization (EM).

    Parameters:
    -----------
    S : np.ndarray
        Sample covariance matrix (p x p).
    q : int
        Latent dimensionality (number of factors).
    method : str, default='FA'
        'FA' for Factor Analysis or 'PPCA'.
    tol : float
        Stopping criterion for EM.
    max_iter : int
        Maximum number of EM iterations.

    Returns:
    --------
    L : np.ndarray
        Factor loadings (p x q).
    psi : np.ndarray
        Diagonal of uniqueness matrix (p x 1).
    log_like : float
        Log likelihood at final EM iteration.
    """
    S = np.asarray(S)
    p = S.shape[0]

    C_MIN_FRAC_VAR = 0.01

    s_diag = np.diag(S)
    epsilon = np.finfo(S.dtype).eps

    valid_mask = np.abs(s_diag) >= np.sqrt(epsilon)

    if not np.all(valid_mask):
        S_filtered = S[valid_mask][:, valid_mask]
        p_filtered = S_filtered.shape[0]
    else:
        S_filtered = S
        p_filtered = p

    rank_s = np.linalg.matrix_rank(S_filtered)

    if rank_s == p_filtered:

        try:
            chol_S = linalg.cholesky(S_filtered, lower=False)
            log_det = 2 * np.sum(np.log(np.diag(chol_S)))
            scale = np.exp(log_det / p_filtered)
        except linalg.LinAlgError:

            eigvals = linalg.eigvalsh(S_filtered)
            scale = np.exp(np.mean(np.log(eigvals[eigvals > 0])))
    else:

        eigvals = linalg.eigvalsh(S_filtered)
        d = np.sort(eigvals)[::-1]
        scale = np.exp(np.mean(np.log(d[:rank_s])))

    L = np.random.randn(p_filtered, q) * np.sqrt(scale / q)
    psi = np.diag(S_filtered).copy()

    var_floor = C_MIN_FRAC_VAR * psi
    I = np.eye(q)
    const_c = -p_filtered / 2.0 * np.log(2 * np.pi)

    log_like = 0.0
    base_log_like = 0.0
    prev_log_like = 0.0

    for i in range(1, max_iter + 1):

        inv_psi_diag = 1.0 / psi
        inv_psi_times_L = L * inv_psi_diag[:, np.newaxis]
        M = I + L.T @ inv_psi_times_L

        term_right = linalg.solve(M.T, inv_psi_times_L.T).T

        inv_C = np.diag(inv_psi_diag) - (inv_psi_times_L @ term_right.T)

        V = inv_C @ L

        S_times_V = S_filtered @ V

        EZZ = I - (V.T @ L) + (V.T @ S_times_V)

        prev_log_like = log_like
        try:
            chol_invC = linalg.cholesky(inv_C, lower=False)
            ldm = np.sum(np.log(np.diag(chol_invC)))
        except linalg.LinAlgError:

            sign, logdet = np.linalg.slogdet(inv_C)
            ldm = 0.5 * logdet

        trace_term = np.sum(inv_C * S_filtered)

        log_like = const_c + ldm - 0.5 * trace_term

        if i <= 2:
            base_log_like = log_like
        elif (log_like - base_log_like) < (1 + tol) * (prev_log_like - base_log_like):
            break

        # === M-step ===

        L = linalg.solve(EZZ.T, S_times_V.T).T

        update_term = np.sum(S_times_V * L, axis=1)
        psi = np.diag(S_filtered) - update_term

        if method.upper() == "PPCA":
            psi[:] = np.mean(psi)
        else:  # 'FA'
            psi = np.maximum(var_floor, psi)

    if not np.all(valid_mask):

        L_final = np.zeros((p, q))
        L_final[valid_mask, :] = L

        psi_final = np.zeros((p,))
        psi_final[valid_mask] = psi

        return L_final, psi_final, log_like

    else:
        return L, psi, log_like
