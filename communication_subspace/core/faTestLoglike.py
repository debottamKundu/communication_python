import numpy as np
from scipy import linalg
from tqdm import tqdm

from communication_subspace.core.factorAnalysis import factor_analysis


def mvn_log_like(X, m, S):
    """
    Computes the log-likelihood of the data X under a multivariate
    Gaussian distribution with mean m and covariance S.

    """
    X = np.asarray(X)
    m = np.asarray(m)
    S = np.asarray(S)

    n, p = X.shape
    X_centered = X - m

    try:

        L = linalg.cholesky(S, lower=True)
    except linalg.LinAlgError:

        return np.nan

    log_det_S = 2 * np.sum(np.log(np.diag(L)))

    Y = linalg.solve_triangular(L, X_centered.T, lower=True)
    mahalanobis_term = np.sum(Y**2)
    const_term = n * p * np.log(2 * np.pi)
    log_det_term = n * log_det_S

    log_like = -0.5 * (const_term + log_det_term + mahalanobis_term)

    return log_like


def factor_analysis_test_log_like(X_train, X_test, q_list, method="FA"):
    """
    Fits FA/PPCA models with latent dims 'q_list' on X_train and
    computes log-likelihood on X_test.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    q_list = np.atleast_1d(q_list)

    m = np.mean(X_train, axis=0)
    S = np.cov(X_train, rowvar=False, bias=True)

    log_likes = np.zeros(len(q_list))

    for i, q in tqdm(enumerate(q_list), desc="Testing Latent Dimensions"):

        if q == 0:
            Psi = np.diag(np.diag(S))
            log_likes[i] = mvn_log_like(X_test, m, Psi)

        else:

            try:
                L, psi, _ = factor_analysis(S, q, method=method)

                epsilon = np.finfo(float).eps
                if np.any(np.abs(psi) < np.sqrt(epsilon)):
                    log_likes[i] = np.nan
                    continue

                Psi_mat = np.diag(psi)
                C = (L @ L.T) + Psi_mat
                log_likes[i] = mvn_log_like(X_test, m, C)
            except Exception:
                log_likes[i] = np.nan

    return log_likes
