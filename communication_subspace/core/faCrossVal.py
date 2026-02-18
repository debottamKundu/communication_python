import numpy as np
from sklearn.model_selection import KFold
from scipy import linalg
from tqdm import tqdm

from communication_subspace.core.factorAnalysis import factor_analysis
from communication_subspace.core.faTestLoglike import factor_analysis_test_log_like


def cross_val_fa(X, q_list, cv_num_folds=10):
    """
    Performs cross-validation for Factor Analysis.

    1. Computes CV log-likelihood for all requested latent dimensions in 'q_list'.
    2. Identifies the optimal dimension 'q_max' (highest likelihood).
    3. Fits a final model on the full data using 'q_max'.
    4. Calculates the cumulative shared variance explained by this optimal model.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix (n_samples, n_features).
    q_list : array-like
        List of latent dimensions to test (e.g., [0, 1, 2, ...]).
    cv_num_folds : int, default=10
        Number of CV folds.

    Returns:
    --------
    cv_loss : np.ndarray
        The cumulative shared variance explained by the q_max model,
        indexed corresponding to the input q_list.
        (Values represent the fraction of variance *unexplained* by the top k factors).
    cv_log_like : np.ndarray
        Matrix of CV log-likelihoods (n_folds, n_dims_tested).
    """
    X = np.asarray(X)

    q_list = np.sort(np.atleast_1d(q_list).astype(int))

    kf = KFold(n_splits=cv_num_folds, shuffle=True, random_state=42)

    cv_log_like = np.zeros((cv_num_folds, len(q_list)))

    for i, (train_idx, test_idx) in tqdm(
        enumerate(kf.split(X)), total=cv_num_folds, desc="Cross-Validation"
    ):
        X_train, X_test = X[train_idx], X[test_idx]

        fold_ll = factor_analysis_test_log_like(X_train, X_test, q_list)
        cv_log_like[i, :] = fold_ll

    mean_ll = np.nanmean(cv_log_like, axis=0)

    q_max_idx = np.nanargmax(mean_ll)
    q_max = q_list[q_max_idx]

    if q_max == 0:
        return np.nan, cv_log_like

    S = np.cov(X, rowvar=False, bias=True)

    L, _, _ = factor_analysis(S, q_max)

    shared_cov_small = L.T @ L
    d = linalg.eigvalsh(shared_cov_small)

    d = np.sort(d)[::-1]

    total_shared_var = np.sum(d)
    variance_curve = 1.0 - (np.cumsum(d) / total_shared_var)

    cv_loss_result = []

    for q_val in q_list:
        if q_val == 0:
            cv_loss_result.append(1.0)
        elif q_val <= len(variance_curve):
            cv_loss_result.append(variance_curve[q_val - 1])
        else:
            cv_loss_result.append(np.nan)

    return np.array(cv_loss_result), cv_log_like


def factor_analysis_model_select(cv_loss, q_list, var_threshold=0.95):
    """
    Selects the optimal FA dimensionality based on the 'Parsimony Rule'.

    cv_loss: The fraction of UNEXPLAINED shared variance relative to the
             max-likelihood model (returned by cross_val_fa).
    q_list:  The list of dimensions tested.
    """

    cv_loss = np.asarray(cv_loss)
    q_list = np.asarray(q_list)

    if np.all(np.isnan(cv_loss)):
        return 0

    explained_var = 1.0 - cv_loss

    valid_indices = np.where(explained_var > var_threshold)[0]

    if len(valid_indices) > 0:
        best_idx = valid_indices[0]
        q_opt = int(q_list[best_idx])
    else:
        valid_mask = ~np.isnan(cv_loss)
        q_opt = int(q_list[valid_mask][-1])

    return q_opt
