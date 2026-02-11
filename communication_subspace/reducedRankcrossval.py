import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from communication_subspace.reducedRank import reduced_rank_regression

def cross_validate_rrr(X, Y, dims, k_folds=10, ridge_init=False, scale=False):
    """
    Performs K-Fold Cross Validation for Reduced Rank Regression.
    Evaluates the model for every dimension specified in 'dims'.

    Returns:
    --------
    results : dict
        Contains 'mean_loss', 'ste_loss', 'loss_per_fold'
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    loss_matrix = np.zeros((k_folds, len(dims)))

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        B_list = reduced_rank_regression(X_train, Y_train, dims, ridge_init=ridge_init)

        for i, B in enumerate(B_list):

            intercept = B[0, :]
            slopes = B[1:, :]

            Y_pred = intercept + (X_test @ slopes)

            r2 = r2_score(Y_test, Y_pred, multioutput="variance_weighted")
            nse = 1.0 - r2
            loss_matrix[fold_idx, i] = nse

    mean_loss = np.mean(loss_matrix, axis=0)
    ste_loss = np.std(loss_matrix, axis=0, ddof=1) / np.sqrt(k_folds)

    return {"dims": dims, "mean_loss": mean_loss, "ste_loss": ste_loss, "raw_losses": loss_matrix}


def select_optimal_dimension(cv_results):
    """
    Selects the optimal dimension using the "One Standard Error Rule".
    """
    mean_loss = cv_results["mean_loss"]
    ste_loss = cv_results["ste_loss"]
    dims = cv_results["dims"]

    best_idx = np.argmin(mean_loss)
    min_loss = mean_loss[best_idx]
    threshold = min_loss + ste_loss[best_idx]

    candidates = np.where(mean_loss <= threshold)[0]
    opt_idx = candidates[0]

    return dims[opt_idx], opt_idx, threshold


