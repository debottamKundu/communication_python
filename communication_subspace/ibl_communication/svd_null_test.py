import numpy as np
import time
import sys
import os
from communication_subspace.ibl_communication.utils import compute_regionwise_null_r2
from sklearn.model_selection import KFold
from scipy.stats import sem


def compute_regionwise_null_r2_svd(
    data_a, data_b, frameidx, frameidy, candidate_trials_matrix, n_iterations=200
):
    n_regions = len(data_a)
    null_distributions = np.zeros((n_regions, n_regions, n_iterations))
    alphas = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for idy in range(n_regions):
        base_region_y = data_b[idy][frameidy, :, :].astype(np.float64)

        shifted_targets = []
        for iter_idx in range(n_iterations):
            trial_order = candidate_trials_matrix[:, iter_idx]
            shifted_targets.append(base_region_y[trial_order, :])

        for idx in range(n_regions):
            if idx == idy:
                continue

            region_x = data_a[idx][frameidx, :, :].astype(np.float64)

            # Precompute SVD-based Projection Matrices for each fold
            fold_data = []
            for train_idx, test_idx in kf.split(region_x):
                X_train, X_test = region_x[train_idx], region_x[test_idx]

                mean_X = np.mean(X_train, axis=0)
                std_X = np.std(X_train, axis=0)
                std_X[std_X == 0] = 1.0

                X_tr_sc = (X_train - mean_X) / std_X
                X_te_sc = (X_test - mean_X) / std_X

                U, S, Vt = np.linalg.svd(X_tr_sc, full_matrices=False)

                alpha_projections = []
                for alpha in alphas:
                    D = S / (S**2 + alpha)
                    Proj = Vt.T @ np.diag(D) @ U.T
                    alpha_projections.append(Proj)

                fold_data.append((train_idx, test_idx, X_te_sc, alpha_projections))

            for iter_idx in range(n_iterations):
                Y_null = shifted_targets[iter_idx]

                mean_scores = np.zeros(len(alphas))
                sem_scores = np.zeros(len(alphas))

                alpha_fold_scores = [[] for _ in range(len(alphas))]

                for train_idx, test_idx, X_te_sc, alpha_projections in fold_data:
                    Y_train, Y_test = Y_null[train_idx], Y_null[test_idx]

                    mean_Y = np.mean(Y_train, axis=0)
                    std_Y = np.std(Y_train, axis=0)
                    std_Y[std_Y == 0] = 1.0

                    Y_tr_sc = (Y_train - mean_Y) / std_Y
                    Y_te_sc = (Y_test - mean_Y) / std_Y

                    for a_idx, Proj in enumerate(alpha_projections):
                        Y_pred = X_te_sc @ (Proj @ Y_tr_sc)

                        ss_res = np.sum((Y_te_sc - Y_pred) ** 2, axis=0)
                        ss_tot = np.sum((Y_te_sc - np.mean(Y_te_sc, axis=0)) ** 2, axis=0)
                        nonzero = ss_tot > 1e-10
                        r2_arr = np.zeros(Y_te_sc.shape[1])
                        r2_arr[nonzero] = 1 - ss_res[nonzero] / ss_tot[nonzero]

                        alpha_fold_scores[a_idx].append(np.mean(r2_arr))

                for a_idx in range(len(alphas)):
                    mean_scores[a_idx] = np.mean(alpha_fold_scores[a_idx])
                    sem_scores[a_idx] = sem(alpha_fold_scores[a_idx])

                peak_idx = np.argmax(mean_scores)
                threshold = mean_scores[peak_idx] - sem_scores[peak_idx]

                valid_indices = np.where(mean_scores >= threshold)[0]
                optimal_alpha = np.max(alphas[valid_indices])

                optimal_idx = np.where(alphas == optimal_alpha)[0][0]
                null_distributions[idx, idy, iter_idx] = mean_scores[optimal_idx]

    return null_distributions


def run_benchmark():
    np.random.seed(42)
    n_regions = 2
    n_trials = 80
    n_features = 20
    n_iterations = 20
    frameidx = 0
    frameidy = 0

    print(
        f"Generating synthetic widefield dataset with {n_regions} regions, {n_trials} trials, {n_iterations} null iterations..."
    )

    # Mock data structure: regions x (frames x trials x voxels)
    data_a = [np.random.randn(1, n_trials, n_features) for _ in range(n_regions)]
    data_b = [np.random.randn(1, n_trials, n_features) for _ in range(n_regions)]

    candidate_trials_matrix = np.random.randint(0, n_trials, size=(n_trials, n_iterations))

    start_time = time.time()
    orig_results = compute_regionwise_null_r2(
        data_a, data_b, frameidx, frameidy, candidate_trials_matrix, n_iterations=n_iterations
    )
    orig_time = time.time() - start_time

    start_time = time.time()
    svd_results = compute_regionwise_null_r2_svd(
        data_a, data_b, frameidx, frameidy, candidate_trials_matrix, n_iterations=n_iterations
    )
    svd_time = time.time() - start_time

    print(f"speedup: {orig_time / svd_time:.1f}")

    try:
        np.testing.assert_allclose(orig_results, svd_results, rtol=1e-5, atol=1e-8)
        print("The SVD logic replicates the scikit-learn pipeline")
    except AssertionError as e:
        print("Numerical mismatch detected.")
        print(e)


if __name__ == "__main__":
    run_benchmark()
