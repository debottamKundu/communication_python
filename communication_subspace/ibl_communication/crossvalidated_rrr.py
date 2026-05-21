import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import sem
from sklearn.preprocessing import StandardScaler
from communication_subspace.adapted.rrr import svd_RRR
from matplotlib import pyplot as plt


def optimize_rrr_rank(X, Y, optimal_lambda=0, n_splits=5, viz=True, detailed=True, max_rank=None):
    """
    Finds the optimal communication rank using 5-fold CV and the 1-SEM rule.
    Plots the R^2 curve against rank dimensionality.
    """
    limit = min(X.shape[1], Y.shape[1])
    if max_rank is not None:
        limit = min(limit, max_rank)
    ranks_to_test = np.arange(1, limit + 1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_r2 = []
    sem_r2 = []

    for rank in ranks_to_test:
        fold_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # try catch
            try:

                scaler_X = StandardScaler()
                scaler_Y = StandardScaler()
                X_train_scaled = scaler_X.fit_transform(X_train)
                Y_train_scaled = scaler_Y.fit_transform(Y_train)
                X_test_scaled = scaler_X.transform(X_test)
                Y_test_scaled = scaler_Y.transform(Y_test)
                w0, _, _ = svd_RRR(
                    X_train_scaled, Y_train_scaled, rnk=rank, lambda_=optimal_lambda
                )
                Y_pred = X_test_scaled @ w0
                score = r2_score(Y_test_scaled, Y_pred)
                fold_scores.append(score)
            except Exception as e:
                print(f"Warning: Fold failed at rank {rank} (Error: {e}). Skipping fold.")
                fold_scores.append(np.nan)

        mean_r2.append(np.nanmean(fold_scores))
        sem_r2.append(sem(fold_scores, nan_policy="omit"))

    mean_r2 = np.array(mean_r2)
    sem_r2 = np.array(sem_r2)

    peak_idx = np.argmax(mean_r2)
    peak_score = mean_r2[peak_idx]
    peak_sem = sem_r2[peak_idx]

    threshold = peak_score - peak_sem
    valid_indices = np.where(mean_r2 >= threshold)[0]
    optimal_idx = valid_indices[0]
    optimal_rank = ranks_to_test[optimal_idx]
    final_r2 = mean_r2[optimal_idx]

    if viz:
        plt.figure(figsize=(8, 5))

        plt.errorbar(
            ranks_to_test,
            mean_r2,
            yerr=sem_r2,
            fmt="-o",
            color="royalblue",
            capsize=4,
            linewidth=2,
        )

        plt.axhline(y=threshold, color="gray", linestyle="--", alpha=0.7, label="1-SEM Threshold")
        plt.plot(
            ranks_to_test[peak_idx],
            peak_score,
            "s",
            color="orange",
            markersize=8,
            label="Absolute Peak",
        )
        plt.plot(
            optimal_rank,
            final_r2,
            "D",
            color="red",
            markersize=10,
            label=f"Optimal Rank ({optimal_rank})",
        )

        plt.title("Reduced-Rank Regression")
        plt.xlabel("Number of Predictive Dimensions (Rank)")
        plt.ylabel("Cross-Validated $R^2$")
        plt.xticks(ranks_to_test)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        plt.show()

    if detailed:
        w0_final, urrr_final, vrrr_final = svd_RRR(X, Y, rnk=optimal_rank, lambda_=optimal_lambda)

        subspace_data = {
            "optimal_rank": optimal_rank,
            "cv_r2": final_r2,
            "source_axes_U": urrr_final,
            "target_axes_V": vrrr_final,
            "full_weight_matrix": w0_final,
            "mean_r2": mean_r2,
            "sem_r2": sem_r2,
        }

    else:
        subspace_data = {
            "optimal_rank": optimal_rank,
            "cv_r2": final_r2,
            "mean_r2": mean_r2,
            "sem_r2": sem_r2,
        }

    return subspace_data
