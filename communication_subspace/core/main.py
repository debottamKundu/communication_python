from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from communication_subspace.core.faCrossVal import cross_val_fa, factor_analysis_model_select
from communication_subspace.core.reducedRankcrossval import (
    cross_validate_rrr,
    select_optimal_dimension,
)


if __name__ == "__main__":

    data = loadmat("./data/sample_data.mat")

    X = data["X"]
    Y_V1 = data["Y_V1"]
    Y_V2 = data["Y_V2"]

    dims_to_test = np.arange(1, 11)
    cv_folds = 10

    # print("Running Cross Validation...")
    # cv_results = cross_validate_rrr(X, Y_V1, dims_to_test, k_folds=cv_folds)

    # opt_dim, opt_idx, threshold = select_optimal_dimension(cv_results)
    # print(f"Optimal Dimension selected: {opt_dim}")

    # x = cv_results["dims"]
    # y = 1.0 - cv_results["mean_loss"]  # Convert Loss (NSE) back to Performance (R2)
    # error = cv_results["ste_loss"]

    # plt.figure(figsize=(8, 5))

    # plt.errorbar(
    #     x, y, yerr=error, fmt="o--", capsize=5, color="blue", ecolor="gray", label="CV Performance"
    # )

    # plt.plot(x[opt_idx], y[opt_idx], "ro", markersize=10, label=f"Optimal Dim ({opt_dim})")

    # # Formatting
    # plt.xlabel("Number of predictive dimensions")
    # plt.ylabel("Predictive performance ($R^2$)")
    # plt.title("Reduced Rank Regression Cross-Validation")
    # plt.grid(True, linestyle=":", alpha=0.6)
    # plt.legend()
    # plt.tight_layout()

    # plt.show()
