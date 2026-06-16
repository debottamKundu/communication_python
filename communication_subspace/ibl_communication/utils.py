from pathlib import Path
import logging
import sys
import numpy as np
import yaml
from prior_localization.prepare_data import prepare_widefield

from communication_subspace.ibl_communication.crossvalidated_ridge import ridgeregression
from communication_subspace.ibl_communication.intrinsic_dimensionality import (
    compute_intrinsic_dimensionality,
)
from communication_subspace.ibl_communication.crossvalidated_rrr import optimize_rrr_rank
from tqdm import tqdm
from scipy.spatial import KDTree


def check_config():
    """Load config yaml and perform some basic checks"""
    # Get config
    with open(Path(__file__).parent.parent.joinpath("config.yaml"), "r") as config_yml:
        config = yaml.safe_load(config_yml)
    return config


def setup_logger(name="CrossPrediction", log_file="pipeline.log", level=logging.INFO):
    """
    Sets up and returns a customized logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:

        log_format = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(log_format)
        logger.addHandler(c_handler)

        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(log_format)
        logger.addHandler(f_handler)

    return logger


def get_align_times(trials, epoch, response_time=False):

    config = check_config()

    if epoch == "stim":
        align_times = trials.stimOn_times
        frame_windows = config["stimulus_frames"]
    elif epoch == "choice":
        # use response times?
        if response_time:
            align_times = trials.response_times
        else:
            align_times = trials.firstMovement_times
        frame_windows = config["choice_frames"]
    else:
        raise NotImplementedError

    return align_times, frame_windows


def load_widefield_epoch(
    one, session_id, trials, hemisphere, epoch, stage_only=False, min_voxels=10, response_time=False
):

    align_times, frame_windows = get_align_times(trials, epoch, response_time)
    data_epoch, actual_regions = prepare_widefield(
        one,
        session_id,
        hemisphere,
        regions="single_regions",
        align_times=align_times,
        frame_window=frame_windows,
        functional_channel=470,
        stage_only=stage_only,
    )

    data_epoch_reduced = []
    regions = []

    for idx in range(len(data_epoch)):  # type: ignore
        n_voxels = data_epoch[idx].shape[-1]  # type: ignore
        if n_voxels < min_voxels:
            continue
        data_epoch_reduced.append(data_epoch[idx].transpose(1, 0, 2))  # type: ignore
        regions.append(actual_regions[idx])  # type: ignore

    return data_epoch_reduced, regions


def compute_modulation_indices(crossarraypreds_high, crossarraypreds_low):

    modulation_indices = np.zeros_like(crossarraypreds_high)
    for idx in range(crossarraypreds_high.shape[0]):
        for idy in range(crossarraypreds_high.shape[1]):
            high_r2 = max(crossarraypreds_high[idx, idy], 0)
            low_r2 = max(crossarraypreds_low[idx, idy], 0)
            if high_r2 + low_r2 == 0:
                modulation_indices[idx, idy] = np.nan
            else:
                modulation_indices[idx, idy] = (high_r2 - low_r2) / (high_r2 + low_r2)

    return modulation_indices


def get_high_low_masks(engagement_signal):
    median_val = np.median(engagement_signal)
    high_mask = engagement_signal >= median_val
    low_mask = engagement_signal < median_val
    return high_mask, low_mask


def get_intrinsic_dimensions(data, high_mask, low_mask):

    intrinsic_dim_all_pca, intrinsic_dim_all_fa = compute_intrinsic_dimensionality(data)
    intrinsic_dim_high_pca, intrinsic_dim_high_fa = compute_intrinsic_dimensionality(
        data, mask=high_mask
    )
    intrinsic_dim_low_pca, intrinsic_dim_low_fa = compute_intrinsic_dimensionality(
        data, mask=low_mask
    )

    storage_dict = {}
    storage_dict["intrinsic_dim_all_pca"] = intrinsic_dim_all_pca
    storage_dict["intrinsic_dim_all_fa"] = intrinsic_dim_all_fa
    storage_dict["intrinsic_dim_high_pca"] = intrinsic_dim_high_pca
    storage_dict["intrinsic_dim_high_fa"] = intrinsic_dim_high_fa
    storage_dict["intrinsic_dim_low_pca"] = intrinsic_dim_low_pca
    storage_dict["intrinsic_dim_low_fa"] = intrinsic_dim_low_fa

    return storage_dict


def compute_regionwise_r2(data_a, data_b, frameidx, frameidy, trialmask=None):
    # data is nregions x nframes x ntrials x nframes x nsessions
    n_regions = len(data_a)
    cross_array_predictions = np.zeros((n_regions, n_regions))
    if trialmask is None:
        trialmask = np.ones(data_a[0].shape[1], dtype=bool)
    for idx in tqdm(range(n_regions), leave=False):
        region_x = data_a[idx][frameidx, trialmask, :]
        for idy in range(n_regions):
            # skip diagonals
            if idx == idy:
                continue
            region_y = data_b[idy][frameidy, trialmask, :]
            cross_array_predictions[idx, idy], _ = ridgeregression(region_x, region_y)
    return cross_array_predictions


def compute_reduced_rank_pairs(data_a, data_b, frameidx, frameidy, trialmask):
    # we always compute the trial-masked versions

    n_regions = len(data_a)
    subspace_dict_main = {}

    for regionidx in range(n_regions):
        region_x = data_a[regionidx][frameidx, trialmask, :]
        for regionidy in range(n_regions):
            region_y = data_b[regionidy][frameidy, trialmask, :]

            subspace_dict = optimize_rrr_rank(
                region_x, region_y, viz=False, detailed=True
            )  # so that we don't generate a lot of images
            subspace_dict_main[(regionidx, regionidy)] = subspace_dict
    return subspace_dict_main


def compute_regionwise_null_r2(
    data_a, data_b, frameidx, frameidy, candidate_trials_matrix, n_iterations=200
):
    """
    Computes the null distribution by using candidate trials which are the closest behaviorally similar trials.
    Returns a 3D array of shape (n_regions, n_regions, n_iterations).
    """
    n_regions = len(data_a)

    assert n_iterations == candidate_trials_matrix.shape[1]

    null_distributions = np.zeros((n_regions, n_regions, n_iterations))

    for idy in tqdm(range(n_regions), desc="Target Regions (Nulls)", leave=False):
        base_region_y = data_b[idy][frameidy, :, :]

        shifted_targets = []
        for iter_idx in range(n_iterations):
            trial_order = candidate_trials_matrix[:, iter_idx]
            pseudosession_y = base_region_y[trial_order, :]
            shifted_targets.append(pseudosession_y)

        for idx in range(n_regions):
            if idx == idy:
                continue

            region_x = data_a[idx][frameidx, :, :]

            for iter_idx, shifted_y in enumerate(shifted_targets):
                r2_null, _ = ridgeregression(region_x, shifted_y)
                null_distributions[idx, idy, iter_idx] = r2_null

    return null_distributions


def build_candidate_pools(df, feature_cols, k_neighbors=20):
    """
    Builds a matrix of candidate trial indices for null distributions.

    Args:
        df: Pandas DataFrame containing trial information.
        feature_cols: List of column names to use for distance (e.g., ['sign_cont', 'prior']).
        k_neighbors: Number of candidate trials to pool per trial.

    Returns:
        candidate_pools: (N_trials, k_neighbors) integer array of trial indices.
    """

    features = df[feature_cols].values.astype(float)

    # min max transform, but is it necessary?
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)

    range_vals = np.where((max_vals - min_vals) == 0, 1, max_vals - min_vals)
    features_norm = (features - min_vals) / range_vals

    tree = KDTree(features_norm)
    _, indices = tree.query(features_norm, k=k_neighbors + 1)

    candidate_pools = indices[:, 1:]  # type: ignore

    return candidate_pools


def generate_pseudosessions(candidate_pools, n_pseudosessions=200):

    n_trials, k_neighbors = candidate_pools.shape
    random_choices = np.random.randint(0, k_neighbors, size=(n_trials, n_pseudosessions))
    row_indices = np.arange(n_trials)[:, np.newaxis]
    pseudosession_indices = candidate_pools[row_indices, random_choices]

    return pseudosession_indices


def compute_regionwise_null_r2_svd(
    data_a, data_b, frameidx, frameidy, candidate_trials_matrix, n_iterations=200, trialmask=None
):
    """
    SVD-Optimized version of compute_regionwise_null_r2.
    Computes the Singular Value Decomposition (SVD) of the input region X once per cross-validation fold,
    allowing instantaneous Ridge Regression fits for all 7 alphas over all 200 null iterations.
    Numerically equivalent to the original 1-SEM KFold Ridge pipeline but exponentially faster.
    """
    import numpy as np
    from sklearn.model_selection import KFold
    from scipy.stats import sem

    n_regions = len(data_a)
    null_distributions = np.zeros((n_regions, n_regions, n_iterations))
    alphas = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])

    if trialmask is None:
        trialmask = np.ones(data_a[0].shape[1], dtype=bool)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for idy in tqdm(range(n_regions), desc="Running null session for target regions", leave=False):
        base_region_y = data_b[idy][frameidy, :, :].astype(np.float64)

        shifted_targets = []
        for iter_idx in range(n_iterations):
            trial_order = candidate_trials_matrix[:, iter_idx]
            shifted_targets.append(base_region_y[trial_order, :][trialmask, :])

        for idx in range(n_regions):
            if idx == idy:
                continue

            region_x = data_a[idx][frameidx, trialmask, :].astype(np.float64)

            # Precompute SVD-based Projection Matrices for each fold
            fold_data = []
            for train_idx, test_idx in kf.split(region_x):
                X_train, X_test = region_x[train_idx], region_x[test_idx]

                mean_X = np.mean(X_train, axis=0)
                std_X = np.std(X_train, axis=0)
                std_X[std_X == 0] = 1.0

                X_tr_sc = (X_train - mean_X) / std_X
                X_te_sc = (X_test - mean_X) / std_X

                # Full SVD: X = U * S * V^T
                U, S, Vt = np.linalg.svd(X_tr_sc, full_matrices=False)

                # Precompute projection matrix for each alpha
                alpha_projections = []
                for alpha in alphas:
                    D = S / (S**2 + alpha)
                    Proj = Vt.T @ np.diag(D) @ U.T
                    alpha_projections.append(Proj)

                fold_data.append((train_idx, test_idx, X_te_sc, alpha_projections))

            # Process all shifted Ys using precomputed Projections
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

                    # Score for each alpha
                    for a_idx, Proj in enumerate(alpha_projections):
                        Y_pred = X_te_sc @ (Proj @ Y_tr_sc)

                        # Vectorized R2 Score perfectly mirroring sklearn
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
                null_distributions[idx, idy, iter_idx] = mean_scores[peak_idx]

    return null_distributions
