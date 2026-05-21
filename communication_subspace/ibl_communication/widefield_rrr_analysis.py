import os
import pickle as pkl
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from brainwidemap import bwm_loading

# # Suppress sklearn and runtime warnings to keep logs clean
warnings.filterwarnings("ignore")

from one.api import ONE
from communication_subspace.ibl_communication.utils import (
    check_config,
    load_widefield_epoch,
    setup_logger,
    build_candidate_pools,
    generate_pseudosessions,
    get_high_low_masks,
    compute_regionwise_r2,
)
from communication_subspace.ibl_communication.intrinsic_dimensionality import (
    compute_intrinsic_dimensionality,
)
from communication_subspace.ibl_communication.crossvalidated_rrr import optimize_rrr_rank
from communication_subspace.adapted.alignment import alignment_input, alignment_output
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

logger = setup_logger("WidefieldRRR")

LOCAL_EIDS = ["f7d46a15-9498-40dc-90da-fb977ce844be", "76448b54-0d56-469a-9c5b-6bdd3b7bce3d"]


def ridgeregression_fixed_alpha(X, Y, alpha, n_splits=5):
    """
    Fits Ridge regression using a single pre-determined alpha.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    model = Ridge(alpha=alpha)
    fold_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        Y_train_scaled = scaler_Y.fit_transform(Y_train)
        X_test_scaled = scaler_X.transform(X_test)
        Y_test_scaled = scaler_Y.transform(Y_test)

        model.fit(X_train_scaled, Y_train_scaled)
        Y_pred = model.predict(X_test_scaled)
        fold_scores.append(r2_score(Y_test_scaled, Y_pred))

    return np.mean(fold_scores)


def compute_regionwise_r2_with_alphas(data_a, data_b, frameidx, frameidy, trialmask=None):
    """
    Computes region-to-region Ridge regressions and records both R2 and optimal alphas.
    """
    from communication_subspace.ibl_communication.crossvalidated_ridge import ridgeregression

    n_regions = len(data_a)
    cross_array_predictions = np.zeros((n_regions, n_regions))
    optimal_alphas = np.zeros((n_regions, n_regions))

    if trialmask is None:
        trialmask = np.ones(data_a[0].shape[1], dtype=bool)

    for idx in tqdm(range(n_regions), desc="True Ridge Regressions", leave=False):
        region_x = data_a[idx][frameidx, trialmask, :]
        for idy in range(n_regions):
            if idx == idy:
                continue
            region_y = data_b[idy][frameidy, trialmask, :]
            r2, alpha = ridgeregression(region_x, region_y)
            cross_array_predictions[idx, idy] = r2
            optimal_alphas[idx, idy] = alpha

    return cross_array_predictions, optimal_alphas


def compute_regionwise_null_r2_fast(
    data_a, data_b, frameidx, frameidy, candidate_trials_matrix, optimal_alphas, n_iterations=50
):
    """
    Blazing fast computation of null distribution.
    Pre-computes the (X^T X + alpha I)^-1 X^T projection matrix and StandardScalers for X
    once per fold, since X never changes during permutation testing.
    Reduces complexity from O(Nulls * Features^3) to O(Features^3 + Nulls * Features^2).
    """
    n_regions = len(data_a)
    null_distributions = np.zeros((n_regions, n_regions, n_iterations))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for idy in tqdm(range(n_regions), desc="Null Ridge Regressions (Target)", leave=False):
        base_region_y = data_b[idy][frameidy, :, :].astype(np.float64)

        # Pre-generate shifted targets for all iterations
        shifted_targets = []
        for iter_idx in range(n_iterations):
            trial_order = candidate_trials_matrix[:, iter_idx]
            shifted_targets.append(base_region_y[trial_order, :])

        for idx in range(n_regions):
            if idx == idy:
                continue

            region_x = data_a[idx][frameidx, :, :].astype(np.float64)
            alpha = optimal_alphas[idx, idy]

            # 1. Precompute X fold structures and projection matrices
            fold_data = []
            for train_idx, test_idx in kf.split(region_x):
                X_train, X_test = region_x[train_idx], region_x[test_idx]

                mean_X = np.mean(X_train, axis=0)
                std_X = np.std(X_train, axis=0)
                std_X[std_X == 0] = 1.0

                X_tr_sc = (X_train - mean_X) / std_X
                X_te_sc = (X_test - mean_X) / std_X

                n_feat = X_tr_sc.shape[1]
                XTX = X_tr_sc.T @ X_tr_sc
                XTX[np.diag_indices(n_feat)] += alpha

                try:
                    Proj = np.linalg.solve(XTX, X_tr_sc.T)
                except np.linalg.LinAlgError:
                    Proj = np.linalg.pinv(XTX) @ X_tr_sc.T

                fold_data.append((train_idx, test_idx, X_te_sc, Proj))

            # 2. Iterate through permuted Y targets using precomputed X Projections
            for iter_idx in range(n_iterations):
                Y_null = shifted_targets[iter_idx]
                fold_scores = []

                for train_idx, test_idx, X_te_sc, Proj in fold_data:
                    Y_train, Y_test = Y_null[train_idx], Y_null[test_idx]

                    mean_Y = np.mean(Y_train, axis=0)
                    std_Y = np.std(Y_train, axis=0)
                    std_Y[std_Y == 0] = 1.0

                    Y_tr_sc = (Y_train - mean_Y) / std_Y
                    Y_te_sc = (Y_test - mean_Y) / std_Y

                    # W = Proj @ Y_tr_sc
                    Y_pred = X_te_sc @ (Proj @ Y_tr_sc)

                    # Vectorized R2 Score perfectly mirroring sklearn logic
                    ss_res = np.sum((Y_te_sc - Y_pred) ** 2, axis=0)
                    ss_tot = np.sum((Y_te_sc - np.mean(Y_te_sc, axis=0)) ** 2, axis=0)
                    nonzero = ss_tot > 1e-10
                    r2_arr = np.zeros(Y_te_sc.shape[1])
                    r2_arr[nonzero] = 1 - ss_res[nonzero] / ss_tot[nonzero]

                    fold_scores.append(np.mean(r2_arr))

                null_distributions[idx, idy, iter_idx] = np.mean(fold_scores)

    return null_distributions


def process_single_session_worker(args):
    (
        eid,
        trials,
        config,
        n_pseudosessions,
        max_rank_cap,
        p_threshold,
        output_dir,
        run_behavioral_alignment,
        fast_null,
        min_r2_threshold,
    ) = args
    from one.api import ONE
    import traceback
    import time

    start_time = time.time()
    try:
        one = ONE(mode="local")
        logger.info(f"========== Processing Session {eid} ==========")

        _, mask = bwm_loading.load_trials_and_mask(
            one,
            eid,
            exclude_nochoice=True,
            exclude_unbiased=False,
        )

        if np.sum(~mask) > 0:
            logger.warning(
                f"Session {eid}: Dropping {np.sum(~mask)} out of {len(trials)} trials using canonical IBL mask."
            )
            trials = trials[mask].reset_index(drop=True)

        # Load widefield data for stimulus and choice epochs
        logger.info(f"Loading widefield data for session {eid}")

        stimulus_data, region_names_stim = load_widefield_epoch(
            one, eid, trials, config["hemisphere"], epoch="stim"
        )
        choice_data, region_names_choice = load_widefield_epoch(
            one, eid, trials, config["hemisphere"], epoch="choice"
        )

        assert (
            region_names_stim == region_names_choice
        ), "Stimulus and Choice region names do not match"
        regions = [r[0] for r in region_names_stim]
        n_regions = len(regions)
        logger.info(f"Loaded {n_regions} regions: {regions}")

        # Get high and low engagement masks
        if "engagement" in trials.columns:
            engagement_signal = trials["engagement"].values
        elif "motivation" in trials.columns:
            engagement_signal = trials["motivation"].values
        else:
            raise KeyError("Neither 'engagement' nor 'motivation' found in trials columns")
        high_mask, low_mask = get_high_low_masks(engagement_signal)
        logger.info(f"Engagement split: {np.sum(high_mask)} high, {np.sum(low_mask)} low trials")

        # Compute intrinsic dimensionality
        logger.info("Computing intrinsic dimensionality...")
        stim_dim_pca, _ = compute_intrinsic_dimensionality(stimulus_data)
        stim_dim_pca_high, _ = compute_intrinsic_dimensionality(stimulus_data, mask=high_mask)
        stim_dim_pca_low, _ = compute_intrinsic_dimensionality(stimulus_data, mask=low_mask)

        choice_dim_pca, _ = compute_intrinsic_dimensionality(choice_data)
        choice_dim_pca_high, _ = compute_intrinsic_dimensionality(choice_data, mask=high_mask)
        choice_dim_pca_low, _ = compute_intrinsic_dimensionality(choice_data, mask=low_mask)

        # Compute true and null Ridge regression for all trials
        # We look at frame_idx=0 (stimOn) and frame_idy=1 (movement)
        frame_idx = 1
        frame_idy = 1

        logger.info("Computing true Ridge R^2 and optimal alphas...")
        true_ridge_r2, optimal_alphas = compute_regionwise_r2_with_alphas(
            stimulus_data, choice_data, frame_idx, frame_idy
        )

        logger.info("Computing true Ridge R^2 for High and Low engagement splits...")
        high_ridge_r2 = compute_regionwise_r2(
            stimulus_data, choice_data, frame_idx, frame_idy, trialmask=high_mask
        )
        low_ridge_r2 = compute_regionwise_r2(
            stimulus_data, choice_data, frame_idx, frame_idy, trialmask=low_mask
        )

        logger.info(
            f"Generating proper null distributions (N={n_pseudosessions} pseudo-sessions)..."
        )

        # Dynamically build features for the null distribution KDTree
        kd_features = ["sign_cont", "prior"]
        if "engagement" in trials.columns:
            kd_features.append("engagement")
        elif "motivation" in trials.columns:
            kd_features.append("motivation")

        candidate_trials = build_candidate_pools(trials, kd_features)
        pseudosession_matrix = generate_pseudosessions(
            candidate_trials, n_pseudosessions=n_pseudosessions
        )
        if fast_null:
            null_ridge_r2 = compute_regionwise_null_r2_fast(
                stimulus_data,
                choice_data,
                frame_idx,
                frame_idy,
                candidate_trials_matrix=pseudosession_matrix,
                optimal_alphas=optimal_alphas,
                n_iterations=n_pseudosessions,
            )
        else:
            from communication_subspace.ibl_communication.utils import (
                compute_regionwise_null_r2,
                compute_regionwise_null_r2_svd,
            )

            null_ridge_r2 = compute_regionwise_null_r2_svd(
                stimulus_data,
                choice_data,
                frame_idx,
                frame_idy,
                candidate_trials_matrix=pseudosession_matrix,
                n_iterations=n_pseudosessions,
            )

        # Calculate p-values for All trials
        logger.info("Calculating empirical p-values from null distributions...")
        p_values = np.ones((n_regions, n_regions))
        for idx in range(n_regions):
            for idy in range(n_regions):
                if idx == idy:
                    continue
                true_val = true_ridge_r2[idx, idy]
                null_vals = null_ridge_r2[idx, idy, :]
                p_values[idx, idy] = (np.sum(null_vals >= true_val) + 1) / (len(null_vals) + 1)

        # Apply FDR correction (Benjamini-Hochberg)
        logger.info("Applying Benjamini-Hochberg FDR correction to region pairs...")
        try:
            from statsmodels.stats.multitest import fdrcorrection

            valid_p_indices = []
            valid_p_vals = []
            for idx in range(n_regions):
                for idy in range(n_regions):
                    if idx != idy and true_ridge_r2[idx, idy] > min_r2_threshold:
                        valid_p_indices.append((idx, idy))
                        valid_p_vals.append(p_values[idx, idy])

            if len(valid_p_vals) > 0:
                rejected, pvals_corrected = fdrcorrection(valid_p_vals, alpha=p_threshold)

                p_values_fdr = np.ones((n_regions, n_regions))
                for (idx, idy), p_corr in zip(valid_p_indices, pvals_corrected):
                    p_values_fdr[idx, idy] = p_corr

                p_values = p_values_fdr
        except ImportError:
            logger.warning("statsmodels not installed, skipping FDR correction!")

        # Filter for significant region pairs and run RRR / subspace alignment
        logger.info(
            f"Filtering significant pairs (FDR p < {p_threshold} and R^2 > {min_r2_threshold})..."
        )
        significant_pairs = []
        for idx in range(n_regions):
            for idy in range(n_regions):
                if idx == idy:
                    continue
                if p_values[idx, idy] < p_threshold and true_ridge_r2[idx, idy] > min_r2_threshold:
                    significant_pairs.append((idx, idy))

        logger.info(
            f"Found {len(significant_pairs)} significant region pairs out of {n_regions * (n_regions - 1)}"
        )

        rrr_results = {}
        for idx, idy in tqdm(
            significant_pairs, desc=f"Fitting RRR & Subspace Alignment for {eid}"
        ):
            X_all = stimulus_data[idx][frame_idx, :, :]
            Y_all = choice_data[idy][frame_idy, :, :]

            conditions = {
                "all": (X_all, Y_all, None),
                "high": (X_all[high_mask, :], Y_all[high_mask, :], high_mask),
                "low": (X_all[low_mask, :], Y_all[low_mask, :], low_mask),
            }

            pair_res = {}
            for cond_name, (X, Y, mask) in conditions.items():
                if X.shape[0] < 10 or Y.shape[0] < 10:
                    # Not enough trials to split
                    continue
                try:
                    # Fit RRR
                    rrr_opt = optimize_rrr_rank(
                        X, Y, n_splits=5, viz=False, detailed=True, max_rank=max_rank_cap
                    )
                    W = rrr_opt["full_weight_matrix"]

                    # Compute alignment indices
                    align_in, _, _ = alignment_input(X, W)
                    align_out, comm_frac = alignment_output(X, Y, W)

                    pair_res[cond_name] = {
                        "optimal_rank": rrr_opt["optimal_rank"],
                        "cv_r2_rrr": rrr_opt["cv_r2"],
                        "input_alignment": align_in,
                        "output_alignment": align_out,
                        "comm_fraction": comm_frac,
                        "mean_r2_curve": rrr_opt["mean_r2"],
                        "full_weight_matrix": W,
                    }
                except Exception as ex:
                    logger.error(
                        f"Error in RRR for pair ({regions[idx]} -> {regions[idy]}) [{cond_name}]: {ex}"
                    )

            if pair_res:
                rrr_results[(idx, idy)] = pair_res

        # Save all results to pickle
        storage_dict = {
            "session_id": eid,
            "regions": regions,
            "true_ridge_r2": true_ridge_r2,
            "high_ridge_r2": high_ridge_r2,
            "low_ridge_r2": low_ridge_r2,
            "p_values": p_values,
            "null_ridge_r2": null_ridge_r2,
            "significant_pairs": significant_pairs,
            "rrr_results": rrr_results,
            "intrinsic_dimensionality": {
                "stim": {
                    "all": (stim_dim_pca),
                    "high": (stim_dim_pca_high),
                    "low": (stim_dim_pca_low),
                },
                "choice": {
                    "all": (choice_dim_pca),
                    "high": (choice_dim_pca_high),
                    "low": (choice_dim_pca_low),
                },
            },
        }

        filename = os.path.join(output_dir, f"{eid}_rrr_results_svd_null_frame1.pkl")
        with open(filename, "wb") as f:
            pkl.dump(storage_dict, f)

        end_time = time.time()
        duration_mins = (end_time - start_time) / 60.0
        logger.info(f"Saved session results to {filename}")
        logger.info(
            f"====== Session {eid} finished successfully in {duration_mins:.2f} minutes ======"
        )
        return eid, True

    except Exception as e:
        end_time = time.time()
        duration_mins = (end_time - start_time) / 60.0
        logger.error(f"Failed to process session {eid} after {duration_mins:.2f} minutes: {e}")
        logger.error(traceback.format_exc())
        return eid, False


def run_full_analysis(
    session_ids=None,
    n_pseudosessions=200,
    max_rank_cap=15,
    p_threshold=0.05,
    output_dir="./data/generated/rrr_analysis",
    run_behavioral_alignment=False,
    fast_null=True,
    min_r2_threshold=0.0,
):
    """
    Loads widefield data, partitions trials into engagement states,
    identifies significant region pairs using a null distribution,
    fits RRR and calculates subspace alignment for significant pairs,
    and saves results. Runs across sessions in parallel.
    """
    if session_ids is None:
        session_ids = LOCAL_EIDS

    os.makedirs(output_dir, exist_ok=True)
    config = check_config()

    # Load trials data
    trials_path = (
        "/usr/people/kundu/code/communication_python/data/processed/wifi_trials_df_all.pkl"
    )
    # local:
    # trials_path = (
    #     "/Users/dkundu/Documents/phd/communication_python/data/processed/wifi_trials_df_all.pkl"
    # )
    logger.info(f"Loading trials data from {trials_path}")
    with open(trials_path, "rb") as f:
        all_trials = pkl.load(f)

    # Prepare arguments for multiprocessing
    tasks = []
    for eid in session_ids:
        if eid not in all_trials:
            logger.warning(f"Session {eid} not found in trials dictionary. Skipping.")
            continue
        tasks.append(
            (
                eid,
                all_trials[eid],
                config,
                n_pseudosessions,
                max_rank_cap,
                p_threshold,
                output_dir,
                run_behavioral_alignment,
                fast_null,
                min_r2_threshold,
            )
        )

    # Run in parallel using ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor, as_completed

    logger.info(f"Starting parallel processing for {len(tasks)} sessions...")

    # Use max_workers=None to use all available cores, or set a specific number
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_session_worker, task): task[0] for task in tasks}

        for future in as_completed(futures):
            eid = futures[future]
            try:
                result_eid, success = future.result()
                if success:
                    logger.info(f"Successfully finished processing session {result_eid}")
                else:
                    logger.error(f"Session {result_eid} failed during processing.")
            except Exception as exc:
                logger.error(f"Session {eid} generated an exception: {exc}")


if __name__ == "__main__":

    # Run full pipeline
    # Use 50 null iterations, and cap RRR rank at 15
    # find sessions ids
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    sessions = one.search(datasets="widefieldU.images.npy")

    session_eids = np.asarray([str(sess) for sess in sessions])  # type: ignore
    single = False
    if single:
        run_full_analysis(
            session_ids=[session_eids[0]],
            n_pseudosessions=200,
            max_rank_cap=15,
            p_threshold=0.05,
            fast_null=False,
        )
    else:
        run_full_analysis(
            session_ids=session_eids,
            n_pseudosessions=200,
            max_rank_cap=15,
            p_threshold=0.05,
            fast_null=False,
        )
