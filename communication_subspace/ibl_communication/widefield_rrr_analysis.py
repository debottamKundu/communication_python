import os
import pickle as pkl
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Suppress sklearn and runtime warnings to keep logs clean
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

    for idx in range(n_regions):
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
    Fast computation of null distribution using pre-computed optimal alphas.
    """
    n_regions = len(data_a)
    null_distributions = np.zeros((n_regions, n_regions, n_iterations))

    for idy in range(n_regions):
        base_region_y = data_b[idy][frameidy, :, :]

        # Pre-generate shifted targets for all iterations
        shifted_targets = []
        for iter_idx in range(n_iterations):
            trial_order = candidate_trials_matrix[:, iter_idx]
            shifted_targets.append(base_region_y[trial_order, :])

        for idx in range(n_regions):
            if idx == idy:
                continue

            region_x = data_a[idx][frameidx, :, :]
            alpha = optimal_alphas[idx, idy]

            for iter_idx in range(n_iterations):
                r2_null = ridgeregression_fixed_alpha(region_x, shifted_targets[iter_idx], alpha)
                null_distributions[idx, idy, iter_idx] = r2_null

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
    ) = args
    from one.api import ONE
    import traceback

    try:
        one = ONE(mode="local")
        logger.info(f"========== Processing Session {eid} ==========")
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
        frame_idx = 0
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
        candidate_trials = build_candidate_pools(trials, ["sign_cont", "prior"])
        pseudosession_matrix = generate_pseudosessions(
            candidate_trials, n_pseudosessions=n_pseudosessions
        )
        null_ridge_r2 = compute_regionwise_null_r2_fast(
            stimulus_data,
            choice_data,
            frame_idx,
            frame_idy,
            candidate_trials_matrix=pseudosession_matrix,
            optimal_alphas=optimal_alphas,
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

        # Filter for significant region pairs and run RRR / subspace alignment
        logger.info(f"Filtering significant pairs (p < {p_threshold} and R^2 > 0)...")
        significant_pairs = []
        for idx in range(n_regions):
            for idy in range(n_regions):
                if idx == idy:
                    continue
                if p_values[idx, idy] < p_threshold and true_ridge_r2[idx, idy] > 0:
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

        filename = os.path.join(output_dir, f"{eid}_rrr_results.pkl")
        with open(filename, "wb") as f:
            pkl.dump(storage_dict, f)
        logger.info(f"Saved session results to {filename}")
        return eid, True

    except Exception as e:
        logger.error(f"Failed to process session {eid}: {e}")
        logger.error(traceback.format_exc())
        return eid, False


def run_full_analysis(
    session_ids=None,
    n_pseudosessions=200,
    max_rank_cap=15,
    p_threshold=0.05,
    output_dir="./data/generated/rrr_analysis",
    run_behavioral_alignment=False,
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
        "/Users/dkundu/Documents/phd/communication_python/data/processed/wifi_trials_df_all.pkl"
    )
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
    run_full_analysis(
        n_pseudosessions=200,
        max_rank_cap=15,
        p_threshold=0.05,
    )
