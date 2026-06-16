import os
import pickle as pkl
import logging
import warnings
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from joblib import Parallel, delayed

from one.api import ONE
from iblatlas.regions import BrainRegions
from brainwidemap.bwm_loading import load_good_units, merge_probes, load_trials_and_mask, bwm_units
from brainbox.population.decode import get_spike_counts_in_bins

from communication_subspace.ibl_communication.utils import (
    build_candidate_pools,
    generate_pseudosessions,
    compute_regionwise_null_r2_svd,
    setup_logger,
)
from communication_subspace.ibl_communication.crossvalidated_ridge import ridgeregression
from communication_subspace.ibl_communication.crossvalidated_rrr import optimize_rrr_rank

logger = setup_logger("EphysRegionCommunication")


def prepare_ephys_epochs(one, eid, trials_df, regions_stim, regions_choice):
    """
    Loads ephys spikes for a given session and bins them for the requested regions.
    epochs:
      'stim': 0 to 100ms after stimOn_times
      'choice': -100ms to 0ms relative to firstMovement_times
    """
    pids, pnames = one.eid2pid(eid)
    to_merge = [
        load_good_units(one, pid, pname=pname, eid=eid) for pid, pname in zip(pids, pnames)
    ]
    spikes, clusters = merge_probes([s for s, _ in to_merge], [c for _, c in to_merge])

    brainreg = BrainRegions()
    beryl_regions = brainreg.acronym2acronym(clusters["acronym"], mapping="Beryl")

    intervals_stim = np.c_[trials_df.stimOn_times.values, trials_df.stimOn_times.values + 0.1]
    intervals_choice = np.c_[
        trials_df.firstMovement_times.values - 0.1, trials_df.firstMovement_times.values
    ]

    stim_data = []
    stim_valid_regions = []
    for region in regions_stim:
        # Wrap region in list if it's a string, assuming single region here
        if isinstance(region, str):
            region = [region]

        region_mask = np.isin(beryl_regions, region)
        if sum(region_mask) < 10:
            continue

        spike_mask = np.isin(spikes["clusters"], clusters[region_mask].index)
        binned, _ = get_spike_counts_in_bins(
            spikes["times"][spike_mask], spikes["clusters"][spike_mask], intervals_stim
        )

        # Filter out silent neurons (must have spiked at least once in these intervals)
        valid_units = np.sum(binned, axis=1) > 0
        binned = binned[valid_units, :]

        if binned.shape[0] < 10:
            continue

        # reshape to (1, n_trials, n_units) to be consistent with downstream arrays
        stim_data.append(np.expand_dims(binned.T, axis=0))
        stim_valid_regions.append(region[0])

    choice_data = []
    choice_valid_regions = []
    for region in regions_choice:
        if isinstance(region, str):
            region = [region]

        region_mask = np.isin(beryl_regions, region)
        if sum(region_mask) < 10:
            continue

        spike_mask = np.isin(spikes["clusters"], clusters[region_mask].index)
        binned, _ = get_spike_counts_in_bins(
            spikes["times"][spike_mask], spikes["clusters"][spike_mask], intervals_choice
        )

        # Filter out silent neurons (must have spiked at least once in these intervals)
        valid_units = np.sum(binned, axis=1) > 0
        binned = binned[valid_units, :]

        if binned.shape[0] < 10:
            continue

        choice_data.append(np.expand_dims(binned.T, axis=0))
        choice_valid_regions.append(region[0])

    return stim_data, stim_valid_regions, choice_data, choice_valid_regions


def compute_r2(data_stim, data_choice, trial_mask):
    n_stim = len(data_stim)
    n_choice = len(data_choice)
    r2_matrix = np.zeros((n_stim, n_choice))
    for i in range(n_stim):
        reg_x = data_stim[i][0, trial_mask, :]
        for j in range(n_choice):
            reg_y = data_choice[j][0, trial_mask, :]
            r2, _ = ridgeregression(reg_x, reg_y)
            r2_matrix[i, j] = r2
    return r2_matrix


def compute_rrr(data_stim, data_choice, trial_mask):
    n_stim = len(data_stim)
    n_choice = len(data_choice)
    rrr_dict = {}
    for i in range(n_stim):
        reg_x = data_stim[i][0, trial_mask, :]
        for j in range(n_choice):
            reg_y = data_choice[j][0, trial_mask, :]
            try:
                subspace = optimize_rrr_rank(reg_x, reg_y, viz=False, detailed=True)
                rrr_dict[(i, j)] = subspace
            except Exception as e:
                logger.warning(f"RRR failed for pair ({i}, {j}): {e}")
                rrr_dict[(i, j)] = None
    return rrr_dict


def process_session(eid, df_ephys, n_pseudosessions=200):
    """
    Main processing function for a single animal (eid).
    """
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    try:
        trials_df, mask = load_trials_and_mask(one, eid)
    except Exception as e:
        logger.error(f"Failed to load trials for {eid}: {e}")
        return False

    trials_df = trials_df[mask].copy()

    # Pre-process trials: Contrast, Side, Feedback
    contrastLeft = trials_df["contrastLeft"].fillna(0)
    contrastRight = trials_df["contrastRight"].fillna(0)
    trials_df["signed_contrast"] = contrastLeft - contrastRight

    # Filter valid times (No NaNs)
    valid_times = ~np.isnan(trials_df["stimOn_times"]) & ~np.isnan(
        trials_df["firstMovement_times"]
    )
    trials_df = trials_df[valid_times].reset_index(drop=True)

    # Keep 0.5 probability trials for the complete dataset, but identify block side for congruent/incongruent
    block_side = np.zeros(len(trials_df))
    block_side[trials_df["probabilityLeft"] == 0.8] = 1
    block_side[trials_df["probabilityLeft"] == 0.2] = -1

    # 0 contrast is filtered out for all sets (stimulus must be visible)
    stim_mask = trials_df["signed_contrast"] != 0
    stim_side = np.sign(trials_df["signed_contrast"])

    # Create masks
    complete_mask = stim_mask

    # Congruent/Incongruent drops the 0.5 trials (where block_side == 0)
    congruent_mask = (block_side == stim_side) & (block_side != 0) & stim_mask
    incongruent_mask = (block_side != stim_side) & (block_side != 0) & stim_mask

    # Subsample congruent to match incongruent (we will do this 10 times later)
    n_incongruent = incongruent_mask.sum()
    congruent_indices = np.where(congruent_mask)[0]

    # Subset to complete data array
    # Everything downstream relies on the index in the complete_data subset
    complete_idx = np.where(complete_mask)[0]

    trials_complete = trials_df.iloc[complete_idx].reset_index(drop=True)

    # Mask of incongruent relative to the complete data
    incongruent_sub_mask = incongruent_mask[complete_idx]

    # Regions to evaluate for this eid based on the valid region pairs
    valid_rows = df_ephys[df_ephys["eids_list"].apply(lambda x: eid in x)]
    regions_stim = valid_rows["Beryl_stim"].unique().tolist()
    regions_choice = valid_rows["Beryl_choice"].unique().tolist()

    if len(regions_stim) == 0 or len(regions_choice) == 0:
        logger.warning(f"No valid regions found for {eid}")
        return False

    # Load Ephys Data
    logger.info(f"Loading ephys data for {eid}")
    try:
        data_stim, valid_stim, data_choice, valid_choice = prepare_ephys_epochs(
            one, eid, trials_complete, regions_stim, regions_choice
        )
    except Exception as e:
        logger.error(f"Failed to prepare ephys for {eid}: {e}")
        return False

    if not data_stim or not data_choice:
        logger.warning(f"No valid data returned after ephys loading for {eid}")
        return False

    # 1. Complete Data Analysis
    logger.info(f"Running complete data analysis for {eid}")
    all_trials_mask = np.ones(len(trials_complete), dtype=bool)
    true_r2_complete = compute_r2(data_stim, data_choice, all_trials_mask)
    rrr_complete = compute_rrr(data_stim, data_choice, all_trials_mask)

    # Generate Pseudosessions
    logger.info(f"Generating null distributions for {eid}")
    candidate_trials = build_candidate_pools(trials_complete, ["signed_contrast", "choice"])
    pseudosession_matrix = generate_pseudosessions(
        candidate_trials, n_pseudosessions=n_pseudosessions
    )

    null_r2_complete = compute_regionwise_null_r2_svd(
        data_stim,
        data_choice,
        0,
        0,
        pseudosession_matrix,
        n_iterations=n_pseudosessions,
        trialmask=all_trials_mask,
    )

    # 2. Congruent Data Analysis
    logger.info(f"Running congruent data analysis for {eid} (10 subsamples)")
    true_r2_congruent_runs = []
    rrr_congruent_runs = []

    for _ in range(10):
        if len(congruent_indices) > n_incongruent:
            subsampled_congruent_idx = np.random.choice(
                congruent_indices, n_incongruent, replace=False
            )
        else:
            subsampled_congruent_idx = congruent_indices

        subsampled_congruent_mask = np.zeros(len(trials_df), dtype=bool)
        subsampled_congruent_mask[subsampled_congruent_idx] = True
        congruent_sub_mask = subsampled_congruent_mask[complete_idx]

        true_r2_congruent_runs.append(compute_r2(data_stim, data_choice, congruent_sub_mask))
        rrr_congruent_runs.append(compute_rrr(data_stim, data_choice, congruent_sub_mask))

    # 3. Incongruent Data Analysis
    logger.info(f"Running incongruent data analysis for {eid}")
    true_r2_incongruent = compute_r2(data_stim, data_choice, incongruent_sub_mask)
    rrr_incongruent = compute_rrr(data_stim, data_choice, incongruent_sub_mask)

    # Save results
    storage_dict = {
        "eid": eid,
        "regions_stim": valid_stim,
        "regions_choice": valid_choice,
        "complete": {
            "true_r2": true_r2_complete,
            "rrr": rrr_complete,
            "null_r2": null_r2_complete,
        },
        "congruent": {
            "true_r2_runs": true_r2_congruent_runs,
            "rrr_runs": rrr_congruent_runs,
            "n_trials": n_incongruent,
            "n_runs": 10,
        },
        "incongruent": {
            "true_r2": true_r2_incongruent,
            "rrr": rrr_incongruent,
            "n_trials": incongruent_sub_mask.sum(),
        },
    }

    output_dir = (
        "/Users/dkundu/Documents/phd/communication_python/data/generated/ephys_communication"
    )
    filename = os.path.join(output_dir, f"{eid}_region_predictions.pkl")

    with open(filename, "wb") as f:
        pkl.dump(storage_dict, f)

    logger.info(f"Finished processing {eid}")
    return True


if __name__ == "__main__":
    # We load the dataframe specifying pairs of regions
    # The pairs were found in ephys_pairwise_sessions.ipynb

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    df_ephys = pd.read_csv("/Users/dkundu/Documents/phd/communication_python/data/generated/ephys_communication/df_stim_choice_pairs.csv")
    
    # When loading from CSV, lists are saved as strings. We need to convert them back to actual lists.
    df_ephys["eids_list"] = df_ephys["eids_list"].apply(ast.literal_eval)

    all_eids = []
    for eids in df_ephys["eids_list"]:
        all_eids.extend(eids)
    eids_to_process = np.unique(all_eids)

    logger.info(f"Found {len(eids_to_process)} sessions to process.")

    multiprocess = False

    if not multiprocess:
        eid = eids_to_process[0]
        print(eid)
        process_session(eid, df_ephys, n_pseudosessions=2)
    else:
        results = Parallel(n_jobs=8)(
            delayed(process_session)(eid, df_ephys) for eid in eids_to_process
        )
