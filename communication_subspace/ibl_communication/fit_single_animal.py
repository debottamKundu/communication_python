import concurrent
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
from ibl_info.decoder_pid import compute_decoder_pid
from ibl_info.decoder_utils import load_specific_regions
from ibl_info.dual_decoders import complete_decoder_pid_with_null, compute_null_distribution
from ibl_info.prepare_data_pid import get_new_cinc_intervals, get_new_cinc_intervals_choice
from ibl_info.utils import epoch_events
from one.api import ONE
from prior_localization.prepare_data import prepare_widefield
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask
import numpy as np
import pickle as pkl
import os
from joblib import Parallel, delayed
from communication_subspace.ibl_communication.utils import (
    check_config,
    compute_reduced_rank_pairs,
    compute_regionwise_r2,
    get_high_low_masks,
    get_intrinsic_dimensions,
    load_widefield_epoch,
)
from communication_subspace.ibl_communication.utils import setup_logger
import concurrent.futures
from tqdm import tqdm

config = check_config()
logger = setup_logger("RidgeRegressors")

# order of business.
""" 
1. For a given epoch
    - for each region, estimate their intrinsic dimensionality
    - median split engagement
        - also estimate intrinsic dimensionality
    - for each region,
        - do a full ridge regression to find the optimal R2. (maybe we do subsampling later)
        - do reduced rank regression, stop when it is 1 sem of peak ridge performance
        - do this also after median engagement split
"""


def fit_single_animal(session_id, engagement_signal, include_reduced=False, stage_only=False):

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    trials, mask = load_trials_and_mask(
        one,
        session_id,
        exclude_nochoice=True,
        exclude_unbiased=False,
    )

    trials = trials[mask]
    engagement_signal = engagement_signal[mask.values]  # atleast we get the same masks

    logger.info(f"Loading stimulus data for {session_id}")

    stimulus_data, region_names_stim = load_widefield_epoch(
        one, session_id, trials, config["hemisphere"], epoch="stim"
    )

    logger.info(f"Loading choice data for {session_id}")
    choice_data, region_names_choice = load_widefield_epoch(
        one, session_id, trials, config["hemisphere"], epoch="choice"
    )

    if stage_only:
        return
    # everything is staged now.
    assert region_names_stim == region_names_choice

    # get high engagement and low engagement masks
    high_mask, low_mask = get_high_low_masks(engagement_signal)
    # logger.info(f"Computing intrinsic dimensions for {session_id}")

    # stimulus_intrinsic_dimensions = get_intrinsic_dimensions(stimulus_data, high_mask, low_mask)
    # choice_intrinsic_dimensions = get_intrinsic_dimensions(choice_data, high_mask, low_mask)

    # now for simple regressions: for all pairs of frames, we have 0,1 and 0,1
    logger.info(f"Running pairwise regressions for {session_id}")

    ridge_regression_dict = {}
    for frameidx in range(0, 2):  # we have two stim frames
        for frameidy in range(0, 2):  # we have two choice frames
            # cpa -> cross prediction array
            all_cpa_high = compute_regionwise_r2(
                stimulus_data, choice_data, frameidx, frameidy, high_mask
            )
            all_cpa_low = compute_regionwise_r2(
                stimulus_data, choice_data, frameidx, frameidy, low_mask
            )
            ridge_regression_dict[(frameidx, frameidy)] = (all_cpa_high, all_cpa_low)

    # now for reduced rank
    # let's keep this restricted

    if include_reduced:
        reduced_rank_dict = {}

        for frameidx in range(0, 2):
            for frameidy in range(0, 2):
                logger.info(f"Running reduced rank regressions for {session_id}")

                reduced_rank_high = compute_reduced_rank_pairs(
                    stimulus_data, choice_data, frameidx, frameidy, high_mask
                )
                reduced_rank_low = compute_reduced_rank_pairs(
                    stimulus_data, choice_data, frameidx, frameidy, low_mask
                )
                reduced_rank_dict[(frameidx, frameidy)] = (reduced_rank_high, reduced_rank_low)

    # save
    storage_dict = {}
    storage_dict["ridge_regression_dict"] = ridge_regression_dict
    storage_dict["stimulus_intrinsic_dimensions"] = stimulus_intrinsic_dimensions
    storage_dict["choice_intrinsic_dimensions"] = choice_intrinsic_dimensions
    if include_reduced:
        storage_dict["reduced_rank_dict"] = reduced_rank_dict
    storage_dict["regions"] = region_names_stim

    # save here
    filename = f"./data/generated/wifi_accuracy_modulation/{session_id}.pkl"
    with open(filename, "wb") as f:
        pkl.dump(storage_dict, f)

    return 1


if __name__ == "__main__":

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    sessions = one.search(datasets="widefieldU.images.npy")

    engagement_dir = "/usr/people/kundu/code/communication_python/data/generated"

    with open(f"{engagement_dir}/wifimicemotivation.pkl", "rb") as f:
        engagement_pickle = pkl.load(f)

    def process_eid(eid):
        engagement_signal = engagement_pickle[str(eid)]

        fit_single_animal(
            session_id=eid,
            engagement_signal=engagement_signal,
        )

    # run a single one
    process_eid(sessions[0])

    multiprocess = False
    if multiprocess:
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:

            futures = {executor.submit(process_eid, eid): eid for eid in sessions}

            for future in concurrent.futures.as_completed(futures):
                eid = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.exception(f"Session {eid} generated an exception: {exc}")
