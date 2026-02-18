import itertools
import concurrent.futures
from joblib import Parallel, delayed
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
from ibl_info.utils import check_config, epoch_events
from one.api import ONE
from prior_localization.prepare_data import prepare_widefield
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask
import numpy as np
import pickle as pkl
import os
from joblib import Parallel, delayed
from communication_subspace.core.reducedRankcrossval import (
    cross_validate_rrr,
    select_optimal_dimension,
)
from communication_subspace.core.runFA import extract_fa_latents
from ibl_info.decoder_pid_wifi import region_combinations, check_minimum

# flow:
# for a single animal
# load wifi data based on epoch
# for each pair of region, find whatever is the best dimension
# save R2, FA and dimension.

config = check_config()


def process_single_animal(eid, epoch):

    align_event = epoch_events(epoch)  # should default to stimon
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    sl = SessionLoader(one, eid=eid)
    trials, mask = load_trials_and_mask(
        one,
        eid,
        sess_loader=sl,  # using session loader to load trials so that we get proper probability
        exclude_nochoice=True,
        exclude_unbiased=True,
    )
    trials = trials[mask]
    align_times = trials[align_event].values

    all_regions = config["widefield_regions"]
    if epoch == "stim":
        frames = config["stimulus_frames"]
    elif epoch == "choice":
        frames = config["choice_frames"]

    data_epoch, actual_regions = prepare_widefield(
        one,
        eid,
        hemisphere=config["hemisphere"],
        regions=all_regions,
        align_times=align_times,
        frame_window=frames,
        functional_channel=470,
        stage_only=False,
    )

    total_frames = data_epoch[0].shape[1]  # type: ignore
    data_epoch, used_regions = check_minimum(data_epoch, actual_regions)

    framewise_data = {}
    for frame_idx in range(total_frames):
        frame_pickle_fa = {}
        frame_pickle_rrr = {}
        for region_a_idx in range(len(used_regions)):
            # run fa here, because we need to run it only once
            region_a = data_epoch[region_a_idx][frame_idx, :].T
            Z, U, Q, q_opt = extract_fa_latents(region_a, q=None, var_threshold=0.95)
            fa_dict = {"Z": Z, "U": U, "Q": Q, "q_opt": q_opt}
            frame_pickle_fa[used_regions[region_a_idx]] = fa_dict

            for region_b_idx in range(len(used_regions)):
                if region_a_idx == region_b_idx:
                    continue
                # region, frame x neurons x trials -> trials x neurons

                region_b = data_epoch[region_b_idx][frame_idx, :].T

                key = f"{used_regions[region_a_idx]}_{used_regions[region_b_idx]}"

                # run fa and rrr.
                cv_folds = config["cv_folds"]
                dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
                n_source_voxels = region_a.shape[1]
                n_target_voxels = region_b.shape[1]
                max_possible_rank = min(n_source_voxels, n_target_voxels)
                valid_dims = [d for d in dimensions if d <= max_possible_rank]

                cv_results = cross_validate_rrr(region_a, region_b, valid_dims, k_folds=cv_folds)
                # just keep the cv_results
                rrr_dict = {"cv_results": cv_results}
                frame_pickle_rrr[key] = rrr_dict

        framewise_data[frame_idx] = {"fa": frame_pickle_fa, "rrr": frame_pickle_rrr}

    return framewise_data


def process_session(session_id):
    try:
        frame_data_stim = process_single_animal(session_id, "stim")
        with open(f"./data/generated/{session_id}_rrr_fa_results_stim.pkl", "wb") as f:
            pkl.dump(frame_data_stim, f)
    except Exception as e:
        print(e)
        return -1

    try:
        frame_data_choice = process_single_animal(session_id, "choice")
        with open(f"./data/generated/{session_id}_rrr_fa_results_choice.pkl", "wb") as f:
            pkl.dump(frame_data_choice, f)
    except Exception as e:
        print(e)
        return -1

    return 1


if __name__ == "__main__":

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    sessions = one.search(datasets="widefieldU.images.npy")
    print(f"{len(sessions)} sessions with widefield data found")  # type: ignore
    n_cores = 8  # type: ignore
    results = Parallel(n_jobs=n_cores, verbose=10)(delayed(process_session)(session) for session in sessions)  # type: ignore

    print(f"Successes: {results.count(1)}")  # type: ignore
    print(f"Failures: {results.count(-1)}")  # type: ignore
