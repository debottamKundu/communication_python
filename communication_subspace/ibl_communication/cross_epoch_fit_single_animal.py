import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
from sklearn.decomposition import PCA
from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask
from prior_localization.prepare_data import prepare_widefield

from ibl_info.utils import epoch_events
from ibl_info.decoder_pid_wifi import check_minimum
from communication_subspace.core.reducedRankcrossval import cross_validate_rrr
from communication_subspace.ibl_communication.utils import check_config

config = check_config()


def get_epoch_data(one, eid, align_times, epoch):
    """
    Helper function to load data for a specific epoch.
    """
    all_regions = config["widefield_regions"]

    if epoch == "stim":
        frames = config["stimulus_frames"]
    elif epoch == "choice":
        frames = config["choice_frames"]
    elif epoch == "prior":
        frames = config["prior_frames"]
    else:
        # Fallback for dynamic/prior epochs based on config format
        frames = config.get(f"{epoch}_frames", [-1, 0])

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

    data_epoch, used_regions = check_minimum(data_epoch, actual_regions)
    return data_epoch, used_regions


def process_cross_epoch_animal(
    eid, epoch_1, epoch_2, prior_epoch=None, pca_components=10
):
    """
    Evaluates directed communication subspaces from a source region in Epoch 1
    to target regions in Epoch 2.
    Uses PCA to control for varying numbers of voxels across regions.
    """
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
        sess_loader=sl,
        exclude_nochoice=True,
        exclude_unbiased=True,
    )
    trials = trials[mask]

    align_event_1 = epoch_events(epoch_1)
    align_times_1 = trials[align_event_1].values
    data_ep1, regions_1 = get_epoch_data(one, eid, align_times_1, epoch_1)

    align_event_2 = epoch_events(epoch_2)
    align_times_2 = trials[align_event_2].values
    data_ep2, regions_2 = get_epoch_data(one, eid, align_times_2, epoch_2)

    data_prior, regions_prior = None, None
    if prior_epoch is not None:
        align_event_prior = epoch_events(prior_epoch)
        align_times_prior = trials[align_event_prior].values
        data_prior, regions_prior = get_epoch_data(
            one, eid, align_times_prior, prior_epoch
        )

    # Find intersection of regions to ensure we iterate safely
    used_regions = [r for r in regions_1 if r in regions_2]

    cross_epoch_data = {}
    pca = PCA(n_components=pca_components)

    for region_a_name in tqdm(
        used_regions, desc=f"Processing Source Regions ({epoch_1})"
    ):
        idx_1 = regions_1.index(region_a_name)
        # Temporally average the frames window -> (voxels, trials). Transpose to -> (trials, voxels)
        X_ep1 = np.mean(data_ep1[idx_1], axis=0).T

        if prior_epoch is not None and region_a_name in regions_prior:
            idx_prior = regions_prior.index(region_a_name)
            X_prior = np.mean(data_prior[idx_prior], axis=0).T
            X_combined = np.hstack([X_prior, X_ep1])
        else:
            X_combined = X_ep1

        # Apply PCA to control for the number of voxels driving RRR results
        X_reduced = (
            pca.fit_transform(X_combined)
            if X_combined.shape[1] > pca_components
            else X_combined
        )

        for region_b_name in used_regions:
            if region_a_name == region_b_name:
                continue

            idx_2 = regions_2.index(region_b_name)
            Y_ep2 = np.mean(data_ep2[idx_2], axis=0).T

            Y_reduced = (
                pca.fit_transform(Y_ep2) if Y_ep2.shape[1] > pca_components else Y_ep2
            )

            key = f"{region_a_name}_{region_b_name}"

            cv_folds = config.get("cv_folds", 5)
            max_possible_rank = min(X_reduced.shape[1], Y_reduced.shape[1])
            valid_dims = np.arange(1, max_possible_rank)

            cv_results = cross_validate_rrr(
                X_reduced, Y_reduced, valid_dims, k_folds=cv_folds
            )

            cross_epoch_data[key] = {"cv_results": cv_results}

    return cross_epoch_data


def process_session_cross_epoch(session_id):
    try:
        cross_results = process_cross_epoch_animal(
            session_id, epoch_1="stim", epoch_2="choice", prior_epoch=None
        )
        os.makedirs("./data/generated", exist_ok=True)
        with open(f"./data/generated/{session_id}_rrr_cross_epoch.pkl", "wb") as f:
            pkl.dump(cross_results, f)
    except Exception as e:
        print(f"Error processing session {session_id}: {e}")
