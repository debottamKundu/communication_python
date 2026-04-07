from collections import defaultdict
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
from communication_subspace.ibl_communication.utils import check_config, setup_logger

config = check_config()
logger = setup_logger(name="IBL_Decoding", log_file="crossprediction_results.log")


from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def cross_predict_regions(neuron_data_a, neuron_data_b, n_outer_splits=5, n_inner_splits=5):
    """
    Performs nested cross-validation to predict Region B from Region A using Ridge Regression.

    Parameters:
    -----------
    neuron_data_a : np.ndarray
        Shape (trials, neurons). The predictor region.
    neuron_data_b : np.ndarray
        Shape (trials, neurons). The target region.
    n_outer_splits : int
        Number of folds for the outer CV loop (evaluating generalization).
    n_inner_splits : int
        Number of folds for the inner CV loop (tuning hyperparameters).

    Returns:
    --------
    results : dict
        Dictionary containing test indices, true/predicted values, and R2 scores.
    """

    X = np.asarray(neuron_data_a)
    Y = np.asarray(neuron_data_b)

    alphas_to_test = np.logspace(-3, 4, 20)

    kf_outer = KFold(n_splits=n_outer_splits, shuffle=True, random_state=42)

    results = {"test_indices": [], "y_true": [], "y_pred": [], "r2_scores": [], "best_alphas": []}

    for train_idx, test_idx in kf_outer.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        model = RidgeCV(alphas=alphas_to_test, cv=n_inner_splits)
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        score = r2_score(Y_test, Y_pred, multioutput="uniform_average")

        results["test_indices"].append(test_idx)
        results["y_true"].append(Y_test)
        results["y_pred"].append(Y_pred)
        results["r2_scores"].append(score)
        results["best_alphas"].append(model.alpha_)

    # Calculate the overall mean R2 across all outer folds
    results["overall_mean_r2"] = np.mean(results["r2_scores"])  # type: ignore

    return results


def compute_cross_temporal_matrix(
    data_epoch_a, regions_a, data_epoch_b, regions_b, desc_label="Processing"
):
    """
    Computes the cross-temporal generalization matrix between two sets of regions and epochs.
    """
    framewise_data = defaultdict(lambda: defaultdict(dict))
    total_frames_a = data_epoch_a[0].shape[1]
    total_frames_b = data_epoch_b[0].shape[1]

    for region_a_idx, reg_a_name in enumerate(tqdm(regions_a, desc=desc_label)):
        for region_b_idx, reg_b_name in enumerate(regions_b):
            for f_a in range(total_frames_a):
                region_a = data_epoch_a[region_a_idx][f_a, :].T
                for f_b in range(total_frames_b):
                    region_b = data_epoch_b[region_b_idx][f_b, :].T

                    results = cross_predict_regions(region_a, region_b)
                    framewise_data[reg_a_name][reg_b_name][(f_a, f_b)] = results

    return framewise_data


def cross_epoch_predict_animal(eid):

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
    align_event_stim = epoch_events(epoch="stim")
    align_event_choice = epoch_events(epoch="choice")
    align_event_prior = epoch_events(epoch="stim")  # the same align, we pick up a different frame

    align_times_stim = trials[align_event_stim].values
    align_times_choice = trials[align_event_choice].values
    align_times_prior = trials[align_event_prior].values

    all_regions = config["widefield_regions"]
    stimulus_frames = config["stimulus_frames"]
    choice_frames = config["choice_frames"]
    prior_frames = config["prior_frames"]

    data_epoch_stim, actual_regions_stim = prepare_widefield(
        one,
        eid,
        hemisphere=config["hemisphere"],
        regions=all_regions,
        align_times=align_times_stim,
        frame_window=stimulus_frames,
        functional_channel=470,
        stage_only=False,
    )

    logger.info(f"Loaded stimulus data for {eid}")

    data_epoch_choice, actual_regions_choice = prepare_widefield(
        one,
        eid,
        hemisphere=config["hemisphere"],
        regions=all_regions,
        align_times=align_times_choice,
        frame_window=choice_frames,
        functional_channel=470,
        stage_only=False,
    )

    logger.info(f"Loaded choice data for {eid}")

    data_epoch_prior, actual_regions_prior = prepare_widefield(
        one,
        eid,
        hemisphere=config["hemisphere"],
        regions=all_regions,
        align_times=align_times_prior,
        frame_window=prior_frames,
        functional_channel=470,
        stage_only=False,
    )

    logger.info(f"Loaded prior data for {eid}")

    # now what
    # stim predicts choice?
    # prior predicts choice?

    data_epoch_stim, used_regions_stim = check_minimum(data_epoch_stim, actual_regions_stim)
    data_epoch_choice, used_regions_choice = check_minimum(
        data_epoch_choice, actual_regions_choice
    )
    data_epoch_prior, used_regions_prior = check_minimum(data_epoch_prior, actual_regions_prior)

    framewise_data_stim_choice = defaultdict(lambda: defaultdict(dict))
    framewise_data_prior_choice = defaultdict(lambda: defaultdict(dict))

    logger.info("\nRunning Prior -> Choice Models...")
    framewise_data_prior_choice = compute_cross_temporal_matrix(
        data_epoch_a=data_epoch_prior,
        regions_a=used_regions_prior,
        data_epoch_b=data_epoch_choice,
        regions_b=used_regions_choice,
        desc_label="Prior Regions",
    )

    logger.info("\nRunning Stim -> Choice Models...")
    framewise_data_stim_choice = compute_cross_temporal_matrix(
        data_epoch_a=data_epoch_stim,
        regions_a=used_regions_stim,
        data_epoch_b=data_epoch_choice,
        regions_b=used_regions_choice,
        desc_label="Stim Regions",
    )

    return framewise_data_prior_choice, framewise_data_stim_choice


def process_session(session_id):
    try:
        logger.info(f"Starting processing for session: {session_id}")
        framewise_data_prior_choice, framewise_data_stim_choice = cross_epoch_predict_animal(
            session_id
        )
        with open(f"./data/crossprediction/{session_id}_results_prior_choice.pkl", "wb") as f:
            pkl.dump(framewise_data_prior_choice, f)
        with open(f"./data/crossprediction/{session_id}_results_stim_choice.pkl", "wb") as f:
            pkl.dump(framewise_data_stim_choice, f)
    except Exception as e:
        logger.exception(f"Error processing session: {session_id}")
        return -1
    logger.info(f"Successfully finished session: {session_id}")
    return 1


if __name__ == "__main__":

    logger.info("Initializing pipeline and ONE API...")
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    sessions = one.search(datasets="widefieldU.images.npy")
    print(f"{len(sessions)} sessions with widefield data found")  # type: ignore

    session_id = sessions[0]  # type: ignore
    process_session(session_id)

    # run for a single animal:

    # n_cores = 8  # type: ignore
    # results = Parallel(n_jobs=n_cores, verbose=10)(delayed(process_session)(session) for session in sessions)  # type: ignore

    # print(f"Successes: {results.count(1)}")  # type: ignore
    # print(f"Failures: {results.count(-1)}")  # type: ignore
