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

    # SAFETY CHECK: Only add handlers if the logger doesn't already have them.
    # This prevents duplicate log lines if the function is accidentally called twice.
    if not logger.handlers:
        # Create formatting
        log_format = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. Console Handler (Prints to your terminal)
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(log_format)
        logger.addHandler(c_handler)

        # 2. File Handler (Saves to a text file)
        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(log_format)
        logger.addHandler(f_handler)

    return logger


def get_align_times(trials, epoch):

    config = check_config()

    if epoch == "stim":
        align_times = trials.stimOn_times
        frame_windows = config["stimulus_frames"]
    elif epoch == "choice":
        align_times = trials.firstMovement_times
        frame_windows = config["choice_frames"]
    else:
        raise NotImplementedError

    return align_times, frame_windows


def load_widefield_epoch(
    one, session_id, trials, hemisphere, epoch, stage_only=False, min_voxels=10
):

    align_times, frame_windows = get_align_times(trials, epoch)
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
