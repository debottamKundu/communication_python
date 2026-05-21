from communication_subspace.core.faCrossVal import cross_val_fa, factor_analysis_model_select
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis  # check which one to use

# from communication_subspace.ibl_communication.utils import setup_logger

# log = setup_logger()


def compute_intrinsic_dimensionality_pca(X, variance_threshold=0.95):
    """
    returns PCA dimensions that explain 95 percent on the variance

    Args:
        X (np.array): n_trials x n_neurons
    """

    pca = PCA()
    pca.fit(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1

    pca_optimal = PCA(n_components=n_components)
    projected_components = pca_optimal.fit(X)

    return n_components, projected_components


def compute_intrinsic_dimensionality_fa(X, max_components=50, variance_threshold=0.95):
    """
    returns FA components that explain 95 percent of variance

    Args:
        X (np.array): n_trials x n_neurons
        max_components (int): maximum number of components to test
    """

    n_components = min(X.shape[1], max_components)
    fa = FactorAnalysis(n_components=n_components)
    fa.fit(X)

    L = fa.components_
    shared_cov_small = L.T @ L
    eigenvalues = np.sort(np.linalg.eigvalsh(shared_cov_small))[::-1]
    total_shared_var = np.sum(eigenvalues)

    if total_shared_var == 0:
        log.error("Zero share variance, weird : check")
        return 0, None

    explained_variance_ratio = eigenvalues / total_shared_var
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    n_components_reqd = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    fa_optimal = FactorAnalysis(n_components=n_components_reqd)
    projected_components = fa_optimal.fit(X)

    return n_components_reqd, projected_components


def compute_intrinsic_dimensionality(data, mask=None):

    # NOTE: skip fa for now
    engagement_dimensions_pca = []
    engagement_dimensions_fa = []
    if mask is None:
        mask = np.ones(data[0].shape[1], dtype=bool)

    from tqdm import tqdm
    # data is now regions x (frames x trials x voxels)
    for neural_data_frames in tqdm(data, desc="Computing Intrinsic Dim", leave=False):
        # first choose the proper frame
        region_wise_data_pca = []
        region_wise_data_fa = []
        for frameidx in range(len(neural_data_frames)):
            neural_data = neural_data_frames[frameidx, :, :]
            # now in trials x neurons

            pca_components, _ = compute_intrinsic_dimensionality_pca(neural_data[mask, :])
            #fa_components, _ = compute_intrinsic_dimensionality_fa(neural_data[mask, :])
            region_wise_data_pca.append(pca_components)
            #region_wise_data_fa.append(fa_components)

        #engagement_dimensions_fa.append(region_wise_data_fa)
        engagement_dimensions_pca.append(region_wise_data_pca)

    engagement_dimensions_fa = np.asarray(engagement_dimensions_fa)
    engagement_dimensions_pca = np.asarray(engagement_dimensions_pca)

    return engagement_dimensions_pca, engagement_dimensions_fa
