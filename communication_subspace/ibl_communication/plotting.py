import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl
from scipy import stats
import numpy as np


def plot_filtered_mi(r2_high, r2_low, labels, threshold=0.10):

    high = np.clip(r2_high, 0, None)
    low = np.clip(r2_low, 0, None)
    labels = [x[0] for x in labels]
    denom = high + low
    mi = np.divide(high - low, denom, out=np.zeros_like(denom), where=denom != 0)

    quality_mask = (high > threshold) | (low > threshold)
    filtered_mi = np.where(quality_mask, mi, np.nan)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        filtered_mi,
        cmap="viridis",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Modulation Index"},
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Inter-areal Modulation (R² > {threshold})")
    plt.show()

    return filtered_mi[quality_mask]


from scipy.stats import wilcoxon


def run_significance_tests(r2_high, r2_low, r2_threshold=0.01):
    """
    Performs dual-level significance testing (MI and Raw R2)
    on region pairs passing a baseline quality threshold.
    """

    n = r2_high.shape[0]
    off_diagonal_mask = ~np.eye(n, dtype=bool)

    high_flat = r2_high[off_diagonal_mask]
    low_flat = r2_low[off_diagonal_mask]

    quality_mask = (high_flat >= r2_threshold) | (low_flat >= r2_threshold)

    high_clean = np.clip(high_flat[quality_mask], 0, None)
    low_clean = np.clip(low_flat[quality_mask], 0, None)

    if len(high_clean) == 0:
        print("No region pairs passed the R2 threshold.")
        return None

    denom = high_clean + low_clean
    mi_clean = np.divide(high_clean - low_clean, denom, out=np.zeros_like(denom), where=denom != 0)

    mi_stat, mi_p = wilcoxon(mi_clean)

    raw_stat, raw_p = wilcoxon(high_clean, low_clean, alternative="greater")

    print(f"--- Statistical Report (N={len(high_clean)} pairs) ---")
    print(f"Threshold used: R² > {r2_threshold}")
    print(f"\n1. MODULATION INDEX (Relative Shift)")
    print(f"   Median MI: {np.median(mi_clean):.4f}")
    print(f"   p-value:   {mi_p:.2e} ({'Significant' if mi_p < 0.05 else 'NS'})")  # type: ignore

    print(f"\n2. RAW R² (Absolute Shift)")
    print(f"   Mean Engaged:    {np.mean(high_clean):.4f}")
    print(f"   Mean Disengaged: {np.mean(low_clean):.4f}")
    print(f"   p-value (one-sided 'greater'): {raw_p:.2e}")

    return {
        "mi_p": mi_p,
        "raw_p": raw_p,
        "mi_data": mi_clean,
        "high_data": high_clean,
        "low_data": low_clean,
    }


def plot_engagement_results(r2_high, r2_low, r2_threshold=0.01):
    """
    Plots the Unity Scatter and MI Histogram for inter-areal communication.
    """
    # 1. Pre-process (Remove diagonal and apply threshold)
    n = r2_high.shape[0]
    mask = ~np.eye(n, dtype=bool)

    high_flat = r2_high[mask]
    low_flat = r2_low[mask]

    q_mask = (high_flat > r2_threshold) | (low_flat > r2_threshold)
    h_clean = high_flat[q_mask]
    l_clean = low_flat[q_mask]

    denom = h_clean + l_clean
    mi = np.divide(h_clean - l_clean, denom, out=np.zeros_like(denom), where=denom != 0)

    # 2. Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(l_clean, h_clean, alpha=0.4, s=25, color="teal", edgecolor="none")

    lims = [0, max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
    ax1.plot(lims, lims, "k--", alpha=0.7, zorder=0)

    ax1.set_title(f"$R^2$ over all pairs", fontsize=14)
    ax1.set_xlabel("Disengaged $R^2$", fontsize=12)
    ax1.set_ylabel("Engaged $R^2$", fontsize=12)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.grid(True, alpha=0.2)

    sns.histplot(mi, bins=40, kde=True, ax=ax2, color="darkorange", alpha=0.6)
    ax2.axvline(
        np.median(mi),
        color="k",
        linestyle="-",
        label=f"Median MI: {np.median(mi):.3f}",
    )

    ax2.set_title("", fontsize=14)
    ax2.set_xlabel("Modulation Index", fontsize=12)
    ax2.set_ylabel("Number of Pairs", fontsize=12)
    ax2.set_xlim([-1, 1])  # MI is naturally bounded here
    ax2.legend()

    plt.tight_layout()
    plt.show()


def aggregate_and_plot_cohort(animal_data_dict, r2_threshold=0.01):

    group_low = []
    group_high = []
    group_mi = []

    for animal_id, matrices in animal_data_dict.items():
        r2_high = matrices["high"]
        r2_low = matrices["low"]

        n = r2_high.shape[0]
        mask = ~np.eye(n, dtype=bool)
        h_flat = r2_high[mask]
        l_flat = r2_low[mask]

        q_mask = (h_flat > r2_threshold) | (l_flat > r2_threshold)
        h_clean = np.clip(h_flat[q_mask], 0, None)
        l_clean = np.clip(l_flat[q_mask], 0, None)

        if len(h_clean) > 10:
            animal_median_high = np.mean(h_clean)
            animal_median_low = np.mean(l_clean)

            denom = animal_median_high + animal_median_low
            animal_mi = (animal_median_high - animal_median_low) / denom if denom != 0 else 0

            group_high.append(animal_median_high)
            group_low.append(animal_median_low)
            group_mi.append(animal_mi)

    group_high = np.array(group_high)
    group_low = np.array(group_low)
    group_mi = np.array(group_mi)
    n_animals = len(group_high)

    stat, p_val = wilcoxon(group_high, group_low, alternative="two-sided")
    mi_stat, mi_p_val = wilcoxon(group_mi, alternative="greater")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(group_low, group_high, color="indigo", s=60, alpha=0.7, edgecolor="white")
    max_val = max(np.max(group_low), np.max(group_high)) * 1.1
    ax1.plot([0, max_val], [0, max_val], "k--", linewidth=1)

    ax1.set_title(f"Engagement modulation, paired wilcoxon= {p_val:.3f}")
    ax1.set_xlabel("Median $R^2$ (Disengaged)")
    ax1.set_ylabel("Median $R^2$ (Engaged)")
    ax1.set_xlim([0, max_val])
    ax1.set_ylim([0, max_val])

    sns.histplot(group_mi, bins=15, kde=True, ax=ax2, color="mediumseagreen")
    ax2.axvline(0, color="black", linestyle="--", linewidth=2)
    ax2.axvline(
        np.median(group_mi),
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Cohort Median MI: {np.median(group_mi):.3f}",
    )

    ax2.set_title(f"Animal modulation index, Wilcoxon p = {mi_p_val:.3f}")
    ax2.set_xlabel("Animal Median Modulation Index")
    ax2.set_ylabel("Number of Animals")
    ax2.legend()
    sns.despine()
    plt.tight_layout()


def convert_to_universal_modules(animal_r2_matrix, animal_regions, module_dict):
    """
    Converts an animal-specific NxN matrix into a universal MxM module matrix.
    Missing modules are filled with np.nan.
    """
    modules = list(module_dict.keys())
    n_modules = len(modules)

    universal_matrix = np.full((n_modules, n_modules), np.nan)

    for i, source_mod in enumerate(modules):
        for j, target_mod in enumerate(modules):

            source_idx = [
                idx for idx, reg in enumerate(animal_regions) if reg in module_dict[source_mod]
            ]
            target_idx = [
                idx for idx, reg in enumerate(animal_regions) if reg in module_dict[target_mod]
            ]

            if len(source_idx) == 0 or len(target_idx) == 0:
                continue

            block = animal_r2_matrix[np.ix_(source_idx, target_idx)]

            if i != j:
                universal_matrix[i, j] = np.nanmean(block)
            else:
                if block.shape[0] < 2:
                    continue
                mask = ~np.eye(block.shape[0], dtype=bool)
                universal_matrix[i, j] = np.nanmean(block[mask])

    return universal_matrix, modules


def process_cohort_modules(cohort_data, allen_modules):
    all_high_modules = []
    all_low_modules = []

    for animal in cohort_data.values():
        high_7x7, mod_names = convert_to_universal_modules(
            animal["high_r2"], animal["regions"], allen_modules
        )
        low_7x7, _ = convert_to_universal_modules(
            animal["low_r2"], animal["regions"], allen_modules
        )

        all_high_modules.append(high_7x7)
        all_low_modules.append(low_7x7)

    stack_high = np.array(all_high_modules)
    stack_low = np.array(all_low_modules)

    group_mean_high = np.nanmean(stack_high, axis=0)
    group_mean_low = np.nanmean(stack_low, axis=0)

    group_delta = group_mean_high - group_mean_low
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        group_delta,
        xticklabels=mod_names,
        yticklabels=mod_names,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "$\Delta R^2$ (Engaged - Disengaged)"},
    )

    plt.title("Cohort Average: Inter-Module Communication Shift (N=50)")
    plt.xlabel("Target Module")
    plt.ylabel("Source Module")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
