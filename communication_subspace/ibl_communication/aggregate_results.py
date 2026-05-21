def aggregate_and_plot_results(
    results_dir="./data/generated/rrr_analysis", output_plot_dir="./reports/figures"
):
    """
    Loads saved analysis outcomes, aggregates metrics, and generates comparison plots.
    """
    import scipy.stats as stats

    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        multipletests = None

    os.makedirs(output_plot_dir, exist_ok=True)
    files = [f for f in os.listdir(results_dir) if f.endswith("_rrr_results.pkl")]
    if not files:
        print("No results files found in", results_dir)
        return

    # Data lists to collect values across all sessions and region pairs
    comparison_data = []
    dim_data = []

    # Lists of matrices across sessions to compute region-by-region heatmaps
    ridge_all_list = []
    null_ridge_all_list = []
    ridge_high_list = []
    ridge_low_list = []

    rrr_all_list = []
    rrr_high_list = []
    rrr_low_list = []

    rank_high_list = []
    rank_low_list = []

    align_in_high_list = []
    align_in_low_list = []

    align_out_high_list = []
    align_out_low_list = []

    regions = None

    for file in files:
        filepath = os.path.join(results_dir, file)
        with open(filepath, "rb") as f:
            res = pkl.load(f)

        regions = res["regions"]
        n_regions = len(regions)
        rrr_res = res["rrr_results"]
        true_ridge = res["true_ridge_r2"]
        high_ridge = res["high_ridge_r2"]
        low_ridge = res["low_ridge_r2"]
        p_vals = res["p_values"]
        null_ridge = res["null_ridge_r2"]

        # Add Ridge matrices directly
        ridge_all_list.append(true_ridge)
        null_ridge_all_list.append(np.mean(null_ridge, axis=-1))
        ridge_high_list.append(high_ridge)
        ridge_low_list.append(low_ridge)

        # Initialize matrices for this session
        rrr_all_mat = np.full((n_regions, n_regions), np.nan)
        rrr_high_mat = np.full((n_regions, n_regions), np.nan)
        rrr_low_mat = np.full((n_regions, n_regions), np.nan)
        rank_high_mat = np.full((n_regions, n_regions), np.nan)
        rank_low_mat = np.full((n_regions, n_regions), np.nan)
        align_in_high_mat = np.full((n_regions, n_regions), np.nan)
        align_in_low_mat = np.full((n_regions, n_regions), np.nan)
        align_out_high_mat = np.full((n_regions, n_regions), np.nan)
        align_out_low_mat = np.full((n_regions, n_regions), np.nan)

        # 1. Collect regression and alignment metrics for significant pairs
        for (idx, idy), cond_dict in rrr_res.items():
            source = regions[idx]
            target = regions[idy]

            # Ridge R^2 values
            r2_ridge_all = true_ridge[idx, idy]
            r2_ridge_high = high_ridge[idx, idy]
            r2_ridge_low = low_ridge[idx, idy]

            # Extract metrics for all, high, low conditions
            row = {
                "session_id": res["session_id"],
                "source": source,
                "target": target,
                "pair": f"{source}->{target}",
                "p_value": p_vals[idx, idy],
                # Ridge R2
                "ridge_r2_all": r2_ridge_all,
                "ridge_r2_high": r2_ridge_high,
                "ridge_r2_low": r2_ridge_low,
            }

            for cond in ["all", "high", "low"]:
                if cond in cond_dict:
                    row[f"rrr_r2_{cond}"] = cond_dict[cond]["cv_r2_rrr"]
                    row[f"rank_{cond}"] = cond_dict[cond]["optimal_rank"]
                    row[f"align_in_{cond}"] = cond_dict[cond]["input_alignment"]
                    row[f"align_out_{cond}"] = cond_dict[cond]["output_alignment"]
                    row[f"comm_frac_{cond}"] = cond_dict[cond]["comm_fraction"]
                    if "stim_alignment" in cond_dict[cond]:
                        row[f"stim_alignment_{cond}"] = cond_dict[cond]["stim_alignment"]
                        row[f"stim_alignment_pval_{cond}"] = cond_dict[cond]["stim_alignment_pval"]

            comparison_data.append(row)

            # Fill matrices
            if "all" in cond_dict:
                rrr_all_mat[idx, idy] = cond_dict["all"]["cv_r2_rrr"]
            if "high" in cond_dict:
                rrr_high_mat[idx, idy] = cond_dict["high"]["cv_r2_rrr"]
                rank_high_mat[idx, idy] = cond_dict["high"]["optimal_rank"]
                align_in_high_mat[idx, idy] = cond_dict["high"]["input_alignment"]
                align_out_high_mat[idx, idy] = cond_dict["high"]["output_alignment"]
            if "low" in cond_dict:
                rrr_low_mat[idx, idy] = cond_dict["low"]["cv_r2_rrr"]
                rank_low_mat[idx, idy] = cond_dict["low"]["optimal_rank"]
                align_in_low_mat[idx, idy] = cond_dict["low"]["input_alignment"]
                align_out_low_mat[idx, idy] = cond_dict["low"]["output_alignment"]

        rrr_all_list.append(rrr_all_mat)
        rrr_high_list.append(rrr_high_mat)
        rrr_low_list.append(rrr_low_mat)
        rank_high_list.append(rank_high_mat)
        rank_low_list.append(rank_low_mat)
        align_in_high_list.append(align_in_high_mat)
        align_in_low_list.append(align_in_low_mat)
        align_out_high_list.append(align_out_high_mat)
        align_out_low_list.append(align_out_low_mat)

        # 2. Collect Intrinsic Dimensionality data
        int_dim = res["intrinsic_dimensionality"]
        for epoch in ["stim", "choice"]:
            frame_idx = 0 if epoch == "stim" else 1
            for cond in ["all", "high", "low"]:
                pca_arr = int_dim[epoch][cond]
                for idx, rname in enumerate(regions):
                    dim_data.append(
                        {
                            "session_id": res["session_id"],
                            "region": rname,
                            "epoch": epoch,
                            "condition": cond,
                            "pca_dim": pca_arr[idx, frame_idx],
                        }
                    )

    df_comp = pd.DataFrame(comparison_data)
    df_dim = pd.DataFrame(dim_data)

    print(
        f"Aggregated {len(df_comp)} significant communication channels across {len(files)} session(s)."
    )

    n_sessions = len(files)

    # 1. Compute significance mask (consistency across animals)
    valid_counts = np.sum(~np.isnan(np.array(rrr_all_list)), axis=0)
    consistency_frac = valid_counts / n_sessions

    # 2. Group-level Paired Tests
    group_pval_ridge = np.ones((n_regions, n_regions))
    group_pval_engagement = np.ones((n_regions, n_regions))

    if n_sessions >= 2:
        ridge_arr = np.array(ridge_all_list)
        null_arr = np.array(null_ridge_all_list)
        high_arr = np.array(ridge_high_list)
        low_arr = np.array(ridge_low_list)

        for i in range(n_regions):
            for j in range(n_regions):
                if i == j:
                    continue
                # Test 1: True Ridge R2 > Null Ridge R2
                diff_ridge = ridge_arr[:, i, j] - null_arr[:, i, j]
                if np.std(diff_ridge) > 1e-6:
                    if n_sessions >= 10:
                        _, p_ridge = stats.wilcoxon(
                            ridge_arr[:, i, j], null_arr[:, i, j], alternative="greater"
                        )
                    else:
                        _, p_ridge = stats.ttest_rel(
                            ridge_arr[:, i, j], null_arr[:, i, j], alternative="greater"
                        )
                    group_pval_ridge[i, j] = p_ridge

                # Test 2: High Engagement > Low Engagement
                diff_eng = high_arr[:, i, j] - low_arr[:, i, j]
                if np.std(diff_eng) > 1e-6:
                    if n_sessions >= 10:
                        _, p_eng = stats.wilcoxon(
                            high_arr[:, i, j], low_arr[:, i, j], alternative="greater"
                        )
                    else:
                        _, p_eng = stats.ttest_rel(
                            high_arr[:, i, j], low_arr[:, i, j], alternative="greater"
                        )
                    group_pval_engagement[i, j] = p_eng

        # FDR Correction
        mask_off_diag = ~np.eye(n_regions, dtype=bool)
        if multipletests is not None:
            _, fdr_pval_ridge, _, _ = multipletests(
                group_pval_ridge[mask_off_diag], method="fdr_bh"
            )
            group_pval_ridge_corrected = np.ones_like(group_pval_ridge)
            group_pval_ridge_corrected[mask_off_diag] = fdr_pval_ridge

            _, fdr_pval_engagement, _, _ = multipletests(
                group_pval_engagement[mask_off_diag], method="fdr_bh"
            )
            group_pval_engagement_corrected = np.ones_like(group_pval_engagement)
            group_pval_engagement_corrected[mask_off_diag] = fdr_pval_engagement
        else:
            group_pval_ridge_corrected = group_pval_ridge
            group_pval_engagement_corrected = group_pval_engagement

    else:
        group_pval_ridge_corrected = np.ones((n_regions, n_regions))
        group_pval_engagement_corrected = np.ones((n_regions, n_regions))

    # Determine plotting mask (consistent in >= 50% and FDR p < 0.05, unless N < 10 then just use consistency)
    if n_sessions >= 10:
        sig_mask = (consistency_frac >= 0.5) & (group_pval_ridge_corrected < 0.05)
        eng_mask = sig_mask & (group_pval_engagement_corrected < 0.05)
    else:
        sig_mask = consistency_frac >= 0.5
        eng_mask = sig_mask

    # Compute session-averaged matrices
    mean_ridge_all = np.nanmean(ridge_all_list, axis=0)
    mean_ridge_high = np.nanmean(ridge_high_list, axis=0)
    mean_ridge_low = np.nanmean(ridge_low_list, axis=0)

    # Apply masks
    mean_ridge_all[~sig_mask] = np.nan
    diff_engagement_mat = mean_ridge_high - mean_ridge_low
    diff_engagement_mat[~eng_mask] = np.nan

    mean_rrr_all = np.nanmean(rrr_all_list, axis=0)
    mean_rrr_high = np.nanmean(rrr_high_list, axis=0)
    mean_rrr_low = np.nanmean(rrr_low_list, axis=0)

    mean_rank_high = np.nanmean(rank_high_list, axis=0)
    mean_rank_low = np.nanmean(rank_low_list, axis=0)

    mean_align_in_high = np.nanmean(align_in_high_list, axis=0)
    mean_align_in_low = np.nanmean(align_in_low_list, axis=0)
    mean_align_out_high = np.nanmean(align_out_high_list, axis=0)
    mean_align_out_low = np.nanmean(align_out_low_list, axis=0)

    # ------------------ HEATMAP PLOTTING HELPER ------------------
    def plot_heatmap(matrix, title, filename, cmap="viridis", center=None, cbar_label=""):
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            xticklabels=regions,
            yticklabels=regions,
            cmap=cmap,
            center=center,
            annot=False,
            cbar_kws={"label": cbar_label},
        )
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Target Region (Choice Epoch)", fontsize=12)
        plt.ylabel("Source Region (Stimulus Epoch)", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

    # ------------------ MATRIX PLOTS ------------------
    # Plot A: Mean Ridge R2 (All)
    plot_heatmap(
        mean_ridge_all,
        "Inter-areal Communication Strength: Mean Ridge R² (All Trials)",
        os.path.join(output_plot_dir, "region_matrix_ridge_r2.png"),
        cmap="rocket",
        cbar_label="Predictive R²",
    )

    # Plot B: Ridge R2 Difference: High - Low Engagement
    plot_heatmap(
        diff_engagement_mat,
        "Engagement Modulation of Inter-areal Communication: Ridge R² (High - Low)",
        os.path.join(output_plot_dir, "region_matrix_engagement_diff.png"),
        cmap="coolwarm",
        center=0,
        cbar_label="Δ R² (High - Low)",
    )

    # Plot C: Subplot panel for RRR Optimal Rank
    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    sns.heatmap(
        mean_rank_high,
        xticklabels=regions,
        yticklabels=regions,
        cmap="viridis",
        ax=axes[0],
        cbar_kws={"label": "Optimal Rank"},
    )
    axes[0].set_title("Optimal RRR Rank (High Engagement)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Target Region")
    axes[0].set_ylabel("Source Region")
    axes[0].set_xticklabels(regions, rotation=45, ha="right")

    sns.heatmap(
        mean_rank_low,
        xticklabels=regions,
        yticklabels=regions,
        cmap="viridis",
        ax=axes[1],
        cbar_kws={"label": "Optimal Rank"},
    )
    axes[1].set_title("Optimal RRR Rank (Low Engagement)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Target Region")
    axes[1].set_ylabel("Source Region")
    axes[1].set_xticklabels(regions, rotation=45, ha="right")

    sns.heatmap(
        mean_rank_high - mean_rank_low,
        xticklabels=regions,
        yticklabels=regions,
        cmap="bwr",
        center=0,
        ax=axes[2],
        cbar_kws={"label": "Δ Rank (High - Low)"},
    )
    axes[2].set_title("Optimal Rank Difference (High - Low)", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Target Region")
    axes[2].set_ylabel("Source Region")
    axes[2].set_xticklabels(regions, rotation=45, ha="right")

    plt.tight_layout()
    plot_path_rank = os.path.join(output_plot_dir, "region_matrix_rrr_ranks.png")
    plt.savefig(plot_path_rank, dpi=150)
    plt.close()

    # Plot D: Subplot panel for Subspace Alignments (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    # Row 0: Input Alignment High & Low
    sns.heatmap(
        mean_align_in_high,
        xticklabels=regions,
        yticklabels=regions,
        cmap="mako",
        vmin=0,
        vmax=0.005,
        ax=axes[0, 0],
        cbar_kws={"label": "Input Alignment"},
    )
    axes[0, 0].set_title(
        "Source Subspace Input Alignment (High Engagement)", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Target Region")
    axes[0, 0].set_ylabel("Source Region")
    axes[0, 0].set_xticklabels(regions, rotation=45, ha="right")

    sns.heatmap(
        mean_align_in_low,
        xticklabels=regions,
        yticklabels=regions,
        cmap="mako",
        vmin=0,
        vmax=0.005,
        ax=axes[0, 1],
        cbar_kws={"label": "Input Alignment"},
    )
    axes[0, 1].set_title(
        "Source Subspace Input Alignment (Low Engagement)", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Target Region")
    axes[0, 1].set_ylabel("Source Region")
    axes[0, 1].set_xticklabels(regions, rotation=45, ha="right")

    # Row 1: Output Alignment High & Low
    sns.heatmap(
        mean_align_out_high,
        xticklabels=regions,
        yticklabels=regions,
        cmap="flare",
        vmin=0.5,
        vmax=1.0,
        ax=axes[1, 0],
        cbar_kws={"label": "Output Alignment"},
    )
    axes[1, 0].set_title(
        "Target Subspace Output Alignment (High Engagement)", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Target Region")
    axes[1, 0].set_ylabel("Source Region")
    axes[1, 0].set_xticklabels(regions, rotation=45, ha="right")

    sns.heatmap(
        mean_align_out_low,
        xticklabels=regions,
        yticklabels=regions,
        cmap="flare",
        vmin=0.5,
        vmax=1.0,
        ax=axes[1, 1],
        cbar_kws={"label": "Output Alignment"},
    )
    axes[1, 1].set_title(
        "Target Subspace Output Alignment (Low Engagement)", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Target Region")
    axes[1, 1].set_ylabel("Source Region")
    axes[1, 1].set_xticklabels(regions, rotation=45, ha="right")

    plt.tight_layout()
    plot_path_align = os.path.join(output_plot_dir, "region_matrix_subspace_align.png")
    plt.savefig(plot_path_align, dpi=150)
    plt.close()

    # Plot E: Intrinsic Dimensionality per Region (High vs Low)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Stimulus Epoch PCA Dim
    df_dim_stim = df_dim[df_dim["epoch"] == "stim"]
    sns.barplot(
        data=df_dim_stim,
        x="region",
        y="pca_dim",
        hue="condition",
        ax=axes[0],
        palette="muted",
        capsize=0.1,
        errorbar="se",
    )
    axes[0].set_title(
        "Stimulus Epoch Intrinsic Dimensionality (PCA 95%) per Region",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].set_xlabel("Brain Region", fontsize=11)
    axes[0].set_ylabel("PCA Dimensionality", fontsize=11)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].grid(True, alpha=0.3)

    # Choice Epoch PCA Dim
    df_dim_choice = df_dim[df_dim["epoch"] == "choice"]
    sns.barplot(
        data=df_dim_choice,
        x="region",
        y="pca_dim",
        hue="condition",
        ax=axes[1],
        palette="muted",
        capsize=0.1,
        errorbar="se",
    )
    axes[1].set_title(
        "Choice Epoch Intrinsic Dimensionality (PCA 95%) per Region",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("Brain Region", fontsize=11)
    axes[1].set_ylabel("PCA Dimensionality", fontsize=11)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path_dim = os.path.join(output_plot_dir, "region_intrinsic_dimensionality.png")
    plt.savefig(plot_path_dim, dpi=150)
    plt.close()

    # ------------------ ORIGINAL CORRELATION PLOTS (Averages) ------------------
    # Keep Plot 1 (Ridge vs RRR scatter)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_comp, x="ridge_r2_all", y="rrr_r2_all", hue="session_id", alpha=0.8, s=60
    )
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, "k--", alpha=0.5, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title("Predictive Accuracy: Full-Rank (Ridge) vs. Reduced-Rank (RRR) R²")
    plt.xlabel("Full-Rank Ridge R²")
    plt.ylabel("Reduced-Rank Regression (RRR) R²")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Session ID")
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, "ridge_vs_rrr_r2.png"), dpi=150)
    plt.close()

    # Keep Plot 2 (Engagement predictive accuracy scatter)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(
        data=df_comp,
        x="ridge_r2_low",
        y="ridge_r2_high",
        hue="session_id",
        ax=axes[0],
        alpha=0.8,
        s=60,
    )
    lims = [
        min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
        max(axes[0].get_xlim()[1], axes[0].get_ylim()[1]),
    ]
    axes[0].plot(lims, lims, "k--", alpha=0.5, zorder=0)
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_title("Full-Rank Ridge R²: High vs. Low Engagement")
    axes[0].set_xlabel("Low Engagement R²")
    axes[0].set_ylabel("High Engagement R²")
    axes[0].grid(True, alpha=0.3)

    sns.scatterplot(
        data=df_comp,
        x="rrr_r2_low",
        y="rrr_r2_high",
        hue="session_id",
        ax=axes[1],
        alpha=0.8,
        s=60,
    )
    lims = [
        min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
        max(axes[1].get_xlim()[1], axes[1].get_ylim()[1]),
    ]
    axes[1].plot(lims, lims, "k--", alpha=0.5, zorder=0)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_title("Reduced-Rank RRR R²: High vs. Low Engagement")
    axes[1].set_xlabel("Low Engagement R²")
    axes[1].set_ylabel("High Engagement R²")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, "engagement_predictive_accuracy.png"), dpi=150)
    plt.close()

    # ------------------ DETAILED REGIONAL STATS & FILE WRITE ------------------
    # Calculate top senders and receivers (Ridge R2)
    # Source regions average predictability to others: mean along target dimension (axis 1)
    mean_ridge_source = np.nanmean(mean_ridge_all, axis=1)
    # Target regions average predictability from others: mean along source dimension (axis 0)
    mean_ridge_target = np.nanmean(mean_ridge_all, axis=0)

    # Sort to find top senders and receivers
    top_senders_idx = np.argsort(mean_ridge_source)[::-1]
    top_receivers_idx = np.argsort(mean_ridge_target)[::-1]

    # Calculate engagement modulation
    diff_mat = diff_engagement_mat
    # Find top region-to-region connections
    pairs_flat_idx = np.argsort(diff_mat.ravel())[::-1]
    top_modulated_pairs = []
    for p_idx in pairs_flat_idx:
        src_idx, tgt_idx = np.unravel_index(p_idx, diff_mat.shape)
        if (
            src_idx != tgt_idx
            and not np.isnan(diff_mat[src_idx, tgt_idx])
            and len(top_modulated_pairs) < 10
        ):
            top_modulated_pairs.append((src_idx, tgt_idx, diff_mat[src_idx, tgt_idx]))

    # Intrinsic dimensionalities by region (All, High, Low)
    pca_stim_mean = (
        df_dim[(df_dim["epoch"] == "stim")]
        .groupby(["region", "condition"])["pca_dim"]
        .mean()
        .unstack()
    )
    pca_choice_mean = (
        df_dim[(df_dim["epoch"] == "choice")]
        .groupby(["region", "condition"])["pca_dim"]
        .mean()
        .unstack()
    )

    # RRR Rank differences
    rank_diff_mat = mean_rank_high - mean_rank_low
    rank_diffs_flat_idx = np.argsort(np.abs(rank_diff_mat.ravel()))[::-1]
    top_rank_diff_pairs = []
    for r_idx in rank_diffs_flat_idx:
        src_idx, tgt_idx = np.unravel_index(r_idx, rank_diff_mat.shape)
        if (
            src_idx != tgt_idx
            and not np.isnan(rank_diff_mat[src_idx, tgt_idx])
            and len(top_rank_diff_pairs) < 10
        ):
            top_rank_diff_pairs.append(
                (
                    src_idx,
                    tgt_idx,
                    mean_rank_high[src_idx, tgt_idx],
                    mean_rank_low[src_idx, tgt_idx],
                )
            )

    # Save summary stats as csv/text
    summary_stats_path = os.path.join(output_plot_dir, "summary_statistics.txt")
    with open(summary_stats_path, "w") as f:
        f.write("========== ANALYSIS COHORT SUMMARY STATISTICS ==========\n\n")
        f.write(f"Total significant communication channels: {len(df_comp)}\n")
        f.write(f"Refined local sessions: {df_comp['session_id'].nunique()}\n\n")

        f.write("--- Cohort Predictive Accuracy (R^2) ---\n")
        f.write(
            f"Full-Rank Ridge R^2 (All): Mean = {df_comp['ridge_r2_all'].mean():.4f}, Std = {df_comp['ridge_r2_all'].std():.4f}\n"
        )
        f.write(
            f"Reduced-Rank RRR R^2 (All): Mean = {df_comp['rrr_r2_all'].mean():.4f}, Std = {df_comp['rrr_r2_all'].std():.4f}\n"
        )
        f.write(
            f"Ridge R^2 (High): Mean = {df_comp['ridge_r2_high'].mean():.4f}, Std = {df_comp['ridge_r2_high'].std():.4f}\n"
        )
        f.write(
            f"Ridge R^2 (Low): Mean = {df_comp['ridge_r2_low'].mean():.4f}, Std = {df_comp['ridge_r2_low'].std():.4f}\n"
        )
        f.write(
            f"RRR R^2 (High): Mean = {df_comp['rrr_r2_high'].mean():.4f}, Std = {df_comp['rrr_r2_high'].std():.4f}\n"
        )
        f.write(
            f"RRR R^2 (Low): Mean = {df_comp['rrr_r2_low'].mean():.4f}, Std = {df_comp['rrr_r2_low'].std():.4f}\n\n"
        )

        f.write("--- Top 5 Information Senders (Highest Average Outgoing Ridge R²) ---\n")
        for i, s_idx in enumerate(top_senders_idx[:5]):
            f.write(
                f"{i+1}. {regions[s_idx]}: Mean outgoing R² = {mean_ridge_source[s_idx]:.4f}\n"
            )
        f.write("\n")

        f.write("--- Top 5 Information Receivers (Highest Average Incoming Ridge R²) ---\n")
        for i, r_idx in enumerate(top_receivers_idx[:5]):
            f.write(
                f"{i+1}. {regions[r_idx]}: Mean incoming R² = {mean_ridge_target[r_idx]:.4f}\n"
            )
        f.write("\n")

        f.write(
            "--- Top 10 Region Pairs Most Modulated by Engagement (Largest Δ R² High - Low) ---\n"
        )
        for i, (src_idx, tgt_idx, diff) in enumerate(top_modulated_pairs[:10]):
            f.write(
                f"{i+1}. {regions[src_idx]} -> {regions[tgt_idx]}: Δ R² = {diff:.4f} (High={mean_ridge_high[src_idx, tgt_idx]:.4f}, Low={mean_ridge_low[src_idx, tgt_idx]:.4f})\n"
            )
        f.write("\n")

        f.write("--- Top 10 Region Pairs with Largest Subspace Optimal Rank Shifts ---\n")
        for i, (src_idx, tgt_idx, r_high, r_low) in enumerate(top_rank_diff_pairs[:10]):
            diff = r_high - r_low
            f.write(
                f"{i+1}. {regions[src_idx]} -> {regions[tgt_idx]}: Δ Rank = {diff:.1f} (High Rank={r_high:.1f}, Low Rank={r_low:.1f})\n"
            )
        f.write("\n")

        f.write("--- Cohort Communication Subspace Alignment Index ---\n")
        f.write(f"Input Alignment (High): Mean = {df_comp['align_in_high'].mean():.4f}\n")
        f.write(f"Input Alignment (Low): Mean = {df_comp['align_in_low'].mean():.4f}\n")
        f.write(f"Output Alignment (High): Mean = {df_comp['align_out_high'].mean():.4f}\n")
        f.write(f"Output Alignment (Low): Mean = {df_comp['align_out_low'].mean():.4f}\n\n")

        f.write("--- Region-by-Region Intrinsic Dimensionality (PCA 95% Variance) ---\n")
        f.write(
            "Region | Stim (All) | Stim (High) | Stim (Low) || Choice (All) | Choice (High) | Choice (Low)\n"
        )
        f.write("-" * 100 + "\n")
        for rname in regions:
            stim_all = pca_stim_mean.loc[rname, "all"]
            stim_high = pca_stim_mean.loc[rname, "high"]
            stim_low = pca_stim_mean.loc[rname, "low"]
            choice_all = pca_choice_mean.loc[rname, "all"]
            choice_high = pca_choice_mean.loc[rname, "high"]
            choice_low = pca_choice_mean.loc[rname, "low"]
            f.write(
                f"{rname:<6} | {stim_all:10.2f} | {stim_high:11.2f} | {stim_low:10.2f} || {choice_all:12.2f} | {choice_high:13.2f} | {choice_low:12.2f}\n"
            )

    print(f"Saved summary statistics to {summary_stats_path}")
