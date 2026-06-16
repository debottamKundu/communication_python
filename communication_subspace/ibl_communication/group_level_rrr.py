import os
import glob
import pickle as pkl
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_cohort_data(data_dir):
    """
    Loads all .pkl files from the data directory and structures them into pandas DataFrames.
    Returns:
        df_pairs: DataFrame where each row is a region-pair (source->target) in a specific session.
        df_regions: DataFrame where each row is a single region in a specific session (for ID).
        region_names: List of all brain regions in consistent order.
    """
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    if not files:
        raise ValueError(f"No .pkl files found in {data_dir}")
        
    logger.info(f"Loading {len(files)} sessions from {data_dir}")
    
    pair_rows = []
    region_rows = []
    region_names = None
    
    for file in files:
        try:
            with open(file, "rb") as f:
                data = pkl.load(f)
                
            eid = data["session_id"]
            regions = data["regions"]
            n_regions = len(regions)
            
            if region_names is None:
                region_names = list(regions)
                
            # --- 1. Extract Pairwise Data (Communication) ---
            for idx in range(n_regions):
                for idy in range(n_regions):
                    if idx == idy:
                        continue
                        
                    source = regions[idx]
                    target = regions[idy]
                    is_sig = (idx, idy) in data["significant_pairs"]
                    
                    row = {
                        "eid": eid,
                        "source_idx": idx,
                        "target_idx": idy,
                        "source": source,
                        "target": target,
                        "is_significant": is_sig,
                        "p_value": data["p_values"][idx, idy],
                        "ridge_r2_all": data["true_ridge_r2"][idx, idy],
                        "ridge_r2_high": data["high_ridge_r2"][idx, idy],
                        "ridge_r2_low": data["low_ridge_r2"][idx, idy],
                    }
                    
                    # Extract RRR metrics if available
                    rrr_pair = data["rrr_results"].get((idx, idy), {})
                    for cond in ["all", "high", "low"]:
                        cond_res = rrr_pair.get(cond, None)
                        if cond_res is not None:
                            row[f"rrr_rank_{cond}"] = cond_res["optimal_rank"]
                            row[f"rrr_r2_{cond}"] = cond_res["cv_r2_rrr"]
                            row[f"align_in_{cond}"] = cond_res["input_alignment"]
                            row[f"align_out_{cond}"] = cond_res["output_alignment"]
                            row[f"comm_frac_{cond}"] = cond_res["comm_fraction"]
                        else:
                            row[f"rrr_rank_{cond}"] = np.nan
                            row[f"rrr_r2_{cond}"] = np.nan
                            row[f"align_in_{cond}"] = np.nan
                            row[f"align_out_{cond}"] = np.nan
                            row[f"comm_frac_{cond}"] = np.nan
                            
                    pair_rows.append(row)
                    
            # --- 2. Extract Regional Data (Intrinsic Dimensionality) ---
            id_data = data.get("intrinsic_dimensionality", {})
            has_id = len(id_data) > 0
            
            for idx in range(n_regions):
                r_row = {
                    "eid": eid,
                    "region": regions[idx],
                }
                if has_id:
                    # Depending on how the dict was saved, handle arrays
                    stim_all = id_data["stim"]["all"]
                    stim_high = id_data["stim"]["high"]
                    stim_low = id_data["stim"]["low"]
                    
                    choice_all = id_data["choice"]["all"]
                    choice_high = id_data["choice"]["high"]
                    choice_low = id_data["choice"]["low"]
                    
                    # They might be tuples (dim, var_exp) or just raw values. If tuple, get first element
                    get_val = lambda x, i: x[i] if isinstance(x, (list, np.ndarray)) else x[0][i] if isinstance(x, tuple) else np.nan
                    
                    try:
                        r_row["stim_dim_all"] = get_val(stim_all, idx)
                        r_row["stim_dim_high"] = get_val(stim_high, idx)
                        r_row["stim_dim_low"] = get_val(stim_low, idx)
                        
                        r_row["choice_dim_all"] = get_val(choice_all, idx)
                        r_row["choice_dim_high"] = get_val(choice_high, idx)
                        r_row["choice_dim_low"] = get_val(choice_low, idx)
                    except Exception as e:
                        logger.warning(f"Could not parse ID for {eid}: {e}")
                
                region_rows.append(r_row)
                
        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")
            
    df_pairs = pd.DataFrame(pair_rows)
    df_regions = pd.DataFrame(region_rows)
    
    logger.info(f"Loaded {len(df_pairs)} pair records and {len(df_regions)} region records.")
    return df_pairs, df_regions, region_names

def _apply_masking(df, region_names, min_prevalence=0.5, min_r2=0.0):
    """
    Helper function: Identifies which (source, target) pairs meet the criteria:
    1. is_significant == True in >= min_prevalence * N_animals
    OR
    2. ridge_r2_all > min_r2 in >= min_prevalence * N_animals
    """
    n_animals = df["eid"].nunique()
    
    # Calculate prevalences
    sig_counts = df.groupby(["source", "target"])["is_significant"].sum()
    r2_counts = df[df["ridge_r2_all"] > min_r2].groupby(["source", "target"]).size()
    
    # Default to 0 if not present
    sig_counts = sig_counts.reindex(pd.MultiIndex.from_product([region_names, region_names]), fill_value=0)
    r2_counts = r2_counts.reindex(pd.MultiIndex.from_product([region_names, region_names]), fill_value=0)
    
    sig_frac = sig_counts / n_animals
    r2_frac = r2_counts / n_animals
    
    # Valid mask
    valid_mask = (sig_frac >= min_prevalence) | (r2_frac >= min_prevalence)
    
    # Extract list of valid (source, target) tuples
    valid_pairs = valid_mask[valid_mask].index.tolist()
    return valid_pairs

def analyze_communication_efficiency(df_pairs, region_names, min_prevalence=0.5, min_r2=0.0):
    """
    Tests if Ridge R2 changes between High and Low engagement.
    """
    logger.info("Analyzing Communication Efficiency (High vs Low R2)...")
    valid_pairs = _apply_masking(df_pairs, region_names, min_prevalence, min_r2)
    logger.info(f"Analyzing {len(valid_pairs)} robust region pairs.")
    
    results = []
    for source, target in valid_pairs:
        if source == target:
            continue
            
        pair_data = df_pairs[(df_pairs["source"] == source) & (df_pairs["target"] == target)]
        # Drop NaNs
        pair_data = pair_data.dropna(subset=["ridge_r2_high", "ridge_r2_low"])
        
        if len(pair_data) < 5:  # Need minimum N for stats
            continue
            
        diffs = pair_data["ridge_r2_high"] - pair_data["ridge_r2_low"]
        mean_diff = diffs.mean()
        
        # Wilcoxon signed-rank
        try:
            stat, pval = stats.wilcoxon(pair_data["ridge_r2_high"], pair_data["ridge_r2_low"])
        except ValueError:
            pval = 1.0  # Zero variance
            
        results.append({
            "source": source,
            "target": target,
            "mean_high": pair_data["ridge_r2_high"].mean(),
            "mean_low": pair_data["ridge_r2_low"].mean(),
            "mean_diff": mean_diff,
            "n_animals": len(pair_data),
            "p_val": pval
        })
        
    df_res = pd.DataFrame(results)
    if len(df_res) > 0:
        rejected, pvals_fdr = fdrcorrection(df_res["p_val"], alpha=0.05)
        df_res["p_val_fdr"] = pvals_fdr
        df_res["is_sig_fdr"] = rejected
    
    return df_res

def analyze_subspace_modulation(df_pairs, region_names, min_prevalence=0.5, min_r2=0.0):
    """
    Analyzes shifts in Optimal Rank, Input Alignment, and Output Alignment.
    """
    logger.info("Analyzing Subspace Modulation (Rank and Alignment)...")
    valid_pairs = _apply_masking(df_pairs, region_names, min_prevalence, min_r2)
    
    metrics = [
        ("rrr_rank", "Optimal Rank"),
        ("align_in", "Input Alignment"),
        ("align_out", "Output Alignment")
    ]
    
    results = []
    for source, target in valid_pairs:
        if source == target:
            continue
            
        pair_data = df_pairs[(df_pairs["source"] == source) & (df_pairs["target"] == target)]
        
        res_dict = {"source": source, "target": target, "n_animals": len(pair_data)}
        
        for prefix, name in metrics:
            col_high = f"{prefix}_high"
            col_low = f"{prefix}_low"
            
            valid_data = pair_data.dropna(subset=[col_high, col_low])
            if len(valid_data) >= 5:
                res_dict[f"{prefix}_mean_high"] = valid_data[col_high].mean()
                res_dict[f"{prefix}_mean_low"] = valid_data[col_low].mean()
                res_dict[f"{prefix}_diff"] = valid_data[col_high].mean() - valid_data[col_low].mean()
                
                try:
                    _, pval = stats.wilcoxon(valid_data[col_high], valid_data[col_low])
                except ValueError:
                    pval = 1.0
                res_dict[f"{prefix}_pval"] = pval
            else:
                res_dict[f"{prefix}_pval"] = 1.0
                
        results.append(res_dict)
        
    df_res = pd.DataFrame(results)
    if len(df_res) > 0:
        for prefix, _ in metrics:
            if f"{prefix}_pval" in df_res.columns:
                rej, fdr = fdrcorrection(df_res[f"{prefix}_pval"], alpha=0.05)
                df_res[f"{prefix}_pval_fdr"] = fdr
                df_res[f"{prefix}_is_sig_fdr"] = rej
                
    return df_res

def plot_group_matrices(df_res, region_names, value_col, sig_col, title, output_path, cmap="coolwarm"):
    """
    Plots an N x N matrix heatmap.
    Only colors the square if sig_col == True, otherwise grey/white.
    """
    n_regions = len(region_names)
    matrix = np.full((n_regions, n_regions), np.nan)
    
    # Map regions to indices
    reg_to_idx = {r: i for i, r in enumerate(region_names)}
    
    for _, row in df_res.iterrows():
        if sig_col is None or row[sig_col]:
            idx = reg_to_idx[row["source"]]
            idy = reg_to_idx[row["target"]]
            matrix[idx, idy] = row[value_col]
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=region_names, yticklabels=region_names, cmap=cmap, center=0 if "diff" in value_col else None)
    plt.title(title)
    plt.xlabel("Target Region")
    plt.ylabel("Source Region")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved matrix to {output_path}")



def analyze_communication_fractions(df_pairs, region_names, min_prevalence=0.5, min_r2=0.0):
    """
    Analyzes the fraction of predictable variance captured by the subspace bottleneck (RRR R2 / Ridge R2).
    """
    logger.info("Analyzing Communication Fractions...")
    valid_pairs = _apply_masking(df_pairs, region_names, min_prevalence, min_r2)
    
    results = []
    for source, target in valid_pairs:
        if source == target:
            continue
            
        pair_data = df_pairs[(df_pairs["source"] == source) & (df_pairs["target"] == target)].copy()
        
        # Calculate fractions
        pair_data["frac_all"] = pair_data["rrr_r2_all"] / pair_data["ridge_r2_all"].replace(0, np.nan)
        pair_data["frac_high"] = pair_data["rrr_r2_high"] / pair_data["ridge_r2_high"].replace(0, np.nan)
        pair_data["frac_low"] = pair_data["rrr_r2_low"] / pair_data["ridge_r2_low"].replace(0, np.nan)
        
        valid_data = pair_data.dropna(subset=["frac_high", "frac_low"])
        if len(valid_data) >= 5:
            diffs = valid_data["frac_high"] - valid_data["frac_low"]
            try:
                _, pval = stats.wilcoxon(valid_data["frac_high"], valid_data["frac_low"])
            except ValueError:
                pval = 1.0
                
            results.append({
                "source": source,
                "target": target,
                "n_animals": len(valid_data),
                "frac_mean_all": pair_data["frac_all"].mean(),
                "frac_mean_high": valid_data["frac_high"].mean(),
                "frac_mean_low": valid_data["frac_low"].mean(),
                "frac_diff": diffs.mean(),
                "frac_pval": pval
            })
            
    df_res = pd.DataFrame(results)
    if len(df_res) > 0:
        rej, fdr = fdrcorrection(df_res["frac_pval"], alpha=0.05)
        df_res["frac_pval_fdr"] = fdr
        df_res["frac_is_sig_fdr"] = rej
        
    return df_res

def analyze_intrinsic_dimensionality(df_regions):
    """
    Analyzes shifts in local state complexity (Intrinsic Dimensionality) across regions.
    """
    logger.info("Analyzing Intrinsic Dimensionality...")
    results = []
    
    for region in df_regions["region"].unique():
        reg_data = df_regions[df_regions["region"] == region]
        reg_data = reg_data.dropna(subset=["stim_dim_high", "stim_dim_low"])
        
        if len(reg_data) >= 5:
            try:
                _, pval_stim = stats.wilcoxon(reg_data["stim_dim_high"], reg_data["stim_dim_low"])
            except ValueError:
                pval_stim = 1.0
                
            results.append({
                "region": region,
                "n_animals": len(reg_data),
                "stim_dim_all": reg_data["stim_dim_all"].mean(),
                "stim_dim_high": reg_data["stim_dim_high"].mean(),
                "stim_dim_low": reg_data["stim_dim_low"].mean(),
                "stim_dim_diff": (reg_data["stim_dim_high"] - reg_data["stim_dim_low"]).mean(),
                "stim_pval": pval_stim
            })
            
    df_res = pd.DataFrame(results)
    if len(df_res) > 0:
        rej, fdr = fdrcorrection(df_res["stim_pval"], alpha=0.05)
        df_res["stim_pval_fdr"] = fdr
        df_res["stim_is_sig_fdr"] = rej
        
    return df_res

def plot_brain_maps(df_regions_res, df_pairs_res, output_dir):
    """
    Plots regional metrics onto the Allen CCF Swanson Flat Map using iblatlas.
    """
    logger.info("Generating Swanson Flat Maps...")
    try:
        from iblatlas.plots import plot_swanson_vector
        import matplotlib as mpl
    except ImportError:
        logger.warning("iblatlas is not installed or available in this environment. Skipping Swanson plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot Intrinsic Dimensionality (All Trials)
    if df_regions_res is not None and not df_regions_res.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        acronyms = df_regions_res["region"].values
        amps = np.array([float(np.mean(x)) if isinstance(x, (list, np.ndarray)) else float(x) for x in df_regions_res["stim_dim_all"].values], dtype=float)
        
        plot_swanson_vector(acronyms, amps, cmap="viridis", ax=ax, orientation="landscape")
        
        norm = mpl.colors.Normalize(vmin=np.nanmin(amps), vmax=np.nanmax(amps))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, orientation="horizontal", fraction=0.05, pad=0.05)
        cbar.set_label("PCA Dimensionality (95% Var)")
        ax.set_title("Intrinsic Dimensionality per Region")
        ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "swanson_intrinsic_dim_all.png"), dpi=300)
        plt.close()
        
    # 2. Plot "Hubness" (Total Incoming R2 per Region)
    if df_pairs_res is not None and not df_pairs_res.empty:
        # Sum incoming R2 across all sources for each target
        incoming_hub = df_pairs_res.groupby("target")["mean_high"].sum().reset_index()
        incoming_hub.columns = ["region", "total_in_r2"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        acronyms = incoming_hub["region"].values
        amps = np.array([float(np.mean(x)) if isinstance(x, (list, np.ndarray)) else float(x) for x in incoming_hub["total_in_r2"].values], dtype=float)
        
        plot_swanson_vector(acronyms, amps, cmap="magma", ax=ax, orientation="landscape")
        
        norm = mpl.colors.Normalize(vmin=np.nanmin(amps), vmax=np.nanmax(amps))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax, orientation="horizontal", fraction=0.05, pad=0.05)
        cbar.set_label("Total Incoming Communication (Sum R2)")
        ax.set_title("Major Receivers (High Engagement)")
        ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "swanson_hub_receivers_high.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    data_dir = "/Users/dkundu/Documents/phd/communication_python/data/generated/rrr_analysis/frame1_stim"
    out_dir = "/Users/dkundu/Documents/phd/communication_python/data/generated/group_analysis"
    os.makedirs(out_dir, exist_ok=True)
    
    df_pairs, df_regions, region_names = load_cohort_data(data_dir)
    
    # 1. Communication Efficiency
    df_eff = analyze_communication_efficiency(df_pairs, region_names, min_prevalence=0.5, min_r2=0.0)
    df_eff.to_csv(os.path.join(out_dir, "communication_efficiency_stats.csv"), index=False)
    
    plot_group_matrices(df_eff, region_names, "mean_diff", "is_sig_fdr", "Engagement Modulation (High - Low R2)", os.path.join(out_dir, "engagement_modulation_fdr.png"))
    plot_group_matrices(df_eff, region_names, "mean_high", None, "High Engagement R2", os.path.join(out_dir, "high_engagement_r2.png"), cmap="viridis")
    
    # 2. Subspace Modulation
    df_sub = analyze_subspace_modulation(df_pairs, region_names, min_prevalence=0.5, min_r2=0.0)
    df_sub.to_csv(os.path.join(out_dir, "subspace_modulation_stats.csv"), index=False)
    
    plot_group_matrices(df_sub, region_names, "rrr_rank_diff", "rrr_rank_is_sig_fdr", "Rank Difference (High - Low)", os.path.join(out_dir, "rank_difference_fdr.png"))
    plot_group_matrices(df_sub, region_names, "align_out_diff", "align_out_is_sig_fdr", "Output Alignment Shift", os.path.join(out_dir, "align_out_difference_fdr.png"))
    
    # 3. Communication Fractions
    df_frac = analyze_communication_fractions(df_pairs, region_names, min_prevalence=0.5, min_r2=0.0)
    df_frac.to_csv(os.path.join(out_dir, "communication_fractions_stats.csv"), index=False)
    
    # 4. Intrinsic Dimensionality
    df_id = analyze_intrinsic_dimensionality(df_regions)
    df_id.to_csv(os.path.join(out_dir, "intrinsic_dimensionality_stats.csv"), index=False)
    
    # 5. Swanson Flat Maps
    plot_brain_maps(df_id, df_eff, out_dir)
    
    logger.info("Group level pipeline completed!")

