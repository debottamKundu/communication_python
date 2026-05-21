import os
import pickle as pkl
import numpy as np

results_dir = "./data/generated/rrr_analysis"
files = [f for f in os.listdir(results_dir) if f.endswith("_rrr_results.pkl")]

comm_frac_all = []
comm_frac_high = []
comm_frac_low = []

for file in files:
    with open(os.path.join(results_dir, file), "rb") as f:
        res = pkl.load(f)
        rrr_res = res["rrr_results"]
        for (idx, idy), cond_dict in rrr_res.items():
            if "all" in cond_dict:
                comm_frac_all.append(cond_dict["all"]["comm_fraction"])
            if "high" in cond_dict:
                comm_frac_high.append(cond_dict["high"]["comm_fraction"])
            if "low" in cond_dict:
                comm_frac_low.append(cond_dict["low"]["comm_fraction"])

print(f"Communication Fraction (All): Mean = {np.nanmean(comm_frac_all):.4f}, Std = {np.nanstd(comm_frac_all):.4f}")
print(f"Communication Fraction (High Eng): Mean = {np.nanmean(comm_frac_high):.4f}, Std = {np.nanstd(comm_frac_high):.4f}")
print(f"Communication Fraction (Low Eng): Mean = {np.nanmean(comm_frac_low):.4f}, Std = {np.nanstd(comm_frac_low):.4f}")
