import pickle as pkl
import os
import numpy as np

results_dir = "./data/generated/rrr_analysis"
files = [f for f in os.listdir(results_dir) if f.endswith("_rrr_results.pkl")]

all_sig_pairs = []

for file in files:
    with open(os.path.join(results_dir, file), "rb") as f:
        res = pkl.load(f)
        regions = res["regions"]
        sig_pairs = res["significant_pairs"]
        n_pairs = len(regions) * len(regions)
        frac = len(sig_pairs) / n_pairs
        print(f"Session {res['session_id']}: {len(sig_pairs)} significant pairs out of {n_pairs} ({frac:.1%})")
        
        # Convert pairs to strings for easy intersection
        pair_strings = set(f"{regions[idx]}->{regions[idy]}" for idx, idy in sig_pairs)
        all_sig_pairs.append(pair_strings)

if len(all_sig_pairs) == 2:
    intersection = len(all_sig_pairs[0].intersection(all_sig_pairs[1]))
    union = len(all_sig_pairs[0].union(all_sig_pairs[1]))
    print(f"\nOverlap (Intersection): {intersection} pairs")
    print(f"Total Unique Pairs (Union): {union} pairs")
    print(f"Jaccard Similarity (Consistency): {intersection / union:.1%}")

