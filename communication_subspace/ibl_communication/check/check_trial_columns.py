import pickle as pkl

trials_path = "/Users/dkundu/Documents/phd/communication_python/data/processed/wifi_trials_df_all.pkl"
with open(trials_path, "rb") as f:
    all_trials = pkl.load(f)

eid = list(all_trials.keys())[0]
trials = all_trials[eid]
print(f"Columns for session {eid}:")
print(trials.columns.tolist())
print("\nSample values for 'choice':")
if "choice" in trials.columns:
    print(trials["choice"].head())
print("\nSample values for 'sign_cont':")
if "sign_cont" in trials.columns:
    print(trials["sign_cont"].head())
