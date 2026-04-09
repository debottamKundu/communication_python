from glob import glob
import pickle as pkl
import pandas as pd
import numpy as np
from one.api import ONE
from tqdm import tqdm


def flatten_prediction_dict(animal_dict, source_type, animal_id):
    """Flattens your nested dictionary into a Pandas DataFrame."""
    records = []

    for region_1, nested_dict in animal_dict.items():
        for region_2, frame_dict in nested_dict.items():
            for frame_pair, metrics in frame_dict.items():

                source_frame, choice_frame = frame_pair
                r2 = metrics.get("r2_scores")

                if isinstance(r2, (list, np.ndarray)):
                    for score in r2:
                        records.append(
                            {
                                "session": animal_id,
                                "region_1": region_1,
                                "region_2": region_2,
                                "source_type": source_type,  # "Prior" or "Stimulus"
                                "source_frame": source_frame,  # Prior=0; Stimulus=0,1,2
                                "choice_frame": choice_frame,  # 0, 1, or 2
                                "r2_score": score,
                            }
                        )
                else:
                    records.append(
                        {
                            "session": animal_id,
                            "region_1": region_1,
                            "region_2": region_2,
                            "source_type": source_type,
                            "source_frame": source_frame,
                            "choice_frame": choice_frame,
                            "r2_score": r2,
                        }
                    )
    return pd.DataFrame(records)


if __name__ == "__main__":

    all_eids = glob("./data/crossprediction/*")
    sessions = [eid.split("/")[-1].split("_")[0] for eid in all_eids]
    sessions = np.unique(sessions)

    df_all = []
    for idx, eid in tqdm(enumerate(sessions)):

        filea = f"./data/crossprediction/{eid}_results_stim_choice.pkl"
        fileb = f"./data/crossprediction/{eid}_results_prior_choice.pkl"

        try:

            data_stim = pkl.load(open(filea, "rb"))
            data_prior = pkl.load(open(fileb, "rb"))

            df_stim = flatten_prediction_dict(data_stim, "Stimulus", idx)
            df_prior = flatten_prediction_dict(data_prior, "Prior", idx)

            df_raw = pd.concat([df_prior, df_stim], ignore_index=True)

            # do some housekeeping, throw away frames I think are overlapping
            df_raw = df_raw[
                (~df_raw["choice_frame"].isin([0, 1]))
                & ((df_raw["source_frame"] == 0) | (df_raw["source_frame"] == 1))
            ]  # keep it for now
            df_all.append(df_raw)
        except Exception as e:
            print(f"Error with {eid}, probably not generated yet", e)
            continue

    df_all = pd.concat(df_all, ignore_index=True)
    df_all.to_parquet("./data/crossprediction/crossprediction_results.pqt")
