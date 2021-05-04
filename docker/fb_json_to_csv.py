#!/opt/conda/bin/python

import argparse
import json
import os.path as op
import pandas as pd

from glob import glob
from tqdm import tqdm


def parse_firebase_json(
    json_file,
    votes_file="votes.csv",
    users_file="users.csv",
    sample_summary_file="sample_summary.csv",
    user_seen_samples="user_seen_samples.csv",
):
    with tqdm(total=5, ncols=50) as pbar:
        with open(json_file, "r") as fp:
            data = json.load(fp)
        pbar.update(1)

        df_votes = pd.DataFrame(data["votes"].values())
        df_votes.to_csv(votes_file, index=False)
        pbar.update(1)

        df_users = pd.DataFrame(data["users"]).T
        df_users.index.name = "user"
        df_users.to_csv(users_file)
        pbar.update(1)

        df_sample_summary = pd.DataFrame(data["sampleSummary"]).T
        df_sample_summary.index.name = "sample"
        df_sample_summary.to_csv(sample_summary_file)
        pbar.update(1)

        df_user_seen_samples = pd.DataFrame(data["userSeenSamples"])
        df_user_seen_samples.columns.name = "user"
        df_user_seen_samples.index.name = "sample"
        df_user_seen_samples.to_csv(user_seen_samples)
        pbar.update(1)

    return {
        "votes": df_votes,
        "users": df_users,
        "sample_summary": df_sample_summary,
        "df_user_seen_samples": df_user_seen_samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="Directory containing SwipesForScience json file to convert to multiple csv files.",
    )
    args = parser.parse_args()
    out_dir = op.abspath(args.input_dir)
    if not op.isdir(out_dir):
        out_dir = op.dirname(out_dir)

    input_json = glob(op.join(out_dir, "*.json"))
    if len(input_json) != 1:
        raise ValueError(
            f"There must be one json file in the input/output directory. "
            f"Found {input_json} instead."
        )

    input_json = input_json[0]
    print(f"Parsing {input_json} to csv files.")

    parse_firebase_json(
        input_json,
        votes_file=op.join(out_dir, "votes.csv"),
        users_file=op.join(out_dir, "users.csv"),
        sample_summary_file=op.join(out_dir, "sample_summary.csv"),
        user_seen_samples=op.join(out_dir, "user_seen_samples.csv"),
    )