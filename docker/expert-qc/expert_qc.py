#!/opt/conda/bin/python

import argparse
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import pingouin as pg
import re
import seaborn as sns

from glob import glob
from sklearn.metrics import cohen_kappa_score


def process_expert_qc(qc_dir):
    qc_files = glob(op.join(qc_dir, "expertqc*.csv"))

    expert_qc = dd.read_csv(
        qc_files, usecols=["subject", "rating"], include_path_column=True
    ).compute()
    expert_qc["rater"] = (
        expert_qc["path"]
        .apply(lambda s: op.basename(s).replace("expertqc-", "").replace(".csv", ""))
        .astype(str)
    )
    expert_qc.drop("path", axis="columns", inplace=True)
    expert_qc.reset_index(drop=True, inplace=True)

    mean_expert_rating = expert_qc.groupby("subject").agg("mean")
    mean_expert_rating["subject"] = mean_expert_rating.index

    r = re.compile("ses-HBNsite([a-zA-Z0-9]*)")
    mean_expert_rating["site"] = mean_expert_rating["subject"].apply(
        lambda s: r.search(s).group(1)
    )

    r = re.compile("sub-([a-zA-Z0-9]*)_")
    mean_expert_rating["subject"] = mean_expert_rating["subject"].apply(
        lambda s: r.search(s).group(1)
    )

    mean_expert_rating.set_index("subject", drop=True, inplace=True)
    mean_expert_rating.to_csv(op.join(qc_dir, "expert_ratings.csv"))

    ratings = expert_qc.pivot(index="subject", columns="rater", values="rating")
    cohen_kappas = {}
    for rater in ratings.columns:
        cohen_kappas[rater] = [
            cohen_kappa_score(
                ratings[rater].to_numpy(),
                ratings[other_rater].to_numpy(),
                labels=[-2, -1, 0, 1, 2],
                weights="quadratic",
            )
            for other_rater in ratings.columns
        ]

    df_kappa = pd.DataFrame(cohen_kappas, index=ratings.columns)
    for rater in ratings.columns:
        df_kappa.loc[rater, rater] = np.nan

    df_kappa["mean"] = df_kappa.mean(axis="columns")
    df_kappa.to_csv(op.join(qc_dir, "cohens_kappa.csv"))

    icc = (
        pg.intraclass_corr(
            data=expert_qc, targets="subject", raters="rater", ratings="rating"
        )
        .round(3)
        .set_index("Type")
        .filter(like="ICC3", axis="index")
    )
    icc.to_csv(op.join(qc_dir, "icc.csv"))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    _ = sns.histplot(
        data=expert_qc, x="rating", bins=np.linspace(-2.5, 2.5, 6), ax=ax[0]
    )

    _ = sns.histplot(
        data=expert_qc,
        x="rating",
        bins=np.linspace(-2.5, 2.5, 6),
        hue="rater",
        multiple="dodge",
        shrink=0.8,
        ax=ax[1],
    )

    _ = ax[0].set_title("Average expert ratings")
    _ = ax[1].set_title("Expert rating distribution")
    fig.savefig(op.join(qc_dir, "expert_rating_distributions.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="Directory containing expert QC csv files.",
    )
    args = parser.parse_args()
    in_dir = op.abspath(args.input_dir)
    if not op.isdir(in_dir):
        in_dir = op.dirname(in_dir)

    process_expert_qc(in_dir)