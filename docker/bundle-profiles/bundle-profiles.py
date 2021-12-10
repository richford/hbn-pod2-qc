#!/opt/conda/bin/python

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as op
import pandas as pd
import seaborn as sns

from afqinsight.datasets import AFQDataset
from afqinsight.plot import plot_tract_profiles
from neurocombat_sklearn import CombatModel
from sklearn.impute import SimpleImputer


def plot_qc_bundle_profiles(fig_dir):
    dataset = AFQDataset(
        fn_nodes="s3://hbn-afq/derivatives/afq_like_hcp/combined_tract_profiles.csv",
        fn_subjects="s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv",
        target_cols=["dl_qc_score", "scan_site_id"],
        dwi_metrics=["dki_fa", "dki_md"],
        label_encode_cols=["scan_site_id"],
        index_col="subject_id",
    )
    dataset.drop_target_na()

    imputer = SimpleImputer(strategy="median")
    dataset.X = imputer.fit_transform(dataset.X)
    site_ids = dataset.y[:, 1].reshape(-1, 1)

    X_harmonized = CombatModel().fit_transform(
        dataset.X,
        site_ids,
        None,
        None,
    )

    qc_bin_figs = plot_tract_profiles(
        X_harmonized,
        groups=dataset.groups,
        group_names=dataset.group_names,
        group_by=dataset.y[:, 0],
        group_by_name="QC",
        bins=4,
        palette="plasma",
        legend_kwargs={"bbox_to_anchor": (0.5, 0.02)},
    )

    for metric, fig in qc_bin_figs.items():
        fig.savefig(
            op.join(fig_dir, f"qc-bins-{metric.replace('_', '-')}.pdf"),
            bbox_inches="tight",
        )


def plot_qc_stats(fig_dir):
    participants = pd.read_csv(
        "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv",
        sep=None,
        engine="python",
        usecols=["subject_id", "dl_qc_score", "sex", "scan_site_id", "age"],
    )
    participants.dropna(inplace=True)
    participants["age_bin"] = pd.cut(participants["age"], bins=4)
    participants["age_round"] = participants["age"].round().astype(int)
    participants.columns = [
        "subject_id",
        "Scan Site",
        "QC Score",
        "Sex",
        "Age raw",
        "Age Bin",
        "Age",
    ]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.tight_layout()

    hist_kws = dict(
        data=participants,
        x="QC Score",
        multiple="dodge",
        shrink=0.8,
        stat="probability",
        common_norm=False,
        bins=10,
    )

    jointgrid = sns.jointplot(
        data=participants,
        x="Age",
        y="QC Score",
        kind="reg",
        joint_kws=dict(
            x_estimator=np.mean,
            truncate=False,
        ),
        marginal_kws=dict(
            kde=False,
            discrete=True,
        ),
        height=5,
    )

    # Set ylim and customize the xticks
    jointgrid.ax_joint.set_ylim(0.38, 1)
    jointgrid.ax_joint.set_xticks(list(range(5, 23, 5)))

    # Remove the right marginal plot
    jointgrid.fig.delaxes(jointgrid.ax_marg_y)

    # Expand the width of the remaining axes to fill up the space left by the right marginal plot
    # First get the bounding boxes of the remaining axes
    joint_bounds = jointgrid.ax_joint.get_position().bounds
    marg_bounds = jointgrid.ax_marg_x.get_position().bounds
    total_width = joint_bounds[-1] + marg_bounds[-1]
    joint_bounds = joint_bounds[:2] + (total_width, joint_bounds[-1])
    marg_bounds = marg_bounds[:2] + (total_width, marg_bounds[-1])
    jointgrid.ax_joint.set_position(joint_bounds)
    jointgrid.ax_marg_x.set_position(marg_bounds)

    _ = sns.histplot(hue="Sex", ax=ax[0], **hist_kws)
    _ = sns.histplot(hue="Scan Site", ax=ax[1], **hist_kws)

    fig.savefig(op.join(fig_dir, "qc-hist.pdf"), bbox_inches="tight")
    jointgrid.fig.savefig(op.join(fig_dir, "qc-age-jointplot.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fig_dir",
        help="Figure directory.",
    )

    args = parser.parse_args()

    plot_qc_bundle_profiles(fig_dir=args.fig_dir)
    plot_qc_stats(fig_dir=args.fig_dir)