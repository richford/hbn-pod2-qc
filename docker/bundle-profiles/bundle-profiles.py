#!/opt/conda/bin/python

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import os.path as op
import pandas as pd
import seaborn as sns

from afqinsight.datasets import AFQDataset
from afqinsight.plot import plot_tract_profiles
from neurocombat_sklearn import CombatModel
from plot_formatting import set_size, FULL_WIDTH, TEXT_WIDTH
from sklearn.impute import SimpleImputer

BBOX = dict(
    linewidth=1,
    facecolor="white",
    edgecolor="black",
    boxstyle="round,pad=0.25",
)

TEXT_KWARGS = dict(
    x=0,
    y=1,
    ha="center",
    va="center",
    zorder=100,
    fontweight="bold",
    bbox=BBOX,
    alpha=1.0,
)


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

    fig, ax = plt.subplots(
        1, 2, figsize=set_size(width=TEXT_WIDTH, subplots=(1, 2)), sharey=True
    )
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

    grid_kws = {"height_ratios": (0.18, 0.82), "hspace": 0}
    width, height = set_size(width=0.5 * TEXT_WIDTH)
    fig_reg, (dist_ax, reg_ax) = plt.subplots(
        2, 1, gridspec_kw=grid_kws, figsize=(width, height), sharex=True
    )

    _ = sns.histplot(
        data=participants, x=participants["Age"], kde=False, discrete=True, ax=dist_ax
    )

    linewidth = mpl.rcParams["lines.linewidth"]
    with mpl.rc_context({"lines.linewidth": linewidth / 2.5}):
        _ = sns.regplot(
            data=participants,
            x="Age",
            y="QC Score",
            x_estimator=np.mean,
            truncate=False,
            ax=reg_ax,
            scatter_kws=dict(s=11),
        )

    mpl.rcParams["lines.linewidth"] = linewidth

    dist_ax.spines["top"].set_visible(False)
    dist_ax.spines["right"].set_visible(False)
    dist_ax.set_ylabel("")
    dist_ax.spines["left"].set_visible(False)
    dist_ax.get_yaxis().set_ticks([])

    # Set ylim and customize the xticks
    reg_ax.set_ylim(0.38, 1)
    reg_ax.set_xticks(list(range(5, 23, 5)))
    dist_ax.set_xticks(list(range(5, 23, 5)))

    visible_ticks = {"top": False, "bottom": False}
    dist_ax.tick_params(axis="x", which="both", direction="out", **visible_ticks)

    _ = sns.histplot(hue="Sex", ax=ax[0], **hist_kws)
    _ = sns.histplot(hue="Scan Site", ax=ax[1], **hist_kws)

    for axis, letter in zip(ax, "cd"):
        _ = axis.text(
            s=letter,
            transform=axis.transAxes,
            **TEXT_KWARGS,
        )

    fig.savefig(op.join(fig_dir, "qc-hist.pdf"), bbox_inches="tight")

    _ = dist_ax.text(
        s="b",
        transform=dist_ax.transAxes,
        **TEXT_KWARGS,
    )
    fig_reg.savefig(op.join(fig_dir, "qc-age-jointplot.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fig_dir",
        help="Figure directory.",
    )

    args = parser.parse_args()

    with plt.style.context("/tex.mplstyle"):
        # plot_qc_bundle_profiles(fig_dir=args.fig_dir)
        plot_qc_stats(fig_dir=args.fig_dir)
