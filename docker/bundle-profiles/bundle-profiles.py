#!/opt/conda/bin/python

import argparse
import dask.dataframe as dd
import json
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import s3fs
import seaborn as sns

from afqinsight.datasets import AFQDataset
from afqinsight.plot import plot_tract_profiles
from matplotlib.lines import Line2D
from neurocombat_sklearn import CombatModel
from plot_formatting import set_size, FULL_WIDTH, TEXT_WIDTH
from scipy.stats import pearsonr
from seaborn.categorical import categorical_order
from seaborn.palettes import color_palette
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, roc_curve, roc_auc_score

BBOX = dict(
    linewidth=1,
    facecolor="white",
    edgecolor="black",
    boxstyle="round,pad=0.25",
)

TEXT_KWARGS = dict(
    x=-0.125,
    y=0.975,
    ha="center",
    va="center",
    zorder=100,
    fontweight="bold",
    bbox=BBOX,
    alpha=1.0,
    fontsize=9,
)


def plot_qc_bundle_profiles(fig_dir):
    dataset = AFQDataset(
        fn_nodes="s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/afq/combined_tract_profiles.csv",
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

    figsize = set_size(width=FULL_WIDTH, subplots=(6, 4))
    qc_bin_figs = plot_tract_profiles(
        X_harmonized,
        groups=dataset.groups,
        group_names=dataset.group_names,
        group_by=dataset.y[:, 0],
        group_by_name="QC",
        bins=4,
        palette="plasma",
        legend_kwargs={"bbox_to_anchor": (0.5, 0.02)},
        figsize=figsize,
        fig_tight_layout_kws=dict(h_pad=0.5, w_pad=0),
    )

    for metric, fig in qc_bin_figs.items():
        fig.savefig(
            op.join(fig_dir, f"qc-bins-{metric.replace('_', '-')}.pdf"),
            bbox_inches="tight",
        )


def plot_qc_stats(fig_dir, hist_ax=None, dist_ax=None, reg_ax=None):
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
        "Sex",
        "Age raw",
        "QC Score",
        "Age Bin",
        "Age",
    ]

    if hist_ax is None:
        fig, hist_ax = plt.subplots(
            1, 2, figsize=set_size(width=TEXT_WIDTH, subplots=(1, 2)), sharey=True
        )
        fig.tight_layout()
    else:
        fig = None

    hist_kws = dict(
        data=participants,
        x="QC Score",
        multiple="dodge",
        shrink=0.8,
        stat="probability",
        common_norm=False,
        bins=10,
    )

    _ = sns.histplot(hue="Sex", ax=hist_ax[0], **hist_kws)
    _ = sns.histplot(hue="Scan Site", ax=hist_ax[1], **hist_kws)

    for axis, letter in zip(hist_ax, "cd"):
        these_kwargs = TEXT_KWARGS.copy()

        _ = axis.text(
            s=letter,
            transform=axis.transAxes,
            **these_kwargs,
        )

    if fig is not None:
        fig.savefig(op.join(fig_dir, "qc-hist.pdf"), bbox_inches="tight")

    if dist_ax is None and reg_ax is None:
        grid_kws = {"height_ratios": (0.18, 0.82), "hspace": 0}
        width, height = set_size(width=0.5 * TEXT_WIDTH)
        fig_reg, (dist_ax, reg_ax) = plt.subplots(
            2, 1, gridspec_kw=grid_kws, figsize=(width, height), sharex=True
        )
    else:
        fig_reg = None

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

    r, p = pearsonr(participants["Age"], participants["QC Score"])
    reg_ax.text(
        x=0.05,
        y=0.95,
        s="r={:.2f}, p={:.2g}".format(r, p),
        transform=reg_ax.transAxes,
        ha="left",
        va="top",
    )

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

    _ = dist_ax.text(
        s="b",
        transform=dist_ax.transAxes,
        **TEXT_KWARGS,
    )

    if fig_reg is not None:
        fig_reg.savefig(op.join(fig_dir, "qc-age-jointplot.pdf"), bbox_inches="tight")

    return {
        "hist_ax": hist_ax,
        "dist_ax": dist_ax,
        "reg_ax": reg_ax,
    }


def plot_qsiprep_stats(fig_dir):
    participants = pd.read_csv(
        "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv",
        sep=None,
        engine="python",
        usecols=["subject_id", "sex", "scan_site_id", "age"],
    )
    participants.dropna(inplace=True)
    participants["age_bin"] = pd.cut(participants["age"], bins=4)
    participants["age_round"] = participants["age"].round().astype(int)
    participants.columns = [
        "subject_id",
        "Scan Site",
        "Sex",
        "Age raw",
        "Age Bin",
        "Age",
    ]
    participants["subject_id"] = participants["subject_id"].apply(
        lambda s: s.replace("sub-", "")
    )
    participants.set_index("subject_id", inplace=True)

    fs = s3fs.S3FileSystem(anon=True)
    dwiqc_s3_uri = (
        "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/dwiqc.json"
    )
    with fs.open(dwiqc_s3_uri) as fp:
        qc_json = json.load(fp)

    df_qc = pd.DataFrame(qc_json["subjects"])
    df_qc["subject_id"] = df_qc["subject_id"].apply(lambda s: s.replace("sub-", ""))
    df_qc.set_index("subject_id", drop=True, inplace=True)
    df_qc.drop(
        [
            "participant_id",
            "session_id",
            "task_id",
            "dir_id",
            "acq_id",
            "file_name",
            "run_id",
            "space_id",
            "rec_id",
        ],
        axis="columns",
        inplace=True,
    )
    df_qc = df_qc[
        ["raw_neighbor_corr", "raw_num_bad_slices", "max_rel_translation"]
    ].copy()
    df_qc.columns = [
        "Neighboring DWI correlation",
        "Num outlier slices",
        "Max rel. translation",
    ]
    df_merged = df_qc.merge(participants, left_index=True, right_index=True)

    width, height = set_size(width=TEXT_WIDTH, subplots=(2, 2))
    fig, axes = plt.subplots(
        3,
        2,
        gridspec_kw=dict(height_ratios=[0.47, 0.47, 0.06]),
        figsize=(width, 1.2 * height),
    )

    gs = axes[2, 0].get_gridspec()
    # remove the underlying axes
    for ax in axes[2, :]:
        ax.remove()

    cax = fig.add_subplot(gs[2, :])
    fig.tight_layout(pad=2)

    _ = sns.violinplot(
        x="Scan Site",
        y="Age raw",
        hue="Sex",
        data=df_merged,
        palette="colorblind",
        split=True,
        inner="quartile",
        ax=axes[0, 0],
        lw=1,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(
        handles,
        labels,
        loc="upper right",
        ncol=2,
        title="Sex",
        borderaxespad=0,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=1,
        borderpad=0.2,
    )

    x_vars = [
        "Neighboring DWI correlation",
        "Neighboring DWI correlation",
        "Num outlier slices",
    ]
    y_vars = ["Num outlier slices", "Max rel. translation", "Max rel. translation"]

    hue_order = categorical_order(df_merged["Age"])
    palette = color_palette("viridis", len(hue_order))

    for ax, x_var, y_var in zip(axes.flatten()[1:4], x_vars, y_vars):
        for idx, age in enumerate(hue_order):
            _df = df_merged[df_merged["Age"] == age].copy()
            _ = sns.scatterplot(
                data=_df,
                x=x_var,
                y=y_var,
                ax=ax,
                legend=False,
                color=palette[idx],
                zorder=idx,
                alpha=0.9,
                s=11,
            )

    norm = plt.Normalize(df_merged["Age"].min(), df_merged["Age"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    fig.colorbar(
        sm,
        orientation="horizontal",
        cax=cax,
        ticks=hue_order,
        label="Age",
    )

    # handles = [
    #     Line2D([0], [0], color=color, ls="", marker="o", ms=4) for color in palette
    # ]

    # _ = axes[1, 0].legend(
    #     handles,
    #     hue_order,
    #     title="Age",
    #     loc="upper center",
    #     bbox_to_anchor=(1.05, -0.25),
    #     ncol=len(hue_order),
    #     handletextpad=-0.5,
    #     columnspacing=0.3,
    # )

    for ax, letter in zip(axes.flatten(), "abcd"):
        these_kwargs = TEXT_KWARGS.copy()
        these_kwargs.update(dict(x=-0.125))
        _ = ax.text(
            s=letter,
            transform=ax.transAxes,
            **these_kwargs,
        )

    fig.savefig(
        op.join(fig_dir, "qsiprep-metric-distributions.pdf"), bbox_inches="tight"
    )


def _get_report_df(glob):
    df_report = dd.read_csv(glob, include_path_column=True).drop(
        "Unnamed: 0", axis="columns"
    )

    df_report["seed"] = df_report["path"].apply(
        lambda s: int(s.split(".csv")[0].split("-")[-1]), meta=("category")
    )

    df_report = df_report.drop("path", axis="columns").compute()
    return df_report


def visualize_auc_curves(report_set_dir, output_dir, ax=None):
    df_report = {
        "with_qc": _get_report_df(op.join(report_set_dir, "with-qc-metrics", "*.csv")),
        "no_qc": _get_report_df(op.join(report_set_dir, "without-qc-metrics", "*.csv")),
    }

    colors = plt.get_cmap("tab10").colors
    labels = ["CNN-i+q", "CNN-i"]

    mean_fpr = np.linspace(0, 1, 100)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=set_size(width=TEXT_WIDTH / 2))
    else:
        fig = None

    for df, label, color in zip(
        df_report.values(),
        labels,
        colors,
    ):
        tprs = []
        aucs = []

        for seed in df["seed"].unique():
            _df = df[df["seed"] == seed]
            y_true = _df["y_true"]
            y_prob = _df["y_prob"]

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(y_true, y_prob))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        _ = ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            label=r"%s (%0.3f $\pm$ %0.3f)" % (label, mean_auc, std_auc),
            lw=1.5,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        _ = ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev." if label == labels[0] else None,
        )

    _ = ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=1.5,
        color=colors[3],
        label="Chance",
        alpha=0.8,
    )

    _ = ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    _ = ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.01, -0.025),
        framealpha=1.0,
        title="Model (AUC)",
        labelspacing=0.25,
        handlelength=1.25,
        handletextpad=0.5,
    )
    _ = ax.set_xlabel("False Positive Rate")
    _ = ax.set_ylabel("True Positive Rate")
    # _ = ax.set_title("Mean ROC")
    ax.text(
        s="a",
        transform=ax.transAxes,
        **TEXT_KWARGS,
    )

    if fig is not None:
        fig.savefig(op.join(output_dir, "dl_roc_auc_curve.pdf"), bbox_inches="tight")

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fig_dir",
        help="Figure directory.",
    )
    parser.add_argument(
        "report_set_dir",
        help="Directory containing the saved tensorflow report set predictions",
    )

    args = parser.parse_args()

    with plt.style.context("/tex.mplstyle"):
        figsize = set_size(width=FULL_WIDTH, subplots=(2, 2))
        fig = plt.figure(figsize=figsize)
        gs0 = gridspec.GridSpec(2, 2, figure=fig)

        gs00 = gs0[0, 0].subgridspec(1, 1)
        auc_ax = fig.add_subplot(gs00[0])

        gs10 = gs0[1, 0].subgridspec(1, 1)
        ax1 = fig.add_subplot(gs10[0])

        gs11 = gs0[1, 1].subgridspec(1, 1)
        ax2 = fig.add_subplot(gs11[0], sharey=ax1)

        hist_ax = np.array([ax1, ax2])

        grid_kws = {"height_ratios": (0.18, 0.82), "hspace": 0}
        gs01 = gs0[0, 1].subgridspec(2, 1, **grid_kws)
        dist_ax = fig.add_subplot(gs01[0])
        reg_ax = fig.add_subplot(gs01[1], sharex=dist_ax)

        # plot_qc_bundle_profiles(fig_dir=args.fig_dir)
        plot_qc_stats(
            fig_dir=args.fig_dir, hist_ax=hist_ax, dist_ax=dist_ax, reg_ax=reg_ax
        )
        # plot_qsiprep_stats(fig_dir=args.fig_dir)
        visualize_auc_curves(
            report_set_dir=args.report_set_dir,
            output_dir=args.fig_dir,
            ax=auc_ax,
        )

        fig.savefig(op.join(args.fig_dir, "deep_learning_qc.pdf"), bbox_inches="tight")
