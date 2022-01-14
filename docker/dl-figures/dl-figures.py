#!/opt/conda/bin/python

import argparse
import dask.dataframe as dd
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import os.path as op
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import warnings

from glob import glob
from plot_formatting import set_size, FULL_WIDTH, TEXT_WIDTH

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from nilearn import plotting

from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

BBOX = dict(
    linewidth=1,
    facecolor="white",
    edgecolor="black",
    boxstyle="round,pad=0.25",
)

TEXT_KWARGS = dict(
    x=-0.15,
    y=1,
    ha="center",
    va="center",
    zorder=100,
    fontweight="bold",
    bbox=BBOX,
    alpha=1.0,
    fontsize=9,
)


def save_hbn_pod2_sankey(out_dir):
    fig = go.Figure(
        data=[
            go.Sankey(
                textfont=dict(size=32, color="black"),
                node=dict(
                    pad=50,
                    thickness=25,
                    line=dict(color="black", width=0.5),
                    label=[
                        "Raw (2,747)",
                        "BIDS pass (2,615)",
                        "Preproc pass (2,136)",
                        "Expert QC (200)",
                        "Community QC (1,653)",
                        "NN QC (2,136)",
                        "Excluded (611)",
                    ],
                    # color = "blue"
                ),
                link=dict(
                    source=[
                        0,
                        1,
                        1,
                        0,
                        2,
                        3,
                        2,
                        4,
                        2,
                    ],  # indices correspond to labels, eg A1, A2, A1, B1, ...
                    target=[1, 2, 6, 6, 3, 4, 4, 5, 5],
                    value=[2615, 2136, 479, 132, 200, 200, 1453, 1653, 483],
                ),
            )
        ]
    )

    fig.update_layout(title_text="HBN-POD2 Data Workflow", font_size=12)
    fig.write_html(op.join(out_dir, "hbn-pod2-sankey.html"))


def _get_report_df(glob):
    df_report = dd.read_csv(glob, include_path_column=True).drop(
        "Unnamed: 0", axis="columns"
    )

    df_report["seed"] = df_report["path"].apply(
        lambda s: int(s.split(".csv")[0].split("-")[-1]), meta=("category")
    )

    df_report = df_report.drop("path", axis="columns").compute()
    return df_report


def visualize_auc_curves(report_set_dir, output_dir):
    df_report = {
        "with_qc": _get_report_df(op.join(report_set_dir, "with-qc-metrics", "*.csv")),
        "no_qc": _get_report_df(op.join(report_set_dir, "without-qc-metrics", "*.csv")),
    }

    colors = plt.get_cmap("tab10").colors
    labels = ["CNN-i+q", "CNN-i"]

    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(1, 1, figsize=set_size(width=TEXT_WIDTH / 2))

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

    fig.savefig(op.join(output_dir, "dl_roc_auc_curve.pdf"), bbox_inches="tight")


def _melt_splits(df_input, metric):
    df = df_input.melt(
        id_vars=["epoch", "seed"],
        value_vars=[metric, "val_" + metric],
        var_name="split",
    )

    df["split"] = df["split"].map({metric: "training", "val_" + metric: "validation"})

    df.rename(columns={"value": metric}, inplace=True)

    return df


def visualize_loss_curves(log_dir, output_dir):
    log_paths = {
        "with_qc": op.join(log_dir, "with-qc-metrics", "*.csv"),
        "without_qc": op.join(log_dir, "without-qc-metrics", "*.csv"),
    }

    df_training = {
        qc_key: dd.read_csv(path, include_path_column=True).compute()
        for qc_key, path in log_paths.items()
    }

    for qc_key in log_paths.keys():
        df_training[qc_key]["seed"] = df_training[qc_key]["path"].apply(
            lambda s: int(s.split(".csv")[0].split("-")[-1])
        )

        df_training[qc_key].drop(
            ["path", "loss", "val_loss"], axis="columns", inplace=True
        )

    for (qc_key, training_df), letter, title in zip(
        df_training.items(), "ab", ["CNN-i+q", "CNN-i"]
    ):
        width, height = set_size(width=0.45 * TEXT_WIDTH, subplots=(3, 1))
        fig, axes = plt.subplots(3, 1, figsize=(width, height * 0.8), sharex=True)

        fig.tight_layout(h_pad=0)

        these_kwargs = TEXT_KWARGS.copy()
        these_kwargs.update(dict(x=-0.2, y=1.1))
        axes[0].text(
            s=letter,
            transform=axes[0].transAxes,
            **these_kwargs,
        )

        axes[0].set_title(title)

        metrics = [
            "binary_crossentropy",
            "accuracy",
            "auc",
        ]

        for ax, metric in zip(axes, metrics):
            df = _melt_splits(training_df, metric=metric)

            _ = sns.lineplot(
                data=df,
                x="epoch",
                y=metric,
                hue="split",
                ax=ax,
            )

            mean_val_metric = df[df["split"] == "validation"].groupby(["epoch"]).mean()
            if metric == "binary_crossentropy":
                best_val_metric = mean_val_metric.min()[0]
            else:
                best_val_metric = mean_val_metric.max()[0]

            ax.set_xlim(0)

            if metric in ["accuracy", "auc"]:
                ax.set_ylim(0, 1)

            ax.set_ylabel(ax.get_ylabel().replace("_", " "))

        axes[0].set_ylim(0.17, 1.27)
        axes[0].get_legend().remove()
        axes[2].get_legend().remove()

        fig.savefig(
            op.join(output_dir, f"dl_learning_curve_{qc_key}.pdf"), bbox_inches="tight"
        )


def visualize_model_architecture(saved_model_dir, output_dir):
    model = tf.keras.models.load_model(saved_model_dir)
    tf.keras.utils.plot_model(
        model,
        to_file=op.join(output_dir, "model.png"),
        show_shapes=True,
        show_layer_names=False,
        expand_nested=True,
    )


def attribution_diff(fn_pass, channel=None):
    pass_vol = nib.load(fn_pass)

    affine = pass_vol.affine
    diff = pass_vol.get_fdata()

    channel_idx = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    if channel in channel_idx.keys():
        diff = diff[:, :, :, channel_idx[channel]]
    if channel == "fa_norm":
        diff = np.linalg.norm(diff, axis=-1)
    if channel == "fa_sum":
        diff = np.sum(diff, axis=-1)

    return nib.Nifti1Image(diff, affine)


def get_bg_volume(fname, channel):
    nifti = nib.load(fname)
    affine = nifti.affine
    volume = nifti.get_fdata()

    channel_idx = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    if channel in channel_idx.keys():
        volume = volume[:, :, :, channel_idx[channel]]
    if channel == "fa_norm":
        volume = np.linalg.norm(volume, axis=-1)
    if channel == "fa_sum":
        volume = np.sum(volume, axis=-1)

    return nib.Nifti1Image(volume, affine)


def plot_bg(
    volumes,
    conf_class,
    channel="b0",
    subject_slice=slice(None),
    cut_coords=np.arange(46, 74, 6),
    display_mode="z",
    title=None,
):
    title = title or "_".join([conf_class, channel]).replace("_", " ")

    return [
        plotting.plot_anat(
            anat_img=channel_dict[channel],
            cut_coords=cut_coords,
            display_mode=display_mode,
            title=title,
        )
        for channel_dict in volumes[conf_class][subject_slice]
    ]


def plot_attribution(
    attribution_volumes,
    bg_volumes,
    conf_class,
    channel="b0",
    subject_idx=0,
    cut_coords=np.arange(46, 74, 6),
    display_mode="z",
    threshold=None,
    title=None,
    arcsinh_transform=False,
    cmap=None,
    axes=None,
    colorbar=True,
    annotate=True,
    dim="auto",
):
    if title == "auto":
        title = "_".join([conf_class, channel]).replace("_", " ")

    def transform(img):
        if arcsinh_transform:
            x = img.get_fdata()
            return nib.Nifti1Image(np.arcsinh(5e5 * x) / 5.5, img.affine)
        else:
            return img

    def get_threshold(img, percentile):
        if percentile is None:
            thresh = None
        else:
            thresh = np.nanpercentile(np.abs(img.get_fdata()), percentile)

        return thresh

    kwargs = dict(
        cut_coords=cut_coords,
        display_mode=display_mode,
        title=title,
        axes=axes,
        colorbar=colorbar,
        annotate=annotate,
        dim=dim,
    )

    if cmap is not None:
        kwargs["cmap"] = cmap

    ig_dict = attribution_volumes[conf_class][subject_idx]
    bg_dict = bg_volumes[conf_class][subject_idx]

    return plotting.plot_stat_map(
        stat_map_img=transform(ig_dict[channel]),
        bg_img=bg_dict.get(channel, bg_dict["b0"]),
        threshold=get_threshold(transform(ig_dict[channel]), threshold),
        **kwargs,
    )


def save_attribution_maps(nifti_dir, out_dir):
    confusion_classes = ["true_pos", "true_neg", "false_pos", "false_neg"]
    channel2label_map = {
        "b0": "b0",
        "x": "color_fa",
        "y": "color_fa",
        "z": "color_fa",
        "fa_norm": "color_fa",
        "fa_sum": "color_fa",
    }

    attribution_maps = {
        conf_class: [
            {
                channel: attribution_diff(
                    fn_pass=op.join(
                        nifti_dir, f"{conf_class}_attribution_pass_{label}_{idx}.nii.gz"
                    ),
                    channel=channel,
                )
                for channel, label in channel2label_map.items()
            }
            for idx in range(3)
        ]
        for conf_class in confusion_classes
    }

    bg_volumes = {
        conf_class: [
            {
                channel: get_bg_volume(
                    fname=op.join(nifti_dir, f"{conf_class}_{label}_{idx}.nii.gz"),
                    channel=channel,
                )
                for channel, label in channel2label_map.items()
            }
            for idx in range(3)
        ]
        for conf_class in confusion_classes
    }

    subject_indices = {
        "true_pos": 1,
        "true_neg": 1,
        "false_pos": 0,
        "false_neg": 0,
    }

    cut_coord = {
        "true_pos": [64],
        "true_neg": [64],
        "false_pos": [58],
        "false_neg": [58],
    }

    channels = {
        "b0": "B0",
        "x": "DTI FA_x",
        "y": "DTI FA_y",
        "z": "DTI FA_z",
    }

    for conf_class in confusion_classes:
        fig, ax_row = plt.subplots(1, 4, figsize=(18, 4.5))
        fig.tight_layout(h_pad=-2, w_pad=-2.75)

        for channel, ax in zip(channels.keys(), ax_row):
            _ = plot_attribution(
                attribution_maps,
                bg_volumes,
                conf_class=conf_class,
                channel=channel,
                display_mode="y" if channel == "z" else "z",
                threshold=98,
                arcsinh_transform=True,
                cmap="bwr_r",
                axes=ax,
                subject_idx=subject_indices[conf_class],
                cut_coords=cut_coord[conf_class],
                colorbar=False,
                title=None,
                annotate=True,
            )

        if conf_class == "true_pos":
            for ax, label in zip(ax_row, channels.values()):
                ax.set_title(label)

        fig.savefig(
            op.join(out_dir, f"attribution-maps-{conf_class.replace('_', '-')}.pdf"),
            bbox_inches="tight",
        )


def visualize_site_generalization(output_dir):
    participants = pd.read_csv(
        "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv",
        sep="\t",
    )
    participants["XGB pass?"] = participants["xgb_qc_score"] >= 0.5
    participants["Expert pass?"] = participants["expert_qc_score"] >= 0.5
    participants.rename(
        columns={
            "xgb_qc_score": "XGB rating",
            "expert_qc_score": "Expert rating",
            "dl_qc_score": "CNN-i rating",
            "scan_site_id": "Site",
        },
        inplace=True,
    )

    width, height = set_size(width=0.25 * TEXT_WIDTH)
    fig, axes = plt.subplots(3, 1, figsize=(width, height * 0.8), sharex=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    _ = sns.kdeplot(
        data=participants,
        x="XGB rating",
        y="CNN-i rating",
        hue="XGB pass?",
        cut=0.25,
        ax=ax,
    )
    sns.move_legend(ax, "lower right")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ratings = ["Expert rating", "XGB rating", "CNN-i rating"]
    hist_df = participants.drop(
        columns=[
            "sex",
            "age",
            "ehq_total",
            "commercial_use",
            "full_pheno",
            "xgb_qsiprep_qc_score",
            "XGB pass?",
            "Expert pass?",
        ]
    )
    hist_df = hist_df[hist_df["Site"] != "SI"].copy()
    hist_melt = hist_df.melt(
        id_vars=["subject_id", "Site"], var_name="type", value_name="rating"
    )
    bins = 12

    for x, ax in zip(ratings, axes[0]):
        _ = sns.histplot(
            data=hist_df,
            x=x,
            hue="Site",
            element="step",
            stat="probability",
            fill=False,
            common_norm=False,
            ax=ax,
            lw=2,
            bins=bins,
        )

    for site, ax in zip(["RU", "CBIC", "CUNY"], axes[1]):
        _ = sns.histplot(
            data=hist_melt[hist_melt["Site"] == site],
            x="rating",
            hue="type",
            element="step",
            stat="probability",
            fill=False,
            common_norm=False,
            ax=ax,
            lw=2,
            bins=bins,
        )

        ax.set_title(site)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "saved_model_dir",
        help="Directory containing the saved tensorflow model",
    )
    parser.add_argument(
        "training_log_dir",
        help="Directory containing the saved tensorflow training logs",
    )
    parser.add_argument(
        "report_set_dir",
        help="Directory containing the saved tensorflow report set predictions",
    )
    parser.add_argument(
        "nifti_dir",
        help="Directory for attribution map nifti files.",
    )
    parser.add_argument(
        "fig_dir",
        help="Directory for output figures.",
    )

    args = parser.parse_args()

    dl_fig_dir = op.join(args.fig_dir, "deep-learning-qc")

    # visualize_model_architecture(
    #     saved_model_dir=args.saved_model_dir,
    #     output_dir=dl_fig_dir,
    # )

    with plt.style.context("/tex.mplstyle"):
        # visualize_loss_curves(
        #     log_dir=args.training_log_dir,
        #     output_dir=dl_fig_dir,
        # )

        # visualize_auc_curves(
        #     report_set_dir=args.report_set_dir,
        #     output_dir=dl_fig_dir,
        # )

        visualize_site_generalization(output_dir=dl_fig_dir)

    # save_hbn_pod2_sankey(args.fig_dir)

    # with plt.style.context("/tex.mplstyle"):
    #     save_attribution_maps(
    #         nifti_dir=args.nifti_dir,
    #         out_dir=dl_fig_dir,
    #     )