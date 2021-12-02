#!/opt/conda/bin/python

import argparse
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import seaborn as sns

from glob import glob
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
    labels = ["DL: imaging + QC", "DL: imaging only"]

    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(8, 5))

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
            label=r"%s (AUC = %0.3f $\pm$ %0.3f)" % (label, mean_auc, std_auc),
            lw=2,
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
        [0, 1], [0, 1], linestyle="--", lw=2, color=colors[3], label="Chance", alpha=0.8
    )

    _ = ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    _ = ax.legend(loc="lower right", fontsize=12)
    _ = ax.set_xlabel("False Positive Rate", fontsize=12)
    _ = ax.set_ylabel("True Positive Rate", fontsize=12)
    _ = ax.set_title("Mean receiver operating characteristic (ROC)", fontsize=14)

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

    for qc_key, training_df in df_training.items():
        fig, axes = plt.subplots(3, 1, figsize=(8, 16))

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

            ax.axhline(
                best_val_metric, ls="--", color="black", label="best validation score"
            )

            ax.annotate(
                text=f"{best_val_metric:5.3f}",
                xy=(0, best_val_metric),
                xytext=(5, 5) if metric == "binary_crossentropy" else (5, -5),
                textcoords="offset points",
                ha="left",
                va="bottom" if metric == "binary_crossentropy" else "top",
            )

            ax.set_xlim(0)

            if metric in ["accuracy", "auc"]:
                ax.set_ylim(0, 1)

            ax.set_xlabel(ax.get_xlabel(), fontsize=14)
            ax.set_ylabel(ax.get_ylabel(), fontsize=14)

            ax.legend()

        fig.savefig(
            op.join(output_dir, f"dl_learning_curve_{qc_key}.png"), bbox_inches="tight"
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
        "fig_dir",
        help="Directory for output figures.",
    )

    args = parser.parse_args()

    visualize_model_architecture(
        saved_model_dir=args.saved_model_dir,
        output_dir=args.fig_dir,
    )

    visualize_loss_curves(
        log_dir=args.training_log_dir,
        output_dir=args.fig_dir,
    )

    visualize_auc_curves(
        report_set_dir=args.report_set_dir,
        output_dir=args.fig_dir,
    )